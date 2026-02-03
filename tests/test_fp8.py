import copy
import random
import time
import torch
import os

# 设置CUDA同步模式，确保能看到kernel的printf输出
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import deep_gemm
from deep_gemm.testing import (
    bench, bench_kineto,
    calc_diff, count_bytes
)

from generators import (
    KernelType, get_ue8m0_usage,
    enumerate_normal, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked, enumerate_k_grouped_contiguous,
    generate_normal, generate_m_grouped_contiguous, generate_m_grouped_masked, generate_k_grouped_contiguous,enumerate_128_layout_compatible, enumerate_128_layout_compatible_debug
)

def generate_random_fp4_as_int32(m, n, device='cuda'):
    """
    generate a m×(n/8) int32 matrix, which represents the FP4 data of m×n (Not E2M1, only 4-bit random values)
    each int32 value packs 8 FP4 values
    """
    assert n % 8 == 0, "n must be divisible by 8"
    
    # generate random FP4 values (0-15, 4-bit range)
    fp4_values = torch.randint(0, 16, (m, n), dtype=torch.uint8, device=device)
    
    # pack 8 FP4 values into one int32
    packed_n = n // 8
    packed_matrix = torch.zeros(m, packed_n, dtype=torch.int32, device=device)
    
    for i in range(8):
        packed_matrix += (fp4_values[:, i::8].to(torch.int32) << (i * 4))
    
    return packed_matrix, fp4_values


def fp4_e2m1_bits_to_float(bits_4bit):
    """
    将 4-bit 整数值（0-15）转换为真正的 E2M1 FP4 浮点数值
    
    E2M1 格式：
    - 4 bits: SEEM (S=符号 1bit, E=指数 2bits, M=尾数 1bit)
    - bias = 1
    
    数值计算：
    - 如果 E == 0 (subnormal): value = (-1)^S * 0.5 * M
    - 如果 E != 0 (normal):    value = (-1)^S * 2^(E-1) * (1 + M*0.5)
    
    所有 16 种可能值的映射表：
    bits | S | E | M | 数值
    0000 | 0 | 0 | 0 |  0.0
    0001 | 0 | 0 | 1 |  0.5
    0010 | 0 | 1 | 0 |  1.0
    0011 | 0 | 1 | 1 |  1.5
    0100 | 0 | 2 | 0 |  2.0
    0101 | 0 | 2 | 1 |  3.0
    0110 | 0 | 3 | 0 |  4.0
    0111 | 0 | 3 | 1 |  6.0
    1000 | 1 | 0 | 0 | -0.0
    1001 | 1 | 0 | 1 | -0.5
    1010 | 1 | 1 | 0 | -1.0
    1011 | 1 | 1 | 1 | -1.5
    1100 | 1 | 2 | 0 | -2.0
    1101 | 1 | 2 | 1 | -3.0
    1110 | 1 | 3 | 0 | -4.0
    1111 | 1 | 3 | 1 | -6.0
    
    Args:
        bits_4bit: int 或 torch.Tensor，4-bit 值（0-15）
    
    Returns:
        对应的 E2M1 FP4 浮点数值
    """
    # 查找表：直接映射 4-bit 值到浮点数
    E2M1_LUT = torch.tensor([
         0.0,   0.5,   1.0,   1.5,   2.0,   3.0,   4.0,   6.0,  # 正数 (S=0)
        -0.0,  -0.5,  -1.0,  -1.5,  -2.0,  -3.0,  -4.0,  -6.0   # 负数 (S=1)
    ], dtype=torch.float32)
    
    if isinstance(bits_4bit, torch.Tensor):
        # 确保输入在合法范围内
        bits_4bit = bits_4bit.long() & 0xF
        return E2M1_LUT[bits_4bit]
    else:
        # 标量情况
        return E2M1_LUT[int(bits_4bit) & 0xF].item()


def fp4_e2m1_gemm_reference(a_packed, b_packed, m, n, k_packed, debug=False):
    """
    真正的 E2M1 FP4 GEMM Reference：
    1. 解包 int32 中的 4-bit 值
    2. 将 4-bit 值转换为真正的 E2M1 FP4 浮点数
    3. 进行矩阵乘法
    
    计算 C[m,n] = A[m,k] × B^T[n,k]
    
    Args:
        a_packed: [m, k_packed] int32 tensor, 每个int32包含8个FP4
        b_packed: [n, k_packed] int32 tensor, 每个int32包含8个FP4
        m, n: 输出矩阵维度
        k_packed: K维度（int32单位），实际K = k_packed × 8
        debug: 是否打印调试信息
    
    Returns:
        c_result: [m, n] float32 tensor
    """
    print(f"    [E2M1 FP4 GEMM] 解包 4-bit 值并转换为真正的 E2M1 浮点数...")
    
    # 将数据移到 CPU 进行计算
    a_cpu = a_packed.cpu()
    b_cpu = b_packed.cpu()
    
    # E2M1 查找表
    E2M1_LUT = torch.tensor([
         0.0,   0.5,   1.0,   1.5,   2.0,   3.0,   4.0,   6.0,  # 正数 (S=0)
        -0.0,  -0.5,  -1.0,  -1.5,  -2.0,  -3.0,  -4.0,  -6.0   # 负数 (S=1)
    ], dtype=torch.float32)
    
    def unpack_and_convert_to_e2m1(packed_tensor):
        """
        解包 int32 为 8 个 4-bit 值，并转换为 E2M1 浮点数
        [batch, k_packed] int32 -> [batch, k_packed*8] float32
        """
        shape = packed_tensor.shape
        # 转换为 int64 避免符号问题
        packed_int64 = packed_tensor.to(torch.int64) & 0xFFFFFFFF
        
        # 解包 8 个 FP4 值
        unpacked_bits = []
        for i in range(8):
            fp4_bits = (packed_int64 >> (i * 4)) & 0xF
            unpacked_bits.append(fp4_bits)
        
        # 堆叠成 [batch, k_packed, 8] 然后重塑为 [batch, k_packed*8]
        unpacked_bits = torch.stack(unpacked_bits, dim=-1)  # [batch, k_packed, 8]
        unpacked_bits = unpacked_bits.reshape(*shape[:-1], -1)  # [batch, k_packed*8]
        
        # 使用查找表转换为 E2M1 浮点数
        e2m1_values = E2M1_LUT[unpacked_bits.long()]
        
        return e2m1_values
    
    print(f"      解包并转换 A 矩阵...")
    a_e2m1 = unpack_and_convert_to_e2m1(a_cpu)  # [m, k_packed*8]
    print(f"      解包并转换 B 矩阵...")
    b_e2m1 = unpack_and_convert_to_e2m1(b_cpu)  # [n, k_packed*8]
    
    if debug:
        print(f"\n=== E2M1 FP4 GEMM Debug: C[0][1] 计算详情 ===")
        print(f"  A[0] (第0行前16个E2M1值): {a_e2m1[0, :16].tolist()}")
        print(f"  B[1] (第1行前16个E2M1值): {b_e2m1[1, :16].tolist()}")
        
        # 详细计算 C[0][1]
        a_row = a_e2m1[0]  # A 的第 0 行
        b_row = b_e2m1[1]  # B 的第 1 行
        
        print(f"\n  逐元素乘积 (前16个):")
        for i in range(min(16, a_row.shape[0])):
            prod = a_row[i] * b_row[i]
            print(f"    k={i:2d}: A={a_row[i]:6.2f} × B={b_row[i]:6.2f} = {prod:8.2f}")
        
        c_01 = (a_row * b_row).sum()
        print(f"\n  C[0][1] = sum(A[0] * B[1]) = {c_01.item():.4f}")
    
    # 矩阵乘法: C = A @ B^T
    print(f"      执行矩阵乘法...")
    c_result = torch.matmul(a_e2m1, b_e2m1.T)  # [m, n]
    
    print(f"    [E2M1 FP4 GEMM] 计算完成！")
    print(f"    结果形状: {c_result.shape}")
    print(f"    结果前 4x4:\n{c_result[:4, :4]}")
    
    return c_result



def simple_fp4_gemm_reference_packed(a_packed, b_packed, m, n, k_packed):
    """
    简化版FP4 GEMM Reference：直接对int32做乘法（不解包FP4，向量化版本）
    
    计算 C[m,n] = A[m,k] × B^T[n,k]
    
    Args:
        a_packed: [m, k_packed] int32 tensor
        b_packed: [n, k_packed] int32 tensor  
        m, n: 输出矩阵维度
        k_packed: K维度（int32单位）
    
    Returns:
        c_result: [m, n] float tensor
    """
    print(f"    使用向量化计算（直接int32相乘）...")
    
    # 将数据移到CPU并转换为float进行计算
    a_cpu = a_packed.cpu().to(torch.float32)  # [m, k_packed]
    b_cpu = b_packed.cpu().to(torch.float32)  # [n, k_packed]
    
    # 使用矩阵乘法: C = A @ B^T
    c_result = torch.matmul(a_cpu, b_cpu.T)  # [m, n]
    
    print(f"    计算完成！")
    return c_result


def fp4_gemm_reference_unpacked(a_packed, b_packed, m, n, k_packed):
    """
    完整版FP4 GEMM Reference：解包FP4后计算真正的矩阵乘法（向量化版本）
    
    计算 C[m,n] = A[m,k] × B^T[n,k]
    
    Args:
        a_packed: [m, k_packed] int32 tensor, 每个int32包含8个FP4
        b_packed: [n, k_packed] int32 tensor, 每个int32包含8个FP4
        m, n: 输出矩阵维度
        k_packed: K维度（int32单位），实际K = k_packed × 8
    
    Returns:
        c_result: [m, n] float tensor
    """
    print(f"    解包FP4并使用向量化计算...")
    
    # 将数据移到CPU进行计算
    a_cpu = a_packed.cpu()
    b_cpu = b_packed.cpu()
    
    # 向量化解包函数：将 [m, k_packed] int32 解包为 [m, k_packed*8] uint8
    def unpack_fp4_vectorized(packed_tensor):
        """
        向量化解包：[..., k_packed] int32 -> [..., k_packed*8] uint8
        每个int32包含8个4-bit值
        """
        shape = packed_tensor.shape
        # 转换为uint32避免符号问题
        packed_uint32 = packed_tensor.to(torch.int64) & 0xFFFFFFFF
        
        # 解包8个FP4值
        unpacked = []
        for i in range(8):
            # 提取第i个4-bit值
            fp4_val = (packed_uint32 >> (i * 4)) & 0xF
            unpacked.append(fp4_val)
        
        # 堆叠成 [..., k_packed, 8] 然后重塑为 [..., k_packed*8]
        unpacked = torch.stack(unpacked, dim=-1)  # [..., k_packed, 8]
        unpacked = unpacked.reshape(*shape[:-1], -1)  # [..., k_packed*8]
        
        return unpacked.to(torch.float32)
    
    print(f"      解包A矩阵...")
    a_unpacked = unpack_fp4_vectorized(a_cpu)  # [m, k_packed*8]
    print(f"      解包B矩阵...")
    b_unpacked = unpack_fp4_vectorized(b_cpu)  # [n, k_packed*8]
    
    # 调试：详细输出C[0,1]的计算过程（前16个FP4元素）
    print(f"\n=== PYTHON: Computing C[0][1], showing first 16 FP4 elements ===")
    print(f"  (使用 A的第0行 × B的第1行)")
    
    # 打印前2个k_packed的打包值（C[0,1]使用A的第0行和B的第1行）
    for kp_idx in range(min(2, k_packed)):
        a_val = a_cpu[0, kp_idx].item()  # A的第0行
        b_val = b_cpu[1, kp_idx].item()  # B的第1行
        # 转换为无符号
        a_val_u = a_val if a_val >= 0 else a_val + 2**32
        b_val_u = b_val if b_val >= 0 else b_val + 2**32
        print(f"k_packed={kp_idx}: a_packed=0x{a_val_u:08x}, b_packed=0x{b_val_u:08x}")
    
    # 详细输出前16个FP4元素的计算过程
    acc = 0.0
    for k in range(16):  # 前16个FP4元素（2个int32 × 8个FP4）
        a_fp4 = a_unpacked[0, k].item()  # A的第0行
        b_fp4 = b_unpacked[1, k].item()  # B的第1行
        product = a_fp4 * b_fp4
        acc_before = acc
        acc += product
        print(f"  k={k:2d}: a_fp4={int(a_fp4):2d}, b_fp4={int(b_fp4):2d}, product={int(product):3d}, acc_before={acc_before:.1f}, acc_after={acc:.1f}")
    
    print(f"Partial sum (first 16 elements): {acc:.2f}")
    print(f"Total K elements: {k_packed * 8} FP4")
    
    # 使用矩阵乘法: C = A @ B^T
    print(f"      计算矩阵乘法...")
    c_result = torch.matmul(a_unpacked, b_unpacked.T)  # [m, n]
    
    print(f"Full C[0,1] result: {c_result[0,1].item():.2f}")
    print(f"    计算完成！")
    return c_result

def test_gemm() -> None:
    print('Testing GEMM:')
    for kernel_type, m, n, k, major_a, major_b, accumulate, out_dtype in enumerate_128_layout_compatible_debug():
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        # for test_alias in (False, True):
        #     a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, use_ue8m0=use_ue8m0)
        #     func_name = f'fp8_gemm_{major_opt.lower() if test_alias else "nt"}'
        #     if test_alias:
        #         a = a if major_a.is_k_major() else (a[0].T, a[1].T)
        #         b = b if major_b.is_k_major() else (b[0].T, b[1].T)
        #         assert a[0].is_contiguous() and b[0].is_contiguous()
        #     getattr(deep_gemm, func_name)(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast)
        #     diff = calc_diff(d, ref_d)
        #     assert diff < 0.001, (f'{m=}, {n=}, {k=}, {kernel_opt}, {major_opt=}, {accumulate=}, {out_dtype=}, '
        #                           f'{diff:.5f}, alias={test_alias}')
        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, use_ue8m0=use_ue8m0)

        # 先不移除原来的fp8数据和量化部分，仅替换a和b的矩阵类型。
        # m*n -> m*(n/8)
        a_packed, a_fp4_raw = generate_random_fp4_as_int32(m, k)  # a (m, k) -> (m, k/8)
        b_packed, b_fp4_raw = generate_random_fp4_as_int32(n, k)  # b (n, k) -> (n, k/8)
        a = (a_packed, a[1])
        b = (b_packed, b[1])
        
        # 计算Host端参考结果（简单验证方法避免溢出）
        k_packed = k // 8  # 打包后的K维度
        sum_result, xor_result = simple_data_verification_host(a_packed, b_packed, m, n, k_packed)
        print(f"HOST_DEBUG: Verification results shape: sum={sum_result.shape}, xor={xor_result.shape}")
        print(f"HOST_DEBUG: First 4x4 elements (SUM method):")
        for i in range(min(4, m)):
            for j in range(min(4, n)):
                print(f"SUM[{i}][{j}] = {sum_result[i, j].item()}")
        print(f"HOST_DEBUG: First 4x4 elements (XOR method):")
        for i in range(min(4, m)):
            for j in range(min(4, n)):
                # 将int32转换为uint32显示，避免负数显示
                xor_val = xor_result[i, j].item()
                if xor_val < 0:
                    xor_val = xor_val + 2**32  # 转换为对应的uint32值
                print(f"XOR[{i}][{j}] = {xor_val}")

        print(a[0].shape, b[0].shape)
        print(f"sf_a={a[1].shape}, sf_b={b[1].shape}")
        print(f"M={m}, N={n}, K={k}, K_packed={k_packed}")
        
        # ========== 调试输出：A矩阵左上角4x4 ==========
        print("HOST_DEBUG: A Matrix debug start")
        print(f"HOST_DEBUG: A tensor shape: {a[0].shape}, dtype: {a[0].dtype}")
        print(f"HOST_DEBUG: Logical K: {k}, Physical K: {a[0].shape[1]} (packed)")
        
        # 只输出前4个int32元素，与kernel端保持一致
        print("HOST_DEBUG: First 4 int32 elements:")
        linear_idx = 0
        for row in range(a[0].shape[0]):
            for col in range(a[0].shape[1]):
                if linear_idx < 4:
                    packed_val = a[0][row, col].item()
                    unsigned_val = packed_val & 0xFFFFFFFF
                    print(f"HOST_DEBUG: [{linear_idx}] = 0x{unsigned_val:08x}")
                    linear_idx += 1
                else:
                    break
            if linear_idx >= 4:
                break
        
        print("HOST_DEBUG: A Matrix debug end")
        print()

        # Test launch overhead
        launch_start_t = time.time_ns()
        deep_gemm.fp8_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast)
        launch_end_t = time.time_ns()
        torch.cuda.synchronize()

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.fp8_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf (m={m:5}, n={n:5}, k={k:5}, {kernel_opt}, layout={major_opt}, {out_opt}, {acc_opt}): '
              f'launch {(launch_end_t - launch_start_t) / 1e3:4.0f} us | {t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s')
    print()

def test_gemm_single_tile():
    """
    测试单个tile的FP4 GEMM
    Step 1: 限制为单CTA单Tile（128×256×128）
    Step 2: 计算Python Reference
    Step 5: 运行kernel并对比结果
    """
    print('='*80)
    print('Testing Single Tile FP4 GEMM')
    print('='*80)
    
    # ========== Step 1: 限制为单CTA单Tile ==========
    # 固定为一个tile的大小（根据实际的kernel配置）
    # 实际配置：block_k = 128字节 / 4字节(int32) = 32个int32
    # m = 128          # BLOCK_M (输出矩阵C的M维度)
    # ========== 单CTA单块测试配置 ==========
    # 完全匹配一个CTA块，排除多CTA问题
    # m = 128          # BLOCK_M (一个CTA块)
    # n = 16           # BLOCK_N (一个CTA块)
    # k = 256          # BLOCK_K × 8 = 32 × 8 (一次K迭代)
    # k_packed = 32    # BLOCK_K (int32单位)
    
    # ========== 多CTA测试配置（注释掉）==========
    m = 256          # 2个CTA块 (M维度)
    n = 256           # 2个CTA块 (N维度)  
    k = 512          # 多次K迭代
    k_packed = 64 

    print(f"\n[Step 1] 生成单Tile数据")
    print(f"  矩阵维度: M={m}, N={n}, K={k} (FP4元素)")
    print(f"  打包维度: M={m}, N={n}, K_packed={k_packed} (int32)")
    print(f"  期望输出: C[{m}, {n}]")
    print(f"  注意：这是根据实际kernel配置调整的tile大小")
    
    # 使用与test_gemm相同的方式生成数据
    # 首先用generate_normal生成正确格式的scaling factors和输出张量
    from generators import KernelType, MajorTypeAB
    kernel_type = KernelType.Kernel1D1D
    major_a = MajorTypeAB.KMajor
    major_b = MajorTypeAB.KMajor
    accumulate = False
    out_dtype = torch.float32
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    disable_ue8m0_cast = not use_ue8m0

    # 生成原始FP8数据（只是为了获取正确格式的scaling factors和tensors）
    a_orig, b_orig, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, use_ue8m0=use_ue8m0)
    
    # 生成随机FP4数据（打包成int32）
    device = 'cuda'
    # A矩阵: [M, K] FP4 → [M, K/8] int32
    # B矩阵: [N, K] FP4 → [N, K/8] int32（K-major格式）
    a_packed, a_fp4_raw = generate_random_fp4_as_int32(m, k, device=device)      # [256, 64] int32
    b_packed, b_fp4_raw = generate_random_fp4_as_int32(n, k, device=device)      # [32, 64] int32
    
    # 替换数据部分，保留scaling factors（与test_gemm一致）
    a = (a_packed, a_orig[1])  # 使用原来的scaling factor
    b = (b_packed, b_orig[1])  # 使用原来的scaling factor
    
    print(f"  A_packed shape: {a_packed.shape}, dtype: {a_packed.dtype}")
    print(f"  B_packed shape: {b_packed.shape}, dtype: {b_packed.dtype}")
    print(f"  Scaling factors: sf_a shape={a[1].shape}, dtype={a[1].dtype}")
    print(f"  Scaling factors: sf_b shape={b[1].shape}, dtype={b[1].dtype}")
    print(f"  Output tensor D: shape={d.shape}, dtype={d.dtype}")
    
    # ========== 调试输出：A和B矩阵的前4个int32元素 ==========
    print(f"\n[Debug] 矩阵数据预览 (前4个int32元素):")
    print(f"  A_packed (first 4 int32):")
    for i in range(min(4, a_packed.numel())):
        row = i // k_packed
        col = i % k_packed
        val = a_packed[row, col].item()
        unsigned_val = val if val >= 0 else val + 2**32
        print(f"    [{i}] A[{row},{col}] = 0x{unsigned_val:08x} ({val})")
    
    print(f"  B_packed (first 4 int32):")
    for i in range(min(4, b_packed.numel())):
        row = i // k_packed
        col = i % k_packed
        val = b_packed[row, col].item()
        unsigned_val = val if val >= 0 else val + 2**32
        print(f"    [{i}] B[{row},{col}] = 0x{unsigned_val:08x} ({val})")
    
    # 输出B[1]的前4个int32元素（用于和kernel对比）
    print(f"  B[1] (first 4 int32):")
    for col in range(min(4, k_packed)):
        val = b_packed[1, col].item()
        unsigned_val = val if val >= 0 else val + 2**32
        print(f"    B[1][{col}] = 0x{unsigned_val:08x} ({val})")
    
    # ========== Step 2: 计算Python Reference ==========
    print(f"\n[Step 2] 计算Python端Reference")
    
    
    print(f"  简单验证结果（前4x4）:")
    for i in range(min(4, m)):
        for j in range(min(4, n)):
            sum_val = sum_result[i, j].item()
            xor_val = xor_result[i, j].item()
            xor_unsigned = xor_val if xor_val >= 0 else xor_val + 2**32
            print(f"    [{i},{j}] SUM={sum_val:12d}, XOR=0x{xor_unsigned:08x}")
    
    # 方法2：真正的GEMM reference（解包FP4版本）
    print(f"\n  计算FP4 GEMM Reference (unpacked)...")
    gemm_ref_unpacked = fp4_gemm_reference_unpacked(a_packed, b_packed, m, n, k_packed)
    
    print(f"  GEMM Reference结果（前4x4）:")
    for i in range(min(4, m)):
        for j in range(min(4, n)):
            val = gemm_ref_unpacked[i, j].item()
            print(f"    GEMM_REF[{i},{j}] = {val:12.2f}")
    
    # 方法3：简化的GEMM reference（直接int32相乘，备用）
    # print(f"\n  计算简化GEMM Reference (packed)...")
    # gemm_ref_packed = simple_fp4_gemm_reference_packed(a_packed, b_packed, m, n, k_packed)
    
    # ========== Step 5: 运行Kernel并对比结果 ==========
    print(f"\n[Step 5] 运行Kernel")
    
    # kernel输入已经在上面准备好了：a = (a_packed, a_orig[1]), b = (b_packed, b_orig[1])
    
    # 调用kernel（与test_gemm保持一致）
    print(f"  调用 deep_gemm.fp8_gemm_nt()...")
    try:
        launch_start_t = time.time_ns()
        deep_gemm.fp8_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast)
        launch_end_t = time.time_ns()
        torch.cuda.synchronize()
        
        launch_time_us = (launch_end_t - launch_start_t) / 1e3
        print(f"  Kernel完成！Launch time: {launch_time_us:.0f} us")
    except Exception as e:
        print(f"  Kernel调用失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== 对比结果 ==========
    print(f"\n[Result Comparison] 对比Kernel输出与Reference")
    
    # 将结果移到CPU
    d_cpu = d.cpu()
    
    # 检查前4x4元素
    print(f"\n  前4x4元素对比:")
    print(f"  {'[i,j]':<8} {'GPU_Output':<15} {'GEMM_Ref':<15} {'Diff':<15} {'Status'}")
    print(f"  {'-'*70}")
    
    max_diff = 0.0
    num_errors = 0
    
    for i in range(min(4, m)):
        for j in range(min(4, n)):
            gpu_val = d_cpu[i, j].item()
            ref_val = gemm_ref_unpacked[i, j].item()
            diff = abs(gpu_val - ref_val)
            max_diff = max(max_diff, diff)
            
            # 检查是否匹配（允许小误差）
            is_match = diff < 1e-3 or (ref_val != 0 and diff / abs(ref_val) < 0.01)
            status = "✓ OK" if is_match else "✗ FAIL"
            if not is_match:
                num_errors += 1
            
            print(f"  [{i},{j}]   {gpu_val:12.2f}    {ref_val:12.2f}    {diff:12.2f}    {status}")
    
    # 统计全矩阵的差异
    print(f"\n  全矩阵统计:")
    all_diff = torch.abs(d_cpu - gemm_ref_unpacked)
    mean_diff = all_diff.mean().item()
    max_diff_full = all_diff.max().item()
    
    print(f"    平均差异: {mean_diff:.6f}")
    print(f"    最大差异: {max_diff_full:.6f}")
    print(f"    前4x4错误数: {num_errors}/16")
    
    # ========== 按CTA区域打印C矩阵 ==========
    print(f"\n[按CTA区域打印C矩阵 {m}×{n}]")
    block_m = 128  # 从config获取
    block_n = 16
    num_m_blocks = (m + block_m - 1) // block_m
    num_n_blocks = (n + block_n - 1) // block_n
    
    for m_block in range(num_m_blocks):
        for n_block in range(num_n_blocks):
            m_start = m_block * block_m
            m_end = min((m_block + 1) * block_m, m)
            n_start = n_block * block_n
            n_end = min((n_block + 1) * block_n, n)
            
            cta_idx = m_block * num_n_blocks + n_block
            print(f"\n=== CTA{cta_idx}: C[{m_start}:{m_end}, {n_start}:{n_end}] (m_block={m_block}, n_block={n_block}) ===")
            
            # 提取这个tile
            tile_gpu = d_cpu[m_start:m_end, n_start:n_end]
            tile_ref = gemm_ref_unpacked[m_start:m_end, n_start:n_end]
            tile_diff = torch.abs(tile_gpu - tile_ref)
            
            # 统计信息
            print(f"  GPU: min={tile_gpu.min():.4f}, max={tile_gpu.max():.4f}, mean={tile_gpu.mean():.4f}")
            print(f"  Ref: min={tile_ref.min():.4f}, max={tile_ref.max():.4f}, mean={tile_ref.mean():.4f}")
            print(f"  Diff: max={tile_diff.max():.6f}, mean={tile_diff.mean():.6f}")
            
            if tile_diff.max() > 0.001:
            # 打印前4行和最后4行的对比
                print(f"  前4行×全部{n_end-n_start}列 (GPU vs Ref):")
                for row in range(min(4, tile_gpu.shape[0])):
                    gpu_str = " ".join([f"{tile_gpu[row, col]:8.1f}" for col in range(tile_gpu.shape[1])])
                    ref_str = " ".join([f"{tile_ref[row, col]:8.1f}" for col in range(tile_ref.shape[1])])
                    match = "✓" if torch.allclose(tile_gpu[row], tile_ref[row], atol=1.0) else "✗"
                    print(f"    [{row:3d}] GPU: {gpu_str} {match}")
                    print(f"          Ref: {ref_str}")
                
                # 打印最后4行（检查M维度末尾）
                if tile_gpu.shape[0] > 8:
                    print(f"  后4行×全部{n_end-n_start}列 (GPU vs Ref):")
                    for row in range(max(0, tile_gpu.shape[0]-4), tile_gpu.shape[0]):
                        gpu_str = " ".join([f"{tile_gpu[row, col]:8.1f}" for col in range(tile_gpu.shape[1])])
                        ref_str = " ".join([f"{tile_ref[row, col]:8.1f}" for col in range(tile_ref.shape[1])])
                        match = "✓" if torch.allclose(tile_gpu[row], tile_ref[row], atol=1.0) else "✗"
                        print(f"    [{row:3d}] GPU: {gpu_str} {match}")
                        print(f"          Ref: {ref_str}")
    
    # ========== 关键位置采样 ==========
    print(f"\n[关键位置采样验证]")
    # 根据当前测试维度动态生成采样点
    sample_points = [
        (0, 0, "左上角"),
        (0, min(n-1, 15), "右上角"),
        (min(m-1, 127), 0, "左下角"),
        (min(m-1, 127), min(n-1, 15), "右下角"),
        (m//2, n//2, "中心点"),
    ]
    # 如果有多个CTA，添加更多采样点
    if m > 128:
        sample_points.extend([
            (128, 0, "CTA2左上角"),
            (min(m-1, 255), min(n-1, 15), "CTA2右下角"),
        ])
    if n > 16:
        sample_points.extend([
            (0, 16, "CTA1左上角"),
            (min(m-1, 127), min(n-1, 31), "CTA1右下角"),
        ])
    
    print(f"  {'Position':<12} {'Description':<15} {'GPU':<12} {'Ref':<12} {'Diff':<12} {'Status'}")
    print(f"  {'-'*75}")
    
    for i, j, desc in sample_points:
        if i < m and j < n:
            gpu_val = d_cpu[i, j].item()
            ref_val = gemm_ref_unpacked[i, j].item()
            diff = abs(gpu_val - ref_val)
            is_ok = diff < 1.0
            status = "✓" if is_ok else "✗"
            print(f"  [{i:3d},{j:2d}]   {desc:<15} {gpu_val:10.2f}  {ref_val:10.2f}  {diff:10.6f}  {status}")
    
    # 检查kernel是否有printf输出
    print(f"\n[Kernel Debug Output]")
    print(f"  查看上方是否有 'KERNEL_DEBUG:' 或 'KERNEL_VERIFICATION:' 输出")
    print(f"  如果没有，说明kernel内部的printf可能未执行或被优化掉")
    
    # 最终判断
    print(f"\n" + "="*80)
    if max_diff_full < 1.0:
        print(f"✓ 测试可能通过（最大差异 {max_diff_full:.2f} < 1.0）")
    else:
        print(f"✗ 测试失败（最大差异 {max_diff_full:.2f} >= 1.0）")
    print(f"="*80)


def generate_mxf4_scale_factors(m, n, k, block_k=32, device='cuda'):
    """
    为 MXF4 生成正确格式的 Scale Factor
    
    MXF4 使用 UE8M0 格式的 SF：
    - UE8M0: 8位无符号指数，bias=127
    - 值 = 2^(exp - 127)
    - exp=127 表示 2^0 = 1（不缩放）
    
    SF 布局（与原 FP8 kernel 兼容）：
    - SFA: [m, sf_k] float32，per-token scaling
    - SFB: [sf_n, sf_k] float32，per-block scaling
    
    为简化测试，生成全 1.0 的 SF（不影响计算）
    """
    from deep_gemm.utils import ceil_div
    
    # SF 的 K 维度计算
    # kNumSFStagesPerLoad = 4 (sizeof(uint32_t) / sizeof(float_ue8m0_t))
    kNumSFStagesPerLoad = 4
    sf_k = ceil_div(k, block_k * kNumSFStagesPerLoad)
    
    # SF 的 M/N 维度（per-token for A, per-128-block for B）
    sf_n = ceil_div(n, 128)
    
    # 生成全 1.0 的 SF（不缩放）
    sf_a = torch.ones((m, sf_k), dtype=torch.float32, device=device)
    sf_b = torch.ones((sf_n, sf_k), dtype=torch.float32, device=device)
    
    print(f"  [MXF4 SF] Generated scale factors:")
    print(f"    sf_a shape: {sf_a.shape} (per-token)")
    print(f"    sf_b shape: {sf_b.shape} (per-block)")
    print(f"    sf_k = ceil_div({k}, {block_k} * {kNumSFStagesPerLoad}) = {sf_k}")
    
    return sf_a, sf_b


def test_fp4_e2m1_gemm():
    """
    简洁测试：对比 kernel 输出与 E2M1 FP4 GEMM reference
    """
    print('='*60)
    print('Test: FP4 E2M1 GEMM vs Kernel')
    print('='*60)
    
    # 配置
    m, n, k = 256, 256, 512
    k_packed = k // 8
    block_k = 32  # kernel 中的 BLOCK_K（int32 单位）
    
    # 数据生成
    from generators import KernelType, MajorTypeAB
    kernel_type = KernelType.Kernel1D1D
    major_a, major_b = MajorTypeAB.KMajor, MajorTypeAB.KMajor
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    disable_ue8m0_cast = not use_ue8m0
    
    # 生成 FP4 打包数据
    a_packed, _ = generate_random_fp4_as_int32(m, k, device='cuda')
    b_packed, _ = generate_random_fp4_as_int32(n, k, device='cuda')
    
    # 生成 MXF4 格式的 Scale Factor（全 1.0，不影响计算）
    sf_a, sf_b = generate_mxf4_scale_factors(m, n, k, block_k=block_k, device='cuda')
    
    # 组装输入元组
    a = (a_packed, sf_a)
    b = (b_packed, sf_b)
    
    # 创建输出张量
    d = torch.empty((m, n), device='cuda', dtype=torch.float32)
    c = None  # 不累加
    
    print(f"Config: M={m}, N={n}, K={k}, K_packed={k_packed}")
    
    # 计算 E2M1 FP4 reference
    print("\n[1] Computing E2M1 FP4 reference...")
    ref_e2m1 = fp4_e2m1_gemm_reference(a_packed, b_packed, m, n, k_packed, debug=False)
    
    # 调用 kernel
    print("\n[2] Running kernel...")
    try:
        deep_gemm.fp8_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast)
        torch.cuda.synchronize()
        print("Kernel completed.")
    except Exception as e:
        print(f"Kernel failed: {e}")
        return
    
    # 对比结果
    print("\n[3] Comparing results...")
    d_cpu = d.cpu().float()
    ref_e2m1_cpu = ref_e2m1.cpu().float()
    
    diff = torch.abs(d_cpu - ref_e2m1_cpu)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"  Max  diff: {max_diff:.4f}")
    print(f"  Mean diff: {mean_diff:.4f}")
    
    # 显示前 4x4 对比
    print(f"\n  First 4x4 comparison:")
    print(f"  {'[i,j]':<6} {'Kernel':<12} {'E2M1_Ref':<12} {'Diff':<12}")
    for i in range(min(4, m)):
        for j in range(min(4, n)):
            k_val = d_cpu[i, j].item()
            r_val = ref_e2m1_cpu[i, j].item()
            d_val = abs(k_val - r_val)
            print(f"  [{i},{j}]  {k_val:<12.2f} {r_val:<12.2f} {d_val:<12.4f}")
    
    # 判断是否通过
    print(f"\n{'='*60}")
    if max_diff < 1.0:
        print(f"✓ PASS (max_diff={max_diff:.4f} < 1.0)")
    else:
        print(f"✗ FAIL (max_diff={max_diff:.4f} >= 1.0)")
    print('='*60)


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    # 测试 E2M1 FP4 GEMM（新增简洁版）
    test_fp4_e2m1_gemm()
    
    # 测试单个tile的FP4 GEMM（详细版）
    # test_gemm_single_tile()
    
    # 原来的测试（暂时注释）
    # test_gemm()
    # test_m_grouped_gemm_contiguous()
    # test_m_grouped_gemm_masked()
    # test_k_grouped_gemm_contiguous()
