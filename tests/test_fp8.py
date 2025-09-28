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

def simple_data_verification_host(a_packed, b_packed, m, n, k_packed):
    """简单的数据验证：计算每个位置的和与异或，避免溢出"""
    # 方法1：计算每个矩阵元素的和（模运算避免溢出）
    sum_result = torch.zeros(m, n, dtype=torch.int64)
    
    # 方法2：计算每个矩阵元素的异或（不会溢出）
    xor_result = torch.zeros(m, n, dtype=torch.int32)
    
    for i in range(m):
        for j in range(n):
            sum_val = 0
            xor_val = 0
            
            for k in range(k_packed):
                a_val = a_packed[i, k].item()
                b_val = b_packed[j, k].item()
                
                # 计算和（模2^32避免溢出）
                sum_val = (sum_val + a_val + b_val) % (2**32)
                
                # 计算异或
                xor_val ^= (a_val ^ b_val)
            
            sum_result[i, j] = sum_val
            xor_result[i, j] = xor_val
    
    return sum_result, xor_result

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


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing m-grouped contiguous GEMM:')

    for kernel_type, num_groups, expected_m_per_group, n, k, major_a, major_b in enumerate_m_grouped_contiguous():
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        for test_alias in (False, True):
            m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b, use_ue8m0=use_ue8m0)
            func_name = f"m_grouped_fp8_gemm_{(major_opt.lower() if test_alias else 'nt')}_contiguous"
            if test_alias:
                assert major_a.is_k_major()
                b = b if major_b.is_k_major() else (b[0].mT, b[1].mT)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(a, b, d, m_indices, disable_ue8m0_cast=disable_ue8m0_cast)
            d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
            diff = calc_diff(d, ref_d)
            assert diff < 0.001, f'{m=}, {n=}, {k=}, {major_opt}, {kernel_opt}, {diff:.5f}, alias={test_alias}'
        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b, use_ue8m0=use_ue8m0)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, m_indices, disable_ue8m0_cast=disable_ue8m0_cast)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}, {kernel_opt}, layout={major_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for kernel_type, num_groups, max_m, expected_m_per_group, n, k in enumerate_m_grouped_masked():
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        # Test correctness
        for i in range(10):
            a, b, masked_m, d, ref_d = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0)
            deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast)
            for j in range(num_groups):
                diff = calc_diff(d[j, :masked_m[j].item()], ref_d[j, :masked_m[j].item()])
                assert diff < 0.001, f'{max_m=}, {n=}, {k=}, {j=}, masked_m={masked_m[j]}, {kernel_opt}, {num_groups=}, {diff:.5f}'

        # Construct full cases
        a, b, masked_m, d, ref_d = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast)

        # Test performance with fixed shapes
        valid_m = masked_m.sum().item()
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, {kernel_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)) / 1e9 / t:4.0f} GB/s')
    print()


def test_k_grouped_gemm_contiguous() -> None:
    print('Testing k-grouped contiguous GEMM:')

    for num_groups, m, n, ks, expected_k_per_group in enumerate_k_grouped_contiguous():
        use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)

        for test_empty_groups in (False, True):
            new_ks = copy.deepcopy(ks)
            if test_empty_groups:
                new_ks[random.randint(0, num_groups - 1)] = 0
            k, a, b, c, d, ref_d = generate_k_grouped_contiguous(num_groups, m, n, new_ks, use_ue8m0=use_ue8m0)
            new_ks_tensor = torch.tensor(new_ks, dtype=torch.int, device='cuda')
            deep_gemm.k_grouped_fp8_gemm_tn_contiguous(a, b, d, new_ks, new_ks_tensor, c=c)
            diff = calc_diff(d, ref_d)
            assert diff < 0.001, f'{m=}, {n=}, {k=}, {i=}, {diff:.5f}'

        # Test performance
        k, a, b, c, d, ref_d = generate_k_grouped_contiguous(num_groups, m, n, ks, use_ue8m0=use_ue8m0)
        ks_tensor = torch.tensor(ks, dtype=torch.int, device='cuda')

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.k_grouped_fp8_gemm_tn_contiguous(a, b, d, ks, ks_tensor, c=c)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=:2}, m={m:5}, n={n:5}, k={k:5}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, c, d) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm()
    # test_m_grouped_gemm_contiguous()
    # test_m_grouped_gemm_masked()
    # test_k_grouped_gemm_contiguous()
