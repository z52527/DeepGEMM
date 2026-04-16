"""
FP4 (E2M1) GEMM correctness test for SM100 MXF4 block-scaled kernel.

Usage:
    python tests/test_fp4.py
"""

import torch
import random
import deep_gemm
from generators import KernelType, get_ue8m0_usage

# ============================================================
# E2M1 FP4 查找表
# ============================================================
E2M1_LUT = torch.tensor([
     0.0,   0.5,   1.0,   1.5,   2.0,   3.0,   4.0,   6.0,   # bits 0-7  (S=0)
    -0.0,  -0.5,  -1.0,  -1.5,  -2.0,  -3.0,  -4.0,  -6.0    # bits 8-15 (S=1)
], dtype=torch.float32)

# ============================================================
# 工具函数
# ============================================================

def pack_fp4_random(m: int, k_fp4: int, device='cuda'):
    """生成随机 E2M1 FP4 数据并打包为 int32。每个 int32 包含 8 个 FP4 值。"""
    assert k_fp4 % 8 == 0
    raw = torch.randint(0, 16, (m, k_fp4), dtype=torch.uint8, device=device)
    packed = torch.zeros(m, k_fp4 // 8, dtype=torch.int32, device=device)
    for i in range(8):
        packed += (raw[:, i::8].to(torch.int32) << (i * 4))
    return packed


def pack_fp4_constant(m: int, k_fp4: int, fp4_bits: int = 0x2, device='cuda'):
    """生成常量 FP4 打包数据。fp4_bits=0x2 -> E2M1 1.0"""
    assert k_fp4 % 8 == 0
    word = 0
    for i in range(8):
        word |= (fp4_bits & 0xF) << (i * 4)
    return torch.full((m, k_fp4 // 8), word, dtype=torch.int32, device=device)


def generate_mxf4_scale_factors(m, n, k_fp4, device='cuda', random_sf=False):
    """为 MXF4 (VS=32) 生成 scale factor (float32 格式)。

    Host 端 C++ API (transform_sf_into_required_layout) 会在调用 kernel 之前
    将 float32 SF 转换为 packed UE8M0 int32:
      1. 从 float32 IEEE 754 中提取 8-bit 指数字段 (bitwise_right_shift(23))
      2. 每 4 个 UE8M0 打包为 1 个 int32
      3. 转置为 MN-major + TMA 对齐
    然后 kernel 通过 TMA 加载到 SMEM, 经 UTCCP 写入 TMEM 供 MMA 使用。

    UE8M0: value = 2^(exp - 127)。每 VS=32 个 FP4 元素共享一个 SF。

    Args:
        random_sf: 如果 True, 生成随机的 2 的幂次 SF (0.25, 0.5, 1.0, 2.0, 4.0)
    """
    VS = 32
    sf_k = ((k_fp4 // VS + 3) // 4) * 4
    if random_sf:
        powers = torch.randint(-2, 3, (m, sf_k), device=device).float()
        sf_a = torch.pow(2.0, powers)
        powers = torch.randint(-2, 3, (n, sf_k), device=device).float()
        sf_b = torch.pow(2.0, powers)
    else:
        sf_a = torch.ones((m, sf_k), dtype=torch.float32, device=device)
        sf_b = torch.ones((n, sf_k), dtype=torch.float32, device=device)
    return sf_a, sf_b


def fp4_reference(a_packed, b_packed, m, n, sf_a=None, sf_b=None):
    """CPU 端 E2M1 FP4 GEMM reference: C = A @ B^T, 支持 block-scaled SF。

    Block-scaled MXF4: C[m,n] = sum_g SF_A[m,g] * SF_B[n,g] * dot(A_g, B_g)
    其中 g 是 VS=32 元素的组。sf_a/sf_b 为 float32 (pre-transform), 每组一个值。
    """
    VS = 32
    a_cpu = a_packed.cpu().to(torch.int64) & 0xFFFFFFFF
    b_cpu = b_packed.cpu().to(torch.int64) & 0xFFFFFFFF
    bits_a = torch.stack([(a_cpu >> (i*4)) & 0xF for i in range(8)], dim=-1).reshape(m, -1)
    bits_b = torch.stack([(b_cpu >> (i*4)) & 0xF for i in range(8)], dim=-1).reshape(n, -1)
    a_float = E2M1_LUT[bits_a.long()]  # [m, k_fp4]
    b_float = E2M1_LUT[bits_b.long()]  # [n, k_fp4]

    if sf_a is None:
        return torch.matmul(a_float, b_float.T)

    k_fp4 = a_float.shape[1]
    sf_a_cpu = sf_a.cpu().float()
    sf_b_cpu = sf_b.cpu().float()
    c = torch.zeros(m, n, dtype=torch.float32)
    num_groups = k_fp4 // VS
    for g in range(num_groups):
        k_start, k_end = g * VS, (g + 1) * VS
        a_g = a_float[:, k_start:k_end]
        b_g = b_float[:, k_start:k_end]
        sf_col = g
        if sf_col < sf_a_cpu.shape[1]:
            sfa_g = sf_a_cpu[:, sf_col].unsqueeze(1)
            sfb_g = sf_b_cpu[:, sf_col].unsqueeze(1)
        else:
            sfa_g = torch.ones(m, 1)
            sfb_g = torch.ones(n, 1)
        c += (sfa_g * sfb_g.T) * torch.matmul(a_g, b_g.T)
    return c


def run_kernel(a_packed, b_packed, sf_a, sf_b, m, n, recipe=(1, 1, 128)):
    """调用 FP4 GEMM kernel (复用 fp8_gemm_nt 入口, int32 dtype 触发 FP4 路径)"""
    duc = not get_ue8m0_usage(KernelType.Kernel1D1D)
    d = torch.empty((m, n), device='cuda', dtype=torch.float32)
    deep_gemm.fp8_gemm_nt((a_packed, sf_a), (b_packed, sf_b), d, c=None,
                          recipe=recipe, disable_ue8m0_cast=duc)
    torch.cuda.synchronize()
    return d


# ============================================================
# 测试用例
# ============================================================

def test_constant():
    """全 1.0 常量测试: C[i,j] = K (因为 1.0 * 1.0 * K 个元素)"""
    print('Test: constant values (all E2M1 1.0)')
    configs = [
        # (M, N, K_fp4) — K 是 FP4 元素个数
        # 单 stage (K <= 256)
        (32,   64,  256),
        (128, 128,  128),
        (128, 256,  256),
        # 多 stage (K > 256)
        (128, 128,  512),
        (256, 256,  512),
        (128, 128, 1024),
        (128, 256, 1024),
        # 大 M (multi-wave: BLOCK_M=128, 所以 M>128 需要多个 wave)
        (256, 128,  256),
        (256, 256, 1024),
    ]
    all_pass = True
    for m, n, k in configs:
        a = pack_fp4_constant(m, k, fp4_bits=0x2)
        b = pack_fp4_constant(n, k, fp4_bits=0x2)
        sf_a, sf_b = generate_mxf4_scale_factors(m, n, k)
        d = run_kernel(a, b, sf_a, sf_b, m, n)
        expected = float(k)
        ok = (d.cpu() == expected).all().item()
        if not ok:
            all_pass = False
        print(f'  M={m:4d} N={n:4d} K={k:4d}: expected={expected:.0f} got={d.cpu()[0,0].item():.0f} {"PASS" if ok else "FAIL"}')
    return all_pass


def test_random():
    """随机数据测试 (SF=1.0): 对比 GPU kernel 与 CPU reference"""
    print('Test: random data (vs CPU reference, SF=1.0)')
    configs = [
        (32,   64,  256),
        (128,  128, 128),
        (128,  256, 256),
        # 多 stage
        (128,  128, 512),
        (256,  256, 512),
        (128,  128, 1024),
        # 大 M
        (256,  128, 256),
        (256,  256, 1024),
        # 较大 N
        (128,  512, 256),
    ]
    all_pass = True
    for m, n, k in configs:
        a = pack_fp4_random(m, k)
        b = pack_fp4_random(n, k)
        sf_a, sf_b = generate_mxf4_scale_factors(m, n, k)
        d = run_kernel(a, b, sf_a, sf_b, m, n)
        ref = fp4_reference(a, b, m, n, sf_a, sf_b)
        max_diff = torch.abs(d.cpu().float() - ref.float()).max().item()
        ok = max_diff < 1.0
        if not ok:
            all_pass = False
        print(f'  M={m:4d} N={n:4d} K={k:4d}: max_diff={max_diff:.4f} {"PASS" if ok else "FAIL"}')
    return all_pass


def test_value_sweep():
    """不同 FP4 值测试: 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0"""
    print('Test: FP4 value sweep')
    m, n, k = 128, 128, 256
    all_pass = True
    for bits, val in [(0x1, 0.5), (0x2, 1.0), (0x3, 1.5), (0x4, 2.0),
                      (0x5, 3.0), (0x6, 4.0), (0x7, 6.0)]:
        a = pack_fp4_constant(m, k, fp4_bits=bits)
        b = pack_fp4_constant(n, k, fp4_bits=bits)
        sf_a, sf_b = generate_mxf4_scale_factors(m, n, k)
        d = run_kernel(a, b, sf_a, sf_b, m, n)
        expected = float(k) * val * val
        actual = d.cpu()[0, 0].item()
        ok = abs(actual - expected) < 1.0
        if not ok:
            all_pass = False
        print(f'  FP4={val:4.1f} (bits=0x{bits:X}): expected={expected:.1f} got={actual:.1f} {"PASS" if ok else "FAIL"}')
    return all_pass


def test_uniform_sf():
    """Uniform SF 测试: 验证 UTCCP SF->TMEM 路径在不同 SF 值下正确"""
    print('Test: uniform scale factors (UTCCP path)')
    configs = [
        (128, 256, 256),
        (256, 128, 256),
        (128, 128, 512),
    ]
    all_pass = True
    for m, n, k in configs:
        for sf_val in [0.25, 0.5, 1.0, 2.0, 4.0]:
            a = pack_fp4_random(m, k)
            b = pack_fp4_random(n, k)
            sf_a, sf_b = generate_mxf4_scale_factors(m, n, k)
            sf_a.fill_(sf_val)
            sf_b.fill_(sf_val)
            d = run_kernel(a, b, sf_a, sf_b, m, n)
            # uniform SF: kernel result = unscaled_result * sf_a * sf_b
            ref = fp4_reference(a, b, m, n, sf_a, sf_b)
            max_diff = torch.abs(d.cpu().float() - ref.float()).max().item()
            ok = max_diff < 1.0
            if not ok:
                all_pass = False
            print(f'  M={m:4d} N={n:4d} K={k:4d} SF={sf_val:5.2f}: max_diff={max_diff:.4f} {"PASS" if ok else "FAIL"}')
    return all_pass


def test_asymmetric_values():
    """A 和 B 使用不同 FP4 值"""
    print('Test: asymmetric A/B values')
    m, n, k = 128, 128, 256
    all_pass = True
    cases = [
        (0x2, 0x4, 1.0, 2.0),   # A=1.0, B=2.0
        (0x1, 0x6, 0.5, 4.0),   # A=0.5, B=4.0
        (0x4, 0x1, 2.0, 0.5),   # A=2.0, B=0.5
    ]
    for bits_a, bits_b, val_a, val_b in cases:
        a = pack_fp4_constant(m, k, fp4_bits=bits_a)
        b = pack_fp4_constant(n, k, fp4_bits=bits_b)
        sf_a, sf_b = generate_mxf4_scale_factors(m, n, k)
        d = run_kernel(a, b, sf_a, sf_b, m, n)
        expected = float(k) * val_a * val_b
        actual = d.cpu()[0, 0].item()
        ok = abs(actual - expected) < 1.0
        if not ok:
            all_pass = False
        print(f'  A={val_a}, B={val_b}: expected={expected:.1f} got={actual:.1f} {"PASS" if ok else "FAIL"}')
    return all_pass


def test_random_sf():
    """随机数据 + 随机 per-group SF (powers of 2)"""
    print('Test: random data + random scale factors')
    configs = [
        (128, 128, 256),
        (128, 256, 256),
        (256, 256, 512),
        (128, 128, 512),
        (128, 128, 1024),
        (256, 128, 256),
        (256, 256, 1024),
        (128, 256, 1024),
    ]
    all_pass = True
    for m, n, k in configs:
        a = pack_fp4_random(m, k)
        b = pack_fp4_random(n, k)
        sf_a, sf_b = generate_mxf4_scale_factors(m, n, k, random_sf=True)
        d = run_kernel(a, b, sf_a, sf_b, m, n)
        ref = fp4_reference(a, b, m, n, sf_a, sf_b)
        max_diff = torch.abs(d.cpu().float() - ref.float()).max().item()
        ok = max_diff < 1.0
        if not ok:
            all_pass = False
        print(f'  M={m:4d} N={n:4d} K={k:4d}: max_diff={max_diff:.4f} {"PASS" if ok else "FAIL"}')
    return all_pass


def test_multicast():
    """大 M 测试：触发 B-multicast (M>=512, 2CTA along M, UMMA_M=256)"""
    print('Test: B-multicast (M>=512, 2CTA)')
    configs = [
        (512,  128,  256),
        (512,  128,  512),
        (512,  128, 1024),
        (1024, 128,  256),
        (1024, 128,  512),
    ]
    all_pass = True
    for m, n, k in configs:
        a = pack_fp4_random(m, k)
        b = pack_fp4_random(n, k)
        sf_a, sf_b = generate_mxf4_scale_factors(m, n, k, random_sf=True)
        d = run_kernel(a, b, sf_a, sf_b, m, n)
        ref = fp4_reference(a, b, m, n, sf_a, sf_b)
        max_diff = torch.abs(d.cpu().float() - ref.float()).max().item()
        ok = max_diff < 1.0
        if not ok:
            all_pass = False
        print(f'  M={m:4d} N={n:4d} K={k:4d}: max_diff={max_diff:.4f} {"PASS" if ok else "FAIL"}')
    return all_pass


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print(f'Library: {deep_gemm.__path__}\n')

    results = [
        ('constant',       test_constant()),
        ('random',         test_random()),
        ('sweep',          test_value_sweep()),
        ('asymmetric',     test_asymmetric_values()),
        ('uniform_sf',     test_uniform_sf()),
        ('random_sf',      test_random_sf()),
        ('multicast',      test_multicast()),
    ]

    print()
    passed = all(r for _, r in results)
    for name, ok in results:
        print(f'  {name}: {"PASS" if ok else "FAIL"}')
    print(f'\n{"ALL FP4 TESTS PASSED" if passed else "SOME TESTS FAILED"}')
    if not passed:
        exit(1)
