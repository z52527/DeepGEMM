#!/usr/bin/env python3
"""
Benchmark: DeepGEMM FP4 (MXF4) vs cuBLAS FP4 (NVFP4)

1. Sweep block_n values for DeepGEMM, verify correctness (vs CPU ref)
2. Compare DeepGEMM numerical output against cuBLAS from the same BF16 source
3. Benchmark both kernels

Usage:
    python tests/bench_fp4_vs_cublas.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, '/home/scratch.runchuz_gpu/testspace/test_fp4_benchmark')

import torch
import deep_gemm
from deep_gemm.testing import bench, bench_kineto
from generators import KernelType, get_ue8m0_usage

# ── E2M1 lookup table ────────────────────────────────────────
E2M1_LUT = torch.tensor([
     0.0,   0.5,   1.0,   1.5,   2.0,   3.0,   4.0,   6.0,
    -0.0,  -0.5,  -1.0,  -1.5,  -2.0,  -3.0,  -4.0,  -6.0
], dtype=torch.float32)

E2M1_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

# ── MXF4 quantization (BF16 → E2M1 + UE8M0 SF) ──────────────

def quantize_to_mxf4(x_bf16, device='cuda'):
    """Quantize BF16 [M, K] → (packed_int32 [M, K//8], sf_float32 [M, sf_k]).
    MXF4: VS=32, UE8M0 scale (power-of-2), E2M1 values.
    """
    M, K = x_bf16.shape
    assert K % 32 == 0
    x = x_bf16.float().to(device)

    VS = 32
    num_groups = K // VS
    sf_k = ((num_groups + 3) // 4) * 4  # pad to multiple of 4

    x_groups = x.reshape(M, num_groups, VS)  # [M, ng, 32]

    # UE8M0: scale = 2^exp, choose exp = round(log2(amax / 6.0))
    amax = x_groups.abs().amax(dim=-1).clamp(min=1e-12)  # [M, ng]
    exp = torch.round(torch.log2(amax / 6.0)).clamp(-127, 127)
    sf = torch.pow(2.0, exp)  # [M, ng]

    # Quantize: divide by sf, round to nearest E2M1
    scaled = x_groups / sf.unsqueeze(-1)  # [M, ng, 32]
    signs = (scaled < 0).to(torch.int32)
    abs_val = scaled.abs()
    # Find nearest E2M1 magnitude
    diffs = (abs_val.unsqueeze(-1) - E2M1_VALS.to(device)).abs()  # [M, ng, 32, 8]
    best_idx = diffs.argmin(dim=-1)  # [M, ng, 32] values in 0..7
    fp4_bits = (signs << 3) | best_idx  # 4-bit: S | EE | M

    # Pack 8 FP4 into int32
    fp4_flat = fp4_bits.reshape(M, K)  # [M, K]
    assert K % 8 == 0
    packed = torch.zeros(M, K // 8, dtype=torch.int32, device=device)
    for i in range(8):
        packed |= (fp4_flat[:, i::8] & 0xF) << (i * 4)

    # Pad sf to sf_k
    sf_padded = torch.ones(M, sf_k, dtype=torch.float32, device=device)
    sf_padded[:, :num_groups] = sf

    return packed, sf_padded


def mxf4_dequantize(packed, sf, M, K):
    """Dequantize MXF4 → float32 [M, K]."""
    VS = 32
    cpu = packed.cpu().to(torch.int64) & 0xFFFFFFFF
    bits = torch.stack([(cpu >> (i * 4)) & 0xF for i in range(8)], dim=-1).reshape(M, -1)[:, :K]
    vals = E2M1_LUT[bits.long()]

    num_groups = K // VS
    sf_cpu = sf.cpu().float()[:, :num_groups]
    vals_g = vals.reshape(M, num_groups, VS)
    result = (vals_g * sf_cpu.unsqueeze(-1)).reshape(M, K)
    return result


# ── CPU reference ─────────────────────────────────────────────

def fp4_reference(a_packed, b_packed, sf_a, sf_b, m, n, k_fp4):
    """CPU reference: C = A @ B^T with per-group SF."""
    a_f = mxf4_dequantize(a_packed, sf_a, m, k_fp4)
    b_f = mxf4_dequantize(b_packed, sf_b, n, k_fp4)
    return a_f @ b_f.T


# ── DeepGEMM FP4 runner ──────────────────────────────────────

def run_deepgemm(a_packed, b_packed, sf_a, sf_b, m, n, block_n=None):
    """Run DeepGEMM FP4 kernel. If block_n given, override via env."""
    if block_n is not None:
        os.environ['DG_FP4_BLOCK_N'] = str(block_n)
    else:
        os.environ.pop('DG_FP4_BLOCK_N', None)

    duc = not get_ue8m0_usage(KernelType.Kernel1D1D)
    d = torch.empty((m, n), device='cuda', dtype=torch.float32)
    deep_gemm.fp8_gemm_nt(
        (a_packed, sf_a), (b_packed, sf_b), d, c=None,
        recipe=(1, 1, 128), disable_ue8m0_cast=duc,
    )
    torch.cuda.synchronize()
    return d


def bench_deepgemm(a_packed, b_packed, sf_a, sf_b, m, n, block_n=None):
    """Measure only the main FP4 GEMM kernel time (exclude SF transform).

    Mirrors the FP8 benchmark approach in tests/test_fp8.py (bench_kineto
    filtered to the GEMM kernel name) so the comparison is apples-to-apples
    with cuBLAS/FlashInfer, which also measure only their main GEMM kernel.
    The SF float32->UE8M0 transform (`transpose_and_pack_fp32_into_ue8m0`)
    is excluded — in real inference pipelines SF is typically pre-quantized
    as part of the model weights, not recomputed per call.
    """
    if block_n is not None:
        os.environ['DG_FP4_BLOCK_N'] = str(block_n)
    else:
        os.environ.pop('DG_FP4_BLOCK_N', None)

    duc = not get_ue8m0_usage(KernelType.Kernel1D1D)
    d = torch.empty((m, n), device='cuda', dtype=torch.float32)

    def fn():
        deep_gemm.fp8_gemm_nt(
            (a_packed, sf_a), (b_packed, sf_b), d, c=None,
            recipe=(1, 1, 128), disable_ue8m0_cast=duc,
        )

    # Filter profiler output to only the FP4 GEMM kernel, matching FP8 tests.
    return bench_kineto(fn, 'sm100_fp4_gemm', suppress_kineto_output=True)


# ── cuBLAS NVFP4 runner ──────────────────────────────────────

def run_cublas(a_bf16, b_bf16):
    """Quantize BF16 to NVFP4 and run cuBLAS."""
    from flashinfer import nvfp4_quantize, SfLayout
    from cublas_fp4 import fp4_gemm_blockwise

    M, K = a_bf16.shape
    N = b_bf16.shape[0]

    a_gsf = (448 * 6) / a_bf16.float().abs().nan_to_num().max()
    b_gsf = (448 * 6) / b_bf16.float().abs().nan_to_num().max()

    a_fp4, a_sf = nvfp4_quantize(a_bf16, a_gsf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    b_fp4, b_sf = nvfp4_quantize(b_bf16, b_gsf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)

    if a_sf.dtype != torch.float8_e4m3fn:
        a_sf = a_sf.view(torch.float8_e4m3fn)
    if b_sf.dtype != torch.float8_e4m3fn:
        b_sf = b_sf.view(torch.float8_e4m3fn)

    alpha_val = 1.0 / (a_gsf.item() * b_gsf.item())
    alpha = torch.tensor([alpha_val], dtype=torch.float32, device='cuda')

    # Match DeepGEMM output dtype (float32) for apples-to-apples bandwidth.
    out = fp4_gemm_blockwise(a_fp4, a_sf, b_fp4, b_sf, 16, alpha, torch.float32)
    torch.cuda.synchronize()
    return out, (a_fp4, a_sf, b_fp4, b_sf, alpha)


def bench_cublas(a_fp4, a_sf, b_fp4, b_sf, alpha):
    from cublas_fp4 import fp4_gemm_blockwise

    def fn():
        # Match DeepGEMM output dtype (float32) for apples-to-apples bandwidth.
        fp4_gemm_blockwise(a_fp4, a_sf, b_fp4, b_sf, 16, alpha, torch.float32)

    return bench(fn, num_warmups=5, num_tests=20)


# ── Helpers ───────────────────────────────────────────────────

def calc_diff(x, y):
    """Cosine-based diff: 1 - cosine_similarity."""
    x, y = x.double().flatten(), y.double().flatten()
    denom = (x * x + y * y).sum()
    if denom < 1e-12:
        return 0.0
    return (1 - 2 * (x * y).sum() / denom).item()


def tflops(M, N, K, t):
    return (2 * M * N * K) / (t * 1e12)


def cuda_ok():
    try:
        torch.cuda.synchronize()
        _ = torch.zeros(1, device='cuda')
        return True
    except RuntimeError:
        return False


# ── Test shapes ───────────────────────────────────────────────

CORRECTNESS_SHAPES = [
    # (M, N, K_fp4) — K is number of FP4 elements
    (128,  128,  256),
    (128,  256,  512),
    (128,  128, 1024),
    (128,  512,  256),
    (128,  256, 1024),
]

BENCHMARK_SHAPES = [
    (128,  2048, 7168),
    (128,  4096, 7168),
    (128,  7168, 2048),
    (128, 24576, 1536),
    (256,  4096, 7168),
    (256,  7168, 2048),
    # Large M: triggers B-multicast (2CTA, UMMA_M=256)
    (512,  2048, 7168),
    (512,  4096, 7168),
    (512,  7168, 2048),
    (1024, 4096, 7168),
    (1024, 7168, 2048),
    (4096, 4096, 7168),
    # Square large GEMM
    (8192, 8192, 8192),
]

BLOCK_NS_TO_TEST = [16, 32, 64, 128, 176, 240]


# ── Part 1: Correctness sweep across block_n ──────────────────

def test_correctness():
    print('=' * 70)
    print('PART 1: Correctness sweep across block_n values')
    print('=' * 70)
    print()

    all_pass = True
    for M, N, K in CORRECTNESS_SHAPES:
        a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')

        a_packed, sf_a = quantize_to_mxf4(a_bf16)
        b_packed, sf_b = quantize_to_mxf4(b_bf16)
        ref = fp4_reference(a_packed, b_packed, sf_a, sf_b, M, N, K)

        results_line = f'  M={M:4d} N={N:4d} K={K:4d}:'
        for bn in BLOCK_NS_TO_TEST:
            if bn > N:
                continue
            if not cuda_ok():
                results_line += f'  bn={bn}:CTX_ERR'
                all_pass = False
                continue
            try:
                d = run_deepgemm(a_packed, b_packed, sf_a, sf_b, M, N, block_n=bn)
                diff = calc_diff(d.cpu().float(), ref)
                ok = diff < 0.01
                if not ok:
                    all_pass = False
                results_line += f'  bn={bn}:{"PASS" if ok else "FAIL"}({diff:.1e})'
            except Exception as e:
                results_line += f'  bn={bn}:ERR'
                all_pass = False
        print(results_line)

    # Reset env
    os.environ.pop('DG_FP4_BLOCK_N', None)
    print()
    print(f'  Correctness: {"ALL PASS" if all_pass else "SOME FAILED"}')
    return all_pass


# ── Part 2: Numerical comparison DeepGEMM vs cuBLAS ──────────

def test_numerical_comparison():
    print()
    print('=' * 70)
    print('PART 2: Numerical comparison DeepGEMM vs cuBLAS (from same BF16)')
    print('       (using larger shapes where cuBLAS NVFP4 is known-good)')
    print('=' * 70)
    print()

    # Use larger shapes — cuBLAS NVFP4 needs sufficient N/K for valid SF layout
    shapes = [
        (128, 2048, 7168),
        (128, 4096, 7168),
        (128, 7168, 2048),
    ]

    hdr = f'{"M":>5} {"N":>5} {"K":>5} | {"DG vs BF16ref":>14} | {"CB vs BF16ref":>14} | {"DG vs CB":>14}'
    print(hdr)
    print('-' * len(hdr))

    for M, N, K in shapes:
        if not cuda_ok():
            print(f'{M:>5} {N:>5} {K:>5} | CUDA context error')
            continue

        # Same BF16 source
        a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
        ref_bf16 = (a_bf16.float() @ b_bf16.float().T)

        # DeepGEMM path: BF16 → MXF4 → kernel
        a_packed, sf_a = quantize_to_mxf4(a_bf16)
        b_packed, sf_b = quantize_to_mxf4(b_bf16)
        dg_out = run_deepgemm(a_packed, b_packed, sf_a, sf_b, M, N)

        # cuBLAS path: BF16 → NVFP4 → kernel
        try:
            cb_out, _ = run_cublas(a_bf16, b_bf16)
        except Exception as e:
            print(f'{M:>5} {N:>5} {K:>5} | cuBLAS error: {str(e)[:40]}')
            continue

        dg_vs_ref = calc_diff(dg_out.cpu().float(), ref_bf16.cpu())
        cb_vs_ref = calc_diff(cb_out.cpu().float(), ref_bf16.cpu())
        dg_vs_cb = calc_diff(dg_out.cpu().float(), cb_out.cpu().float())

        print(f'{M:>5} {N:>5} {K:>5} |     {dg_vs_ref:10.2e} |     {cb_vs_ref:10.2e} |     {dg_vs_cb:10.2e}')

        del a_bf16, b_bf16, a_packed, sf_a, b_packed, sf_b
        torch.cuda.empty_cache()


# ── Part 3: Performance benchmark ────────────────────────────

def test_performance():
    print()
    print('=' * 70)
    print('PART 3: Performance benchmark (DeepGEMM best-block_n vs cuBLAS)')
    print('=' * 70)
    print()

    hdr = (f'{"M":>6} {"N":>6} {"K":>6} |'
           f' {"DeepGEMM":>17} | {"cuBLAS":>17} | {"speedup":>9}')
    sep = '-' * len(hdr)
    print(hdr)
    print(sep)

    for M, N, K in BENCHMARK_SHAPES:
        if not cuda_ok():
            print(f'{M:>6} {N:>6} {K:>6} | CUDA context error')
            break

        # Prepare DeepGEMM data
        a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
        a_packed, sf_a = quantize_to_mxf4(a_bf16)
        b_packed, sf_b = quantize_to_mxf4(b_bf16)

        # Bench DeepGEMM (auto block_n selection)
        try:
            dg_t = bench_deepgemm(a_packed, b_packed, sf_a, sf_b, M, N, block_n=None)
            dg_s = f'{dg_t * 1e6:8.0f} us  {tflops(M, N, K, dg_t):7.1f} T'
        except Exception as e:
            dg_t = None
            dg_s = f'{"FAIL":>17}'

        # Bench cuBLAS
        try:
            _, (a_fp4, a_sf, b_fp4, b_sf, alpha) = run_cublas(a_bf16, b_bf16)
            cb_t = bench_cublas(a_fp4, a_sf, b_fp4, b_sf, alpha)
            cb_s = f'{cb_t * 1e6:8.0f} us  {tflops(M, N, K, cb_t):7.1f} T'
        except Exception as e:
            cb_t = None
            cb_s = f'{"FAIL":>17}'

        if dg_t and cb_t:
            # Speedup = cuBLAS_time / DG_time. >1.0 means DG is faster.
            ratio = f'{cb_t / dg_t:8.2f}x'
        else:
            ratio = f'{"--":>9}'

        print(f'{M:>6} {N:>6} {K:>6} | {dg_s} | {cb_s} | {ratio}')

        del a_bf16, b_bf16, a_packed, sf_a, b_packed, sf_b
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            break

    print(sep)
    print('speedup > 1.0 means DeepGEMM is faster (cuBLAS_time / DG_time)')


# ── Main ──────────────────────────────────────────────────────

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(42)

    print(f'GPU:  {torch.cuda.get_device_name(0)}')
    print(f'CUDA: {torch.version.cuda}')
    print()

    test_correctness()
    test_numerical_comparison()
    test_performance()
