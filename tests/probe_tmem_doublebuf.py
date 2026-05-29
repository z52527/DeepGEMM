#!/usr/bin/env python3
"""Compare BLOCK_N=256 (1 epi stage) vs BLOCK_N=240 (2 epi stages).

Uses the same timing methodology as bench_fp4_vs_cublas.py:
  - Allocate `d` once outside _run
  - _run only calls fp8_gemm_nt
  - Use deep_gemm.testing.bench with 5 warmups, 20 tests, high_precision
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))

import deep_gemm
from deep_gemm.testing import bench
from bench_fp4_vs_cublas import quantize_to_mxf4
from generators import KernelType, get_ue8m0_usage

SHAPES = [
    # Shapes where baseline heuristic picks bn=256 (1 epi stage)
    (1024, 4096, 7168),
    (4096, 4096, 7168),
    # Control shape where baseline picks bn!=256 (already 2 epi)
    (512, 4096, 7168),  # bn=112 baseline
]

BNS = [256, 240, 224, 208, 192, 176, 160, 144, 128, 112]

def tflops(M, N, K, t):
    return 2.0 * M * N * K / t / 1e12

def bench_shape(M, N, K, bn):
    torch.manual_seed(42)
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    a_packed, sf_a = quantize_to_mxf4(a_bf16)
    b_packed, sf_b = quantize_to_mxf4(b_bf16)

    duc = not get_ue8m0_usage(KernelType.Kernel1D1D)
    d = torch.empty((M, N), device='cuda', dtype=torch.float32)

    os.environ['DG_FP4_BLOCK_N'] = str(bn)
    def _run():
        deep_gemm.fp8_gemm_nt((a_packed, sf_a), (b_packed, sf_b), d, c=None,
                              recipe=(1, 1, 128), disable_ue8m0_cast=duc)
    # warmup
    for _ in range(10):
        _run()
    torch.cuda.synchronize()
    t = bench(_run, num_warmups=5, num_tests=30, high_precision=True)
    return t

def main():
    print(f'{"Shape":>20} | {"bn":>4} | {"time (us)":>10} | {"TFLOPs":>8} | note')
    print('-' * 68)
    for M, N, K in SHAPES:
        print(f'  {M:>4}x{N:>5}x{K:>5}')
        best_t = 1e9
        best_bn = None
        for bn in BNS:
            t = bench_shape(M, N, K, bn)
            f = tflops(M, N, K, t)
            fits2epi = (2*bn + 8 + (max(bn,128)//32)*2) <= 512
            note = '2-epi' if fits2epi else '1-epi'
            if t < best_t:
                best_t = t
                best_bn = bn
            mark = ' ***' if t == best_t else ''
            print(f'  {"":20} | {bn:>4} | {t*1e6:>10.1f} | {f:>8.1f} | {note}{mark}')
        print(f'  {"→ best":20} | bn={best_bn}')
        print()
    os.environ.pop('DG_FP4_BLOCK_N', None)

if __name__ == '__main__':
    main()
