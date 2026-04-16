#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
// Reuse some types in the JIT modules
#include <deep_gemm/common/types.hpp>

#include "common.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

struct SM100ArchSpec {
    static constexpr int smem_capacity = 232448;

    static int get_ab_load_block_m(const MulticastConfig& config, const int& block_m) {
        return block_m / (config.is_multicast_on_a ? config.num_multicast : 1);
    }

    static int get_ab_load_block_n(const MulticastConfig& config, const int& block_n) {
        return block_n / (config.is_multicast_on_a ? 1 : config.num_multicast);
    }

    static int get_cd_store_block_m(const int& block_m) {
        constexpr int layout_ad_m = 128;
        return std::min(block_m, layout_ad_m);
    }

    static int get_cd_store_block_n(const int& block_n) {
        return block_n;
    }

    static std::pair<int, int> get_sf_uttcp_aligned_block_sizes(
        const int& block_m, const int& block_n, const at::ScalarType& ab_dtype) {
        constexpr int num_utccp_aligned_elems = 128;
        DG_HOST_ASSERT(block_m % num_utccp_aligned_elems == 0);
        switch (ab_dtype) {
            case torch::kBFloat16: return {0, 0};
            case torch::kFloat8_e4m3fn: return {align(block_m, num_utccp_aligned_elems), align(block_n, num_utccp_aligned_elems)};
            case torch::kInt: return {align(block_m, num_utccp_aligned_elems), align(block_n, num_utccp_aligned_elems)};  // FP4 packed data
            default: DG_HOST_UNREACHABLE("Unknown dtype");
        }
    }

    static bool is_block_size_legal(const KernelType& kernel_type,
                                    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                    const at::ScalarType& ab_dtype, const at::ScalarType& cd_dtype,
                                    const int& block_m, const int& block_n, const int& block_k) {
        // TODO: consider more carefully for BF16 GEMMs
        // 2SM BF16 UMMA does not support `N % 32 != 0`
        if (ab_dtype == torch::kBFloat16 and block_n % 32 != 0)
            return false;

        // Layout A/D does not support `block_m == 64` and `block_n % 16 != 0`
        if (block_m == 64 or block_n % 16 != 0)
            return false;

        // Performance is lower with 1D1D and `block_m == 256`
        if (kernel_type == KernelType::Kernel1D1D and major_b == cute::UMMA::Major::K and block_m != 128)
            return false;

        // 1D2D kernels' maximum block N is 128
        // 1D2D kernels require more friendly block Ns
        if (kernel_type == KernelType::Kernel1D2D and (block_n > 128 or 128 % block_n != 0))
            return false;

        // Check tensor memory validity
        int sf_block_m = 0, sf_block_n = 0;
        if (kernel_type == KernelType::Kernel1D1D) {
            const auto& [sf_block_m_, sf_block_n_] = get_sf_uttcp_aligned_block_sizes(block_m, block_n, ab_dtype);
            sf_block_m = sf_block_m_, sf_block_n = sf_block_n_;
        }
        // FP4: sf_packed_k_per_stage = block_k * 8 / 32 / 4 = block_k / 16
        const int sf_tmem_k_mult = (ab_dtype == torch::kInt) ? (block_k / 16) : 1;
        if (((2 * block_n) + (sf_block_m / 32) * sf_tmem_k_mult + (sf_block_n / 32) * sf_tmem_k_mult) > 512)
            return false;

        // NOTES: when B is MN-major, we restrict `block_n` to multiples of 64,
        // since TMA performance degrades when `swizzle_b <= 32B` (i.e., when `block_ns % 64 != 0`), even with 3D TMA
        return major_b == cute::UMMA::Major::K or (block_n * c10::elementSize(ab_dtype)) % 64 == 0;
    }

    static bool is_num_stages_legal(const at::ScalarType& ab_dtype, const at::ScalarType& cd_dtype,
                                    const int& num_stages,
                                    const int& block_m, const int& block_n, const int& block_k) {
        return true;
    }

    static bool should_minimize_num_sms() {
        return false;
    }

    static std::pair<bool, bool> get_multicast_legality(const GemmType& gemm_type,
                                                      const int& m, const int& n, const int& block_m, const int& block_n,
                                                      const int& num_sms) {
        // B-multicast: FP8 supports it (UMMA_M=256 ok for MXF8).
        // FP4 doesn't reach here (dispatched to sm100_fp4_gemm_1d1d which uses its own config with torch::kInt).
        // A-multicast: blocked by kernel static assert.
        return {
            false,
            is_multicast_legal(m, block_m, 2, num_sms, true) and (gemm_type == GemmType::Normal or gemm_type == GemmType::KGroupedContiguous),
        };
    }

    static ThreadConfig get_thread_config(const KernelType& kernel_type,
                                          const int& block_m, const int& block_n) {
        return ThreadConfig::sm100(128, kernel_type == KernelType::Kernel1D2D ? block_m : 128);
    }

    static int get_smem_cd_size(const KernelType& kernel_type,
                                const int& block_m, const int& block_n,
                                const int& swizzle_cd_mode,
                                const at::ScalarType& cd_dtype) {
        constexpr static int layout_ad_m = 128;
        return (kernel_type != KernelType::Kernel1D2D ? std::min(block_m, layout_ad_m) : block_m) * swizzle_cd_mode * 2;
    }

    static std::pair<int, int> get_sf_smem_size_per_stage(const KernelType& kernel_type,
                                                          const int& block_m, const int& block_n, const int& block_k,
                                                          const at::ScalarType& ab_dtype, const at::ScalarType& cd_dtype) {
        if (ab_dtype == torch::kBFloat16)
            return {0, 0};

        int smem_sfa_per_stage = 0;
        int smem_sfb_per_stage = 0;
        if (kernel_type == KernelType::Kernel1D1D) {
            const auto [sf_block_m, sf_block_n] = get_sf_uttcp_aligned_block_sizes(block_m, block_n, ab_dtype);
            const int sf_packed_k_per_stage = (ab_dtype == torch::kInt) ? (block_k / 16) : 1;
            smem_sfa_per_stage = sf_block_m * 4 * sf_packed_k_per_stage;
            smem_sfb_per_stage = sf_block_n * 4 * sf_packed_k_per_stage;
        } else {
            smem_sfa_per_stage = block_m * 4;
            smem_sfb_per_stage = 0;
        }
        return {smem_sfa_per_stage, smem_sfb_per_stage};
    }

    static int get_extra_sfb_smem_size(const int& m, const int& n, const int& k,
                                       const int& block_m, const int& block_n, const int& block_k) {
        return 0;
    }

    static int get_barrier_smem_size(const int& num_stages) {
        // TODO: remove SF barriers for BF16 GEMMs
        // TMA full/empty barriers, with-SF full barriers, tensor memory full/empty barriers
        // NOTES: 1D2D kernel will not use the with-SF full barriers
        // NOTES: some shapes may only have 1 epilogue stage, but we still allocate space for 2 stages
        return num_stages * 8 * 3 + 2 * 8 * 2;
    }

    static int get_tmem_ptr_smem_size() {
        return 4;
    }
};

// ============================================================
// FP4-specific heuristic for SM100
// ============================================================
// FP4 (MXF4 E2M1) has tighter hardware constraints than FP8:
//   - BLOCK_M is fixed to 128 (UMMA_M=128 for MXF4 2-CTA MMA)
//   - BLOCK_K is fixed to 32 int32 (= 256 FP4 elements = 128 bytes)
//   - No multicast (MXF4 2-CTA MMA only supports M=128)
//   - SF occupies 2x TMEM columns vs FP8 (sf_packed_k_per_stage=2)
//   - TMEM capacity limits max BLOCK_N to ~240
//
// Strategy: same wave-minimization as FP8, but with a much narrower
// search space (only BLOCK_N varies). Tie-breaking favors smaller
// BLOCK_N to reduce wasted computation, since FP4's high arithmetic
// intensity means we're less sensitive to launch overhead.

static GemmConfig get_best_fp4_config(const GemmType& gemm_type,
                                      const int& m, const int& n, const int& k, const int& num_groups,
                                      const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                      const at::ScalarType& cd_dtype,
                                      const bool& with_accumulation, const int& num_sms) {
    constexpr auto ab_dtype = at::kInt;  // FP4 packed as int32
    constexpr auto kernel_type = KernelType::Kernel1D1D;
    constexpr int block_m = 128;   // MXF4 UMMA_M is always 128
    constexpr int block_k = 32;    // 128 bytes / sizeof(int32) = 32 (256 FP4 per stage)

    // Build candidate BLOCK_N list with legality check
    // TMEM constraint: epilogue_stages*BN + (SF_BLOCK_M/32)*sf_pk + (SF_BLOCK_N/32)*sf_pk <= 512
    // sf_pk = block_k / 16 (SF_PACKED_K_PER_STAGE)
    constexpr int sf_pk = block_k / 16;
    constexpr int sf_block_m_cols = (128 / 32) * sf_pk;
    auto is_fp4_block_n_legal = [&](const int& block_n) -> bool {
        if (block_n % 16 != 0 || block_n < 16 || block_n > 256)
            return false;
        // N%bn!=0 is OK: B TMA OOB columns read garbage but TMA store drops OOB writes.
        // SFB SMEM padding handled by zero-fill before warp-transpose.
        if (major_b != cute::UMMA::Major::K && (block_n * static_cast<int>(c10::elementSize(ab_dtype))) % 64 != 0)
            return false;
        // TMEM capacity: kernel auto-reduces epilogue stages from 2 to 1 if needed
        const int sf_block_n = align(block_n, 128);
        const int sf_block_n_cols = (sf_block_n / 32) * sf_pk;
        // Check with 1 epilogue stage (minimum)
        if ((1 * block_n + sf_block_m_cols + sf_block_n_cols) > 512)
            return false;
        return true;
    };

    // Wave-based block_n selection (same logic as FP8 but only block_n varies)
    const auto get_num_blocks = [=](const int& bn) {
        return ceil_div(m, block_m) * ceil_div(n, bn) * num_groups;
    };
    const auto get_num_waves = [=](const int& bn) {
        return ceil_div(get_num_blocks(bn), num_sms);
    };
    const auto get_last_wave_util = [=](const int& bn) {
        const auto num_last = get_num_blocks(bn) % num_sms;
        return num_last == 0 ? num_sms : num_last;
    };

    int best_block_n = 0;
    int best_num_waves = 0, best_score = 0;
    for (int bn = 16; bn <= 256; bn += 16) {
        if (!is_fp4_block_n_legal(bn))
            continue;

        const int num_waves = get_num_waves(bn);

        // Estimate pipeline stages for this block_n (conservative: no multicast for smem estimate)
        const int per_stage_approx = block_m * block_k * 4 + bn * block_k * 4
                                   + 128 * sf_pk * 4 + align(bn, 128) * sf_pk * 4;
        const int avail_smem = SM100ArchSpec::smem_capacity - 32768 - 200;
        const int est_stages = std::min(12, std::max(1, avail_smem / per_stage_approx));

        // Composite score: stages² × bn balances pipeline depth vs tile size.
        // Squared stages penalizes low pipeline depth, matching empirical sweep data.
        const int score = est_stages * est_stages * bn;

        bool success = false;
        if (best_block_n == 0 || num_waves < best_num_waves) {
            success = true;
        } else if (num_waves == best_num_waves && bn <= n && score > best_score) {
            success = true;
        }

        if (success) {
            best_block_n = bn;
            best_num_waves = num_waves;
            best_score = score;
        }
    }
    DG_HOST_ASSERT(best_block_n > 0);

    // Allow env override for benchmarking
    if (const auto env_bn = get_env<int>("DG_FP4_BLOCK_N"); env_bn > 0) {
        DG_HOST_ASSERT(env_bn % 16 == 0 && env_bn <= 256);
        best_block_n = env_bn;
    }

    // B-multicast for FP4: 2CTA along M, UMMA_M=256, each CTA loads half of B
    // A-multicast is not supported (2x1SM distributes along M only)
    MulticastConfig multicast_config = {1, false};
    if (m >= 512
        && is_multicast_legal(m, block_m, 2, num_sms, true)
        && (gemm_type == GemmType::Normal || gemm_type == GemmType::KGroupedContiguous)) {
        multicast_config = {2, false};  // B-multicast
    }

    // Find max pipeline stages that fit in shared memory
    constexpr int smem_capacity = SM100ArchSpec::smem_capacity;
    int best_num_stages = 0;
    SharedMemoryConfig best_smem_config;
    for (int num_stages = std::min(12, ceil_div(k, block_k)); num_stages > 0; --num_stages) {
        best_smem_config = get_smem_config<SM100ArchSpec>(kernel_type,
                                                          m, n, k,
                                                          block_m, best_block_n, block_k,
                                                          major_a, major_b,
                                                          ab_dtype, cd_dtype,
                                                          num_stages, multicast_config);
        if (best_smem_config.smem_size <= smem_capacity) {
            best_num_stages = num_stages;
            break;
        }
    }
    DG_HOST_ASSERT(best_num_stages != 0);

    const auto config = GemmConfig {
        .gemm_type = gemm_type,
        .kernel_type = kernel_type,
        .ab_dtype = ab_dtype,
        .cd_dtype = cd_dtype,
        .major_a = major_a,
        .major_b = major_b,
        .with_accumulation = with_accumulation,
        .block_m = block_m,
        .block_n = best_block_n,
        .block_k = block_k,
        .num_stages = best_num_stages,
        .num_last_stages = ceil_div(k, block_k) % best_num_stages,
        .num_sms = num_sms,
        .tc_util = device_runtime->get_tc_util(),
        .multicast_config = multicast_config,
        .smem_config = best_smem_config,
        .thread_config = SM100ArchSpec::get_thread_config(kernel_type, block_m, best_block_n)
    };

    // Print config
    if (get_env<int>("DG_JIT_DEBUG") || get_env<int>("DG_PRINT_CONFIGS")) {
        auto key = std::make_tuple(gemm_type, m, n, k, num_groups);
        static std::set<decltype(key)> printed;
        if (printed.count(key) == 0) {
            printf("FP4 GEMM: M: %d, N: %d, K: %d, groups: %d -> "
                   "block N: %d, stages: %d, last stages: %d, "
                   "shared memory: %d bytes, SMs: %d\n",
                   m, n, k, num_groups, best_block_n,
                   best_num_stages, config.num_last_stages,
                   best_smem_config.smem_size, num_sms);
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
