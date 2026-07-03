#pragma once

#include <cstdlib>
#include <string>

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm120.hpp"

#include "epilogue.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

static constexpr int kSM120FP8FP4AblationNoTMAStore = 1 << 0;
static constexpr int kSM120FP8FP4AblationNoEpilogueStore = 1 << 1;
static constexpr int kSM120FP8FP4AblationNoMMA = 1 << 2;
static constexpr int kSM120FP8FP4AblationNoSFLoad = 1 << 3;
static constexpr int kSM120FP8FP4AblationNoABTMALoad = 1 << 4;
static constexpr int kSM120FP8FP4AblationNoFragmentLoad = 1 << 5;
static constexpr int kSM120FP8FP4AblationNoSFTMALoad = 1 << 6;
static constexpr int kSM120FP8FP4AblationNoKPipeline = 1 << 7;
static constexpr int kSM120FP8FP4AblationScheduleBReuse16 = 1 << 8;
static constexpr int kSM120FP8FP4AblationSFTileCache = 1 << 9;
static constexpr int kSM120FP8FP4AblationNoATMALoad = 1 << 10;
static constexpr int kSM120FP8FP4AblationNoBTMALoad = 1 << 11;
static constexpr int kSM120FP8FP4AblationCTAParityAB = 1 << 12;
static constexpr int kSM120FP8FP4AblationEpiPrepack = 1 << 13;
static constexpr int kSM120FP8FP4AblationTMAStoreRing = 1 << 14;
static constexpr int kSM120FP8FP4AblationEpiPrepackOneMT = 1 << 15;
static constexpr int kSM120FP8FP4AblationTMAStoreRotating4 = 1 << 16;
static constexpr int kSM120FP8FP4AblationDirectGlobalStore = 1 << 17;
static constexpr int kSM120FP8FP4AblationEpi48DoubleBuffer = 1 << 18;
static constexpr int kSM120FP8FP4AblationSFTileCacheOneStage = 1 << 19;
static constexpr int kSM120FP8FP4AblationEpi48DoubleBufferWait0 = 1 << 20;

static int get_sm120_fp8_fp4_ablation_mode() {
    const char* mode_cstr = std::getenv("DG_ABLATION");
    if (mode_cstr == nullptr)
        return 0;

    auto add_token = [](const std::string& token, int& mode) {
        if (token.empty() or token == "baseline" or token == "none")
            return;
        if (token == "no_tma_store")
            mode |= kSM120FP8FP4AblationNoTMAStore;
        else if (token == "no_epilogue_store")
            mode |= kSM120FP8FP4AblationNoEpilogueStore;
        else if (token == "no_mma")
            mode |= kSM120FP8FP4AblationNoMMA;
        else if (token == "no_sf_load")
            mode |= kSM120FP8FP4AblationNoSFLoad;
        else if (token == "no_ab_tma_load")
            mode |= kSM120FP8FP4AblationNoABTMALoad;
        else if (token == "no_a_tma_load")
            mode |= kSM120FP8FP4AblationNoATMALoad;
        else if (token == "no_b_tma_load")
            mode |= kSM120FP8FP4AblationNoBTMALoad;
        else if (token == "no_fragment_load")
            mode |= kSM120FP8FP4AblationNoFragmentLoad;
        else if (token == "no_sf_tma_load")
            mode |= kSM120FP8FP4AblationNoSFTMALoad;
        else if (token == "no_k_pipeline")
            mode |= kSM120FP8FP4AblationNoKPipeline;
        else
            DG_HOST_ASSERT(false and "Unknown DG_ABLATION mode");
    };

    int mode = 0;
    std::string token;
    const std::string modes(mode_cstr);
    for (size_t i = 0; i <= modes.size(); ++i) {
        const char c = (i < modes.size()) ? modes[i] : '+';
        if (c == '+' or c == ',' or c == ';' or c == ' ' or c == '\t') {
            add_token(token, mode);
            token.clear();
        } else {
            token.push_back(c);
        }
    }
    return mode;
}

static bool get_sm120_fp8_fp4_clean_ablation_kernel() {
    const char* value = std::getenv("DG_ABLATION_CLEAN");
    return value != nullptr and std::string(value) != "" and std::string(value) != "0";
}

static bool get_sm120_fp8_fp4_schedule_b_reuse16() {
    const char* value = std::getenv("DG_CTA_SWIZZLE_B_REUSE16");
    return value != nullptr and std::string(value) != "" and std::string(value) != "0";
}

static bool get_sm120_fp8_fp4_sf_tile_cache() {
    const char* value = std::getenv("DG_SF_TILE_CACHE");
    return value != nullptr and std::string(value) != "" and std::string(value) != "0";
}

static bool get_sm120_fp8_fp4_cta_parity_ab() {
    const char* value = std::getenv("DG_TMA_CTA_PARITY_AB");
    return value != nullptr and std::string(value) != "" and std::string(value) != "0";
}

static bool get_sm120_fp8_fp4_epi_prepack() {
    const char* value = std::getenv("DG_EPI_PREPACK_BEFORE_TMA_WAIT");
    return value != nullptr and std::string(value) != "" and std::string(value) != "0";
}

static bool get_sm120_fp8_fp4_tma_store_ring() {
    const char* value = std::getenv("DG_TMA_STORE_RING");
    return value != nullptr and std::string(value) != "" and std::string(value) != "0";
}

static bool get_sm120_fp8_fp4_epi_prepack_one_mt() {
    const char* value = std::getenv("DG_EPI_PREPACK_ONE_MT");
    return value != nullptr and std::string(value) != "" and std::string(value) != "0";
}

static bool get_sm120_fp8_fp4_tma_store_rotating4() {
    const char* value = std::getenv("DG_TMA_STORE_ROTATING_4");
    return value != nullptr and std::string(value) != "" and std::string(value) != "0";
}

static bool get_sm120_fp8_fp4_direct_global_store() {
    const char* value = std::getenv("DG_EPI_DIRECT_GMEM_STORE");
    return value != nullptr and std::string(value) != "" and std::string(value) != "0";
}

static int get_sm120_fp8_fp4_epi48_double_buffer() {
    const char* value = std::getenv("DG_TMA_STORE_EPI48_DOUBLE_BUFFER");
    if (value == nullptr or std::string(value).empty() or std::string(value) == "0")
        return 0;
    if (std::string(value) == "1")
        return 1;
    if (std::string(value) == "wait0" or std::string(value) == "serialized")
        return 2;
    DG_HOST_ASSERT(false and
                   "DG_TMA_STORE_EPI48_DOUBLE_BUFFER must be 0, 1, wait0, or serialized");
    return 0;
}

static bool get_sm120_fp8_fp4_sf_tile_cache_one_stage() {
    const char* value = std::getenv("DG_SF_TILE_CACHE_ONE_STAGE");
    return value != nullptr and std::string(value) != "" and std::string(value) != "0";
}

class SM120FP8FP4Gemm1D1DRuntime final: public LaunchRuntime<SM120FP8FP4Gemm1D1DRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;
        const std::optional<std::string> epilogue_type;

        int gran_k_a, gran_k_b;
        bool is_fp4;
        bool b_is_fp4;
        bool a_is_fp4;
        bool k_grouped_constant_stride;
        int stride_cd_m;
        int stride_cd_n;
        int stride_cd_batch;

        void* gmem_d;
        void* gmem_c;
        void* gmem_a_ptr;
        void* gmem_b_ptr;
        void* gmem_workspace;
        void* grouped_layout;
        void* tensor_map_buffer;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;
        int ablation_mode;
        bool use_clean_ablation_kernel;
    };

    static std::string generate_impl(const Args& args) {
        const char* kernel_name = args.use_clean_ablation_kernel
            ? "sm120_fp8_fp4_gemm_1d1d_clean_ablation_impl"
            : "sm120_fp8_fp4_gemm_1d1d_impl";
        return fmt::format(R"(
#include <deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{}<
        {}, {}, {},
        {}, {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}
    >);
}};
)",
        kernel_name,
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gran_k_a, args.gran_k_b,
        args.gemm_desc.num_groups,
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode,
        args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_tma_threads, args.gemm_config.launch_config.num_math_threads,
        args.gemm_config.launch_config.num_sms,
        to_string(args.gemm_desc.gemm_type), args.gemm_desc.with_accumulation,
        to_string(args.gemm_desc.cd_dtype),
        get_default_epilogue_type(args.epilogue_type),
        args.is_fp4 ? "true" : "false",
        args.b_is_fp4 ? "true" : "false",
        args.a_is_fp4 ? "true" : "false",
        (args.gemm_desc.major_b == cute::UMMA::Major::K) ? "true" : "false",
        args.k_grouped_constant_stride ? "true" : "false",
        args.gemm_config.storage_config.store_block_m,
        args.ablation_mode,
        args.gemm_config.split_k_factor);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.gmem_d, args.gmem_c,
            args.gmem_a_ptr, args.gmem_b_ptr,
            args.grouped_layout,
            args.tensor_map_buffer,
            args.gmem_workspace,
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.stride_cd_m, args.stride_cd_n, args.stride_cd_batch,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd));
    }
};

class SM120SplitKReduceRuntime final: public LaunchRuntime<SM120SplitKReduceRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;
        int stride_cd_m;
        int stride_cd_n;
        void* gmem_d;
        void* workspace;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm120_split_k_reduce.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_split_k_reduce_impl<{}, {}>);
}};
)",
        to_string(args.gemm_desc.cd_dtype),
        args.gemm_config.split_k_factor);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.gmem_d, args.workspace,
            args.gemm_desc.m, args.gemm_desc.n, args.stride_cd_m, args.stride_cd_n));
    }
};

static void sm120_split_k_reduce(const torch::Tensor& workspace, const torch::Tensor& d,
                                  const int& m, const int& n, const int& split_k,
                                  const int stride_cd_m, const int stride_cd_n) {
    const int total = m * n;
    const int threads = 256;
    const int blocks = ceil_div(total, threads);

    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = 0, .num_groups = 1,
        .a_dtype = torch::kFloat, .b_dtype = torch::kFloat,
        .cd_dtype = d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = false,
        .num_sms = blocks,
        .tc_util = 100, .compiled_dims = ""
    };
    const auto config = GemmConfig {
        .split_k_factor = split_k
    };

    const SM120SplitKReduceRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(blocks, threads, 0, 0),
        .stride_cd_m = stride_cd_m,
        .stride_cd_n = stride_cd_n,
        .gmem_d = d.data_ptr(),
        .workspace = workspace.data_ptr(),
    };
    const auto code = SM120SplitKReduceRuntime::generate(args);
    const auto runtime = compiler->build("sm120_split_k_reduce", code);
    SM120SplitKReduceRuntime::launch(runtime, args);
}

static void sm120_fp8_fp4_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                    const torch::Tensor& b, const torch::Tensor& sfb,
                                    const std::optional<torch::Tensor>& c,
                                    const torch::Tensor& d,
                                    const int& m, const int& n, const int& k,
                                    const int& gran_k_a, const int& gran_k_b,
                                    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                    const std::string& compiled_dims,
                                    const std::optional<std::string>& epilogue_type = std::nullopt,
                                    const bool swap_ab = false) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const bool a_dt_fp4 = (a.scalar_type() == kPackedFP4);
    const bool b_dt_fp4 = (b.scalar_type() == kPackedFP4);
    const bool is_fp4 = a_dt_fp4 && b_dt_fp4;       // symmetric FP4xFP4 (k64)
    const bool b_is_fp4 = (!a_dt_fp4) && b_dt_fp4;  // FP8_A x FP4_B (k32 mixed)
    const bool a_is_fp4 = a_dt_fp4 && (!b_dt_fp4);  // FP4_A x FP8_B (k32 mixed, swapAB)

    auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .max_gran_k = std::max(gran_k_a, gran_k_b),
        .cd_n_contiguous = !swap_ab  // strided-store epilogue (AB-swap output is transposed)
    };

    auto config = get_best_config<SM120ArchSpec>(desc);
    config.split_k_factor = SM120ArchSpec::get_split_k_factor(desc, config.layout);

    const bool sf_tile_cache = get_sm120_fp8_fp4_sf_tile_cache();
    const bool cta_parity_ab = get_sm120_fp8_fp4_cta_parity_ab();
    const bool epi_prepack = get_sm120_fp8_fp4_epi_prepack();
    const bool tma_store_ring = get_sm120_fp8_fp4_tma_store_ring();
    const bool epi_prepack_one_mt = get_sm120_fp8_fp4_epi_prepack_one_mt();
    const bool tma_store_rotating4 = get_sm120_fp8_fp4_tma_store_rotating4();
    const bool direct_global_store = get_sm120_fp8_fp4_direct_global_store();
    const int epi48_double_buffer = get_sm120_fp8_fp4_epi48_double_buffer();
    const bool sf_tile_cache_one_stage = get_sm120_fp8_fp4_sf_tile_cache_one_stage();

    const auto cd = c.value_or(d);
    const bool fp4_unpacked = !is_fp4;
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode, 0, false, fp4_unpacked);
    // Mixed FP8xFP4: B uses 16U4_ALIGN16B (fp4_unpacked_smem=true) for .b4x16_p64 padded format
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                              config.storage_config.swizzle_b_mode, 0, false,
                                              b_is_fp4 ? true : fp4_unpacked);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, gran_k_a, 1, 0, 0, false,
                                                 sf_tile_cache ? 4 : 1);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 config.layout.block_n, gran_k_b, 1, 0, 0, false,
                                                 sf_tile_cache ? 4 : 1);
    const int d_n = static_cast<int>(d.size(-1));
    const int d_stride = static_cast<int>(d.stride(-2));
    const int cd_store_m = config.storage_config.store_block_m > 0
        ? config.storage_config.store_block_m : config.layout.block_m;
    const auto tensor_map_cd = make_tma_cd_desc(d, m, d_n,
                                                cd_store_m, config.layout.block_n,
                                                d_stride, 1,
                                                config.storage_config.swizzle_cd_mode);

    const int split_k = config.split_k_factor;
    int ablation_mode = get_sm120_fp8_fp4_ablation_mode();
    const bool schedule_b_reuse16 = get_sm120_fp8_fp4_schedule_b_reuse16();
    if (schedule_b_reuse16)
        ablation_mode |= kSM120FP8FP4AblationScheduleBReuse16;
    if (sf_tile_cache)
        ablation_mode |= kSM120FP8FP4AblationSFTileCache;
    if (cta_parity_ab)
        ablation_mode |= kSM120FP8FP4AblationCTAParityAB;
    if (epi_prepack)
        ablation_mode |= kSM120FP8FP4AblationEpiPrepack;
    if (tma_store_ring)
        ablation_mode |= kSM120FP8FP4AblationTMAStoreRing;
    if (epi_prepack_one_mt)
        ablation_mode |= kSM120FP8FP4AblationEpiPrepackOneMT;
    if (tma_store_rotating4)
        ablation_mode |= kSM120FP8FP4AblationTMAStoreRotating4;
    if (direct_global_store)
        ablation_mode |= kSM120FP8FP4AblationDirectGlobalStore;
    if (epi48_double_buffer != 0)
        ablation_mode |= kSM120FP8FP4AblationEpi48DoubleBuffer;
    if (sf_tile_cache_one_stage)
        ablation_mode |= kSM120FP8FP4AblationSFTileCacheOneStage;
    if (epi48_double_buffer == 2)
        ablation_mode |= kSM120FP8FP4AblationEpi48DoubleBufferWait0;
    const bool clean_ablation_requested = get_sm120_fp8_fp4_clean_ablation_kernel();
    const bool clean_ablation_tile =
        (config.layout.block_m == 192 and config.layout.block_n == 96 and config.layout.block_k == 64 and
         config.storage_config.swizzle_cd_mode == 64 and
         (config.storage_config.store_block_m == 48 or config.storage_config.store_block_m == 64 or
          config.storage_config.store_block_m == 96) and
         config.pipeline_config.num_stages == 4) or
        (config.layout.block_m == 192 and config.layout.block_n == 96 and config.layout.block_k == 128 and
         config.storage_config.swizzle_cd_mode == 64 and
         (config.storage_config.store_block_m == 64 or config.storage_config.store_block_m == 96) and
         config.pipeline_config.num_stages == 2) or
        (config.layout.block_m == 128 and config.layout.block_n == 128 and config.layout.block_k == 64 and
         config.storage_config.swizzle_cd_mode == 128 and config.storage_config.store_block_m == 64 and
         config.pipeline_config.num_stages == 4);
    const bool use_clean_ablation_kernel = clean_ablation_requested and
        desc.gemm_type == GemmType::Normal and
        desc.kernel_type == KernelType::Kernel1D1D and
        desc.m == 3072 and desc.n == 3072 and desc.k == 2048 and
        gran_k_a == 128 and gran_k_b == 128 and
        desc.a_dtype == torch::kFloat8_e4m3fn and desc.b_dtype == torch::kFloat8_e4m3fn and
        desc.cd_dtype == torch::kBFloat16 and not desc.with_accumulation and
        major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K and
        config.storage_config.swizzle_a_mode == config.storage_config.swizzle_b_mode and
        (config.storage_config.swizzle_a_mode == 64 or config.storage_config.swizzle_a_mode == 128) and
        clean_ablation_tile and
        config.launch_config.num_tma_threads == 128 and config.launch_config.num_math_threads == 256 and
        split_k == 1 and not epilogue_type.has_value() and not swap_ab;
    if (clean_ablation_requested)
        DG_HOST_ASSERT(use_clean_ablation_kernel and
                       "DG_ABLATION_CLEAN=1 only supports normal FP8xFP8 BF16 3072x3072x2048 with 192x96x64 stage4 epilogue48/64/96, 192x96x128 stage2, or 128x128x64 stage4");
    if (schedule_b_reuse16)
        DG_HOST_ASSERT(use_clean_ablation_kernel and
                       "DG_CTA_SWIZZLE_B_REUSE16=1 is only enabled for the clean ablation kernel");
    const bool exact_192x96_stage4_target = use_clean_ablation_kernel and
        config.layout.block_m == 192 and config.layout.block_n == 96 and config.layout.block_k == 64 and
        config.pipeline_config.num_stages == 4;
    const bool exact_ab_overlap_target = exact_192x96_stage4_target and
        config.storage_config.store_block_m == 64;
    const bool exact_sf_cache_target = exact_192x96_stage4_target and
        (config.storage_config.store_block_m == 48 or config.storage_config.store_block_m == 64);
    const bool exact_epi48_target = exact_192x96_stage4_target and
        config.storage_config.store_block_m == 48;
    if (sf_tile_cache)
        DG_HOST_ASSERT(exact_sf_cache_target and
                       "DG_SF_TILE_CACHE=1 requires the clean 192x96x64 stage4 epilogue48/64 kernel");
    if (sf_tile_cache_one_stage)
        DG_HOST_ASSERT(sf_tile_cache and exact_epi48_target and
                       "DG_SF_TILE_CACHE_ONE_STAGE=1 requires the clean epilogue48 SF-cache kernel");
    if (epi48_double_buffer != 0)
        DG_HOST_ASSERT(exact_epi48_target and (not sf_tile_cache or sf_tile_cache_one_stage) and
                       "DG_TMA_STORE_EPI48_DOUBLE_BUFFER requires clean epilogue48 and either no SF cache or its one-stage capacity control");
    if (cta_parity_ab)
        DG_HOST_ASSERT(exact_ab_overlap_target and
                       "DG_TMA_CTA_PARITY_AB=1 requires the clean 192x96x64 stage4 epilogue64 kernel");
    if (epi_prepack)
        DG_HOST_ASSERT(exact_ab_overlap_target and
                       "DG_EPI_PREPACK_BEFORE_TMA_WAIT=1 requires the clean 192x96x64 stage4 epilogue64 kernel");
    if (tma_store_ring)
        DG_HOST_ASSERT(exact_ab_overlap_target and
                       "DG_TMA_STORE_RING=1 requires the clean 192x96x64 stage4 epilogue64 kernel");
    if (epi_prepack_one_mt)
        DG_HOST_ASSERT(exact_ab_overlap_target and
                       "DG_EPI_PREPACK_ONE_MT=1 requires the clean 192x96x64 stage4 epilogue64 kernel");
    if (tma_store_rotating4)
        DG_HOST_ASSERT(exact_ab_overlap_target and
                       "DG_TMA_STORE_ROTATING_4=1 requires the clean 192x96x64 stage4 epilogue64 kernel");
    if (direct_global_store)
        DG_HOST_ASSERT(exact_ab_overlap_target and d.stride(-1) == 1 and d_stride % 2 == 0 and
                       "DG_EPI_DIRECT_GMEM_STORE=1 requires the clean 192x96x64 stage4 epilogue64 kernel");
    DG_HOST_ASSERT(static_cast<int>(epi_prepack) + static_cast<int>(tma_store_ring) +
                       static_cast<int>(epi_prepack_one_mt) + static_cast<int>(tma_store_rotating4) +
                       static_cast<int>(direct_global_store) + static_cast<int>(epi48_double_buffer != 0) <= 1 and
                   "Epilogue overlap switches are separate experiments");
    if (ablation_mode & (kSM120FP8FP4AblationNoATMALoad | kSM120FP8FP4AblationNoBTMALoad))
        DG_HOST_ASSERT(use_clean_ablation_kernel and
                       "no_a_tma_load and no_b_tma_load are only enabled for the clean ablation kernel");
    const bool needs_workspace = split_k > 1 and
        not (ablation_mode & kSM120FP8FP4AblationNoEpilogueStore);
    torch::Tensor workspace;
    if (needs_workspace)
        workspace = torch::empty({split_k, m, n}, d.options().dtype(torch::kFloat));

    // Four existing per-stage SF slabs form the first whole-tile bank.
    const int sf_tile_cache_extra_smem = sf_tile_cache and not sf_tile_cache_one_stage
        ? 4 * (config.layout.block_m + config.layout.block_n) * static_cast<int>(sizeof(int32_t))
        : 0;
    // Both TMA overlap rings grow SMEM_D from three to four 64x32 BF16 atoms.
    const int tma_store_ring_extra_smem = (tma_store_ring or tma_store_rotating4)
        ? config.storage_config.swizzle_cd_mode * config.storage_config.store_block_m
        : 0;
    // A full second epilogue48 output bank contains three 64x48 BF16 TMA atoms.
    const int epi48_double_buffer_extra_smem = epi48_double_buffer != 0
        ? 3 * config.storage_config.swizzle_cd_mode * config.storage_config.store_block_m
        : 0;
    const int launch_smem_size = config.pipeline_config.smem_size +
        sf_tile_cache_extra_smem + tma_store_ring_extra_smem + epi48_double_buffer_extra_smem;
    DG_HOST_ASSERT(launch_smem_size <= SM120ArchSpec::smem_capacity);

    const SM120FP8FP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  launch_smem_size,
                                  1),
        .epilogue_type = epilogue_type,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        .is_fp4 = is_fp4,
        .b_is_fp4 = b_is_fp4,
        .a_is_fp4 = a_is_fp4,
        .k_grouped_constant_stride = false,
        .stride_cd_m = swap_ab ? static_cast<int>(d.stride(-1)) : d_stride,
        .stride_cd_n = swap_ab ? static_cast<int>(d.stride(-2)) : 0,
        .stride_cd_batch = 0,
        .gmem_d = d.data_ptr(),
        .gmem_c = c.has_value() ? cd.data_ptr() : nullptr,
        .gmem_a_ptr = nullptr,
        .gmem_b_ptr = nullptr,
        .gmem_workspace = needs_workspace ? workspace.data_ptr() : nullptr,
        .grouped_layout = nullptr,
        .tensor_map_buffer = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd,
        .ablation_mode = ablation_mode,
        .use_clean_ablation_kernel = use_clean_ablation_kernel,
    };

    const auto code = SM120FP8FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm120_fp8_fp4_gemm_1d1d", code);
    SM120FP8FP4Gemm1D1DRuntime::launch(runtime, args);

    if (split_k > 1) {
        const int reduce_stride_m = swap_ab ? static_cast<int>(d.stride(-1)) : d_stride;
        const int reduce_stride_n = swap_ab ? static_cast<int>(d.stride(-2)) : 1;
        sm120_split_k_reduce(workspace, d, m, n, split_k, reduce_stride_m, reduce_stride_n);
    }
}

static void sm120_k_grouped_fp8_fp4_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                               const torch::Tensor& b, const torch::Tensor& sfb,
                                               const std::optional<torch::Tensor>& c,
                                               const torch::Tensor& d,
                                               const int& m, const int& n,
                                               const std::vector<int>& ks, const torch::Tensor& ks_tensor,
                                               const torch::Tensor& tensor_map_buffer,
                                               const int& gran_k_a, const int& gran_k_b,
                                               const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                               const std::string& compiled_dims,
                                               const bool k_grouped_constant_stride = false,
                                               const int outer_stride_k_override = 0) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(c.has_value());

    const bool a_dt_fp4 = (a.scalar_type() == kPackedFP4);
    const bool b_dt_fp4 = (b.scalar_type() == kPackedFP4);
    const bool is_fp4 = a_dt_fp4 && b_dt_fp4;       // symmetric FP4xFP4 (k64)
    const bool b_is_fp4 = (!a_dt_fp4) && b_dt_fp4;  // FP8_A x FP4_B (k32 mixed)
    const bool a_is_fp4 = a_dt_fp4 && (!b_dt_fp4);  // FP4_A x FP8_B (k32 mixed, swapAB)
    const auto num_groups = static_cast<int>(ks.size());
    int first_k = 0, sum_k = 0, max_k = 0;
    int sum_sf_k_a = 0, sum_sf_k_b = 0;
    for (int i = 0; i < num_groups; ++i) {
        if (first_k == 0 and ks[i] != 0)
            first_k = ks[i];
        sum_k += ks[i];
        sum_sf_k_a += ceil_div(ks[i], gran_k_a * 4);
        sum_sf_k_b += ceil_div(ks[i], gran_k_b * 4);
        max_k = std::max(max_k, ks[i]);
        DG_HOST_ASSERT(ks[i] % 64 == 0);
    }

    const auto desc = GemmDesc {
        .gemm_type = GemmType::KGroupedContiguous,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = sum_k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .max_gran_k = std::max(gran_k_a, gran_k_b),
        .expected_m = m, .expected_n = n, .expected_k = max_k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM120ArchSpec>(desc);

    const auto cd = c.value_or(d);
    const bool fp4_unpacked = !is_fp4;
    const int effective_stride = (outer_stride_k_override > 0) ? outer_stride_k_override : first_k;
    const int outer_stride_k_a = is_fp4 ? (effective_stride / 2) : effective_stride;
    const int outer_stride_k_b = (is_fp4 || b_is_fp4) ? (effective_stride / 2) : effective_stride;
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, first_k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k, outer_stride_k_a, 1,
                                              config.storage_config.swizzle_a_mode, 0, false, fp4_unpacked);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, first_k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k, outer_stride_k_b, 1,
                                              config.storage_config.swizzle_b_mode, 0, false,
                                              b_is_fp4 ? true : fp4_unpacked);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, sum_sf_k_a * gran_k_a * 4,
                                                 config.layout.block_m, gran_k_a, 1, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, sum_sf_k_b * gran_k_b * 4,
                                                 config.layout.block_n, gran_k_b, 1, 0);
    const int cd_store_m = config.storage_config.store_block_m > 0
        ? config.storage_config.store_block_m : config.layout.block_m;
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                cd_store_m, config.layout.block_n,
                                                n, num_groups,
                                                config.storage_config.swizzle_cd_mode);

    const SM120FP8FP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  1),
        .epilogue_type = std::nullopt,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        .is_fp4 = is_fp4,
        .b_is_fp4 = b_is_fp4,
        .a_is_fp4 = a_is_fp4,
        .k_grouped_constant_stride = k_grouped_constant_stride,
        .stride_cd_m = n,
        .stride_cd_n = 0,
        .stride_cd_batch = 0,
        .gmem_d = d.data_ptr(),
        .gmem_c = cd.data_ptr(),
        .gmem_a_ptr = a.data_ptr(),
        .gmem_b_ptr = b.data_ptr(),
        .gmem_workspace = nullptr,
        .grouped_layout = ks_tensor.data_ptr(),
        .tensor_map_buffer = tensor_map_buffer.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd,
        .ablation_mode = get_sm120_fp8_fp4_ablation_mode(),
        .use_clean_ablation_kernel = false,
    };
    const auto code = SM120FP8FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm120_k_grouped_fp8_fp4_gemm_1d1d", code);
    SM120FP8FP4Gemm1D1DRuntime::launch(runtime, args);
}

static void sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                         const torch::Tensor& b, const torch::Tensor& sfb,
                                                         const torch::Tensor& d,
                                                         const torch::Tensor& grouped_layout,
                                                         const int& num_groups, const int& m, const int& n, const int& k,
                                                         const int& gran_k_a, const int& gran_k_b,
                                                         const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                         const std::string& compiled_dims,
                                                         const bool& use_psum_layout,
                                                         const std::optional<int>& expected_m_for_psum_layout) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const auto gemm_type = use_psum_layout ?
        GemmType::MGroupedContiguousWithPsumLayout : GemmType::MGroupedContiguous;

    if (expected_m_for_psum_layout)
        DG_HOST_ASSERT(use_psum_layout);

    const bool a_dt_fp4 = (a.scalar_type() == kPackedFP4);
    const bool b_dt_fp4 = (b.scalar_type() == kPackedFP4);
    const bool is_fp4 = a_dt_fp4 && b_dt_fp4;       // symmetric FP4xFP4 (k64)
    const bool b_is_fp4 = (!a_dt_fp4) && b_dt_fp4;  // FP8_A x FP4_B (k32 mixed)
    const bool a_is_fp4 = a_dt_fp4 && (!b_dt_fp4);  // FP4_A x FP8_B (k32 mixed, swapAB)

    const auto desc = GemmDesc {
        .gemm_type = gemm_type,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .max_gran_k = std::max(gran_k_a, gran_k_b),
        .expected_m = expected_m_for_psum_layout.value_or(m),
        .expected_n = n, .expected_k = k,
        .expected_num_groups = expected_m_for_psum_layout.has_value() ? num_groups : 1
    };
    const auto config = get_best_config<SM120ArchSpec>(desc);

    const bool fp4_unpacked = !is_fp4;
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode, 0, false, fp4_unpacked);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode, 0, false,
                                              b_is_fp4 ? true : fp4_unpacked);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, gran_k_a, 1, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 config.layout.block_n, gran_k_b, num_groups, 0);
    const int cd_store_m = config.storage_config.store_block_m > 0
        ? config.storage_config.store_block_m : config.layout.block_m;
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                cd_store_m, config.layout.block_n,
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.storage_config.swizzle_cd_mode);

    const SM120FP8FP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  1),
        .epilogue_type = std::nullopt,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        .is_fp4 = is_fp4,
        .b_is_fp4 = b_is_fp4,
        .a_is_fp4 = a_is_fp4,
        .k_grouped_constant_stride = false,
        .stride_cd_m = n,
        .stride_cd_n = 0,
        .stride_cd_batch = 0,
        .gmem_d = d.data_ptr(),
        .gmem_c = nullptr,
        .gmem_a_ptr = nullptr,
        .gmem_b_ptr = nullptr,
        .gmem_workspace = nullptr,
        .grouped_layout = grouped_layout.data_ptr(),
        .tensor_map_buffer = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd,
        .ablation_mode = get_sm120_fp8_fp4_ablation_mode(),
        .use_clean_ablation_kernel = false,
    };
    const auto code = SM120FP8FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d", code);
    SM120FP8FP4Gemm1D1DRuntime::launch(runtime, args);
}

static void sm120_m_grouped_fp8_fp4_gemm_masked_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                     const torch::Tensor& b, const torch::Tensor& sfb,
                                                     const torch::Tensor& d,
                                                     const torch::Tensor& masked_m,
                                                     const int& num_groups, const int& m, const int& n, const int& k,
                                                     const int& expected_m,
                                                     const int& gran_k_a, const int& gran_k_b,
                                                     const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                     const std::string& compiled_dims) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const bool a_dt_fp4 = (a.scalar_type() == kPackedFP4);
    const bool b_dt_fp4 = (b.scalar_type() == kPackedFP4);
    const bool is_fp4 = a_dt_fp4 && b_dt_fp4;       // symmetric FP4xFP4 (k64)
    const bool b_is_fp4 = (!a_dt_fp4) && b_dt_fp4;  // FP8_A x FP4_B (k32 mixed)
    const bool a_is_fp4 = a_dt_fp4 && (!b_dt_fp4);  // FP4_A x FP8_B (k32 mixed, swapAB)

    const auto desc = GemmDesc {
        .gemm_type = GemmType::MGroupedMasked,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .max_gran_k = std::max(gran_k_a, gran_k_b),
        .expected_m = expected_m, .expected_n = n, .expected_k = k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM120ArchSpec>(desc);

    const bool fp4_unpacked = !is_fp4;
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), num_groups,
                                              config.storage_config.swizzle_a_mode, 0, false, fp4_unpacked);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode, 0, false,
                                              b_is_fp4 ? true : fp4_unpacked);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, gran_k_a, num_groups, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 config.layout.block_n, gran_k_b, num_groups, 0);
    const int cd_store_m = config.storage_config.store_block_m > 0
        ? config.storage_config.store_block_m : config.layout.block_m;
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                cd_store_m, config.layout.block_n,
                                                static_cast<int>(d.stride(-2)), num_groups,
                                                config.storage_config.swizzle_cd_mode);

    const SM120FP8FP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  1),
        .epilogue_type = std::nullopt,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        .is_fp4 = is_fp4,
        .b_is_fp4 = b_is_fp4,
        .a_is_fp4 = a_is_fp4,
        .k_grouped_constant_stride = false,
        .stride_cd_m = n,
        .stride_cd_n = 0,
        .stride_cd_batch = 0,
        .gmem_d = d.data_ptr(),
        .gmem_c = nullptr,
        .gmem_a_ptr = nullptr,
        .gmem_b_ptr = nullptr,
        .gmem_workspace = nullptr,
        .grouped_layout = masked_m.data_ptr(),
        .tensor_map_buffer = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd,
        .ablation_mode = get_sm120_fp8_fp4_ablation_mode(),
        .use_clean_ablation_kernel = false,
    };
    const auto code = SM120FP8FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm120_m_grouped_fp8_fp4_gemm_masked_1d1d", code);
    SM120FP8FP4Gemm1D1DRuntime::launch(runtime, args);
}

static void sm120_fp8_fp4_bmm(const torch::Tensor& a, const torch::Tensor& sfa,
                              const torch::Tensor& b, const torch::Tensor& sfb,
                              const std::optional<torch::Tensor>& c,
                              const torch::Tensor& d,
                              const int& batch_size, const int& m, const int& n, const int& k,
                              const int& gran_k_a, const int& gran_k_b,
                              const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                              const std::string& compiled_dims,
                              const bool swap_ab = false) {
    // Requires K-major operands; callers .contiguous() (MN-major scalar path is ~3x slower).
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const bool a_dt_fp4 = (a.scalar_type() == kPackedFP4);
    const bool b_dt_fp4 = (b.scalar_type() == kPackedFP4);
    const bool is_fp4 = a_dt_fp4 && b_dt_fp4;       // symmetric FP4xFP4 (k64)
    const bool b_is_fp4 = (!a_dt_fp4) && b_dt_fp4;  // FP8_A x FP4_B (k32 mixed)
    const bool a_is_fp4 = a_dt_fp4 && (!b_dt_fp4);  // FP4_A x FP8_B (k32 mixed, swapAB)

    const auto desc = GemmDesc {
        .gemm_type = GemmType::Batched,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = batch_size,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .max_gran_k = std::max(gran_k_a, gran_k_b),
        .cd_n_contiguous = !swap_ab  // strided-store epilogue (AB-swap output is transposed)
    };
    const auto config = get_best_config<SM120ArchSpec>(desc);

    const auto cd = c.value_or(d);
    const bool fp4_unpacked = !is_fp4;
    const auto tensor_map_a = make_tma_3d_desc(a, k, m, batch_size,
                                               config.layout.block_k, config.storage_config.load_block_m, 1,
                                               a.stride(1), a.stride(0),
                                               config.storage_config.swizzle_a_mode, 0, false, fp4_unpacked);
    const auto tensor_map_b = make_tma_3d_desc(b, k, n, batch_size,
                                               config.layout.block_k, config.storage_config.load_block_n, 1,
                                               b.stride(1), b.stride(0),
                                               config.storage_config.swizzle_b_mode, 0, false,
                                               b_is_fp4 ? true : fp4_unpacked);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, gran_k_a, batch_size, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 config.layout.block_n, gran_k_b, batch_size, 0);
    const int cd_store_m = config.storage_config.store_block_m > 0
        ? config.storage_config.store_block_m : config.layout.block_m;
    const auto tensor_map_cd = make_tma_3d_desc(d, n, m, batch_size,
                                                config.layout.block_n, cd_store_m, 1,
                                                d.stride(1), d.stride(0),
                                                config.storage_config.swizzle_cd_mode);

    const SM120FP8FP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  1),
        .epilogue_type = std::nullopt,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        .is_fp4 = is_fp4,
        .b_is_fp4 = b_is_fp4,
        .a_is_fp4 = a_is_fp4,
        .k_grouped_constant_stride = false,
        .stride_cd_m = swap_ab ? static_cast<int>(d.stride(-1)) : static_cast<int>(d.stride(1)),
        .stride_cd_n = swap_ab ? static_cast<int>(d.stride(1)) : 0,
        .stride_cd_batch = static_cast<int>(d.stride(0)),
        .gmem_d = d.data_ptr(),
        .gmem_c = c.has_value() ? cd.data_ptr() : nullptr,
        .gmem_a_ptr = nullptr,
        .gmem_b_ptr = nullptr,
        .gmem_workspace = nullptr,
        .grouped_layout = nullptr,
        .tensor_map_buffer = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd,
        .ablation_mode = get_sm120_fp8_fp4_ablation_mode(),
        .use_clean_ablation_kernel = false,
    };
    const auto code = SM120FP8FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm120_fp8_fp4_bmm", code);
    SM120FP8FP4Gemm1D1DRuntime::launch(runtime, args);
}

} // namespace deep_gemm
