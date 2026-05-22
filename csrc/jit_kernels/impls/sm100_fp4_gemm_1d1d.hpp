#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM100FP4Gemm1D1DRuntime final: public LaunchRuntime<SM100FP4Gemm1D1DRuntime> {
public:
    struct Args {
        int m, n, k, num_groups;
        const std::string& compiled_dims;

        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void* grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_c;
        CUtensorMap tensor_map_d;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp4_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp4_gemm_1d1d_impl<
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {}, {}, {}
    >);
}};
)",
        to_string(args.gemm_config.major_a), to_string(args.gemm_config.major_b),
        get_compiled_dim(args.m, 'm', args.compiled_dims), get_compiled_dim(args.n, 'n', args.compiled_dims), get_compiled_dim(args.k, 'k', args.compiled_dims),
        args.gemm_config.block_m, args.gemm_config.block_n, args.gemm_config.block_k,
        args.num_groups,
        args.gemm_config.smem_config.swizzle_a_mode, args.gemm_config.smem_config.swizzle_b_mode, args.gemm_config.smem_config.swizzle_cd_mode,
        args.gemm_config.num_stages, args.gemm_config.num_last_stages,
        args.gemm_config.thread_config.num_non_epilogue_threads, args.gemm_config.thread_config.num_epilogue_threads,
        args.gemm_config.multicast_config.num_multicast, args.gemm_config.multicast_config.is_multicast_on_a,
        args.gemm_config.num_sms,
        to_string(args.gemm_config.gemm_type), args.gemm_config.with_accumulation, to_string(args.gemm_config.cd_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.grouped_layout, args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_c, args.tensor_map_d));
    }
};

static void sm100_fp4_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                const torch::Tensor& b, const torch::Tensor& sfb,
                                const std::optional<torch::Tensor>& c,
                                const torch::Tensor& d,
                                const int& m, const int& n, const int& k,
                                const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                const std::string& compiled_dims) {
    // 检测数据类型：如果是int32，说明是FP4打包数据
    const bool is_fp4_packed = (a.scalar_type() == torch::kInt);
    const auto actual_ab_dtype = is_fp4_packed ? torch::kInt : torch::kFloat8_e4m3fn;

    // 对齐 K 维度
    // 对于 FP8：对齐到 128 字节（128 FP8 元素）
    // 对于 FP4 打包 (int32)：对齐到 32 int32 元素（= 128 字节 = BLOCK_K）
    const int alignment = is_fp4_packed ? 32 : 128;
    const auto& aligned_k = align(k, alignment);

    // FP4 packed: zero-pad A/B along K to aligned_k so TMA reads zeros (not garbage) for OOB columns.
    // TMA uses OOB_FILL_NONE (required for int32), so without padding the kernel reads stale memory.
    auto a_work = a;
    auto b_work = b;
    int k_work = k;
    if (is_fp4_packed && k < aligned_k) {
        a_work = torch::zeros({m, aligned_k}, a.options());
        a_work.slice(1, 0, k).copy_(a);
        b_work = torch::zeros({n, aligned_k}, b.options());
        b_work.slice(1, 0, k).copy_(b);
        k_work = aligned_k;
    }

    // Use FP4-specific heuristic: block_m=128, B-multicast when M>=512, tighter TMEM
    auto config = is_fp4_packed
        ? get_best_fp4_config(GemmType::Normal,
                              m, n, k, 1, major_a, major_b,
                              d.scalar_type(), c.has_value(),
                              device_runtime->get_num_sms())
        : get_best_config<SM100ArchSpec>(
                              GemmType::Normal, KernelType::Kernel1D1D,
                              m, n, k, 1, major_a, major_b,
                              actual_ab_dtype, d.scalar_type(), c.has_value(),
                              device_runtime->get_num_sms());

    std::cout << "Using config: block_m=" << config.block_m
            << ", block_n=" << config.block_n
            << ", block_k=" << config.block_k
            << ", num_stages=" << config.num_stages
            << ", multicast=" << config.multicast_config.num_multicast
            << (config.multicast_config.is_multicast_on_a ? "(A)" : "(B)");
    if (is_fp4_packed) {
        std::cout << " (FP4 packed mode)";
    }
    std::cout << std::endl;
    const auto& cd = c.value_or(d);
    const auto& tensor_map_a = make_tma_a_desc(major_a, a_work, m, k_work,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a_work.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b_work, n, k_work,
                                               SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b_work.stride(get_non_contiguous_dim(major_b))), 1,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_d = make_tma_cd_desc(d, m, n,
                                                SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.smem_config.swizzle_cd_mode);
    const auto& tensor_map_c = make_tma_cd_desc(cd, m, n,
                                                SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                static_cast<int>(cd.stride(-2)), 1,
                                                config.smem_config.swizzle_cd_mode);
    // For FP4: VS=32 FP4 elements = 4 int32 per scale group.
    // make_tma_sf_desc uses ceil_div(shape_k, sf_block_k * 4) for SF K dim.
    // FP8: sf_block_k = block_k = 128 (VS=128), works: ceil_div(K, 512).
    // FP4: sf_block_k must be VS_in_int32 = 4, so: ceil_div(K_int32, 16) = correct packed cols.
    const int sf_block_k = is_fp4_packed ? 4 : config.block_k;
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, sf_block_k, 1, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                  config.block_n, sf_block_k, 1, 0);

    // Duplicate the accumulator if necessary
    if (c.has_value()) {
        if (c->data_ptr() == d.data_ptr()) {
            DG_HOST_ASSERT(c->sizes() == d.sizes() and c->strides() == d.strides());
        } else {
            // ReSharper disable once CppExpressionWithoutSideEffects
            d.copy_(c.value());
        }
    }

    // Launch
    const SM100FP4Gemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .num_groups = 1,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_c = tensor_map_c,
        .tensor_map_d = tensor_map_d
    };
    const auto& code = SM100FP4Gemm1D1DRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_fp4_gemm_1d1d", code);
    SM100FP4Gemm1D1DRuntime::launch(runtime, args);
}

static void sm100_m_grouped_fp4_gemm_contiguous_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                     const torch::Tensor& b, const torch::Tensor& sfb,
                                                     const torch::Tensor& d,
                                                     const torch::Tensor& m_indices,
                                                     const int& num_groups, const int& m, const int& n, const int& k,
                                                     const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                     const std::string& compiled_dims) {
    // FP4 packed: K is int32 count; align to 32 int32 (= 128 bytes = BLOCK_K)
    const auto& aligned_k = align(k, 32);

    // Zero-pad A/B along K to aligned_k if needed (TMA OOB_FILL_NONE on int32).
    // A: [M, K] → [M, aligned_K]. B: [G, N, K] → [G, N, aligned_K].
    auto a_work = a;
    auto b_work = b;
    int k_work = k;
    if (k < aligned_k) {
        a_work = torch::zeros({m, aligned_k}, a.options());
        a_work.slice(1, 0, k).copy_(a);
        b_work = torch::zeros({num_groups, n, aligned_k}, b.options());
        b_work.slice(2, 0, k).copy_(b);
        k_work = aligned_k;
    }

    const auto& config = get_best_fp4_config(GemmType::MGroupedContiguous,
                                             m, n, k, 1, major_a, major_b,
                                             d.scalar_type(), false,
                                             device_runtime->get_num_sms());

    // Create tensor descriptors. B carries num_groups in the outer dim;
    // A is 2D since the M-dim is already concatenated across groups (m_indices selects the group per row).
    const auto& tensor_map_a = make_tma_a_desc(major_a, a_work, m, k_work,
                                               SM100ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a_work.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    const auto& tensor_map_b = make_tma_b_desc(major_b, b_work, n, k_work,
                                               SM100ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b_work.stride(get_non_contiguous_dim(major_b))), num_groups,
                                               config.smem_config.swizzle_b_mode);
    const auto& tensor_map_d = make_tma_cd_desc(d, m, n,
                                                SM100ArchSpec::get_cd_store_block_m(config.block_m),
                                                SM100ArchSpec::get_cd_store_block_n(config.block_n),
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.smem_config.swizzle_cd_mode);
    // FP4 SF: sf_block_k = 4 (int32 unit), not config.block_k
    const int sf_block_k = 4;
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, sf_block_k, 1, 0);
    const auto& tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                  config.block_n, sf_block_k, num_groups, 0);

    const SM100FP4Gemm1D1DRuntime::Args& args = {
        .m = m, .n = n, .k = aligned_k,
        .num_groups = num_groups,
        .compiled_dims = compiled_dims,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .grouped_layout = m_indices.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_c = tensor_map_d,
        .tensor_map_d = tensor_map_d
    };
    const auto& code = SM100FP4Gemm1D1DRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_m_grouped_fp4_gemm_contiguous_1d1d", code);
    SM100FP4Gemm1D1DRuntime::launch(runtime, args);
}

} // namespace deep_gemm
