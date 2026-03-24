#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/scheduler.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm100;

// E2M1 FP4 到 float 的转换函数
// E2M1 格式: 4 bits = SEEM (S=符号 1bit, E=指数 2bits, M=尾数 1bit)
__device__ __forceinline__ float fp4_e2m1_to_float(uint32_t fp4_bits) {
    constexpr float E2M1_LUT[16] = {
         0.0f,   0.5f,   1.0f,   1.5f,   2.0f,   3.0f,   4.0f,   6.0f,  // 正数 (S=0)
        -0.0f,  -0.5f,  -1.0f,  -1.5f,  -2.0f,  -3.0f,  -4.0f,  -6.0f   // 负数 (S=1)
    };
    return E2M1_LUT[fp4_bits & 0xF];
}

// Swizzle-aware shared memory index for reading TMA-loaded data.
// TMA stores data with bank-group XOR swizzle: physical_bank = logical_bank ^ (row % num_banks).
// swizzle_mode: kSwizzleAMode or kSwizzleBMode (bytes, e.g. 128)
// row: M or N row index, k: K column index, block_k: elements per row
template <uint32_t swizzle_mode>
__device__ __forceinline__ uint32_t swizzled_smem_k_major_idx(uint32_t row, uint32_t k, uint32_t block_k) {
    constexpr uint32_t kElemBytes = sizeof(uint32_t);
    constexpr uint32_t kBankBytes = 16;
    constexpr uint32_t kElemsPerBank = kBankBytes / kElemBytes;            // 4
    constexpr uint32_t kNumBanks = swizzle_mode / kBankBytes;             // e.g. 8 for 128B
    uint32_t bank = k / kElemsPerBank;
    uint32_t in_bank = k % kElemsPerBank;
    uint32_t swizzled_bank = bank ^ (row % kNumBanks);
    return row * block_k + swizzled_bank * kElemsPerBank + in_bank;
}

// SM100 FP4 GEMM 1D1D kernel实现
// 支持 MXF4 block-scaled 矩阵乘法
template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages, uint32_t kNumLastStages,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation, typename cd_dtype_t>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_fp8_gemm_1d1d_impl(int* grouped_layout,
                         uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_c,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_d) {
    
    // ========== 基础配置和类型定义 ==========
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");

    // ========== 核心配置参数 ==========
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t kNumMWaves = BLOCK_M / LAYOUT_AD_M;
    constexpr uint32_t kNumTMAStoreStages = 2;
    
    // ========== MXF4 配置 ==========
    constexpr uint32_t kNumSFAStagesPerLoad = 1;
    constexpr uint32_t kNumSFBStagesPerLoad = 1;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t FP4_ELEMS_PER_INT32 = 8;
    constexpr uint32_t MXF4_VS = 32;
    constexpr uint32_t BLOCK_K_FP4 = BLOCK_K * FP4_ELEMS_PER_INT32;
    constexpr uint32_t UMMA_K_FP4 = 64;
    constexpr uint32_t SF_K_PER_STAGE = BLOCK_K_FP4 / MXF4_VS;
    constexpr uint32_t SF_PACKED_K_PER_STAGE = SF_K_PER_STAGE / 4;
    
    DG_STATIC_ASSERT(BLOCK_M % LAYOUT_AD_M == 0 and 2 % kNumMWaves == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_K == 32, "Packed FP4 path expects BLOCK_K == 32");

    // ========== 动态形状处理 ==========
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    
    // FP4 packed: shape_k 是 int32 个数，每个 int32 有 8 个 FP4。
    // 1 个 scale 覆盖 MXF4_VS=32 个 FP4 (=4 个 int32)，
    // 每个 uint32 打包 4 个 scale → 1 个 packed group = 4*32 = 128 FP4。
    const uint32_t total_scales_k =
    ceil_div(shape_k * FP4_ELEMS_PER_INT32,
             MXF4_VS);  // MXF4_VS 是 uint32_t，OK

    const uint32_t total_packed_k =
        ceil_div(total_scales_k,
                uint32_t(4));  // 把 4 也变成 uint32_t
    
    const uint32_t shape_sfa_k = total_packed_k;
    const uint32_t shape_sfb_k = total_packed_k;

    // ========== 线程和warp信息 ==========
    bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = get_lane_idx();

    // ========== 共享内存分配 ==========
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // ========== 块大小计算 ==========
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    
    DG_STATIC_ASSERT(not kIsMulticastOnA or kNumMulticast == 1, "Invalid multicast");
    DG_STATIC_ASSERT(LOAD_BLOCK_M == BLOCK_M and BLOCK_M % LAYOUT_AD_M == 0, "Only support tensor memory layout A/D");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");

    // ========== 共享内存大小计算 ==========
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_PACKED_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(uint32_t);
    constexpr uint32_t SMEM_B_PACKED_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(uint32_t);
    constexpr uint32_t SF_BLOCK_M = constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * SF_PACKED_K_PER_STAGE * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * SF_PACKED_K_PER_STAGE * sizeof(uint32_t);
    
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0, "Shared memory must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(kNumTMAStoreStages >= 1, "Invalid number of TMA stages");

    // ========== 张量内存配置 ==========
    constexpr uint32_t kNumSFATmemCols = (SF_BLOCK_M / 32) * SF_PACKED_K_PER_STAGE;
    constexpr uint32_t kNumSFBTmemCols = (SF_BLOCK_N / 32) * SF_PACKED_K_PER_STAGE;
    constexpr uint32_t kNumEpilogueStages = (2 * kNumMWaves * BLOCK_N + kNumSFATmemCols + kNumSFBTmemCols) > 512 ? 1 : 2;
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * kNumMWaves * BLOCK_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;

    // ========== TMA描述符预取 ==========
    if (threadIdx.x == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_d);
        if constexpr (kWithAccumulation)
            cute::prefetch_tma_descriptor(&tensor_map_c);
    }

    // ========== 共享内存指针设置 ==========
    cd_dtype_t* smem_cd[kNumTMAStoreStages];
    uint32_t* smem_sfa[kNumStages];
    uint32_t* smem_sfb[kNumStages];
    uint32_t* smem_a_packed[kNumStages];
    uint32_t* smem_b_packed[kNumStages];
     
    #pragma unroll
    for (uint32_t i = 0; i < kNumTMAStoreStages; ++ i)
        smem_cd[i] = reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++ i) {
        smem_a_packed[i] = reinterpret_cast<uint32_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_PACKED_SIZE_PER_STAGE);
        smem_b_packed[i] = reinterpret_cast<uint32_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_PACKED_SIZE_PER_STAGE + i * SMEM_B_PACKED_SIZE_PER_STAGE);
    }
    
    auto sf_start_ptr = smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_PACKED_SIZE_PER_STAGE + SMEM_B_PACKED_SIZE_PER_STAGE);
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++ i) {
        smem_sfa[i] = reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
        smem_sfb[i] = reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    }

    // ========== 屏障初始化 ==========
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer +
        SMEM_CD_SIZE +
        kNumStages * (SMEM_A_PACKED_SIZE_PER_STAGE + SMEM_B_PACKED_SIZE_PER_STAGE) +
        kNumStages * (SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE));
    auto full_barriers         = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers        = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto with_sf_full_barriers = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_full_barriers    = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
    auto tmem_empty_barriers   = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages + i); });

    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    if (threadIdx.x == 0) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            with_sf_full_barriers[i]->init(kNumMulticast * 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumMulticast * kNumEpilogueThreads);
        }
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::fence_barrier_init();
    } 
    else if (threadIdx.x >= 32 and threadIdx.x < 64) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // ========== 块调度器初始化 ==========
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>(shape_m, shape_n, grouped_layout);

    // ========== K维度迭代控制 ==========
    struct DivisibleK {};
    struct NotDivisibleK {};
    uint32_t phase = 0;
    
    auto launch_k_iterations = [&](const auto& func) {
        const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
        const uint32_t num_iterations = ceil_div(current_shape_k, kNumStages * BLOCK_K);
        const uint32_t num_last_stages = ceil_div(current_shape_k, BLOCK_K) % kNumStages;

        if (num_last_stages == 0) {
            for (uint32_t k_iter = 0; k_iter < num_iterations; ++ k_iter, phase ^= 1)
                func(k_iter, DivisibleK{}, k_iter == num_iterations - 1, num_last_stages);
        } else {
            for (uint32_t k_iter = 0; k_iter < num_iterations - 1; ++ k_iter, phase ^= 1)
                func(k_iter, DivisibleK{}, false, num_last_stages);
            func(num_iterations - 1, NotDivisibleK{}, true, num_last_stages), phase ^= 1;
        }
    };

    auto dispatch_accum_stage_idx = [&](uint32_t accum_stage_idx, const auto& func) {
        DG_STATIC_ASSERT(1 <= kNumEpilogueStages and kNumEpilogueStages <= 2, "Too many epilogue stages");
        accum_stage_idx == 0 ? func(0) : func(1);
    };

    // ========== Warp角色分发 ==========
    if (warp_idx == 0) {
        // ========== TMA加载warp ==========
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            launch_k_iterations([&](uint32_t k_iter, auto type, bool is_last_iter, uint32_t num_last_stages) {
                constexpr bool kHasDivisibleStages = cute::is_same_v<decltype(type), DivisibleK>;
                const uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : num_last_stages;

                #pragma unroll
                for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                    empty_barriers[s]->wait(phase ^ 1);

                    uint32_t m_idx = scheduler.template get_global_idx<(kGemmType == GemmType::MGroupedMasked), KGroupedIndexType::MN>(shape_m, BLOCK_M, m_block_idx);
                    uint32_t n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), KGroupedIndexType::MN>(shape_n, BLOCK_N, n_block_idx, m_block_idx);
                    
                    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kMajorA == cute::UMMA::Major::K, "Invalid major");
                    uint32_t k_block_idx = k_iter * kNumStages + s;
                    uint32_t k_idx = k_block_idx * BLOCK_K;
                    uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), KGroupedIndexType::K>(shape_k, BLOCK_K, k_block_idx, m_block_idx);
                    uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), KGroupedIndexType::K>(shape_k, BLOCK_K, k_block_idx, m_block_idx);

                    if constexpr (kNumMulticast > 1) {
                        m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * LOAD_BLOCK_M) : 0;
                        n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                    }

                    if (cute::elect_one_sync()) {
                        if constexpr (kMajorA == cute::UMMA::Major::K)
                            tma_copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, 1>(&tensor_map_a, full_barriers[s], smem_a_packed[s], k_a_idx, m_idx);
                        if constexpr (kMajorA == cute::UMMA::Major::MN)
                            tma_copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, 1>(&tensor_map_a, full_barriers[s], smem_a_packed[s], m_idx, k_a_idx);
                        if constexpr (kMajorB == cute::UMMA::Major::K)
                            tma_copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, 1>(&tensor_map_b, full_barriers[s], smem_b_packed[s], k_b_idx, n_idx);
                        if constexpr (kMajorB == cute::UMMA::Major::MN)
                            tma_copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, 1>(&tensor_map_b, full_barriers[s], smem_b_packed[s], n_idx, k_b_idx);
                    }
                    auto num_arrival_bytes = SMEM_A_PACKED_SIZE_PER_STAGE + SMEM_B_PACKED_SIZE_PER_STAGE;

                    const uint32_t sfa_stage_in_group_idx = (k_iter * kNumStages + s) % kNumSFAStagesPerLoad;
                    if (sfa_stage_in_group_idx == 0 and cute::elect_one_sync()) {
                        uint32_t sf_k_base = k_block_idx / kNumSFAStagesPerLoad * SF_PACKED_K_PER_STAGE;
                        #pragma unroll
                        for (uint32_t pk = 0; pk < SF_PACKED_K_PER_STAGE; ++ pk) {
                            // 计算当前要 load 的 SFA K-slice 下标
                            auto sf_k = scheduler.template get_global_idx<
                            (kGemmType != GemmType::MGroupedContiguous),
                            KGroupedIndexType::SF_K>(
                                shape_sfa_k, 1, sf_k_base + pk);

                            tma_copy<BLOCK_M, 1, 0, 1>(&tensor_map_sfa, full_barriers[s], smem_sfa[s] + pk * SF_BLOCK_M, m_block_idx * BLOCK_M,
                                                       scheduler.template get_global_idx<(kGemmType != GemmType::MGroupedContiguous), KGroupedIndexType::SF_K>(shape_sfa_k, 1, sf_k_base + pk));
                        }
                        num_arrival_bytes += BLOCK_M * SF_PACKED_K_PER_STAGE * sizeof(uint32_t);
                    }
                    const uint32_t sfb_stage_in_group_idx = (k_iter * kNumStages + s) % kNumSFBStagesPerLoad;
                    if (sfb_stage_in_group_idx == 0 and cute::elect_one_sync()) {
                        uint32_t sf_k_base = k_block_idx / kNumSFBStagesPerLoad * SF_PACKED_K_PER_STAGE;
                        #pragma unroll
                        for (uint32_t pk = 0; pk < SF_PACKED_K_PER_STAGE; ++ pk) {
                            auto sf_k = scheduler.template get_global_idx<
                            true, KGroupedIndexType::SF_K>(
                                shape_sfb_k, 1, sf_k_base + pk, m_block_idx);

                            tma_copy<BLOCK_N, 1, 0, 1>(&tensor_map_sfb, full_barriers[s], smem_sfb[s] + pk * SF_BLOCK_N, n_block_idx * BLOCK_N,
                                                       scheduler.template get_global_idx<true, KGroupedIndexType::SF_K>(shape_sfb_k, 1, sf_k_base + pk, m_block_idx));
                        }
                        num_arrival_bytes += BLOCK_N * SF_PACKED_K_PER_STAGE * sizeof(uint32_t);
                    }

                    if (cute::elect_one_sync())
                        full_barriers[s]->arrive_and_expect_tx(num_arrival_bytes);
                }

                #pragma unroll
                for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                    empty_barriers[s]->wait(phase ^ 1);
                    if (cute::elect_one_sync())
                        full_barriers[s]->arrive();
                }
            });
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // ========== MMA发起warp (MXF4) ==========
        using ElemAB = cutlass::float_e2m1_t;  // MXF4 逻辑 A/B 元素类型（4bit 对应的 8bit 容器）
        constexpr uint32_t UMMA_M = LAYOUT_AD_M * (kIsMulticastOnA ? 1 : kNumMulticast);
        constexpr uint32_t UMMA_N = BLOCK_N * (kIsMulticastOnA ? kNumMulticast : 1);
        constexpr uint32_t UMMA_K_INT32 = UMMA_K_FP4 / FP4_ELEMS_PER_INT32;
        constexpr uint32_t NUM_K_ITERS_PER_STAGE = BLOCK_K / UMMA_K_INT32;
        constexpr uint32_t NUM_N_ITERS = BLOCK_N / UMMA_N;
        
        auto instr_desc_mxf4 = cute::UMMA::make_instr_desc_block_scaled<
            cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue8m0_t,
            UMMA_M, UMMA_N, kMajorA, kMajorB>();
        
        using cute_mma_mxf4_t = cute::conditional_t<kNumMulticast == 1,
            cute::SM100_MMA_MXF4_SS<cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
                                    cutlass::float_ue8m0_t, UMMA_M, UMMA_N, MXF4_VS,
                                    kMajorA, kMajorB>,
            cute::SM100_MMA_MXF4_2x1SM_SS<cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
                                          cutlass::float_ue8m0_t, UMMA_M, UMMA_N, MXF4_VS,
                                          kMajorA, kMajorB>>;
        
        auto sf_desc = make_sf_desc(nullptr);
        
        DG_STATIC_ASSERT(UMMA_M == 128, "MXF4 requires M=128");
        DG_STATIC_ASSERT((UMMA_N % 8 == 0) and (8 <= UMMA_N) and (UMMA_N <= 256), "Invalid MXF4 N-mode size");

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            dispatch_accum_stage_idx(scheduler.current_iter % kNumEpilogueStages, [&](uint32_t accum_stage_idx) {
                auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
                tcgen05_after_thread_sync();

                auto empty_barrier_arrive = [&](uint32_t s, bool do_tmem_full_arrive) {
                    auto umma_arrive = [](const uint64_t* barrier) {
                        if constexpr (kNumMulticast == 1) {
                            cutlass::arch::umma_arrive(barrier);
                        } else {
                            constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                            cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                        }
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[s]));
                };

                launch_k_iterations([&](uint32_t k_iter, auto type, bool is_last_iter, uint32_t num_last_stages) {
                    constexpr bool kHasDivisibleStages = cute::is_same_v<decltype(type), DivisibleK>;
                    const uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : num_last_stages;

                    #pragma unroll
                    for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                        with_sf_full_barriers[s]->wait(phase);
                    // ===== A/B SMEM DEBUG: 查看各 stage 的 K-block0 是否正确加载 =====
                    if (cute::elect_one_sync() &&
                        m_block_idx == 0 && n_block_idx == 0 &&
                        k_iter == 0 && s < 2) {   // 只看前两个 stage，避免太多输出
                        printf("DEBUG A: stage=%u, k_block_idx=%u, smem_a_packed[s][0..7]: ",
                               s, /*k_block_idx*/ s);
                        for (int i = 0; i < 8; ++i) {
                            printf("0x%08x ", smem_a_packed[s][i]);
                        }
                        printf("\n");

                        printf("DEBUG B: stage=%u, k_block_idx=%u, smem_b_packed[s][0..7]: ",
                               s, /*k_block_idx*/ s);
                        for (int i = 0; i < 8; ++i) {
                            printf("0x%08x ", smem_b_packed[s][i]);
                        }
                        printf("\n");
                    }
                        // SF复制到TMEM (SF_PACKED_K_PER_STAGE packed groups per stage)
                        const uint32_t sfa_stage_in_group_idx = (k_iter * kNumStages + s) % kNumSFAStagesPerLoad;
                        if (sfa_stage_in_group_idx == 0 and cute::elect_one_sync()) {
                            using cute_utccp_t = cute::conditional_t<kNumMulticast == 1,
                                cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;

                            #pragma unroll
                            for (uint32_t pk = 0; pk < SF_PACKED_K_PER_STAGE; ++ pk) {
                                #pragma unroll
                                for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                                    auto smem_ptr = smem_sfa[s] + pk * SF_BLOCK_M + i * kNumUTCCPAlignedElems;
                                    replace_smem_desc_addr(sf_desc, smem_ptr);
                                    cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + pk * (SF_BLOCK_M / 32) + i * 4);
                                }
                            }
                        }
                        const uint32_t sfb_stage_in_group_idx = (k_iter * kNumStages + s) % kNumSFBStagesPerLoad;
                        if (sfb_stage_in_group_idx == 0 and cute::elect_one_sync()) {
                            using cute_utccp_t = cute::conditional_t<kNumMulticast == 1,
                                cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;

                            #pragma unroll
                            for (uint32_t pk = 0; pk < SF_PACKED_K_PER_STAGE; ++ pk) {
                                #pragma unroll
                                for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                                    auto smem_ptr = smem_sfb[s] + pk * SF_BLOCK_N + i * kNumUTCCPAlignedElems;
                                    replace_smem_desc_addr(sf_desc, smem_ptr);
                                    cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + pk * (SF_BLOCK_N / 32) + i * 4);
                                }
                            }
                        }
                        __syncwarp();
                        tcgen05_after_thread_sync();
                        
                        // 创建 descriptor
                        // 创建 descriptor（按 uint32_t 视角）
                        constexpr uint32_t SMEM_A_SIZE_PER_STAGE_PACKED =
                            LOAD_BLOCK_M * BLOCK_K * sizeof(uint32_t);
                        constexpr uint32_t SMEM_B_SIZE_PER_STAGE_PACKED =
                            LOAD_BLOCK_N * BLOCK_K * sizeof(uint32_t);

                        // desc 直接以 packed uint32_t* 为基址
                        auto a_desc_base =
                            make_umma_desc<kMajorA, BLOCK_M, BLOCK_K, kSwizzleAMode>(
                                smem_a_packed[0], 0, 0);
                        auto b_desc_base =
                            make_umma_desc<kMajorB, BLOCK_N, BLOCK_K, kSwizzleBMode>(
                                smem_b_packed[0], 0, 0);

                        // 每个 stage 在 SMEM 上的字节偏移：仍然按「整块 [BLOCK_M x BLOCK_K uint32]」大小平移
                        uint32_t a_desc_stage_lo =
                            a_desc_base.lo + s * (SMEM_A_SIZE_PER_STAGE_PACKED / 16);
                        uint32_t b_desc_stage_lo =
                            b_desc_base.lo + s * (SMEM_B_SIZE_PER_STAGE_PACKED / 16);

                        uint32_t tmem_sfa_addr = kTmemStartColOfSFA;
                        uint32_t tmem_sfb_addr = kTmemStartColOfSFB;
                        const auto runtime_instr_desc_mxf4 = cute::UMMA::make_runtime_instr_desc_block_scaled(
                            instr_desc_mxf4, tmem_sfa_addr, tmem_sfb_addr);

                        // MMA 循环
                        #pragma unroll
                        for (uint32_t k = 0; k < NUM_K_ITERS_PER_STAGE; ++k) {
                            #pragma unroll
                            for (uint32_t n = 0; n < NUM_N_ITERS; ++n) {
                                auto b_desc = b_desc_base;
                                b_desc.lo = advance_umma_desc_lo<
                                    kMajorB, BLOCK_N, kSwizzleBMode, uint32_t>(
                                        b_desc_stage_lo,
                                        /*n_offset_in_u32*/ n * UMMA_N * BLOCK_K,
                                        /*k_offset_in_u32*/ k * UMMA_K_INT32);
                                
                                #pragma unroll
                                for (uint32_t w = 0; w < kNumMWaves; ++w) {
                                    auto a_desc = a_desc_base;
                                    a_desc.lo = advance_umma_desc_lo<
                                        kMajorA, BLOCK_M, kSwizzleAMode, uint32_t>(
                                            a_desc_stage_lo,
                                            /*m_offset_in_u32*/ w * LAYOUT_AD_M * BLOCK_K,
                                            /*k_offset_in_u32*/ k * UMMA_K_INT32);

                                    uint32_t tmem_col = accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N + n * UMMA_N;
                                    bool do_accumulate = (k_iter > 0 || s > 0 || k > 0);
                                    
                                    // cute_mma_mxf4_t::fma(a_desc, b_desc, tmem_col, do_accumulate,
                                    //                      runtime_instr_desc_mxf4, tmem_sfa_addr, tmem_sfb_addr);
                                    if (cute::elect_one_sync()) {
                                        uint64_t desc_a_raw = static_cast<uint64_t>(a_desc);
                                        uint64_t desc_b_raw = static_cast<uint64_t>(b_desc);
                                        uint32_t idesc_hi = static_cast<uint32_t>(runtime_instr_desc_mxf4 >> 32);
                                        uint32_t acc = do_accumulate ? 1u : 0u;
                                        asm volatile(
                                            "{\n\t"
                                            ".reg .pred p;\n\t"
                                            "setp.ne.b32 p, %4, 0;\n\t"
                                            "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 "
                                            "[%0], %1, %2, %3, [%5], [%6], p; \n\t"
                                            "}\n"
                                            :
                                            : "r"(tmem_col), "l"(desc_a_raw), "l"(desc_b_raw),
                                              "r"(idesc_hi), "r"(acc),
                                              "r"(tmem_sfa_addr), "r"(tmem_sfb_addr));
                                    }
                                }
                            }
                        }
                        
                        tcgen05_before_thread_sync();
                        
                        if (is_last_iter && s == kNumInnerStages - 1) {
                            if (cute::elect_one_sync()) {
                                tmem_full_barriers[accum_stage_idx]->arrive();
                            }
                        }

                        empty_barrier_arrive(s, is_last_iter and s == kNumInnerStages - 1);
                    }

                    #pragma unroll
                    for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                        with_sf_full_barriers[s]->wait(phase);
                        empty_barrier_arrive(s, false);
                    }
                });
            });
        }
    } else if (warp_idx == 2) {
        // ========== UTCCP转置器warp ==========
        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
            __syncwarp();
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
        };

        // When BLOCK_N < 128, TMA only loads BLOCK_N SF values into smem positions 0..BLOCK_N-1.
        // warp_transpose reads smem_old[c*32+t] for c=0..3, t=0..BLOCK_N-1 to populate TMEM.
        // Since all k-groups within one BLOCK_K stage share the same SF (SF granularity = 256 FP4
        // = 1 BLOCK_K > UMMA_K = 64 FP4), fill positions c*32..c*32+BLOCK_N-1 for c=1,2,3
        // with the TMA-loaded values from c=0 before transposing.
        auto fill_sfb_missing_k_groups = [&](uint32_t* smem_ptr) {
            if constexpr (BLOCK_N < kNumUTCCPAlignedElems) {
                constexpr uint32_t kKGroups = kNumUTCCPAlignedElems / 32;  // = 4
                #pragma unroll
                for (uint32_t c = 1; c < kKGroups; ++c) {
                    if (lane_idx < BLOCK_N)
                        st_shared(smem_ptr + c * 32 + lane_idx, ld_shared(smem_ptr + lane_idx));
                }
                __syncwarp();
            }
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            launch_k_iterations([&](uint32_t k_iter, auto type, bool is_last_iter, uint32_t num_last_stages) {
                constexpr bool kHasDivisibleStages = cute::is_same_v<decltype(type), DivisibleK>;
                const uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : num_last_stages;

                #pragma unroll
                for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                    full_barriers[s]->wait(phase);

                    const uint32_t sfa_stage_in_group_idx = (k_iter * kNumStages + s) % kNumSFAStagesPerLoad;
                    if (sfa_stage_in_group_idx == 0) {
                        #pragma unroll
                        for (uint32_t pk = 0; pk < SF_PACKED_K_PER_STAGE; ++ pk) {
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i)
                                utccp_required_smem_warp_transpose(smem_sfa[s] + pk * SF_BLOCK_M + i * kNumUTCCPAlignedElems);
                        }
                        cutlass::arch::fence_view_async_shared();
                    }

                    const uint32_t sfb_stage_in_group_idx = (k_iter * kNumStages + s) % kNumSFBStagesPerLoad;
                    if (sfb_stage_in_group_idx == 0) {
                        #pragma unroll
                        for (uint32_t pk = 0; pk < SF_PACKED_K_PER_STAGE; ++ pk) {
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                                fill_sfb_missing_k_groups(smem_sfb[s] + pk * SF_BLOCK_N + i * kNumUTCCPAlignedElems);
                                utccp_required_smem_warp_transpose(smem_sfb[s] + pk * SF_BLOCK_N + i * kNumUTCCPAlignedElems);
                            }
                        }
                        cutlass::arch::fence_view_async_shared();
                    }

                    with_sf_full_barriers[s]->arrive(0u);
                }

                #pragma unroll
                for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                    full_barriers[s]->wait(phase);
                    with_sf_full_barriers[s]->arrive(0u);
                }
            });
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32) {
        // ========== Epilogue warp组 ==========
        const auto epilogue_thread_idx = threadIdx.x - kNumNonEpilogueThreads;
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);

        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            dispatch_accum_stage_idx(scheduler.current_iter % kNumEpilogueStages, [&](uint32_t accum_stage_idx) {
                auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

                if (epilogue_thread_idx == 0)
                    cute::tma_store_wait<0>();
                cutlass::arch::NamedBarrier(kNumEpilogueThreads).sync();

                tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
                tcgen05_after_thread_sync();

                DG_STATIC_ASSERT(kNumEpilogueThreads == 128, "Epilogue threads not enough");
                DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

                #pragma unroll
                for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                    constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
                    #pragma unroll
                    for (uint32_t s = 0; s < kNumStores; ++ s) {
                        const uint32_t iter_idx = w * kNumStores + s;
                        if (iter_idx >= kNumTMAStoreStages) {
                            if (epilogue_thread_idx == 0)
                                cute::tma_store_wait<kNumTMAStoreStages - 1>();
                            cutlass::arch::NamedBarrier(kNumEpilogueThreads).sync();
                        }

                        const auto tma_stage_idx = iter_idx % kNumTMAStoreStages;
                        const auto m_idx = scheduler.template get_global_idx<(kGemmType != GemmType::MGroupedContiguous), KGroupedIndexType::MN>(shape_m, BLOCK_M, m_block_idx) + w * LAYOUT_AD_M;
                        const auto n_idx = n_block_idx * BLOCK_N + s * STORE_BLOCK_N;

                        #pragma unroll
                        for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                            auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                            constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                            auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                            auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                            col ^= row % (kSwizzleCDMode / 16);

                            uint32_t tmem_addr = accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                            auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +
                                            epilogue_warp_idx * 32 * kSwizzleCDMode +
                                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;

                            uint32_t values[kNumElemsPerBankGroup];
                            if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                                DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4, "Invalid type");
                                cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr, values[0], values[1], values[2], values[3]);
                                cutlass::arch::fence_view_async_tmem_load();
                                st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
                            }
                        }

                        if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
                            tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }
                        __syncwarp();

                        cute::tma_store_fence();
                        cutlass::arch::NamedBarrier(kNumEpilogueThreads).sync();
                        if (epilogue_thread_idx == 0) {
                            using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                            cute_tma_t::copy(&tensor_map_d, smem_cd[tma_stage_idx], n_idx, m_idx);
                            cute::tma_store_arrive();
                        }
                    }
                }
            });
        }

        if (epilogue_thread_idx == 0)
            cute::tma_store_wait<0>();

        if (epilogue_warp_idx == 1)
            Allocator().free(0, kNumTmemCols);
    }

    if constexpr (kNumMulticast > 1)
        cute::cluster_sync();
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
