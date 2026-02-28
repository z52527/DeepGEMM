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
    constexpr uint32_t kNumSFStagesPerLoad = 1;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t FP4_ELEMS_PER_INT32 = 8;
    constexpr uint32_t MXF4_VS = 32;
    constexpr uint32_t BLOCK_K_FP4 = BLOCK_K * FP4_ELEMS_PER_INT32;
    constexpr uint32_t UMMA_K_FP4 = 64;
    
    DG_STATIC_ASSERT(BLOCK_M % LAYOUT_AD_M == 0 and 2 % kNumMWaves == 0, "Invalid block M");
    // TMEM 32dp：每列 32 个 depth，故 BLOCK_M 行需 BLOCK_M/32 个“行块”
    constexpr uint32_t kNumMTiles = BLOCK_M / 32;
    constexpr uint32_t kNumRowBlocksPerMWave = LAYOUT_AD_M / 32;

    // ========== 动态形状处理 ==========
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    const uint32_t shape_sf_k = ceil_div(shape_k, BLOCK_K * kNumSFStagesPerLoad);

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
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0, "Shared memory must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(kNumTMAStoreStages >= 1, "Invalid number of TMA stages");

    // ========== 张量内存配置 ==========
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumEpilogueStages = (2 * kNumMTiles * BLOCK_N + kNumSFATmemCols + kNumSFBTmemCols) > 512 ? 1 : 2;
    // 累加区列数：每 stage 存 BLOCK_M 行×BLOCK_N 列，需 kNumMTiles 个行块×BLOCK_N 列
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * kNumMTiles * BLOCK_N;
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

                    const uint32_t sf_stage_in_group_idx = (k_iter * kNumStages + s) % kNumSFStagesPerLoad;
                    if (sf_stage_in_group_idx == 0 and cute::elect_one_sync()) {
                        tma_copy<BLOCK_M, 1, 0, 1>(&tensor_map_sfa, full_barriers[s], smem_sfa[s], m_block_idx * BLOCK_M,
                                                   scheduler.template get_global_idx<(kGemmType != GemmType::MGroupedContiguous), KGroupedIndexType::SF_K>(shape_sf_k, 1, ceil_div(k_idx, BLOCK_K * kNumSFStagesPerLoad)));
                        tma_copy<BLOCK_N, 1, 0, 1>(&tensor_map_sfb, full_barriers[s], smem_sfb[s], n_block_idx * BLOCK_N,
                                                   scheduler.template get_global_idx<true, KGroupedIndexType::SF_K>(shape_sf_k, 1, ceil_div(k_idx, BLOCK_K * kNumSFStagesPerLoad), m_block_idx));
                        num_arrival_bytes += (BLOCK_M + BLOCK_N) * sizeof(uint32_t);
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

                        // SF复制到TMEM
                        const uint32_t sf_stage_in_group_idx = (k_iter * kNumStages + s) % kNumSFStagesPerLoad;
                        if (sf_stage_in_group_idx == 0 and cute::elect_one_sync()) {
                            using cute_utccp_t = cute::conditional_t<kNumMulticast == 1,
                                cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;

                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = smem_sfa[s] + i * kNumUTCCPAlignedElems;
                                replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                            }
                            #pragma unroll
                            for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                                auto smem_ptr = smem_sfb[s] + i * kNumUTCCPAlignedElems;
                                replace_smem_desc_addr(sf_desc, smem_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                            }
                        }
                        __syncwarp();

                        tcgen05_after_thread_sync();
                        
                        // 创建 descriptor
                        constexpr uint32_t SMEM_A_SIZE_PER_STAGE_PACKED = LOAD_BLOCK_M * BLOCK_K * sizeof(uint32_t);
                        constexpr uint32_t SMEM_B_SIZE_PER_STAGE_PACKED = LOAD_BLOCK_N * BLOCK_K * sizeof(uint32_t);
                        
                        auto a_desc_base = make_umma_desc<kMajorA, BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a_packed[0], 0, 0);
                        auto b_desc_base = make_umma_desc<kMajorB, BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b_packed[0], 0, 0);
                        
                        uint32_t a_desc_stage_lo = a_desc_base.lo + s * SMEM_A_SIZE_PER_STAGE_PACKED / 16;
                        uint32_t b_desc_stage_lo = b_desc_base.lo + s * SMEM_B_SIZE_PER_STAGE_PACKED / 16;
                        
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
                                b_desc.lo = advance_umma_desc_lo<kMajorB, BLOCK_N, kSwizzleBMode, uint32_t>(
                                    b_desc_stage_lo, n * UMMA_N * BLOCK_K, k * UMMA_K_INT32);
                                
                                #pragma unroll
                                for (uint32_t w = 0; w < kNumMWaves; ++w) {
                                    auto a_desc = a_desc_base;
                                    a_desc.lo = advance_umma_desc_lo<kMajorA, BLOCK_M, kSwizzleAMode, uint32_t>(
                                        a_desc_stage_lo, w * LAYOUT_AD_M * BLOCK_K, k * UMMA_K_INT32);
                                    
                                    bool do_accumulate = (k_iter > 0 || k > 0 || s > 0);
                                    
                                    // ========== 仍只用 warp 1：每个 lane 算 4 行（row_group 0..3），共 32×4=128 行覆盖整块 ==========
                                    #pragma unroll
                                    for (uint32_t row_group = 0; row_group < kNumMTiles; ++row_group) {
                                        // TMEM 列基址：按行块划分；row_group 对应本轮 32 行 (row_group*32 .. row_group*32+31)
                                        uint32_t m_block_in_tmem = w * kNumRowBlocksPerMWave + row_group;
                                        uint32_t tmem_col = accum_stage_idx * kNumMTiles * BLOCK_N + m_block_in_tmem * BLOCK_N + n * UMMA_N;
                                        // 本 lane 在本轮负责的 M 行
                                        uint32_t m_row = row_group * 32 + lane_idx;
                                        
                                        float acc[16] = {0};
                                        
                                        // 从 TMEM 读取（如果需要累加）
                                        if (do_accumulate) {
                                            uint32_t r0 = tmem_col + 0u;
                                            uint32_t r1 = tmem_col + 4u;
                                            uint32_t r2 = tmem_col + 8u;
                                            uint32_t r3 = tmem_col + 12u;
                                            uint32_t v0, v1, v2, v3;
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
                                                : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
                                                : "r"(r0)
                                                : "memory"
                                            );
                                            acc[0] = *reinterpret_cast<float*>(&v0);
                                            acc[1] = *reinterpret_cast<float*>(&v1);
                                            acc[2] = *reinterpret_cast<float*>(&v2);
                                            acc[3] = *reinterpret_cast<float*>(&v3);
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
                                                : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
                                                : "r"(r1)
                                                : "memory"
                                            );
                                            acc[4] = *reinterpret_cast<float*>(&v0);
                                            acc[5] = *reinterpret_cast<float*>(&v1);
                                            acc[6] = *reinterpret_cast<float*>(&v2);
                                            acc[7] = *reinterpret_cast<float*>(&v3);
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
                                                : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
                                                : "r"(r2)
                                                : "memory"
                                            );
                                            acc[8] = *reinterpret_cast<float*>(&v0);
                                            acc[9] = *reinterpret_cast<float*>(&v1);
                                            acc[10] = *reinterpret_cast<float*>(&v2);
                                            acc[11] = *reinterpret_cast<float*>(&v3);
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
                                                : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
                                                : "r"(r3)
                                                : "memory"
                                            );
                                            acc[12] = *reinterpret_cast<float*>(&v0);
                                            acc[13] = *reinterpret_cast<float*>(&v1);
                                            acc[14] = *reinterpret_cast<float*>(&v2);
                                            acc[15] = *reinterpret_cast<float*>(&v3);
                                        }
                                        
                                        // 计算：1 行 M × 16 列 N
                                        if (m_row < LOAD_BLOCK_M) {
                                            for (uint32_t n_col = 0; n_col < UMMA_N; ++n_col) {
                                                for (uint32_t k_offset = 0; k_offset < UMMA_K_INT32; ++k_offset) {
                                                    uint32_t k_idx = k * UMMA_K_INT32 + k_offset;
                                                    uint32_t a_packed = smem_a_packed[s][m_row * BLOCK_K + k_idx];
                                                    uint32_t b_packed = smem_b_packed[s][n_col * BLOCK_K + k_idx];
                                                    for (int fp4 = 0; fp4 < 8; ++fp4) {
                                                        uint32_t a_bits = (a_packed >> (fp4 * 4)) & 0xF;
                                                        uint32_t b_bits = (b_packed >> (fp4 * 4)) & 0xF;
                                                        acc[n_col] += fp4_e2m1_to_float(a_bits) * fp4_e2m1_to_float(b_bits);
                                                    }
                                                }
                                            }
                                        }
                                        
                                        // 写回 TMEM
                                        uint32_t w0 = tmem_col + 0u;
                                        uint32_t w1 = tmem_col + 4u;
                                        uint32_t w2 = tmem_col + 8u;
                                        uint32_t w3 = tmem_col + 12u;
                                        // printf("CUDA CORE: w0=%u, w1=%u, w2=%u, w3=%u\n", w0, w1, w2, w3);
                                        asm volatile(
                                            "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};"
                                            : : "r"(w0), "r"(*reinterpret_cast<uint32_t*>(&acc[0])), "r"(*reinterpret_cast<uint32_t*>(&acc[1])), "r"(*reinterpret_cast<uint32_t*>(&acc[2])), "r"(*reinterpret_cast<uint32_t*>(&acc[3]))
                                            : "memory"
                                        );
                                        asm volatile(
                                            "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};"
                                            : : "r"(w1), "r"(*reinterpret_cast<uint32_t*>(&acc[4])), "r"(*reinterpret_cast<uint32_t*>(&acc[5])), "r"(*reinterpret_cast<uint32_t*>(&acc[6])), "r"(*reinterpret_cast<uint32_t*>(&acc[7]))
                                            : "memory"
                                        );
                                        asm volatile(
                                            "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};"
                                            : : "r"(w2), "r"(*reinterpret_cast<uint32_t*>(&acc[8])), "r"(*reinterpret_cast<uint32_t*>(&acc[9])), "r"(*reinterpret_cast<uint32_t*>(&acc[10])), "r"(*reinterpret_cast<uint32_t*>(&acc[11]))
                                            : "memory"
                                        );
                                        asm volatile(
                                            "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};"
                                            : : "r"(w3), "r"(*reinterpret_cast<uint32_t*>(&acc[12])), "r"(*reinterpret_cast<uint32_t*>(&acc[13])), "r"(*reinterpret_cast<uint32_t*>(&acc[14])), "r"(*reinterpret_cast<uint32_t*>(&acc[15]))
                                            : "memory"
                                        );
                                    }
                                }
                            }
                        }
                        
                        tcgen05_before_thread_sync();
                        
                        if (is_last_iter && s == kNumInnerStages - 1) {
                            // ===== Warp 1: TMEM → smem_cd copy =====
                            // TMEM is per-warp: only warp 1 can read its own data.
                            // Epilogue warps cannot see it, so warp 1 copies to smem_cd directly.
                            tcgen05_after_thread_sync();

                            constexpr uint32_t kBGBytes = 16;
                            constexpr uint32_t kElemsPerBG = kBGBytes / sizeof(cd_dtype_t);
                            const uint32_t tma_stage_idx = 0;

                            #pragma unroll
                            for (uint32_t rg = 0; rg < kNumMTiles; ++rg) {
                                uint32_t tmem_base = accum_stage_idx * kNumMTiles * BLOCK_N + rg * BLOCK_N;
                                #pragma unroll
                                for (uint32_t i = 0; i < STORE_BLOCK_N / kElemsPerBG; ++i) {
                                    uint32_t rb_addr = tmem_base + i * kElemsPerBG;
                                    uint32_t rv0, rv1, rv2, rv3;
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
                                        : "=r"(rv0), "=r"(rv1), "=r"(rv2), "=r"(rv3)
                                        : "r"(rb_addr) : "memory");
                                    cutlass::arch::fence_view_async_tmem_load();

                                    // Swizzled smem_cd layout (same formula as epilogue)
                                    auto bgi = i + lane_idx * (kSwizzleCDMode / kBGBytes);
                                    constexpr bool kShortcut = (kSwizzleCDMode / kBGBytes) == 8;
                                    auto srow = kShortcut ? (i / 8 + lane_idx) : (bgi / 8);
                                    auto scol = kShortcut ? (i) : (bgi % 8);
                                    scol ^= srow % (kSwizzleCDMode / 16);

                                    auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +
                                                    rg * 32 * kSwizzleCDMode +
                                                    srow * (kBGBytes * 8) + scol * kBGBytes;
                                    st_shared(smem_ptr, rv0, rv1, rv2, rv3);
                                }
                            }

                            tcgen05_before_thread_sync();
                            cutlass::arch::fence_view_async_shared();

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
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            launch_k_iterations([&](uint32_t k_iter, auto type, bool is_last_iter, uint32_t num_last_stages) {
                constexpr bool kHasDivisibleStages = cute::is_same_v<decltype(type), DivisibleK>;
                const uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : num_last_stages;

                #pragma unroll
                for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                    full_barriers[s]->wait(phase);
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

                // smem_cd is already filled by warp 1 (TMEM is per-warp, epilogue can't read it)
                tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);

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

                        // TMEM read skipped: smem_cd already written by MMA warp (warp 1)

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
