#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/scheduler.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm100;

// SM100 FP8 GEMM 1D1D kernel实现
// 这是一个高度优化的GEMM kernel，支持：
// - FP8精度的矩阵乘法运算
// - 1D布局的输入矩阵A和B
// - 支持多种数据布局（行主序/列主序）
// - 支持分组GEMM和累加操作
// - 使用TMA（Tensor Memory Accelerator）进行高效内存访问
// - 使用TMEM（Tensor Memory）进行中间结果存储
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
// #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    
    // ========== 基础配置和类型定义 ==========
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    // 验证累加操作的数据类型约束：累加操作必须使用FP32输出
    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");

    // ========== 核心配置参数 ==========
    constexpr uint32_t LAYOUT_AD_M = 128;                                              // A/D矩阵的M维度布局
    constexpr uint32_t kNumMWaves = BLOCK_M / LAYOUT_AD_M;                            // M维度的wave数量
    constexpr uint32_t kNumTMAStoreStages = 2;                                        // TMA存储的流水线阶段数
    constexpr uint32_t kNumSFStagesPerLoad = sizeof(uint32_t) / sizeof(cutlass::float_ue8m0_t); // 每次加载的缩放因子阶段数
    constexpr uint32_t kNumUTCCPAlignedElems = 128;                                   // UTCCP对齐的元素数量
    
    // 静态断言验证配置的合法性
    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_M % LAYOUT_AD_M == 0 and 2 % kNumMWaves == 0, "Invalid block M");

    // ========== 动态形状处理 ==========
    // 如果编译时给出了形状常量，则覆盖运行时参数
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    const uint32_t shape_sf_k = ceil_div(shape_k, BLOCK_K * kNumSFStagesPerLoad);

    // ========== 线程和warp信息 ==========
    bool is_leader_cta = cute::block_rank_in_cluster() == 0;                          // 是否为集群中的领导CTA
    const auto warp_idx = cutlass::canonical_warp_idx_sync();                         // 当前warp索引
    const auto lane_idx = get_lane_idx();                                             // 当前lane索引

    // ========== 共享内存分配 ==========
    // 对齐到1024字节以支持swizzle-128B模式
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // ========== 多播配置和块大小计算 ==========
    // 2-CTA MMA配置：根据多播设置调整加载和存储块大小
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    
    // 验证多播配置的合法性
    DG_STATIC_ASSERT(not kIsMulticastOnA or kNumMulticast == 1, "Invalid multicast");
    DG_STATIC_ASSERT(LOAD_BLOCK_M == BLOCK_M and BLOCK_M % LAYOUT_AD_M == 0, "Only support tensor memory layout A/D");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");

    // ========== 共享内存大小计算 ==========
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * kSwizzleCDMode;       // 每个阶段C/D矩阵的共享内存大小
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;    // C/D矩阵总共享内存大小
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3); // 每个阶段A矩阵的共享内存大小
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3); // 每个阶段B矩阵的共享内存大小
    constexpr uint32_t SF_BLOCK_M = constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);  // 对齐后的缩放因子块M大小
    constexpr uint32_t SF_BLOCK_N = constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);  // 对齐后的缩放因子块N大小
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);       // 每个阶段SFA的共享内存大小
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);       // 每个阶段SFB的共享内存大小
    
    // 验证共享内存对齐要求
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(kNumTMAStoreStages >= 1, "Invalid number of TMA stages");

    // ========== 自动推导epilogue阶段数 ==========
    // 根据张量内存大小自动推导epilogue阶段数（1或2）
    // TODO: 测试 kNumMWaves == 2 and kNumEpilogueStages == 2 的情况
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;                             // SFA张量内存列数
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;                             // SFB张量内存列数
    constexpr uint32_t kNumEpilogueStages = (2 * kNumMWaves * BLOCK_N + kNumSFATmemCols + kNumSFBTmemCols) > 512 ? 1 : 2;

    // ========== 张量内存配置 ==========
    // 计算实际的张量内存大小和偏移量
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * kNumMWaves * BLOCK_N; // 累加器张量内存列数
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>(); // 总张量内存列数
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;                        // SFA在张量内存中的起始列
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;      // SFB在张量内存中的起始列

    // ========== TMA描述符预取 ==========
    // 在最开始预取TMA描述符以减少延迟
    if (threadIdx.x == 0) {
        // 注意：这里必须使用 reinterpret_cast，否则NVRTC会失败
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_d);
        if constexpr (kWithAccumulation)
            cute::prefetch_tma_descriptor(&tensor_map_c);
    }

    // ========== 共享内存指针设置 ==========
    // 共享内存上的数据（按以下顺序布局）
    cd_dtype_t* smem_cd[kNumTMAStoreStages];                                          // C/D矩阵共享内存指针数组
    cutlass::float_e4m3_t* smem_a[kNumStages];                                       // A矩阵共享内存指针数组
    cutlass::float_e4m3_t* smem_b[kNumStages];                                       // B矩阵共享内存指针数组
    uint32_t* smem_sfa[kNumStages];                                                   // SFA缩放因子共享内存指针数组
    uint32_t* smem_sfb[kNumStages];                                                   // SFB缩放因子共享内存指针数组

    // 填充D/A/B指针
    #pragma unroll
    for (uint32_t i = 0; i < kNumTMAStoreStages; ++ i)
        smem_cd[i] = reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++ i) {
        smem_a[i] = reinterpret_cast<cutlass::float_e4m3_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i] = reinterpret_cast<cutlass::float_e4m3_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    }

    // 填充SFA/SFB指针
    auto sf_start_ptr = smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++ i) {
        smem_sfa[i] = reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
        smem_sfb[i] = reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    }

    // ========== 屏障初始化 ==========
    // 填充屏障对象
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer +
        SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) +
        kNumStages * (SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE));
    auto full_barriers              = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers             = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto with_sf_full_barriers      = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_full_barriers         = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
    auto tmem_empty_barriers        = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages + i); });

    // ========== 张量内存指针设置 ==========
    // 填充张量内存指针
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // ========== 屏障和张量内存初始化 ==========
    // 初始化屏障
    if (threadIdx.x == 0) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // 在所有CTA上到达
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            // 仅在领导CTA上到达
            with_sf_full_barriers[i]->init(kNumMulticast * 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // 在所有CTA上到达
            tmem_full_barriers[i]->init(1);
            // 仅在领导CTA上到达
            tmem_empty_barriers[i]->init(kNumMulticast * kNumEpilogueThreads);
        }

        // 使初始化的屏障在异步代理中可见
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::fence_barrier_init();
    } else if (threadIdx.x >= 32 and threadIdx.x < 64) {
        // 分配张量内存
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // ========== 块调度器初始化 ==========
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>(shape_m, shape_n, grouped_layout);

    // ========== K维度迭代控制 ==========
    // 用于流水线展开的结构体
    struct DivisibleK {};      // K维度可整除的情况
    struct NotDivisibleK {};   // K维度不可整除的情况
    uint32_t phase = 0;        // 流水线相位
    
    // K维度迭代启动器：处理完整的K维度循环
    auto launch_k_iterations = [&](const auto& func) {
        const uint32_t current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
        const uint32_t num_iterations = ceil_div(current_shape_k, kNumStages * BLOCK_K);
        const uint32_t num_last_stages = ceil_div(current_shape_k, BLOCK_K) % kNumStages;

        // TODO: 重构这里的逻辑
        if (num_last_stages == 0) {
            // K维度完全可整除的情况
            for (uint32_t k_iter = 0; k_iter < num_iterations; ++ k_iter, phase ^= 1)
                func(k_iter, DivisibleK{}, k_iter == num_iterations - 1, num_last_stages);
        } else {
            // K维度不完全整除的情况
            for (uint32_t k_iter = 0; k_iter < num_iterations - 1; ++ k_iter, phase ^= 1)
                func(k_iter, DivisibleK{}, false, num_last_stages);
            func(num_iterations - 1, NotDivisibleK{}, true, num_last_stages), phase ^= 1;
        }
    };

    // 累加阶段索引分发器：根据epilogue阶段数进行分发
    auto dispatch_accum_stage_idx = [&](uint32_t accum_stage_idx, const auto& func) {
        DG_STATIC_ASSERT(1 <= kNumEpilogueStages and kNumEpilogueStages <= 2,
                         "Too many epilogue stages, please modify the Python heuristic as well");
        accum_stage_idx == 0 ? func(0) : func(1);
    };

    // ========== Warp角色分发 ==========
    // 将warp分配到不同的角色
    if (warp_idx == 0) {
        // ========== TMA加载warp ==========
        // 持续调度处理块
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            launch_k_iterations([&](uint32_t k_iter, auto type, bool is_last_iter, uint32_t num_last_stages) {
                constexpr bool kHasDivisibleStages = cute::is_same_v<decltype(type), DivisibleK>;
                const uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : num_last_stages;

                #pragma unroll
                for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                    // 等待消费者释放
                    empty_barriers[s]->wait(phase ^ 1);

                    // ========== 计算全局索引偏移 ==========
                    // 注意：组总是与外部维度连接
                    uint32_t m_idx = scheduler.template get_global_idx<(kGemmType == GemmType::MGroupedMasked), KGroupedIndexType::MN> (
                        shape_m, BLOCK_M, m_block_idx);
                    uint32_t n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), KGroupedIndexType::MN> (
                        shape_n, BLOCK_N, n_block_idx, m_block_idx);

                    // 注意：k_idx实际上是K-major的默认k索引，而k_b_idx可能是MN-major
                    // 对于所有m-grouped GEMM，A必须是K-majored
                    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kMajorA == cute::UMMA::Major::K, "Invalid major");
                    uint32_t k_block_idx = k_iter * kNumStages + s;
                    uint32_t k_idx = k_block_idx * BLOCK_K;
                    uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), KGroupedIndexType::K> (
                        shape_k, BLOCK_K, k_block_idx, m_block_idx);
                    uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), KGroupedIndexType::K> (
                        shape_k, BLOCK_K, k_block_idx, m_block_idx);

                    // 添加2-CTA偏移
                    if constexpr (kNumMulticast > 1) {
                        m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * LOAD_BLOCK_M) : 0;
                        n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                    }

                    // ========== 发起TMA传输 ==========
                    if (cute::elect_one_sync()) {
                        // 根据矩阵布局发起相应的TMA复制
                        if constexpr (kMajorA == cute::UMMA::Major::K)
                            tma_copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, 1>(&tensor_map_a, full_barriers[s], smem_a[s], k_a_idx, m_idx);
                        if constexpr (kMajorA == cute::UMMA::Major::MN)
                            tma_copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, 1>(&tensor_map_a, full_barriers[s], smem_a[s], m_idx, k_a_idx);
                        if constexpr (kMajorB == cute::UMMA::Major::K)
                            tma_copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, 1>(&tensor_map_b, full_barriers[s], smem_b[s], k_b_idx, n_idx);
                        if constexpr (kMajorB == cute::UMMA::Major::MN)
                            tma_copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, 1>(&tensor_map_b, full_barriers[s], smem_b[s], n_idx, k_b_idx);
                    }
                    auto num_arrival_bytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;

                    // ========== 在特定阶段发起SFA和SFB TMA ==========
                    // 无交织，所以一个TMA对应一个SF就足够了
                    const uint32_t sf_stage_in_group_idx = (k_iter * kNumStages + s) % kNumSFStagesPerLoad;
                    if (sf_stage_in_group_idx == 0 and cute::elect_one_sync()) {
                        tma_copy<BLOCK_M, 1, 0, 1>(&tensor_map_sfa, full_barriers[s], smem_sfa[s], m_block_idx * BLOCK_M,
                                                   scheduler.template get_global_idx<(kGemmType != GemmType::MGroupedContiguous), KGroupedIndexType::SF_K>(shape_sf_k, 1, ceil_div(k_idx, BLOCK_K * kNumSFStagesPerLoad)));
                        tma_copy<BLOCK_N, 1, 0, 1>(&tensor_map_sfb, full_barriers[s], smem_sfb[s], n_block_idx * BLOCK_N,
                                                   scheduler.template get_global_idx<true, KGroupedIndexType::SF_K>(shape_sf_k, 1, ceil_div(k_idx, BLOCK_K * kNumSFStagesPerLoad), m_block_idx));
                        num_arrival_bytes += (BLOCK_M + BLOCK_N) * sizeof(uint32_t);
                    }

                    // 到达满屏障
                    if (cute::elect_one_sync())
                        full_barriers[s]->arrive_and_expect_tx(num_arrival_bytes);
                }

                // 等待未对齐的情况
                #pragma unroll
                for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                    empty_barriers[s]->wait(phase ^ 1);
                    if (cute::elect_one_sync())
                        full_barriers[s]->arrive();
                }
            });
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // ========== MMA发起warp ==========
        // 注意：只有领导CTA会执行此操作
        
        // ========== 制作指令描述符 ==========
        // TODO: 重构 UMMA_M 计算
        // constexpr uint32_t UMMA_M = LAYOUT_AD_M * (kIsMulticastOnA ? 1 : kNumMulticast);
        // constexpr uint32_t UMMA_N = BLOCK_N * (kIsMulticastOnA ? kNumMulticast : 1);
        // constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
        // auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
        //                                                            float, cutlass::float_ue8m0_t,
        //                                                            UMMA_M, UMMA_N, kMajorA, kMajorB>();
        // auto sf_desc = make_sf_desc(nullptr);

        // // ========== 创建UMMA描述符 ==========
        // DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        // auto a_desc = make_umma_desc<kMajorA, BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        // auto b_desc = make_umma_desc<kMajorB, BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
        // uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        // uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        // // ========== MMA指令检查 ==========
        // // 注意：CUTLASS除了MMA特征外没有这样的检查，但我们不使用这些特征
        // DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
        //                  (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
        //                  (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
        //                  "Invalid MMA instruction shape");

        // ========== 持续调度处理块 ==========
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            dispatch_accum_stage_idx(scheduler.current_iter % kNumEpilogueStages, [&](uint32_t accum_stage_idx) {
                // 等待张量内存空屏障到达
                auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
                tcgen05_after_thread_sync();

                // ========== 空屏障到达处理 ==========
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

                    // 注意：张量内存累加器流水线与多播无关
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                };

                // ========== 启动MMA ==========
                launch_k_iterations([&](uint32_t k_iter, auto type, bool is_last_iter, uint32_t num_last_stages) {
                    constexpr bool kHasDivisibleStages = cute::is_same_v<decltype(type), DivisibleK>;
                    const uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : num_last_stages;

                    #pragma unroll
                    for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                        // 等待TMA和SF转置到达
                        with_sf_full_barriers[s]->wait(phase);
                        // tcgen05_after_thread_sync();

                        // ========== 在特定阶段执行SF复制 ==========
                        // 注意：CUTLASS UTCCP的接口没有 elect_one_sync，我们必须自己处理
                        // const uint32_t sf_stage_in_group_idx = (k_iter * kNumStages + s) % kNumSFStagesPerLoad;
                        // if (sf_stage_in_group_idx == 0 and cute::elect_one_sync()) {
                            // using cute_utccp_t = cute::conditional_t<kNumMulticast == 1,
                            //     cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;

                            // SFA和SFB复制
                            // TODO: 通过加法处理共享内存描述符
                            // #pragma unroll
                            // for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                            //     auto smem_ptr = smem_sfa[s] + i * kNumUTCCPAlignedElems;
                            //     replace_smem_desc_addr(sf_desc, smem_ptr);
                            //     cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                            // }
                            // #pragma unroll
                            // for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                            //     auto smem_ptr = smem_sfb[s] + i * kNumUTCCPAlignedElems;
                            //     replace_smem_desc_addr(sf_desc, smem_ptr);
                            //     cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                            // }
                        // }
                        __syncwarp();

                        // ========== 在领导CTA中发起UMMA ==========
                        // using cute_mma_t = cute::conditional_t<kNumMulticast == 1,
                        //     cute::SM100_MMA_MXF8F6F4_SS      <cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                        //                                       cutlass::float_ue8m0_t, UMMA_M, UMMA_N, kMajorA, kMajorB>,
                        //     cute::SM100_MMA_MXF8F6F4_2x1SM_SS<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                        //                                       cutlass::float_ue8m0_t, UMMA_M, UMMA_N, kMajorA, kMajorB>>;
                        // const auto& runtime_instr_desc = make_runtime_instr_desc_with_sf_id(instr_desc, sf_stage_in_group_idx);
                        // const auto& a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, s);
                        // const auto& b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, s);
                        
                        // 嵌套循环执行实际的矩阵乘法运算
                        // #pragma unroll
                        // for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                        //     b_desc.lo = advance_umma_desc_lo<kMajorB, BLOCK_N, kSwizzleBMode, cutlass::float_e4m3_t>(b_desc_base_lo, 0, k * UMMA_K);
                        //     #pragma unroll
                        //     for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                        //         a_desc.lo = advance_umma_desc_lo<kMajorA, BLOCK_M, kSwizzleAMode, cutlass::float_e4m3_t>(a_desc_base_lo, w * LAYOUT_AD_M * BLOCK_K, k * UMMA_K);
                        //         cute_mma_t::fma(a_desc, b_desc,
                        //                         accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,
                        //                         k_iter > 0 or s > 0 or k > 0,
                        //                         runtime_instr_desc,
                        //                         kTmemStartColOfSFA + w * (kNumUTCCPAlignedElems / 32),
                        //                         kTmemStartColOfSFB);
                        //     }
                        // }

                        // 提交到mbarrier对象
                        // 不需要显式的 tcgen05.fence::before_thread_sync，因为这已经被 tcgen05.commit 隐式执行
                        empty_barrier_arrive(s, is_last_iter and s == kNumInnerStages - 1);
                    }

                    // 等待未对齐的情况
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
        
        // UTCCP所需的共享内存warp转置函数
        // auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
        //     DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
        //     uint32_t values[4];
            
        //     // 读取数据
        //     #pragma unroll
        //     for (uint32_t i = 0; i < 4; ++ i)
        //         values[i] = ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
        //     __syncwarp();
            
        //     // 写回转置后的数据
        //     #pragma unroll
        //     for (uint32_t i = 0; i < 4; ++ i)
        //         st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
        // };

        // ========== 持续调度处理块 ==========
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            launch_k_iterations([&](uint32_t k_iter, auto type, bool is_last_iter, uint32_t num_last_stages) {
                constexpr bool kHasDivisibleStages = cute::is_same_v<decltype(type), DivisibleK>;
                const uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : num_last_stages;

                #pragma unroll
                for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                    // 等待TMA到达
                    full_barriers[s]->wait(phase);

                    // 在特定阶段为UTCCP执行转置
                    // const uint32_t sf_stage_in_group_idx = (k_iter * kNumStages + s) % kNumSFStagesPerLoad;
                    // if (sf_stage_in_group_idx == 0) {
                    //     // 对SFA和SFB执行转置
                    //     #pragma unroll
                    //     for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i)
                    //         utccp_required_smem_warp_transpose(smem_sfa[s] + i * kNumUTCCPAlignedElems);
                    //     #pragma unroll
                    //     for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i)
                    //         utccp_required_smem_warp_transpose(smem_sfb[s] + i * kNumUTCCPAlignedElems);
                    //     // TODO: 确定代理栅栏对2-CTA情况是否有效
                    //     cutlass::arch::fence_view_async_shared();
                    // }

                    // 到达屏障
                    with_sf_full_barriers[s]->arrive(0u);
                }

                // 等待未对齐的情况
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

        // 注意：张量内存地址被简化，因为硬件会忽略warp索引位，
        // 即不需要 tmem_ptr |= (epilogue_warp_idx * 32) << 16
        // 注意：我们也禁止两个CTA共享相同的SM及其张量内存
        DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);

        // ========== TMA检查 ==========
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
        DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
        DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");

        // ========== 持续调度处理块 ==========
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            dispatch_accum_stage_idx(scheduler.current_iter % kNumEpilogueStages, [&](uint32_t accum_stage_idx) {
                auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

                // ========== 刷新TMA存储 ==========
                // 注意：对于第一次存储，我们必须刷新所有之前的TMA，
                // 因为我们不在两个块之间共享流水线阶段
                if (epilogue_thread_idx == 0)
                    cute::tma_store_wait<0>();
                cutlass::arch::NamedBarrier(kNumEpilogueThreads).sync();

                // 等待UMMA到达
                tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
                tcgen05_after_thread_sync();

                // ========== 从张量内存加载到寄存器，并用STSM写入共享内存 ==========
                DG_STATIC_ASSERT(kNumEpilogueThreads == 128, "Epilogue threads not enough");
                DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

                // 遍历M waves
                #pragma unroll
                for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                    // 发起每个交织原子并流水线化STSM和TMA存储
                    constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
                    #pragma unroll
                    for (uint32_t s = 0; s < kNumStores; ++ s) {
                        // 等待共享内存被释放
                        const uint32_t iter_idx = w * kNumStores + s;
                        if (iter_idx >= kNumTMAStoreStages) {
                            if (epilogue_thread_idx == 0)
                                cute::tma_store_wait<kNumTMAStoreStages - 1>();
                            cutlass::arch::NamedBarrier(kNumEpilogueThreads).sync();
                        }

                        // ========== 计算流水线阶段和索引 ==========
                        const auto tma_stage_idx = iter_idx % kNumTMAStoreStages;
                        const auto m_idx = scheduler.template get_global_idx<(kGemmType != GemmType::MGroupedContiguous), KGroupedIndexType::MN>(shape_m, BLOCK_M, m_block_idx) + w * LAYOUT_AD_M;
                        const auto n_idx = n_block_idx * BLOCK_N + s * STORE_BLOCK_N;

                        // ========== 存储到共享内存 ==========
                        #pragma unroll
                        for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                            // 计算原子中要写入的bank group索引
                            auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);

                            // 在另一个视图中重塑原子并交织
                            //  - 原始：(LAYOUT_AD_M, kSwizzleCDMode / kNumBankGroupBytes)
                            //  - 新：(LAYOUT_AD_M * kSwizzleCDMode / kNumBankGroupBytes / 8, 8)
                            // 注意："8"是bank group的数量，"16"是交织模式
                            constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                            auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                            auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                            col ^= row % (kSwizzleCDMode / 16);

                            // 源和目标内存地址
                            uint32_t tmem_addr = accum_stage_idx * kNumMWaves * BLOCK_N +               // 累加器偏移
                                                 w * BLOCK_N +                                          // Wave偏移
                                                 s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;         // 块内偏移
                            auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +        // 基指针
                                            epilogue_warp_idx * 32 * kSwizzleCDMode +                   // Warp偏移
                                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;  // 原子内偏移

                            // ========== 从张量内存加载，存储到共享内存 ==========
                            uint32_t values[kNumElemsPerBankGroup];
                            if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                                // 对于FP32输出，读取并存储
                                DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4, "Invalid type");
                                cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr,
                                    values[0], values[1], values[2], values[3]);
                                cutlass::arch::fence_view_async_tmem_load();
                                st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
                            } else {
                                // 对于BF16输出，读取、转换并存储
                                DG_STATIC_ASSERT(kNumElemsPerBankGroup == 8 and cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid type");
                                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                                    values[0], values[1], values[2], values[3],
                                    values[4], values[5], values[6], values[7]);
                                cutlass::arch::fence_view_async_tmem_load();
                                st_shared(smem_ptr,
                                          cast_into_bf16_and_pack(values[0], values[1]),
                                          cast_into_bf16_and_pack(values[2], values[3]),
                                          cast_into_bf16_and_pack(values[4], values[5]),
                                          cast_into_bf16_and_pack(values[6], values[7]));
                            }
                        }

                        // ========== 尽快通知张量内存空（仅在领导CTA）到达 ==========
                        // 注意：只有最后一个阶段需要这样做
                        if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
                            tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }
                        __syncwarp();

                        // ========== 同步所有线程并发起TMA ==========
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

        // ========== 刷新流水线中的所有阶段以使TMA存储对下一个kernel可见 ==========
        if (epilogue_thread_idx == 0)
            cute::tma_store_wait<0>();

        // ========== 由warp 1释放张量内存 ==========
        // 注意：warp 0正在等待TMA存储
        if (epilogue_warp_idx == 1)
            Allocator().free(0, kNumTmemCols);
    }

    // ========== 最终同步 ==========
    // 为了安全地析构所有屏障，我们需要集群同步
    // TODO: 通过另一轮屏障等待来优化它
    if constexpr (kNumMulticast > 1)
        cute::cluster_sync();
// #else
//     if (blockIdx.x == 0 and threadIdx.x == 0)
//         DG_DEVICE_ASSERT(false and "This kernel only support sm_100a/sm_101a");
// #endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
