#pragma once

// Cluster Launch Control (CLC) scheduler for SM100.
//
// CLC lets hardware dynamically dispatch work tiles to idle CTAs via a few
// PTX primitives, instead of each CTA computing its own tile assignment from
// blockIdx. This closes the tail-wave imbalance gap that static persistent
// schedulers leave behind.
//
// Design
// ------
// One designated "scheduler warp" in the cluster issues `try_cancel`
// requests to the hardware, which asynchronously writes 128-bit responses
// into a SMEM mailbox gated by mbarriers. Work warps read the mailbox in
// lockstep (each tile is consumed by all 4 work warps: load, mma, sf, epi),
// so every slot is broadcast to `kNumConsumers` arrivers before being
// reclaimed by the producer.
//
// PTX reference: CUTLASS 3.x `PersistentTileSchedulerSm100` in
// `cutlass/gemm/kernel/sm100_tile_scheduler.hpp`.
//
// Availability: only when compiled for sm_100a (CUTLASS_ARCH_CLC_ENABLED).
// When not available, the device-side helpers trap; host-side heuristic must
// not select this scheduler.

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/config.h>

#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

// One CLC response is a 128-bit value laid out as (M_idx, N_idx, L_idx, reserved).
// We keep the full 16B in SMEM so the PTX `ld.shared.b128` in query_cancel
// reads it atomically.
struct alignas(16) ClcResponse {
    uint32_t m_idx;
    uint32_t n_idx;
    uint32_t l_idx;
    uint32_t reserved;
};

// Size in bytes that try_cancel writes — goes into mbarrier expect_tx.
// Matches what CUTLASS uses for the SM100 scheduler.
static constexpr uint32_t kClcResponseBytes = 16;


// ────────────────────────────────────────────────────────────────────────
// PTX primitives
// ────────────────────────────────────────────────────────────────────────

// Issue an async try_cancel. Hardware will:
//   1. Try to grab the next unclaimed tile id from the cluster's launch queue
//   2. Write the 128-bit response into `response_smem_addr`
//   3. Signal `mbarrier_smem_addr` with `kClcResponseBytes` of tx-complete bytes
// Multicast::cluster::all broadcasts the response to every CTA in the cluster,
// so every CTA sees the same tile assignment for this slot.
__device__ __forceinline__ void
clc_try_cancel(uint32_t response_smem_addr, uint32_t mbarrier_smem_addr) {
#if defined(CUTLASS_ARCH_CLC_ENABLED)
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta."
        "mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n"
        :: "r"(response_smem_addr), "r"(mbarrier_smem_addr));
#else
    // Not reachable on SM100A+; host heuristic gates selection.
    __trap();
#endif
}

// Parse a CLC response from SMEM. Returns tile_valid (false means "done, no
// more tiles in queue"); when valid, fills out m/n/l coordinates.
__device__ __forceinline__ bool
clc_query_cancel(uint32_t response_smem_addr,
                 uint32_t& m_idx, uint32_t& n_idx, uint32_t& l_idx) {
    uint32_t valid = 0;
#if defined(CUTLASS_ARCH_CLC_ENABLED)
    asm volatile(
        "{\n"
        ".reg .pred p_valid;\n\t"
        ".reg .b128 clc_result;\n\t"
        "ld.shared.b128 clc_result, [%4];\n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p_valid, clc_result;\n\t"
        "selp.u32 %3, 1, 0, p_valid;\n\t"
        "@p_valid clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 "
        "{%0, %1, %2, _}, clc_result;\n\t"
        "}\n"
        : "=r"(m_idx), "=r"(n_idx), "=r"(l_idx), "=r"(valid)
        : "r"(response_smem_addr)
        : "memory");
    cutlass::arch::fence_view_async_shared();
#else
    __trap();
#endif
    return valid == 1;
}


// ────────────────────────────────────────────────────────────────────────
// SchedulerCLC — pipeline of kNumStages mailbox slots
// ────────────────────────────────────────────────────────────────────────
//
// Barrier protocol per slot s:
//   - `full[s]`:  producer (scheduler warp, via hardware try_cancel) signals
//                 "response written"; consumers wait on it before reading.
//   - `empty[s]`: `kNumConsumers` work warps each arrive after consuming;
//                 producer waits for all before re-issuing try_cancel.
//
// Phase bits alternate 0/1 each full cycle through the pipeline, standard
// mbarrier handshake pattern.
//
// Consumers call get_next_block() in lockstep. All kNumConsumers warps must
// call it the same number of times — the mbarrier arrival count depends on it.

template <uint32_t kNumStages, uint32_t kNumConsumers>
struct SchedulerCLC {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    ClcResponse* mailbox;      // kNumStages slots
    Barrier*     full_bars;    // kNumStages
    Barrier*     empty_bars;   // kNumStages

    // Per-thread counters (each thread tracks its own consumer position)
    uint32_t consumer_idx   = 0;
    uint32_t consumer_phase = 0;

    __device__ __forceinline__
    SchedulerCLC(ClcResponse* mbox, Barrier* full, Barrier* empty)
        : mailbox(mbox), full_bars(full), empty_bars(empty) {}

    // Host/init-side setup. Call ONCE from thread 0 after SMEM allocation.
    // Initializes mbarriers with correct arrival counts.
    static __device__ __forceinline__ void
    init_barriers(Barrier* full, Barrier* empty) {
        #pragma unroll
        for (uint32_t s = 0; s < kNumStages; ++s) {
            // full: arrived by hardware try_cancel (1 arrive/slot, expect_tx in bytes)
            full[s].init(1);
            // empty: arrived by each consumer when done
            empty[s].init(kNumConsumers);
        }
    }

    // Scheduler-warp entry point. Call from exactly ONE thread in the cluster
    // (conventionally leader CTA's scheduler warp, thread 0 of that warp).
    // Runs until the hardware reports "no more tiles" for all in-flight slots.
    __device__ __forceinline__ void
    run_scheduler_warp() {
        // Prime the pipeline: issue kNumStages try_cancels up front.
        #pragma unroll
        for (uint32_t s = 0; s < kNumStages; ++s) {
            auto mbox_addr = cute::cast_smem_ptr_to_uint(&mailbox[s]);
            auto bar_addr  = cute::cast_smem_ptr_to_uint(&full_bars[s]);
            full_bars[s].arrive_and_expect_tx(kClcResponseBytes);
            clc_try_cancel(mbox_addr, bar_addr);
        }

        // Steady state: re-issue a new try_cancel each time a slot is
        // consumed (empty_bars fires).
        uint32_t producer_idx = 0;
        uint32_t producer_phase = 0;
        for (;;) {
            empty_bars[producer_idx].wait(producer_phase);
            // NOTE: once hardware reports "no more tiles" on any slot, all
            // subsequent try_cancels will also report "no more tiles". We
            // keep re-issuing to let the consumer loop exit cleanly; when
            // the consumer detects is_valid=false it signals the scheduler
            // to stop via an external "done" mechanism (see integration).
            auto mbox_addr = cute::cast_smem_ptr_to_uint(&mailbox[producer_idx]);
            auto bar_addr  = cute::cast_smem_ptr_to_uint(&full_bars[producer_idx]);
            full_bars[producer_idx].arrive_and_expect_tx(kClcResponseBytes);
            clc_try_cancel(mbox_addr, bar_addr);

            producer_idx = (producer_idx + 1) % kNumStages;
            if (producer_idx == 0) producer_phase ^= 1;
        }
    }

    // Consumer-side: returns false when CLC reports "no more tiles".
    // Every work warp calls this in lockstep — same number of calls.
    __device__ __forceinline__ bool
    get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        full_bars[consumer_idx].wait(consumer_phase);

        uint32_t m, n, l;
        auto mbox_addr = cute::cast_smem_ptr_to_uint(&mailbox[consumer_idx]);
        bool valid = clc_query_cancel(mbox_addr, m, n, l);

        // One arrival per consumer per slot; when all kNumConsumers arrive,
        // the scheduler warp is released to re-issue try_cancel.
        empty_bars[consumer_idx].arrive();

        consumer_idx = (consumer_idx + 1) % kNumStages;
        if (consumer_idx == 0) consumer_phase ^= 1;

        if (valid) {
            m_block_idx = m;
            n_block_idx = n;
        }
        return valid;
    }
};

}  // namespace deep_gemm
