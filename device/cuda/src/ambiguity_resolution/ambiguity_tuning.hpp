// TRACCC ambiguity-resolver hardware tuning constants.
//
// Single-header tuning lever for the GV100 (Volta, SM 7.0) target.
// Branch: thesis-greedy-hardware-tuning
//   (forked from thesis-novelty-hardware-tuning, greedy-only scope)
//
// Inherited Tier A items (all three backends):
//   A1  Block size for graph kernels — reverted to 64 (ablated).
//   A2  __launch_bounds__ macros for nvcc occupancy hints.
//   A3  Read-only-cache hint helpers (__ldg) + restrict pointer macros.
//   A4  96 KB shared memory opt-in for build_conflict_coo + wider nt_coo.
//   A5  thrust::cuda::par_nosync policy (applied at call sites).
//
// Greedy-only tuning tiers (this branch, remove_tracks / sort_updated_tracks
// / rearrange_tracks only — JP/MIS kernels are NOT changed):
//   GB-1  Two-phase warp-shuffle prefix scan in remove_tracks: Phase 1
//          does an intra-warp inclusive scan via __shfl_up_sync (0 syncs),
//          Phase 2 propagates warp sums serially.  Net: 2 __syncthreads
//          vs 18 in the original Hillis-Steele.
//   GB-2  Warp-only bitonic sort fast path in sort_updated_tracks: when
//          n_updated_tracks ≤ 32 the entire sort runs via __shfl_xor_sync
//          (zero __syncthreads, all in registers), eliminating shared-mem
//          traffic for the common convergence-phase invocations.
//   GB-3  [planned] 96 KB dynamic smem in remove_tracks, bound 512→1024.

#pragma once

#include <cuda_runtime.h>

namespace traccc::cuda::tuning {

// ---- A1: graph kernel launch geometry ------------------------------------
// All graph_mis_* kernels and apply_graph_removals/update_rel_shared use a
// 1D grid with this block size.
// A1 ABLATION: reverted to 64 (warp_size*2, the upstream default) to test
// whether 256 was causing the high-pile-up regression seen in sweep 1.
inline constexpr unsigned int graph_kernel_block_size = 64u;

// ---- A4: build_conflict_coo launch geometry ------------------------------
// nt_coo is the per-CTA gather width for the COO-edge-list builder. The
// fast path requires n_rows <= blockDim.x; the upstream value 128 was the
// limit at which static smem fits in the default 48 KB carveout. With the
// 96 KB opt-in (see kFullSharedMemBytes) we can safely run at 512.
inline constexpr unsigned int build_conflict_coo_block_size = 512u;

// Total shared-memory budget per SM that we explicitly opt into via
// cudaFuncSetAttribute(MaxDynamicSharedMemorySize). 96 KB is the GV100
// per-SM maximum; the kernel's actual smem request must not exceed this.
inline constexpr int kFullSharedMemBytes = 96 * 1024;

// ---- A2: launch-bounds macro ---------------------------------------------
// nvcc treats __launch_bounds__(maxThreadsPerBlock, minBlocksPerSM) as a
// hint to bound register usage for occupancy. minBlocksPerSM=2 is enough
// to keep two blocks resident per SM at our register footprints (~32-40
// regs/thread per the kernels we touch).
#define TRACCC_LAUNCH_BOUNDS(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM) \
    __launch_bounds__((MAX_THREADS_PER_BLOCK), (MIN_BLOCKS_PER_SM))

// ---- A3: read-only-cache helpers -----------------------------------------
// Wrap a single load through the read-only data cache. Equivalent to a
// plain dereference but routes through the texture/L2 read-only path,
// reducing pressure on L1 for kernels that read large neighbour arrays.
template <typename T>
__device__ __forceinline__ T tuned_ldg(const T* ptr) {
    return __ldg(ptr);
}

#define TRACCC_RESTRICT __restrict__

}  // namespace traccc::cuda::tuning
