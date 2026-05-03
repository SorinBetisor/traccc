// TRACCC ambiguity-resolver Tier A hardware tuning constants.
//
// Single-header tuning lever for the GV100 (Volta, SM 7.0) target.
// All values are static so there is no runtime dispatch overhead; switching
// the untuned vs tuned binary is done by checking out a different branch
// (thesis-novelty-conflict-graph vs thesis-novelty-hardware-tuning).
//
// Tier A items applied here:
//   A1  Larger block size for graph kernels (was warp_size*2 = 64).
//   A2  __launch_bounds__ macros for nvcc occupancy hints.
//   A3  Read-only-cache hint helpers (__ldg) + restrict pointer macros.
//   A4  96 KB shared memory opt-in for build_conflict_coo + larger nt_coo.
//   A5  thrust::cuda::par_nosync policy (applied at call sites).

#pragma once

#include <cuda_runtime.h>

namespace traccc::cuda::tuning {

// ---- A1: graph kernel launch geometry ------------------------------------
// All graph_mis_* kernels and apply_graph_removals/update_rel_shared use a
// 1D grid with this block size. 64 (warp_size*2) was the upstream default;
// 256 = 8 warps gives a much better instructions-per-block ratio on Volta
// while still allowing 8 resident blocks per SM (= full 64-warp occupancy
// at typical register usage).
inline constexpr unsigned int graph_kernel_block_size = 256u;

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
