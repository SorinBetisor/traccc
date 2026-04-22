/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "batch_prologue.cuh"

namespace traccc::cuda::kernels {

__global__ void batch_prologue(device::batch_prologue_payload payload) {

    // Single-thread, single-block kernel by launch config. Guard defensively.
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    if (*(payload.max_shared) == 0) {
        *(payload.terminate) = 1;
    }

    // Reset per-iteration counters consumed by the new kernels and the
    // unchanged downstream insertion-sort pipeline.
    *(payload.max_shared) = 0;
    *(payload.n_updated_tracks) = 0;
    *(payload.batch_size) = 0;
}

}  // namespace traccc::cuda::kernels
