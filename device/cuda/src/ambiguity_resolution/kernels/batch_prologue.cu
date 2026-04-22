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

    // Snapshot n_accepted for the rest of this graph replay. PBG identify and
    // apply kernels read this snapshot rather than the live n_accepted, so
    // race-free even when apply removes tracks atomically.
    *(payload.n_acc_snapshot) = *(payload.n_accepted);

    // Reset the first-failure rank to the maximum so batch_confirm's atomicMin
    // can lower it. The conflict-free prefix admitted by apply_batch_removals
    // is [0, *first_fail).
    *(payload.first_fail) = payload.candidate_window_size;
}

}  // namespace traccc::cuda::kernels
