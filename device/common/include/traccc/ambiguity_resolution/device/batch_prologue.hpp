/** TRACCC library, part of the ACTS project (R&D line)
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

/// (Event Data) Payload for the batch_prologue kernel used by the parallel
/// batch greedy (Tier 2a) variant of the CUDA ambiguity resolver.
///
/// This is a single-thread single-block kernel that handles the per-iteration
/// bootstrap that baseline remove_tracks does in its first __syncthreads'd
/// section:
///   - if max_shared == 0: set terminate = 1
///   - reset max_shared = 0, n_updated_tracks = 0, batch_size = 0
///   - snapshot n_accepted into n_acc_snapshot so all PBG kernels in this
///     graph replay see a single, race-free n_accepted value (the actual
///     n_accepted is only mutated post-apply by batch_commit).
/// The claimed_by buffer is reset by an explicit cudaMemsetAsync node placed
/// before this kernel in the graph.
struct batch_prologue_payload {
    int* terminate;
    unsigned int* max_shared;
    unsigned int* n_updated_tracks;
    unsigned int* batch_size;
    const unsigned int* n_accepted;
    unsigned int* n_acc_snapshot;
    /// Reset to candidate_window_size at the start of every iteration so the
    /// downstream batch_confirm can atomicMin into it. apply_batch_removals
    /// admits ranks [0, *first_fail) — the conflict-free prefix of the worst
    /// tracks in this batch.
    unsigned int* first_fail;
    unsigned int candidate_window_size;
};

}  // namespace traccc::device
