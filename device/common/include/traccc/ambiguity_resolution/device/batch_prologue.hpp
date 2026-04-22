/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
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
/// The claimed_by buffer is reset by an explicit cudaMemsetAsync node placed
/// before this kernel in the graph.
struct batch_prologue_payload {
    int* terminate;
    unsigned int* max_shared;
    unsigned int* n_updated_tracks;
    unsigned int* batch_size;
};

}  // namespace traccc::device
