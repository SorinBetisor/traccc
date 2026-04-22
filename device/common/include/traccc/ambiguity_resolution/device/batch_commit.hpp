/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

/// (Event Data) Payload for the batch_commit kernel used by the parallel
/// batch greedy (Tier 2a) variant of the CUDA ambiguity resolver.
///
/// Single-thread single-block kernel that runs after apply_batch_removals.
/// It commits the per-iteration batch by subtracting batch_size from the live
/// n_accepted. Splitting this from apply guarantees that every thread of
/// batch_identify_removals and apply_batch_removals reads the same snapshot
/// of n_accepted (taken in batch_prologue), avoiding the intra-grid race
/// that would otherwise let blocks see drifting n_accepted values.
struct batch_commit_payload {
    int* terminate;
    unsigned int* n_accepted;
    const unsigned int* batch_size;
};

}  // namespace traccc::device
