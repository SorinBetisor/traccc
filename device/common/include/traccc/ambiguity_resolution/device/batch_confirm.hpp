/** TRACCC library, part of the ACTS project (R&D line)
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"

// VecMem include(s).
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for the batch_confirm kernel used by the parallel
/// batch greedy (Tier 2a) variant of the CUDA ambiguity resolver.
///
/// One thread per candidate rank r. The thread re-reads claimed_by for each
/// of its still-contested measurements and decides whether claim/confirm
/// succeeded for r. If r FAILS confirm, it pushes its rank into a global
/// first_fail slot via atomicMin.
///
/// The downstream apply kernel admits ranks [0, *first_fail) only. Constraining
/// admission to a contiguous conflict-free prefix of the worst tracks keeps
/// the invariant that removed tracks live at the tail of sorted_ids, which is
/// what the unchanged rearrange/update_status pipeline expects.
///
/// See docs/analysis/parallel_batch_greedy_design.md Sec. 4 for the full
/// semantics and the why.
struct batch_confirm_payload {

    /// Sorted track ids (worst-first at the tail). Read-only.
    vecmem::data::vector_view<const unsigned int> sorted_ids_view;

    /// Snapshot of n_accepted (taken by batch_prologue). Read-only.
    const unsigned int* n_accepted;

    /// Measurement ids per track (jagged). Read-only.
    vecmem::data::jagged_vector_view<const measurement_id_type> meas_ids_view;

    /// Measurement id -> unique measurement index map. Read-only.
    vecmem::data::vector_view<const unsigned int> meas_id_to_unique_id_view;

    /// Per unique-measurement count of currently-accepted tracks. Read-only.
    vecmem::data::vector_view<const unsigned int>
        n_accepted_tracks_per_measurement_view;

    /// Per-track current number of shared measurements. Read-only.
    vecmem::data::vector_view<const unsigned int> n_shared_view;

    /// Per unique-measurement claim slot from batch_identify_removals.
    /// Read-only here.
    vecmem::data::vector_view<const unsigned int> claimed_by_view;

    /// Termination flag.
    int* terminate;

    /// Maximum number of tail entries of sorted_ids considered this iteration.
    unsigned int candidate_window_size;

    /// Output: rank of the first candidate that failed confirm. Initialised
    /// to candidate_window_size by batch_prologue. apply admits ranks
    /// [0, *first_fail).
    unsigned int* first_fail;
};

}  // namespace traccc::device
