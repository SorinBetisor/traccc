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

/// (Event Data) Payload for the batch_identify_removals kernel used by the
/// parallel batch greedy (Tier 2a) variant of the CUDA ambiguity resolver.
///
/// See docs/analysis/parallel_batch_greedy_design.md Sec. 4.1 for semantics.
struct batch_identify_removals_payload {

    /// Sorted track ids (worst-first at the tail). Read-only.
    vecmem::data::vector_view<const unsigned int> sorted_ids_view;

    /// Snapshot of n_accepted taken by batch_prologue. Read-only here. The
    /// live n_accepted is mutated only by the post-apply commit kernel, so
    /// reading the snapshot guarantees every thread sees the same value
    /// regardless of inter-block scheduling.
    const unsigned int* n_accepted;

    /// Measurement ids per track (jagged). Read-only.
    vecmem::data::jagged_vector_view<const measurement_id_type> meas_ids_view;

    /// Measurement id -> unique measurement index map. Read-only.
    vecmem::data::vector_view<const unsigned int> meas_id_to_unique_id_view;

    /// Per unique-measurement count of currently-accepted tracks. Read-only
    /// in this kernel; a measurement with count > 1 is still contested.
    vecmem::data::vector_view<const unsigned int>
        n_accepted_tracks_per_measurement_view;

    /// Per-track current number of shared measurements. A track is a candidate
    /// for removal if n_shared > 0.
    vecmem::data::vector_view<const unsigned int> n_shared_view;

    /// Per unique-measurement claim slot: lowest priority (= rank from the
    /// tail of sorted_ids) that claimed this measurement this iteration.
    /// Must be reset to UINT_MAX at the start of every outer iteration.
    vecmem::data::vector_view<unsigned int> claimed_by_view;

    /// Termination flag. If set, the kernel returns early (the graph header
    /// has already decided to terminate).
    int* terminate;

    /// Maximum number of tail entries of sorted_ids to consider this
    /// iteration. The host clamps this to min(n_accepted, window_cap).
    unsigned int candidate_window_size;
};

}  // namespace traccc::device
