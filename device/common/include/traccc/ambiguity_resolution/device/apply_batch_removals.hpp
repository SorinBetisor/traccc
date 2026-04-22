/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
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

/// (Event Data) Payload for the apply_batch_removals kernel used by the
/// parallel batch greedy (Tier 2a) variant of the CUDA ambiguity resolver.
///
/// See docs/analysis/parallel_batch_greedy_design.md Sec. 4.2 for semantics.
struct apply_batch_removals_payload {

    /// Sorted track ids (worst-first at the tail).
    vecmem::data::vector_view<const unsigned int> sorted_ids_view;

    /// Current number of accepted tracks (decremented by batch_size).
    unsigned int* n_accepted;

    /// Measurement ids per track.
    vecmem::data::jagged_vector_view<const measurement_id_type> meas_ids_view;

    /// Per-track total number of measurements. Used for rel_shared recompute.
    vecmem::data::vector_view<const unsigned int> n_meas_view;

    /// Measurement id -> unique measurement index map.
    vecmem::data::vector_view<const unsigned int> meas_id_to_unique_id_view;

    /// Tracks sharing each unique measurement.
    vecmem::data::jagged_vector_view<const unsigned int>
        tracks_per_measurement_view;

    /// Per unique-measurement per-entry acceptance flag (1 = still accepted,
    /// 0 = the track was removed). Written for every measurement of each
    /// admitted track.
    vecmem::data::jagged_vector_view<int> track_status_per_measurement_view;

    /// Per unique-measurement count of currently-accepted tracks (atomically
    /// decremented for each admitted track).
    vecmem::data::vector_view<unsigned int>
        n_accepted_tracks_per_measurement_view;

    /// Per-track current number of shared measurements (atomically updated
    /// when an admitted track leaves a neighbour alone on a measurement).
    vecmem::data::vector_view<unsigned int> n_shared_view;

    /// Per-track relative number of shared measurements. Recomputed for every
    /// neighbour track whose n_shared changes.
    vecmem::data::vector_view<traccc::scalar> rel_shared_view;

    /// Claim slots produced by batch_identify_removals. Read-only in this
    /// kernel; used to confirm a candidate is admitted.
    vecmem::data::vector_view<const unsigned int> claimed_by_view;

    /// Output buffer: track ids admitted to this iteration's removal batch.
    vecmem::data::vector_view<unsigned int> batch_ids_view;

    /// Output: the number of tracks admitted this iteration.
    unsigned int* batch_size;

    /// Termination flag. If set, the kernel returns early.
    int* terminate;

    /// Maximum number of tail entries of sorted_ids to consider this
    /// iteration (same value as passed to batch_identify_removals).
    unsigned int candidate_window_size;

    /// Number of tracks whose rel_shared/n_shared changed this iteration.
    unsigned int* n_updated_tracks;

    /// View of updated-track buffer (used by the downstream insertion-sort
    /// pipeline unchanged from baseline).
    vecmem::data::vector_view<unsigned int> updated_tracks_view;

    /// Per-track updated flag.
    vecmem::data::vector_view<int> is_updated_view;

    /// Scratch ref-count to detect "last one standing" duplicates among
    /// neighbours touched multiple times in the same iteration.
    vecmem::data::vector_view<int> track_count_view;
};

}  // namespace traccc::device
