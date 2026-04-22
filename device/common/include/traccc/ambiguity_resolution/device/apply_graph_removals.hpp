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

/// (Event Data) Payload for apply_graph_removals — Tier 2c Stage 2 apply.
///
/// One thread per vertex id. Vertices whose mis_state == IN_MIS are removed:
/// their measurement-level acceptance state is flipped, n_accepted_tracks_
/// per_measurement is atomically decremented, per-measurement transitions of
/// 2 -> 1 mark the surviving neighbour as rel_shared-updated, and the
/// per-track is_removed flag is raised (Stage 1 compaction consumes it).
///
/// Because the MIS is, by construction, a conflict-free independent set in
/// the conflict graph of still-accepted tracks, no two threads can race on
/// the same measurement slot through the IN_MIS path: each contested
/// measurement has at most one IN_MIS track among its accepted rows. The
/// survivor-update step, on the other hand, can race with other removed
/// tracks that happen to share an uncontested side of an edge; atomicCAS on
/// is_updated gates the enqueue to one thread.
struct apply_graph_removals_payload {
    /// Measurement ids per track.
    vecmem::data::jagged_vector_view<const measurement_id_type> meas_ids_view;

    /// Per-track total number of measurements (used for rel_shared recompute).
    vecmem::data::vector_view<const unsigned int> n_meas_view;

    /// Measurement id -> unique index map.
    vecmem::data::vector_view<const unsigned int> meas_id_to_unique_id_view;

    /// Tracks per unique measurement (sorted ascending).
    vecmem::data::jagged_vector_view<const unsigned int>
        tracks_per_measurement_view;

    /// Per-measurement per-entry acceptance flag.
    vecmem::data::jagged_vector_view<int> track_status_per_measurement_view;

    /// Per-measurement accepted-track counter.
    vecmem::data::vector_view<unsigned int>
        n_accepted_tracks_per_measurement_view;

    /// Per-track shared-measurement count.
    vecmem::data::vector_view<unsigned int> n_shared_view;

    /// MIS state from the preceding graph-algorithm rounds.
    vecmem::data::vector_view<const int> mis_state_view;

    /// Written: per-track removal flag consumed by Stage 1 compaction.
    vecmem::data::vector_view<int> is_removed_view;

    /// Written: per-track updated flag used by the insertion-sort pipeline.
    vecmem::data::vector_view<int> is_updated_view;

    /// Written: list of updated-track ids for the insertion-sort pipeline.
    vecmem::data::vector_view<unsigned int> updated_tracks_view;

    /// Written: batch size (IN_MIS vertex count) for logging / diagnostics.
    unsigned int* batch_size;

    /// Written: running number of updated tracks (shared with baseline and
    /// PBG paths; cleared by the host at the start of each outer iteration).
    unsigned int* n_updated_tracks;

    /// Number of vertices in the iteration (== n_tracks).
    unsigned int n_vertices;
};

}  // namespace traccc::device
