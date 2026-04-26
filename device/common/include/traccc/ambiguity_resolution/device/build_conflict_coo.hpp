/** TRACCC library, part of the ACTS project (R&D line)
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for build_conflict_coo — Tier 2c Stage 2.
///
/// One block (or warp) per contested unique measurement. For every ordered
/// pair (a, b) of still-accepted tracks that share this measurement, emit
/// both directed edges (a, b) and (b, a) into the COO edge list. Symmetric
/// storage lets the downstream MIS / JP coloring kernels iterate neighbours
/// with a single CSR row read.
///
/// Vertex indexing: vertices are *track ids* (size n_tracks). The "accepted
/// and still-shared" filter is not materialised here; the graph algorithms
/// operate on all vertex ids but only schedule work for vertices whose
/// `n_shared[t] > 0` (equivalent to rel_shared > 0 in the current pipeline).
///
/// See docs/analysis/novelty_algs/conflict_graph_design.md Sec. 3a.
struct build_conflict_coo_payload {

    /// Tracks sharing each unique measurement (sorted ascending per row).
    vecmem::data::jagged_vector_view<const unsigned int>
        tracks_per_measurement_view;

    /// Per unique-measurement per-entry acceptance flag (1 = still accepted,
    /// 0 = track was removed). Used to filter to still-accepted tracks.
    vecmem::data::jagged_vector_view<const int>
        track_status_per_measurement_view;

    /// Per unique-measurement accepted-track count. Read to skip
    /// uncontested measurements (count <= 1 — no edges to emit).
    vecmem::data::vector_view<const unsigned int>
        n_accepted_tracks_per_measurement_view;

    /// Output: COO source vertex ids. Preallocated to max_edges * 2.
    vecmem::data::vector_view<unsigned int> coo_src_view;

    /// Output: COO destination vertex ids. Preallocated to max_edges * 2.
    vecmem::data::vector_view<unsigned int> coo_dst_view;

    /// Output: live edge counter (atomicAdd-incremented). Must be zeroed
    /// before launch. On completion, *edge_count == number of directed
    /// edges written.
    unsigned int* edge_count;

    /// Number of unique measurements (= row count of the jagged views).
    unsigned int meas_count;
};

}  // namespace traccc::device
