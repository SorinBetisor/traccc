/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// State codes for the Luby-MIS / JP-round-1 state vector.
namespace graph_mis_state {
constexpr int UNDECIDED = 0;
constexpr int IN_MIS = 1;
constexpr int REMOVED_NEIGHBOR = 2;
}  // namespace graph_mis_state

/// (Event Data) Payload for graph_mis_propose / graph_mis_finalize — Tier 2c.
///
/// One round of Luby-style MIS over the explicit CSR conflict graph.
/// Proposal phase marks every UNDECIDED vertex whose deterministic priority
/// (higher inverted_ids rank = worse track = higher priority) exceeds that
/// of all UNDECIDED neighbours; marked vertices transition to IN_MIS.
/// Finalize phase turns every UNDECIDED neighbour of an IN_MIS vertex into
/// REMOVED_NEIGHBOR (not removed from the solver — just excluded from the
/// MIS for this iteration; it stays accepted). A host-side loop repeats the
/// two passes until no vertex remains UNDECIDED.
///
/// JP-round-1 mode reuses graph_mis_propose only and accepts the non-maximal
/// independent set it produces.
struct graph_mis_round_payload {
    /// Only vertices v with mis_active[v] != 0 participate. Non-participating
    /// vertices are skipped (already-removed tracks, or tracks with
    /// n_shared[v] == 0 this iteration). Populated by the host before the
    /// first round from the still-shared accepted track set.
    vecmem::data::vector_view<const int> mis_active_view;

    /// State vector [UNDECIDED, IN_MIS, REMOVED_NEIGHBOR]. Initialised to
    /// UNDECIDED by the host for all active vertices before round 0.
    vecmem::data::vector_view<int> mis_state_view;

    /// Priority per vertex. Higher wins. Set to inverted_ids[v] (distance
    /// from the worst-end of sorted_ids increases priority).
    vecmem::data::vector_view<const unsigned int> priority_view;

    /// CSR row pointer. row_ptr[v] .. row_ptr[v+1] is the neighbour range
    /// for vertex v.
    vecmem::data::vector_view<const unsigned int> row_ptr_view;

    /// CSR column indices. col_idx[k] = neighbour vertex id.
    vecmem::data::vector_view<const unsigned int> col_idx_view;

    /// Number of vertices (== n_tracks). The grid iterates the full range so
    /// that non-active vertices short-circuit cheaply; the kernel treats
    /// mis_active_view == 0 as an early return.
    unsigned int n_vertices;

    /// Progress flag written in the propose kernel. Set to 1 whenever an
    /// UNDECIDED vertex exists; the host reads it after each round to
    /// decide whether to keep iterating.
    int* any_undecided;
};

}  // namespace traccc::device
