/** TRACCC library, part of the ACTS project (R&D line)
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for graph_mis_init — Tier 2c.
///
/// Initialises the per-vertex MIS buffers at the start of an outer iteration:
///   mis_state[v]  = UNDECIDED
///   mis_active[v] = 1 iff v is still accepted AND n_shared[v] > 0
///   priority[v]   = inverted_ids[v]        (worse tracks get higher priority)
///
/// Non-active vertices (status-0 tracks, or accepted tracks with no shared
/// measurements this iteration) carry mis_active = 0 and are skipped by the
/// round kernels.
struct graph_mis_init_payload {
    /// Read: inverted sort index — inverted_ids[v] is v's position in
    /// sorted_ids. Used as MIS priority; higher wins.
    vecmem::data::vector_view<const unsigned int> inverted_ids_view;

    /// Read: per-track shared-measurement count. v is only active if > 0.
    vecmem::data::vector_view<const unsigned int> n_shared_view;

    /// Read: live n_accepted. Vertices with inverted_ids[v] >= n_accepted
    /// are already dropped tracks and are inactive.
    const unsigned int* n_accepted;

    /// Write: priority per vertex.
    vecmem::data::vector_view<unsigned int> priority_view;

    /// Write: activity flag per vertex.
    vecmem::data::vector_view<int> mis_active_view;

    /// Write: MIS state per vertex (UNDECIDED).
    vecmem::data::vector_view<int> mis_state_view;

    /// Total number of vertices in the address space (== n_tracks).
    unsigned int n_vertices;
};

}  // namespace traccc::device
