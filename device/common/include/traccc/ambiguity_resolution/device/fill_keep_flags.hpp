/** TRACCC library, part of the ACTS project (R&D line)
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for fill_keep_flags — Tier 2c Stage 1.
///
/// For each slot i in [0, n_accepted), write
///     keep_flag_per_slot[i] = (is_removed[sorted_ids[i]] == 0)
/// so that a downstream parallel prefix sum + scatter compacts sorted_ids by
/// dropping exactly the removed tracks while preserving relative order. Also
/// clears is_removed[t] after reading it, so the flag is per-iteration.
struct fill_keep_flags_payload {
    /// Current sorted tracks (prefix [0, n_accepted)).
    vecmem::data::vector_view<const unsigned int> sorted_ids_view;

    /// Live n_accepted (read-only).
    const unsigned int* n_accepted;

    /// Termination flag. Short-circuits the kernel body.
    int* terminate;

    /// Per-track removal flag written by apply_graph_removals. Set to 1 for
    /// tracks the current outer iteration has eliminated. Cleared slot-by-
    /// slot as this kernel consumes it, so it remains zero on entry next
    /// iteration without needing an additional memset.
    vecmem::data::vector_view<int> is_removed_view;

    /// Output: 1 = keep the track at slot i, 0 = drop it.
    vecmem::data::vector_view<int> keep_flag_per_slot_view;
};

}  // namespace traccc::device
