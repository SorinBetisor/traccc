/** TRACCC library, part of the ACTS project (R&D line)
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for compact_sorted_ids — Tier 2c Stage 1.
///
/// Scatters the kept tracks of sorted_ids (those with keep_flag_per_slot[i]
/// == 1) into a contiguous prefix of a temporary buffer using the prefix sum
/// of keep_flag_per_slot. Preserves relative order. The final n_accepted
/// value is written as a post-condition.
///
/// Why it exists: Tier 2c algorithms (Luby MIS, Jones–Plassmann coloring)
/// can select tracks to remove that are *not* in the contiguous tail of
/// sorted_ids, which breaks the invariant the baseline rearrange_tracks
/// pipeline depends on. Stage 1 compaction restores the invariant by
/// rebuilding sorted_ids out of the survivors, after which the unchanged
/// rearrange_tracks / update_status path does its job.
struct compact_sorted_ids_payload {
    /// Current sorted tracks (read).
    vecmem::data::vector_view<const unsigned int> sorted_ids_view;

    /// Keep flags per slot (read). 1 = survive, 0 = dropped.
    vecmem::data::vector_view<const int> keep_flag_per_slot_view;

    /// Inclusive prefix sum of keep_flag_per_slot (read).
    vecmem::data::vector_view<const int> keep_prefix_sums_view;

    /// Output: compacted sorted_ids. Slot 0 = first survivor, ...
    vecmem::data::vector_view<unsigned int> compacted_sorted_ids_view;

    /// Live n_accepted (read pre-kernel, overwritten post-kernel by a single
    /// thread with the number of survivors).
    unsigned int* n_accepted;

    /// Termination flag. Short-circuits the kernel body.
    int* terminate;
};

}  // namespace traccc::device
