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
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for the update_rel_shared kernel used by the
/// parallel batch greedy (Tier 2a) variant of the CUDA ambiguity resolver.
///
/// Recomputes rel_shared[t] = n_shared[t] / n_meas[t] for every track t in
/// updated_tracks[0..n_updated_tracks). Required because apply_batch_removals
/// updates n_shared atomically across the grid; rel_shared can only be read
/// back consistently after a kernel boundary.
struct update_rel_shared_payload {
    int* terminate;
    unsigned int* n_updated_tracks;
    vecmem::data::vector_view<const unsigned int> updated_tracks_view;
    vecmem::data::vector_view<const unsigned int> n_shared_view;
    vecmem::data::vector_view<const unsigned int> n_meas_view;
    vecmem::data::vector_view<traccc::scalar> rel_shared_view;
};

}  // namespace traccc::device
