/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/ambiguity_resolution/device/graph_mis_round.hpp"

namespace traccc::cuda::kernels {

/// Proposal pass of one Luby MIS round. For each UNDECIDED active vertex v,
/// set state[v] = IN_MIS if priority[v] > priority[u] for every UNDECIDED
/// neighbour u. Also raises *any_undecided if at least one UNDECIDED vertex
/// was seen.
__global__ void graph_mis_propose(device::graph_mis_round_payload payload);

/// Finalize pass of one Luby MIS round. For each UNDECIDED active vertex v,
/// set state[v] = REMOVED_NEIGHBOR if any neighbour u is IN_MIS.
__global__ void graph_mis_finalize(device::graph_mis_round_payload payload);

}  // namespace traccc::cuda::kernels
