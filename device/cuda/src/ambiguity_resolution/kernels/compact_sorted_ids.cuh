/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/ambiguity_resolution/device/compact_sorted_ids.hpp"

namespace traccc::cuda::kernels {

__global__ void compact_sorted_ids(device::compact_sorted_ids_payload payload);

}  // namespace traccc::cuda::kernels
