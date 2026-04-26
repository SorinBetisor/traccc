/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/ambiguity_resolution/device/build_conflict_coo.hpp"

namespace traccc::cuda::kernels {

__global__ void build_conflict_coo(device::build_conflict_coo_payload payload);

}  // namespace traccc::cuda::kernels
