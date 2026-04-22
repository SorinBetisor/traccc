/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/ambiguity_resolution/device/update_rel_shared.hpp"

namespace traccc::cuda::kernels {

__global__ void update_rel_shared(device::update_rel_shared_payload payload);
}
