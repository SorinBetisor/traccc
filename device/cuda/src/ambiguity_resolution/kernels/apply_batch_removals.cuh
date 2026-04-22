/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/ambiguity_resolution/device/apply_batch_removals.hpp"

namespace traccc::cuda::kernels {

__global__ void apply_batch_removals(
    device::apply_batch_removals_payload payload);
}
