/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/ambiguity_resolution/device/batch_commit.hpp"

namespace traccc::cuda::kernels {

__global__ void batch_commit(device::batch_commit_payload payload);
}
