/** TRACCC library, part of the ACTS project (R&D line)
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/math.hpp"
#include "traccc/definitions/primitives.hpp"

// Local include(s).
#include "../ambiguity_tuning.hpp"
#include "update_rel_shared.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::cuda::kernels {

// A2 (Tier A): occupancy hint at the tuned block size used by both the
// graph mode and the baseline orchestrator (256 threads = 8 warps).
__global__ TRACCC_LAUNCH_BOUNDS(
    ::traccc::cuda::tuning::graph_kernel_block_size,
    2) void update_rel_shared(device::update_rel_shared_payload payload) {

    if (*(payload.terminate) == 1) {
        return;
    }

    const unsigned int globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int n_updated = *(payload.n_updated_tracks);

    if (globalIndex >= n_updated) {
        return;
    }

    vecmem::device_vector<const unsigned int> updated_tracks(
        payload.updated_tracks_view);
    vecmem::device_vector<const unsigned int> n_shared(payload.n_shared_view);
    vecmem::device_vector<const unsigned int> n_meas(payload.n_meas_view);
    vecmem::device_vector<traccc::scalar> rel_shared(payload.rel_shared_view);

    const unsigned int t = updated_tracks[globalIndex];

    rel_shared[t] = math::div_ieee754(static_cast<traccc::scalar>(n_shared[t]),
                                      static_cast<traccc::scalar>(n_meas[t]));
}

}  // namespace traccc::cuda::kernels
