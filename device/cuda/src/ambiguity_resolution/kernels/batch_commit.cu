/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "batch_commit.cuh"

namespace traccc::cuda::kernels {

__global__ void batch_commit(device::batch_commit_payload payload) {

    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (*(payload.terminate) == 1) {
        return;
    }

    *(payload.n_accepted) -= *(payload.batch_size);
}

}  // namespace traccc::cuda::kernels
