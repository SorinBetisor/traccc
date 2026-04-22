/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/definitions/primitives.hpp"

#include "batch_confirm.cuh"

#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>

namespace traccc::cuda::kernels {

__global__ void batch_confirm(device::batch_confirm_payload payload) {

    if (*(payload.terminate) == 1) {
        return;
    }

    const unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= payload.candidate_window_size) {
        return;
    }
    const unsigned int n_acc = *(payload.n_accepted);
    if (r >= n_acc) {
        return;
    }

    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    vecmem::jagged_device_vector<const measurement_id_type> meas_ids(
        payload.meas_ids_view);
    vecmem::device_vector<const unsigned int> meas_id_to_unique_id(
        payload.meas_id_to_unique_id_view);
    vecmem::device_vector<const unsigned int> n_accepted_tracks_per_meas(
        payload.n_accepted_tracks_per_measurement_view);
    vecmem::device_vector<const unsigned int> n_shared(payload.n_shared_view);
    vecmem::device_vector<const unsigned int> claimed_by(
        payload.claimed_by_view);

    const unsigned int t = sorted_ids[n_acc - 1 - r];

    // Tracks with no still-shared measurements are not candidates -- they
    // implicitly succeed (no contention to resolve), so we do NOT push their
    // rank into first_fail. This mirrors the candidate filter in claim.
    if (n_shared[t] == 0u) {
        return;
    }

    const auto& mids = meas_ids[t];
    const unsigned int n_m = static_cast<unsigned int>(mids.size());

    bool ok = true;
    for (unsigned int i = 0; i < n_m; ++i) {
        const unsigned int u = meas_id_to_unique_id[mids[i]];
        if (n_accepted_tracks_per_meas[u] > 1u && claimed_by[u] != r) {
            ok = false;
            break;
        }
    }

    if (!ok) {
        atomicMin(payload.first_fail, r);
    }
}

}  // namespace traccc::cuda::kernels
