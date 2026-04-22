/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/primitives.hpp"

// Local include(s).
#include "batch_identify_removals.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>

namespace traccc::cuda::kernels {

__global__ void batch_identify_removals(
    device::batch_identify_removals_payload payload) {

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

    // Read-only views.
    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    vecmem::jagged_device_vector<const measurement_id_type> meas_ids(
        payload.meas_ids_view);
    vecmem::device_vector<const unsigned int> meas_id_to_unique_id(
        payload.meas_id_to_unique_id_view);
    vecmem::device_vector<const unsigned int> n_accepted_tracks_per_meas(
        payload.n_accepted_tracks_per_measurement_view);
    vecmem::device_vector<const unsigned int> n_shared(payload.n_shared_view);
    vecmem::device_vector<unsigned int> claimed_by(payload.claimed_by_view);

    // Candidate priority = rank from the tail of sorted_ids (0 = worst).
    const unsigned int t = sorted_ids[n_acc - 1 - r];

    // Candidate selection: only tracks that still have shared measurements.
    // This matches the baseline criterion (remove_tracks considers tracks
    // from the tail of sorted_ids whose worst measurement is still shared).
    if (n_shared[t] == 0u) {
        return;
    }

    const auto& mids = meas_ids[t];
    const unsigned int n_m = static_cast<unsigned int>(mids.size());

    // Claim each still-contested measurement using priority = r. Uncontested
    // measurements (n_accepted_tracks_per_measurement == 1) are skipped; a
    // track whose only-still-contested measurements are all un-claimed is
    // admitted (the confirm pass in apply_batch_removals verifies this).
    for (unsigned int i = 0; i < n_m; ++i) {
        const unsigned int u = meas_id_to_unique_id[mids[i]];
        if (n_accepted_tracks_per_meas[u] > 1u) {
            // atomicMin on a global-memory slot: lowest priority wins.
            // operator[] on a vecmem::device_vector returns a reference into
            // the underlying global-memory buffer; taking its address gives a
            // valid device pointer for CUDA's generic-pointer atomics.
            atomicMin(&claimed_by[u], r);
        }
    }
}

}  // namespace traccc::cuda::kernels
