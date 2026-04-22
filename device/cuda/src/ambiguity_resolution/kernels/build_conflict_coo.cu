/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "build_conflict_coo.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>

namespace traccc::cuda::kernels {

/// Grid layout: one CTA per unique measurement. Threads of the CTA cooperate
/// to gather the still-accepted vertex list of that row and emit the
/// directed pair list. With blockDim.x threads per measurement and a fast
/// shared-memory gather, n_m <= blockDim.x measurements are handled in a
/// single iteration; larger rows loop.
__global__ void build_conflict_coo(
    device::build_conflict_coo_payload payload) {

    const unsigned int u = blockIdx.x;
    if (u >= payload.meas_count) {
        return;
    }

    vecmem::device_vector<const unsigned int> n_acc_per_meas(
        payload.n_accepted_tracks_per_measurement_view);

    // Fast reject: uncontested measurements contribute no edges.
    if (n_acc_per_meas[u] <= 1u) {
        return;
    }

    vecmem::jagged_device_vector<const unsigned int> tracks_per_meas(
        payload.tracks_per_measurement_view);
    vecmem::jagged_device_vector<const int> status_per_meas(
        payload.track_status_per_measurement_view);

    auto tracks_u = tracks_per_meas[u];
    auto status_u = status_per_meas[u];

    const unsigned int n_rows = static_cast<unsigned int>(tracks_u.size());

    // Collect still-accepted track ids into shared memory.
    extern __shared__ unsigned int smem_gathered[];
    __shared__ unsigned int smem_count;

    if (threadIdx.x == 0) {
        smem_count = 0u;
    }
    __syncthreads();

    for (unsigned int k = threadIdx.x; k < n_rows; k += blockDim.x) {
        if (status_u[k] == 1) {
            const unsigned int slot = atomicAdd(&smem_count, 1u);
            smem_gathered[slot] = tracks_u[k];
        }
    }
    __syncthreads();

    const unsigned int n_gathered = smem_count;
    if (n_gathered < 2u) {
        return;
    }

    // Emit directed pairs. For each i, emit (i, j) for all j != i in
    // [0, n_gathered). Threads split the i dimension.
    vecmem::device_vector<unsigned int> coo_src(payload.coo_src_view);
    vecmem::device_vector<unsigned int> coo_dst(payload.coo_dst_view);

    for (unsigned int i = threadIdx.x; i < n_gathered; i += blockDim.x) {
        const unsigned int a = smem_gathered[i];
        for (unsigned int j = 0; j < n_gathered; ++j) {
            if (j == i) {
                continue;
            }
            const unsigned int b = smem_gathered[j];
            const unsigned int pos = atomicAdd(payload.edge_count, 1u);
            coo_src[pos] = a;
            coo_dst[pos] = b;
        }
    }
}

}  // namespace traccc::cuda::kernels
