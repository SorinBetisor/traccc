/** TRACCC library, part of the ACTS project (R&D line)
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
/// to gather the still-accepted vertex list for that row and emit the directed
/// pair list into the COO edge buffers.
///
/// Shared-memory safety guarantee
/// ─────────────────────────────
/// The dynamic shared-memory scratch buffer (smem_gathered) has exactly
/// blockDim.x slots, matching the caller's allocation:
///   smem_bytes = nt_coo * sizeof(unsigned int)
///
/// The kernel handles rows wider than blockDim.x via a slow path that reads
/// directly from global memory, completely bypassing the shared-memory buffer.
/// This ensures smem_gathered[slot] is always in bounds regardless of how many
/// tracks share a single measurement.
__global__ void build_conflict_coo(device::build_conflict_coo_payload payload) {

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

    vecmem::device_vector<unsigned int> coo_src(payload.coo_src_view);
    vecmem::device_vector<unsigned int> coo_dst(payload.coo_dst_view);

    // -----------------------------------------------------------------------
    // Fast path: row fits in one shared-memory chunk (common case on real
    // detector geometries — ODD muon n_rows <= 93, Fatras mu=600 n_rows
    // peaks around 1200 across the full sorted list but individual
    // measurement rows are typically much smaller).
    // -----------------------------------------------------------------------
    if (n_rows <= blockDim.x) {
        // smem_gathered: dynamic shared memory, blockDim.x unsigned ints.
        extern __shared__ unsigned int smem_gathered[];
        __shared__ unsigned int smem_count;

        if (threadIdx.x == 0) {
            smem_count = 0u;
        }
        __syncthreads();

        for (unsigned int k = threadIdx.x; k < n_rows; k += blockDim.x) {
            if (status_u[k] == 1) {
                const unsigned int slot = atomicAdd(&smem_count, 1u);
                // slot is bounded by n_rows <= blockDim.x, so always in range.
                smem_gathered[slot] = tracks_u[k];
            }
        }
        __syncthreads();

        const unsigned int n_gathered = smem_count;
        if (n_gathered < 2u) {
            return;
        }

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
        return;
    }

    // -----------------------------------------------------------------------
    // Slow path: row is wider than blockDim.x.  This is not expected on any
    // real physical detector geometry we have tested (max observed n_rows on
    // measurement rows is well below 128 in ODD and Fatras data), but can
    // occur in adversarial synthetic inputs.
    //
    // Instead of using shared memory (which is bounded at blockDim.x slots),
    // iterate directly over global memory. Each thread is responsible for
    // emitting all outgoing edges from the track at its stride position.
    // Global memory reads of status_u / tracks_u are sequential per warp and
    // cached by L2, so the penalty vs the shared-memory fast path is modest.
    // -----------------------------------------------------------------------
    for (unsigned int i = threadIdx.x; i < n_rows; i += blockDim.x) {
        if (status_u[i] != 1) {
            continue;
        }
        const unsigned int a = tracks_u[i];
        for (unsigned int j = 0u; j < n_rows; ++j) {
            if (j == i) {
                continue;
            }
            if (status_u[j] != 1) {
                continue;
            }
            const unsigned int b = tracks_u[j];
            const unsigned int pos = atomicAdd(payload.edge_count, 1u);
            coo_src[pos] = a;
            coo_dst[pos] = b;
        }
    }
}

}  // namespace traccc::cuda::kernels
