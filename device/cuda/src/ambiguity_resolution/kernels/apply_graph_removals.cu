/** TRACCC library, part of the ACTS project (R&D line)
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/ambiguity_resolution/device/graph_mis_round.hpp"
#include "traccc/definitions/primitives.hpp"

// Local include(s).
#include "../ambiguity_tuning.hpp"
#include "apply_graph_removals.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>

// Thrust include(s).
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

namespace traccc::cuda::kernels {

// A2 (Tier A): occupancy hint at the tuned block size (256). The kernel is
// register-heavy because of the inner search loops; minBlocksPerSM=2 keeps
// 50% theoretical occupancy without forcing aggressive register spilling.
__global__ TRACCC_LAUNCH_BOUNDS(
    ::traccc::cuda::tuning::graph_kernel_block_size,
    2) void apply_graph_removals(
    device::apply_graph_removals_payload payload) {

    const unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= payload.n_vertices) {
        return;
    }

    vecmem::device_vector<const int> mis_state(payload.mis_state_view);
    // Remove only IN_MIS vertices. Because graph_mis_propose requires a
    // strict local maximum AND at least one undecided neighbour, IN_MIS is
    // always a subset of still-contested worst tracks AND remains an
    // independent set in the conflict graph. Removing an independent set in
    // parallel is race-free — no two IN_MIS threads share a measurement
    // slot on which they both atomically decrement.
    if (mis_state[t] != device::graph_mis_state::IN_MIS) {
        return;
    }

    // We are a removed track. Propagate the measurement-level status change
    // and raise is_removed so Stage 1 compacts us out of sorted_ids.
    vecmem::jagged_device_vector<const measurement_id_type> meas_ids(
        payload.meas_ids_view);
    vecmem::device_vector<const unsigned int> meas_id_to_unique_id(
        payload.meas_id_to_unique_id_view);
    vecmem::jagged_device_vector<const unsigned int> tracks_per_measurement(
        payload.tracks_per_measurement_view);
    vecmem::jagged_device_vector<int> track_status_per_measurement(
        payload.track_status_per_measurement_view);
    vecmem::device_vector<unsigned int> n_accepted_tracks_per_meas(
        payload.n_accepted_tracks_per_measurement_view);
    vecmem::device_vector<unsigned int> n_shared(payload.n_shared_view);
    vecmem::device_vector<int> is_removed(payload.is_removed_view);
    vecmem::device_vector<int> is_updated(payload.is_updated_view);
    vecmem::device_vector<unsigned int> updated_tracks(
        payload.updated_tracks_view);

    is_removed[t] = 1;
    // Zero our own n_shared so host-side termination checks based on
    // thrust::max_element over the full n_shared buffer do not see stale
    // values from removed tracks.
    n_shared[t] = 0u;
    atomicAdd(payload.batch_size, 1u);

    const auto& mids = meas_ids[t];
    const unsigned int n_m = static_cast<unsigned int>(mids.size());

    for (unsigned int i = 0; i < n_m; ++i) {
        const auto mid = mids[i];
        const unsigned int u = meas_id_to_unique_id[mid];

        auto tracks_on_u = tracks_per_measurement[u];
        auto status_on_u = track_status_per_measurement[u];

        const unsigned int my_idx = static_cast<unsigned int>(
            thrust::lower_bound(thrust::seq, tracks_on_u.begin(),
                                tracks_on_u.end(), t) -
            tracks_on_u.begin());
        status_on_u[my_idx] = 0;

        const unsigned int prev_count =
            atomicSub(&n_accepted_tracks_per_meas[u], 1u);

        if (prev_count == 2u) {
            int alive_idx = -1;
            const unsigned int n_rows =
                static_cast<unsigned int>(status_on_u.size());
            for (unsigned int k = 0; k < n_rows; ++k) {
                if (status_on_u[k] == 1) {
                    alive_idx = static_cast<int>(k);
                    break;
                }
            }
            if (alive_idx < 0) {
                continue;
            }

            const unsigned int alive = tracks_on_u[alive_idx];
            const auto& alive_mids = meas_ids[alive];
            const unsigned int m_count =
                static_cast<unsigned int>(thrust::count(
                    thrust::seq, alive_mids.begin(), alive_mids.end(), mid));
            atomicSub(&n_shared[alive], m_count);

            if (atomicCAS(&is_updated[alive], 0, 1) == 0) {
                const unsigned int pos =
                    atomicAdd(payload.n_updated_tracks, 1u);
                updated_tracks[pos] = alive;
            }
        }
    }
}

}  // namespace traccc::cuda::kernels
