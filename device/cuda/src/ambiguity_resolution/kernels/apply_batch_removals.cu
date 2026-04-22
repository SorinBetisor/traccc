/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/primitives.hpp"

// Local include(s).
#include "apply_batch_removals.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>

// Thrust include(s).
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

namespace traccc::cuda::kernels {

__global__ void apply_batch_removals(
    device::apply_batch_removals_payload payload) {

    if (*(payload.terminate) == 1) {
        return;
    }

    const unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= payload.candidate_window_size) {
        return;
    }

    // Conflict-free prefix admission: only ranks strictly below first_fail
    // are eligible. This keeps removed tracks at the contiguous tail of
    // sorted_ids, which is the invariant the unchanged rearrange_tracks /
    // update_status pipeline expects.
    if (r >= *(payload.first_fail)) {
        return;
    }

    const unsigned int n_acc = *(payload.n_accepted);
    if (r >= n_acc) {
        return;
    }

    // Views.
    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
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
    vecmem::device_vector<const unsigned int> claimed_by(
        payload.claimed_by_view);
    vecmem::device_vector<unsigned int> batch_ids(payload.batch_ids_view);
    vecmem::device_vector<unsigned int> updated_tracks(
        payload.updated_tracks_view);
    vecmem::device_vector<int> is_updated(payload.is_updated_view);

    const unsigned int t = sorted_ids[n_acc - 1 - r];

    // Same candidate criterion as batch_identify: still-shared tracks only.
    if (n_shared[t] == 0u) {
        return;
    }

    const auto& mids = meas_ids[t];
    const unsigned int n_m = static_cast<unsigned int>(mids.size());

    // --- confirm: I am admitted iff every still-contested measurement of
    // mine holds my priority in claimed_by. Uncontested measurements (count
    // == 1) are ignored because no one else is competing there anyway.
    bool ok = true;
    for (unsigned int i = 0; i < n_m; ++i) {
        const unsigned int u = meas_id_to_unique_id[mids[i]];
        if (n_accepted_tracks_per_meas[u] > 1u && claimed_by[u] != r) {
            ok = false;
            break;
        }
    }
    if (!ok) {
        return;
    }

    // --- admit: append to batch_ids. The corresponding n_accepted decrement
    // is deferred to batch_commit (single-thread post-kernel) so threads do
    // not see drifting n_accepted values within this grid.
    const unsigned int slot = atomicAdd(payload.batch_size, 1u);
    batch_ids[slot] = t;

    // --- apply: measurement-level propagation (mirrors baseline remove_tracks
    // second half, but races only across distinct measurements because
    // admitted tracks are measurement-disjoint by construction of the
    // independent set).
    for (unsigned int i = 0; i < n_m; ++i) {
        const auto mid = mids[i];
        const unsigned int u = meas_id_to_unique_id[mid];

        auto tracks_on_u = tracks_per_measurement[u];
        auto status_on_u = track_status_per_measurement[u];

        // Flip my own status slot in the jagged tracks_per_measurement row.
        const unsigned int my_idx = static_cast<unsigned int>(
            thrust::lower_bound(thrust::seq, tracks_on_u.begin(),
                                tracks_on_u.end(), t) -
            tracks_on_u.begin());
        status_on_u[my_idx] = 0;

        // Decrement the count of accepted tracks on this measurement.
        const unsigned int prev_count =
            atomicSub(&n_accepted_tracks_per_meas[u], 1u);

        // prev_count is the value BEFORE this decrement. If it was 2, exactly
        // one accepted track remains on this measurement and its n_shared is
        // reduced.
        if (prev_count == 2u) {
            // Find the unique surviving track on measurement u. Our flip to 0
            // has already committed (same thread), so the surviving entry is
            // the only one still carrying status == 1.
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

            // Number of times measurement u appears in the survivor's meas
            // list. Typically 1, but be safe.
            const auto& alive_mids = meas_ids[alive];
            const unsigned int m_count =
                static_cast<unsigned int>(thrust::count(
                    thrust::seq, alive_mids.begin(), alive_mids.end(), mid));

            atomicSub(&n_shared[alive], m_count);

            // Enqueue the survivor exactly once across the grid.
            if (atomicCAS(&is_updated[alive], 0, 1) == 0) {
                const unsigned int pos =
                    atomicAdd(payload.n_updated_tracks, 1u);
                updated_tracks[pos] = alive;
            }
        }
    }
}

}  // namespace traccc::cuda::kernels
