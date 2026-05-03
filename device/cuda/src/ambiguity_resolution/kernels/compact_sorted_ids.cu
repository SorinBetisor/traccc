/** TRACCC library, part of the ACTS project (R&D line)
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../ambiguity_tuning.hpp"
#include "compact_sorted_ids.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::cuda::kernels {

__global__ TRACCC_LAUNCH_BOUNDS(
    ::traccc::cuda::tuning::graph_kernel_block_size,
    2) void compact_sorted_ids(device::compact_sorted_ids_payload payload) {

    if (*(payload.terminate) == 1) {
        return;
    }

    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int n_acc_old = *(payload.n_accepted);
    if (i >= n_acc_old) {
        return;
    }

    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    vecmem::device_vector<const int> keep_flag(payload.keep_flag_per_slot_view);
    vecmem::device_vector<const int> keep_prefix(payload.keep_prefix_sums_view);
    vecmem::device_vector<unsigned int> compacted(
        payload.compacted_sorted_ids_view);

    if (keep_flag[i] == 1) {
        // Inclusive prefix sum: slot i contributes to keep_prefix[i], so the
        // destination index is keep_prefix[i] - 1.
        const int dst = keep_prefix[i] - 1;
        compacted[dst] = sorted_ids[i];
    }

    // A single thread (the last slot) publishes the new n_accepted. Because
    // the prefix sum is inclusive, keep_prefix[n_acc_old - 1] is the total
    // number of surviving tracks.
    if (i == n_acc_old - 1) {
        *(payload.n_accepted) = static_cast<unsigned int>(keep_prefix[i]);
    }
}

}  // namespace traccc::cuda::kernels
