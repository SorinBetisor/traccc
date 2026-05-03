/** TRACCC library, part of the ACTS project (R&D line)
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../ambiguity_tuning.hpp"
#include "fill_keep_flags.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::cuda::kernels {

__global__ TRACCC_LAUNCH_BOUNDS(
    ::traccc::cuda::tuning::graph_kernel_block_size,
    2) void fill_keep_flags(device::fill_keep_flags_payload payload) {

    if (*(payload.terminate) == 1) {
        return;
    }

    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int n_acc = *(payload.n_accepted);
    if (i >= n_acc) {
        return;
    }

    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    vecmem::device_vector<int> is_removed(payload.is_removed_view);
    vecmem::device_vector<int> keep_flag(payload.keep_flag_per_slot_view);

    const unsigned int t = sorted_ids[i];
    const int removed = is_removed[t];
    keep_flag[i] = removed ? 0 : 1;

    // Clear is_removed for the next outer iteration. Safe because each track
    // appears in sorted_ids at most once per iteration.
    if (removed) {
        is_removed[t] = 0;
    }
}

}  // namespace traccc::cuda::kernels
