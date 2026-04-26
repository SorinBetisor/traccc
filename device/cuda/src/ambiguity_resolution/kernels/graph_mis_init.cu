/** TRACCC library, part of the ACTS project (R&D line)
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/ambiguity_resolution/device/graph_mis_round.hpp"

// Local include(s).
#include "graph_mis_init.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::cuda::kernels {

__global__ void graph_mis_init(device::graph_mis_init_payload payload) {

    const unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= payload.n_vertices) {
        return;
    }

    vecmem::device_vector<const unsigned int> inverted_ids(
        payload.inverted_ids_view);
    vecmem::device_vector<const unsigned int> n_shared(payload.n_shared_view);
    vecmem::device_vector<unsigned int> priority(payload.priority_view);
    vecmem::device_vector<int> mis_active(payload.mis_active_view);
    vecmem::device_vector<int> mis_state(payload.mis_state_view);

    const unsigned int n_acc = *(payload.n_accepted);
    const unsigned int pos = inverted_ids[v];

    priority[v] = pos;
    mis_state[v] = device::graph_mis_state::UNDECIDED;
    mis_active[v] = (pos < n_acc && n_shared[v] > 0u) ? 1 : 0;
}

}  // namespace traccc::cuda::kernels
