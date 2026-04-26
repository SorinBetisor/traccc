/** TRACCC library, part of the ACTS project (R&D line)
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "graph_mis_round.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::cuda::kernels {

__global__ void graph_mis_propose(device::graph_mis_round_payload payload) {

    const unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= payload.n_vertices) {
        return;
    }

    vecmem::device_vector<const int> mis_active(payload.mis_active_view);
    vecmem::device_vector<int> mis_state(payload.mis_state_view);
    vecmem::device_vector<const unsigned int> priority(payload.priority_view);
    vecmem::device_vector<const unsigned int> row_ptr(payload.row_ptr_view);
    vecmem::device_vector<const unsigned int> col_idx(payload.col_idx_view);

    if (mis_active[v] == 0) {
        return;
    }
    if (mis_state[v] != device::graph_mis_state::UNDECIDED) {
        return;
    }

    // At least one UNDECIDED vertex exists. Any thread that gets here would
    // set this; duplicate writes are fine.
    if (*(payload.any_undecided) == 0) {
        atomicExch(payload.any_undecided, 1);
    }

    const unsigned int my_prio = priority[v];
    const unsigned int beg = row_ptr[v];
    const unsigned int end = row_ptr[v + 1];

    // Priority here is `inverted_ids[v]`. Higher rank = later in sorted_ids
    // = worse track (higher rel_shared). The MIS is the set of LOCALLY
    // WORST tracks: a vertex qualifies when its priority strictly exceeds
    // every still-UNDECIDED neighbour's priority. IN_MIS is then the batch
    // we REMOVE, and because the MIS is an independent set by construction
    // no two IN_MIS vertices share a measurement — the apply kernel is
    // race-free.
    //
    // Critical invariant: a vertex must have at least one UNDECIDED
    // neighbour to become IN_MIS. Without this check, a vertex whose graph
    // neighbours all decided in previous rounds (typically as
    // REMOVED_NEIGHBOR survivors) would vacuously become "locally worst"
    // and be removed, even though it is a GOOD track being rescued by its
    // neighbours' fates. Such vertices must remain UNDECIDED and will be
    // re-evaluated in a later outer iteration once the graph is rebuilt.
    bool i_am_local_max = true;
    bool has_undecided_neighbor = false;
    for (unsigned int k = beg; k < end; ++k) {
        const unsigned int u = col_idx[k];
        if (mis_state[u] != device::graph_mis_state::UNDECIDED) {
            continue;
        }
        has_undecided_neighbor = true;
        const unsigned int their_prio = priority[u];
        if (their_prio > my_prio || (their_prio == my_prio && u > v)) {
            i_am_local_max = false;
            break;
        }
    }

    if (i_am_local_max && has_undecided_neighbor) {
        mis_state[v] = device::graph_mis_state::IN_MIS;
    }
}

__global__ void graph_mis_finalize(device::graph_mis_round_payload payload) {

    const unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= payload.n_vertices) {
        return;
    }

    vecmem::device_vector<const int> mis_active(payload.mis_active_view);
    vecmem::device_vector<int> mis_state(payload.mis_state_view);
    vecmem::device_vector<const unsigned int> row_ptr(payload.row_ptr_view);
    vecmem::device_vector<const unsigned int> col_idx(payload.col_idx_view);

    if (mis_active[v] == 0) {
        return;
    }
    if (mis_state[v] != device::graph_mis_state::UNDECIDED) {
        return;
    }

    const unsigned int beg = row_ptr[v];
    const unsigned int end = row_ptr[v + 1];

    for (unsigned int k = beg; k < end; ++k) {
        const unsigned int u = col_idx[k];
        if (mis_state[u] == device::graph_mis_state::IN_MIS) {
            mis_state[v] = device::graph_mis_state::REMOVED_NEIGHBOR;
            return;
        }
    }
}

}  // namespace traccc::cuda::kernels
