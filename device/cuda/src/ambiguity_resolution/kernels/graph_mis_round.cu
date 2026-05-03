/** TRACCC library, part of the ACTS project (R&D line)
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../ambiguity_tuning.hpp"
#include "graph_mis_round.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::cuda::kernels {

__global__ TRACCC_LAUNCH_BOUNDS(
    ::traccc::cuda::tuning::graph_kernel_block_size,
    2) void graph_mis_propose(device::graph_mis_round_payload payload) {

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

    if (*(payload.any_undecided) == 0) {
        atomicExch(payload.any_undecided, 1);
    }

    // A3: hot read-only loads through the read-only cache. The neighbour
    // list (col_idx, priority) is invariant during the propose pass, so
    // __ldg is semantically equivalent and frees up L1 for mis_state.
    const unsigned int* TRACCC_RESTRICT priority_p = priority.data();
    const unsigned int* TRACCC_RESTRICT row_ptr_p = row_ptr.data();
    const unsigned int* TRACCC_RESTRICT col_idx_p = col_idx.data();

    const unsigned int my_prio = ::traccc::cuda::tuning::tuned_ldg(priority_p + v);
    const unsigned int beg = ::traccc::cuda::tuning::tuned_ldg(row_ptr_p + v);
    const unsigned int end = ::traccc::cuda::tuning::tuned_ldg(row_ptr_p + v + 1);

    // Critical invariant: a vertex must have at least one UNDECIDED
    // neighbour to become IN_MIS. Without this check, a vertex whose graph
    // neighbours all decided in previous rounds (typically as
    // REMOVED_NEIGHBOR survivors) would vacuously become "locally worst"
    // and be removed, even though it is a GOOD track being rescued by its
    // neighbours' fates.
    bool i_am_local_max = true;
    bool has_undecided_neighbor = false;
    for (unsigned int k = beg; k < end; ++k) {
        const unsigned int u = ::traccc::cuda::tuning::tuned_ldg(col_idx_p + k);
        // mis_state must NOT use __ldg: it is mutated by other blocks in
        // this same kernel via mis_state[v] = IN_MIS, so the read-only
        // cache would observe stale values.
        if (mis_state[u] != device::graph_mis_state::UNDECIDED) {
            continue;
        }
        has_undecided_neighbor = true;
        const unsigned int their_prio =
            ::traccc::cuda::tuning::tuned_ldg(priority_p + u);
        if (their_prio > my_prio || (their_prio == my_prio && u > v)) {
            i_am_local_max = false;
            break;
        }
    }

    if (i_am_local_max && has_undecided_neighbor) {
        mis_state[v] = device::graph_mis_state::IN_MIS;
    }
}

__global__ TRACCC_LAUNCH_BOUNDS(
    ::traccc::cuda::tuning::graph_kernel_block_size,
    2) void graph_mis_finalize(device::graph_mis_round_payload payload) {

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

    const unsigned int* TRACCC_RESTRICT row_ptr_p = row_ptr.data();
    const unsigned int* TRACCC_RESTRICT col_idx_p = col_idx.data();

    const unsigned int beg = ::traccc::cuda::tuning::tuned_ldg(row_ptr_p + v);
    const unsigned int end = ::traccc::cuda::tuning::tuned_ldg(row_ptr_p + v + 1);

    for (unsigned int k = beg; k < end; ++k) {
        const unsigned int u = ::traccc::cuda::tuning::tuned_ldg(col_idx_p + k);
        if (mis_state[u] == device::graph_mis_state::IN_MIS) {
            mis_state[v] = device::graph_mis_state::REMOVED_NEIGHBOR;
            return;
        }
    }
}

}  // namespace traccc::cuda::kernels
