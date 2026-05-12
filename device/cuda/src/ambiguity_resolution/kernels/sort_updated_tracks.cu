/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/barrier.hpp"
#include "../../utils/global_index.hpp"
#include "sort_updated_tracks.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::cuda::kernels {

__launch_bounds__(512) __global__
    void sort_updated_tracks(device::sort_updated_tracks_payload payload) {

    if (*(payload.terminate) == 1 || *(payload.n_updated_tracks) == 0) {
        return;
    }

    vecmem::device_vector<const traccc::scalar> rel_shared(
        payload.rel_shared_view);
    vecmem::device_vector<const traccc::scalar> pvals(payload.pvals_view);
    vecmem::device_vector<unsigned int> updated_tracks(
        payload.updated_tracks_view);

    const unsigned int tid = threadIdx.x;
    const unsigned int n_updated = *(payload.n_updated_tracks);
    const unsigned int N = 1u << (32 - __clz(n_updated - 1));

    // GB-2: warp-only fast path when n_updated_tracks fits in one warp.
    // Eliminates all __syncthreads by using __shfl_xor_sync for every
    // compare-swap step of the bitonic network.  Only threads with
    // tid < N participate; the rest exit early after the initial load.
    if (N <= 32u) {
        const unsigned int mask = (N == 32u) ? 0xFFFFFFFFu : ((1u << N) - 1u);
        const unsigned int sentinel = std::numeric_limits<unsigned int>::max();

        unsigned int trk = (tid < n_updated) ? updated_tracks[tid] : sentinel;
        traccc::scalar rel  = (trk != sentinel) ? rel_shared[trk]
                                                 : std::numeric_limits<traccc::scalar>::max();
        traccc::scalar pval = (trk != sentinel) ? pvals[trk] : 0.f;

        for (int k = 2; k <= static_cast<int>(N); k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                // Exchange key/value with partner lane.
                unsigned int trk_p  = __shfl_xor_sync(mask, trk,  j);
                traccc::scalar rel_p  = __shfl_xor_sync(mask, rel,  j);
                traccc::scalar pval_p = __shfl_xor_sync(mask, pval, j);

                // Partner's lane index.
                const int ixj = static_cast<int>(tid) ^ j;

                // Decide whether to keep or adopt partner's element.
                // The segment is ascending when (tid & k) == 0.
                const bool ascending = ((tid & k) == 0);
                const bool lower     = (static_cast<int>(tid) < ixj);

                // Comparator: current element should be the smaller one in
                // ascending segments (lower-index position), larger in
                // descending segments.
                const bool cur_beats_partner =
                    (rel < rel_p) || (rel == rel_p && pval >= pval_p);
                const bool want_cur =
                    ascending ? (lower ? cur_beats_partner : !cur_beats_partner)
                              : (lower ? !cur_beats_partner : cur_beats_partner);

                if (!want_cur) {
                    trk  = trk_p;
                    rel  = rel_p;
                    pval = pval_p;
                }
            }
        }

        if (tid < n_updated) {
            updated_tracks[tid] = trk;
        }
        return;
    }

    // Slow path: standard shared-memory bitonic sort for n_updated > 32.
    __shared__ unsigned int shared_mem_tracks[512];
    shared_mem_tracks[tid] = std::numeric_limits<unsigned int>::max();
    if (tid < n_updated) {
        shared_mem_tracks[tid] = updated_tracks[tid];
    }
    __syncthreads();

    traccc::scalar rel_i, rel_j, pval_i, pval_j;

    for (int k = 2; k <= static_cast<int>(N); k <<= 1) {
        const bool ascending = ((tid & k) == 0);

        for (int j = k >> 1; j > 0; j >>= 1) {
            const int ixj = static_cast<int>(tid) ^ j;

            if (ixj > static_cast<int>(tid) &&
                ixj < static_cast<int>(N) &&
                static_cast<int>(tid) < static_cast<int>(N)) {

                unsigned int trk_i = shared_mem_tracks[tid];
                unsigned int trk_j = shared_mem_tracks[ixj];

                if (trk_i == std::numeric_limits<unsigned int>::max()) {
                    rel_i = std::numeric_limits<traccc::scalar>::max();
                    pval_i = 0.f;
                } else {
                    rel_i = rel_shared[trk_i];
                    pval_i = pvals[trk_i];
                }

                if (trk_j == std::numeric_limits<unsigned int>::max()) {
                    rel_j = std::numeric_limits<traccc::scalar>::max();
                    pval_j = 0.f;
                } else {
                    rel_j = rel_shared[trk_j];
                    pval_j = pvals[trk_j];
                }

                bool should_swap =
                    (rel_i > rel_j || (rel_i == rel_j && pval_i < pval_j)) ==
                    ascending;

                if (should_swap) {
                    shared_mem_tracks[tid]  = trk_j;
                    shared_mem_tracks[ixj]  = trk_i;
                }
            }
            __syncthreads();
        }
    }

    if (tid < n_updated) {
        updated_tracks[tid] = shared_mem_tracks[tid];
    }
}

}  // namespace traccc::cuda::kernels
