/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/utils.hpp"
#include "./kernels/add_block_offset.cuh"
#include "./kernels/apply_batch_removals.cuh"
#include "./kernels/apply_graph_removals.cuh"
#include "./kernels/batch_commit.cuh"
#include "./kernels/batch_confirm.cuh"
#include "./kernels/batch_identify_removals.cuh"
#include "./kernels/batch_prologue.cuh"
#include "./kernels/block_inclusive_scan.cuh"
#include "./kernels/build_conflict_coo.cuh"
#include "./kernels/compact_sorted_ids.cuh"
#include "./kernels/count_shared_measurements.cuh"
#include "./kernels/fill_inverted_ids.cuh"
#include "./kernels/fill_keep_flags.cuh"
#include "./kernels/fill_track_candidates.cuh"
#include "./kernels/fill_tracks_per_measurement.cuh"
#include "./kernels/fill_unique_meas_id_map.cuh"
#include "./kernels/fill_vectors.cuh"
#include "./kernels/graph_mis_init.cuh"
#include "./kernels/graph_mis_round.cuh"
#include "./kernels/rearrange_tracks.cuh"
#include "./kernels/remove_tracks.cuh"
#include "./kernels/scan_block_offsets.cuh"
#include "./kernels/sort_tracks_per_measurement.cuh"
#include "./kernels/sort_updated_tracks.cuh"
#include "./kernels/update_rel_shared.cuh"
#include "./kernels/update_status.cuh"
#include "traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/definitions/math.hpp"

// NVTX range helpers — zero overhead when no profiler is attached.
#if __has_include("nvtx3/nvToolsExt.h")
#include "nvtx3/nvToolsExt.h"
#define AR_NVTX_PUSH(name) nvtxRangePushA(name)
#define AR_NVTX_POP() nvtxRangePop()
#else
#define AR_NVTX_PUSH(name) ((void)0)
#define AR_NVTX_POP() ((void)0)
#endif

// Thrust include(s).
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
namespace {

/// Lightweight CUDA event set used for per-phase elapsed-time measurement.
/// All events are created and destroyed inside operator() when profiling is on.
struct PerfEvents {
    cudaEvent_t ev[8]{};
    bool active;

    PerfEvents(bool on, cudaStream_t s) : active(on) {
        if (!active)
            return;
        for (auto& e : ev)
            cudaEventCreate(&e);
        cudaEventRecord(ev[0], s);
    }

    ~PerfEvents() {
        if (!active)
            return;
        for (auto& e : ev)
            cudaEventDestroy(e);
    }

    void mark(int idx, cudaStream_t s) {
        if (active)
            cudaEventRecord(ev[idx], s);
    }

    float elapsed(int a, int b) const {
        float ms = 0.f;
        if (active)
            cudaEventElapsedTime(&ms, ev[a], ev[b]);
        return ms;
    }
};

}  // namespace

namespace traccc::cuda {

struct identity_op {
    template <typename T>
    TRACCC_HOST_DEVICE T operator()(T i) const {
        return i;
    }
};

// Device operator to calculate relative number of shared measurements
struct devide_op {
    TRACCC_HOST_DEVICE
    traccc::scalar operator()(unsigned int a, unsigned int b) const {
        return math::div_ieee754(static_cast<traccc::scalar>(a),
                                 static_cast<traccc::scalar>(b));
    }
};

// Track comparator to sort the track ids
struct track_comparator {
    const traccc::scalar* rel_shared;
    const traccc::scalar* pvals;

    TRACCC_HOST_DEVICE track_comparator(const traccc::scalar* rel_shared_,
                                        const traccc::scalar* pvals_)
        : rel_shared(rel_shared_), pvals(pvals_) {}

    TRACCC_HOST_DEVICE bool operator()(unsigned int a, unsigned int b) const {
        if (rel_shared[a] != rel_shared[b]) {
            return rel_shared[a] < rel_shared[b];
        }
        return pvals[a] > pvals[b];
    }
};

greedy_ambiguity_resolution_algorithm::greedy_ambiguity_resolution_algorithm(
    const config_type& cfg, const traccc::memory_resource& mr,
    vecmem::copy& copy, stream& str, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      m_config(cfg),
      m_mr(mr),
      m_copy(copy),
      m_stream(str),
      m_warp_size(details::get_warp_size(str.device())) {}

greedy_ambiguity_resolution_algorithm::output_type
greedy_ambiguity_resolution_algorithm::operator()(
    const edm::track_container<default_algebra>::const_view& tracks_view)
    const {

    const edm::measurement_collection::const_device measurements(
        tracks_view.measurements);

    auto n_meas_total = m_copy.get().get_size(tracks_view.measurements);

    // Make sure that max_measurement_id = number_of_measurement -1
    // @TODO: More robust way is to assert that measurement id ranges from 0, 1,
    // ..., number_of_measurement - 1
    [[maybe_unused]] auto max_meas_it = thrust::max_element(
        thrust::device, measurements.identifier().begin(),
        // We have to use this ugly form here, because if the measurement
        // collection is resizable (which it often is), the end() function
        // cannot be used in host code.
        measurements.identifier().begin() + n_meas_total);

    unsigned int max_meas_id;
    cudaMemcpy(&max_meas_id, thrust::raw_pointer_cast(&(*max_meas_it)),
               sizeof(unsigned int), cudaMemcpyDeviceToHost);

    if (max_meas_id != n_meas_total - 1) {
        throw std::runtime_error(
            "max measurement id should be equal to (the number of measurements "
            "- 1)");
    }

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // The Thrust policy to use.
    auto thrust_policy =
        thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(m_mr.main)))
            .on(stream);

    const unsigned int n_tracks = tracks_view.tracks.capacity();

    if (n_tracks == 0) {
        return {};
    }

    PerfEvents perf(m_profiling, stream);
    unsigned int n_graph_launches = 0u;

    // Make sure that max_shared_meas is largen than zero
    assert(m_config.max_shared_meas > 0u);

    // Status (1 = Accept, 0 = Reject) vector to count the number of acceptable
    // tracks based on the number of candidates (measurements)
    vecmem::data::vector_buffer<int> status_buffer{n_tracks, m_mr.main};

    vecmem::device_vector<int> status_device(status_buffer);
    thrust::fill(thrust_policy, status_device.begin(), status_device.end(), 1);

    // Get the sizes of the measurement index vector in each track
    const std::vector<unsigned int> candidate_sizes =
        m_copy.get().get_sizes(tracks_view.tracks);

    // Declare the buffer for meas_ids which is a jagged vector
    // Each sub-vector of meas_ids represent measurement IDs of each track
    vecmem::data::jagged_vector_buffer<measurement_id_type> meas_ids_buffer{
        candidate_sizes, m_mr.main, m_mr.host,
        vecmem::data::buffer_type::resizable};
    m_copy.get().setup(meas_ids_buffer)->ignore();

    // The sum of the number of candidates (measurements) of all tracks
    const unsigned int n_cands_total =
        std::accumulate(candidate_sizes.begin(), candidate_sizes.end(), 0u);

    // Declare flat_meas_ids which is just a flattening version of meas_ids with
    // a single vector container. It is used to count the number of unique
    // measurements
    vecmem::data::vector_buffer<measurement_id_type> flat_meas_ids_buffer{
        n_cands_total, m_mr.main, vecmem::data::buffer_type::resizable};
    m_copy.get().setup(flat_meas_ids_buffer)->ignore();
    vecmem::data::vector_buffer<traccc::scalar> pvals_buffer{n_tracks,
                                                             m_mr.main};
    vecmem::data::vector_buffer<unsigned int> n_meas_buffer{n_tracks,
                                                            m_mr.main};
    thrust::fill(thrust_policy, n_meas_buffer.ptr(),
                 n_meas_buffer.ptr() + n_tracks, 0);

    {
        const unsigned int nThreads = m_warp_size * 2;
        const unsigned int nBlocks = (n_tracks + nThreads - 1) / nThreads;

        AR_NVTX_PUSH("gpu_ar_filter_setup");
        kernels::fill_vectors<<<nBlocks, nThreads, 0, stream>>>(
            m_config, device::fill_vectors_payload{
                          .tracks_view = tracks_view,
                          .meas_ids_view = meas_ids_buffer,
                          .flat_meas_ids_view = flat_meas_ids_buffer,
                          .pvals_view = pvals_buffer,
                          .n_meas_view = n_meas_buffer,
                          .status_view = status_buffer});
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        m_stream.get().synchronize();
        AR_NVTX_POP();
        perf.mark(1, stream);
    }

    // Count the number of pre-accepted tracks
    unsigned int n_accepted = static_cast<unsigned int>(thrust::count(
        thrust_policy, status_buffer.ptr(), status_buffer.ptr() + n_tracks, 1));

    vecmem::unique_alloc_ptr<unsigned int> n_accepted_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);
    TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(n_accepted_device.get(),
                                            &n_accepted, sizeof(unsigned int),
                                            cudaMemcpyHostToDevice, stream));

    m_stream.get().synchronize();

    if (n_accepted == 0) {
        return {};
    }

    // Indices of pre-accepted tracks
    vecmem::data::vector_buffer<unsigned int> pre_accepted_ids_buffer{
        n_accepted, m_mr.main};

    m_copy.get().setup(pre_accepted_ids_buffer)->ignore();

    // Find the indices of pre-accepted tracks by checking if status is 1
    auto cit_begin = thrust::counting_iterator<int>(0);
    auto cit_end = cit_begin + n_tracks;
    thrust::copy_if(thrust_policy, cit_begin, cit_end, status_buffer.ptr(),
                    pre_accepted_ids_buffer.ptr(), identity_op{});

    AR_NVTX_PUSH("gpu_ar_unique_meas");
    // Sort the flat measurement id vector, which is required to count the
    // number of unique measurements
    thrust::sort(thrust_policy, flat_meas_ids_buffer.ptr(),
                 flat_meas_ids_buffer.ptr() + n_cands_total);

    // Count the number of unique measurements
    const unsigned int meas_count = static_cast<unsigned int>(
        thrust::unique_count(thrust_policy, flat_meas_ids_buffer.ptr(),
                             flat_meas_ids_buffer.ptr() + n_cands_total,
                             thrust::equal_to<int>()));

    // Unique measurement ids
    vecmem::data::vector_buffer<measurement_id_type> unique_meas_buffer{
        meas_count, m_mr.main};

    // Counts of unique measurement id in flat id vector.
    // This information is used to know the number of tracks associated with a
    // measurement ID.
    vecmem::data::vector_buffer<std::size_t> unique_meas_counts_buffer{
        meas_count, m_mr.main};
    m_copy.get().setup(unique_meas_counts_buffer)->ignore();

    // Counting can be done using reduce_by_key and constant iterator
    thrust::reduce_by_key(thrust_policy, flat_meas_ids_buffer.ptr(),
                          flat_meas_ids_buffer.ptr() + n_cands_total,
                          thrust::make_constant_iterator(1),
                          unique_meas_buffer.ptr(),
                          unique_meas_counts_buffer.ptr());

    // Sort unique meas ids
    thrust::sort_by_key(thrust_policy, unique_meas_buffer.ptr(),
                        unique_meas_buffer.ptr() + meas_count,
                        unique_meas_counts_buffer.ptr());

    // Unique measurement ids
    vecmem::data::vector_buffer<measurement_id_type>
        meas_id_to_unique_id_buffer{max_meas_id + 1, m_mr.main};

    // Make meas_id to unique_meas_id vector
    {
        const unsigned int nThreads = m_warp_size * 2;
        const unsigned int nBlocks = (meas_count + nThreads - 1) / nThreads;

        kernels::fill_unique_meas_id_map<<<nBlocks, nThreads, 0, stream>>>(
            device::fill_unique_meas_id_map_payload{
                .unique_meas_view = unique_meas_buffer,
                .meas_id_to_unique_id_view = meas_id_to_unique_id_buffer});
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        m_stream.get().synchronize();
        AR_NVTX_POP();
        perf.mark(2, stream);
        m_last_profile.unique_meas_count = meas_count;
    }

    // Retreive the counting vector to host for the size allocation of
    // tracks_per_measurement
    std::vector<std::size_t> unique_meas_counts;
    m_copy
        .get()(unique_meas_counts_buffer, unique_meas_counts,
               vecmem::copy::type::device_to_host)
        ->wait();

    // Make the tracks_per_measurement vector
    // Each sub vector contains track ids associated with the unique measurement
    vecmem::data::jagged_vector_buffer<unsigned int>
        tracks_per_measurement_buffer(unique_meas_counts, m_mr.main, m_mr.host,
                                      vecmem::data::buffer_type::resizable);
    m_copy.get().setup(tracks_per_measurement_buffer)->ignore();

    // Make the track_status_per_measurement vector
    // Each sub vector contains whether the track ids is still associated with
    // the unique measurements For example, the value turns into 0 (false) if
    // the track is rejected during the ambiguity solver
    vecmem::data::jagged_vector_buffer<int> track_status_per_measurement_buffer(
        unique_meas_counts, m_mr.main, m_mr.host,
        vecmem::data::buffer_type::resizable);

    m_copy.get().setup(track_status_per_measurement_buffer)->ignore();

    // Make the number of accetped_tracks_per_measurement vector
    // Each element represents the number of associated tracks with the unique
    // measurement (the number of track_status whose value is 1 (true))
    vecmem::data::vector_buffer<unsigned int>
        n_accepted_tracks_per_measurement_buffer(meas_count, m_mr.main);
    thrust::fill(thrust_policy, n_accepted_tracks_per_measurement_buffer.ptr(),
                 n_accepted_tracks_per_measurement_buffer.ptr() + meas_count,
                 0);

    AR_NVTX_PUSH("gpu_ar_inverted_index");
    // Fill tracks_per_measurement, track_status_per_measurement and
    // n_accepted_tracks_per_measurement vectors
    {
        const unsigned int nThreads = m_warp_size * 2;
        const unsigned int nBlocks = (n_accepted + nThreads - 1) / nThreads;

        kernels::fill_tracks_per_measurement<<<nBlocks, nThreads, 0, stream>>>(
            device::fill_tracks_per_measurement_payload{
                .accepted_ids_view = pre_accepted_ids_buffer,
                .meas_ids_view = meas_ids_buffer,
                .meas_id_to_unique_id_view = meas_id_to_unique_id_buffer,
                .tracks_per_measurement_view = tracks_per_measurement_buffer,
                .track_status_per_measurement_view =
                    track_status_per_measurement_buffer,
                .n_accepted_tracks_per_measurement_view =
                    n_accepted_tracks_per_measurement_buffer});
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        m_stream.get().synchronize();
    }

    // Sort tracks per measurement vector
    // @TODO: For the case where the measurement is shared by more than 1024
    // tracks, the tracks need to be sorted again using thrust::sort
    {
        const unsigned int nThreads = 1024;
        const unsigned int nBlocks = meas_count;

        kernels::sort_tracks_per_measurement<<<nBlocks, nThreads, 0, stream>>>(
            device::sort_tracks_per_measurement_payload{
                .tracks_per_measurement_view = tracks_per_measurement_buffer,
            });
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        m_stream.get().synchronize();
    }
    AR_NVTX_POP();
    perf.mark(3, stream);

    // Make vector buffer for the number of shared measurements for each track
    vecmem::data::vector_buffer<unsigned int> n_shared_buffer{n_tracks,
                                                              m_mr.main};
    thrust::fill(thrust_policy, n_shared_buffer.ptr(),
                 n_shared_buffer.ptr() + n_tracks, 0);
    m_copy.get().setup(n_shared_buffer)->ignore();

    AR_NVTX_PUSH("gpu_ar_shared_count");
    // Count the number of shared measurements
    {
        const unsigned int nThreads = m_warp_size * 2;
        const unsigned int nBlocks = (n_accepted + nThreads - 1) / nThreads;

        kernels::count_shared_measurements<<<nBlocks, nThreads, 0, stream>>>(
            device::count_shared_measurements_payload{
                .accepted_ids_view = pre_accepted_ids_buffer,
                .meas_ids_view = meas_ids_buffer,
                .meas_id_to_unique_id_view = meas_id_to_unique_id_buffer,
                .n_accepted_tracks_per_measurement_view =
                    n_accepted_tracks_per_measurement_buffer,
                .n_shared_view = n_shared_buffer});
        TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

        m_stream.get().synchronize();
    }
    AR_NVTX_POP();
    perf.mark(4, stream);

    // Make relative number of shared measurements vector
    // The relative number of shared measurement is defined as the number of
    // shared measurement divided the number of measurements of the track
    vecmem::data::vector_buffer<traccc::scalar> rel_shared_buffer{n_tracks,
                                                                  m_mr.main};

    AR_NVTX_PUSH("gpu_ar_initial_sort");
    // Fill the relative shared number of measurements vector
    thrust::transform(thrust_policy, n_shared_buffer.ptr(),
                      n_shared_buffer.ptr() + n_tracks, n_meas_buffer.ptr(),
                      rel_shared_buffer.ptr(), devide_op{});

    // Make a buffer for track ids sorted based on the relative number of shared
    // measurements and pvalues
    vecmem::data::vector_buffer<unsigned int> sorted_ids_buffer{n_accepted,
                                                                m_mr.main};
    m_copy.get().setup(sorted_ids_buffer)->ignore();

    // Make a temporary buffer for sorted track ids
    vecmem::data::vector_buffer<unsigned int> temp_sorted_ids_buffer{n_accepted,
                                                                     m_mr.main};
    m_copy.get().setup(temp_sorted_ids_buffer)->ignore();

    // track id to the index of sorted ids
    vecmem::data::vector_buffer<unsigned int> inverted_ids_buffer{n_tracks,
                                                                  m_mr.main};
    m_copy.get().setup(inverted_ids_buffer)->ignore();

    // Make a buffer of boolean elements (Whether a corresponding track id is
    // updated after an iteration)
    vecmem::data::vector_buffer<int> is_updated_buffer{n_tracks, m_mr.main};
    m_copy.get().setup(is_updated_buffer)->ignore();
    m_copy.get().memset(is_updated_buffer, 0)->ignore();

    // Count track id apperance during removal process
    vecmem::data::vector_buffer<int> track_count_buffer{n_tracks, m_mr.main};
    m_copy.get().setup(track_count_buffer)->ignore();
    m_copy.get().memset(track_count_buffer, 0)->ignore();

    // Prefix sum buffer used for the insertion sort during an iteration
    vecmem::data::vector_buffer<int> prefix_sums_buffer{n_tracks, m_mr.main};
    m_copy.get().setup(prefix_sums_buffer)->ignore();

    // Fill the sorted ids vector
    thrust::copy(thrust_policy, pre_accepted_ids_buffer.ptr(),
                 pre_accepted_ids_buffer.ptr() + n_accepted,
                 sorted_ids_buffer.ptr());
    m_stream.get().synchronize();

    track_comparator trk_comp(rel_shared_buffer.ptr(), pvals_buffer.ptr());

    // Sort the sorted ids vector based on the relative number of shared
    // measurements and pvalues
    thrust::sort(thrust_policy, sorted_ids_buffer.ptr(),
                 sorted_ids_buffer.ptr() + n_accepted, trk_comp);

    if (m_profiling)
        m_stream.get().synchronize();
    AR_NVTX_POP();
    perf.mark(5, stream);

    // Make a buffer of track ids whose number of shared measurements are
    // updated during an iteration
    vecmem::data::vector_buffer<unsigned int> updated_tracks_buffer{n_accepted,
                                                                    m_mr.main};
    m_copy.get().setup(updated_tracks_buffer)->ignore();

    // Device objects
    vecmem::unique_alloc_ptr<unsigned int> n_removable_tracks_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);
    vecmem::unique_alloc_ptr<unsigned int> n_meas_to_remove_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);
    vecmem::unique_alloc_ptr<unsigned int> n_valid_threads_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);

    // Whether to terminate the iteration process
    int terminate = 0;
    vecmem::unique_alloc_ptr<int> terminate_device =
        vecmem::make_unique_alloc<int>(m_mr.main);
    cudaMemsetAsync(terminate_device.get(), 0, sizeof(int), stream);
    auto max_shared = thrust::max_element(thrust::device, n_shared_buffer.ptr(),
                                          n_shared_buffer.ptr() + n_tracks);

    // The maximum number of shared measurements. The process is terminated if
    // this value is zero
    vecmem::unique_alloc_ptr<unsigned int> max_shared_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);
    cudaMemcpyAsync(max_shared_device.get(), max_shared, sizeof(unsigned int),
                    cudaMemcpyDeviceToDevice, stream);

    // The number of tracks whose number of share measurements is updated
    vecmem::unique_alloc_ptr<unsigned int> n_updated_tracks_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);

    // Thread block size
    unsigned int nThreads_adaptive = m_warp_size;
    unsigned int nBlocks_adaptive =
        (n_accepted + nThreads_adaptive - 1) / nThreads_adaptive;

    unsigned int nThreads_rearrange = 1024;
    unsigned int nBlocks_rearrange =
        (n_accepted + (nThreads_rearrange / kernels::nThreads_per_track) - 1) /
        (nThreads_rearrange / kernels::nThreads_per_track);

    // Compute the threadblock dimension for scanning kernels
    auto compute_scan_config = [&](unsigned int n_accepted) {
        unsigned int nThreads_scan = m_warp_size * 4;
        unsigned int nBlocks_scan =
            (n_accepted + nThreads_scan - 1) / nThreads_scan;

        while (nThreads_scan <= 1024) {
            if (nBlocks_scan > 1024) {
                nThreads_scan *= 2;
                nBlocks_scan = (n_accepted + nThreads_scan - 1) / nThreads_scan;
            } else {
                break;
            }
        }

        return std::make_pair(nThreads_scan, nBlocks_scan);
    };

    auto scan_dim = compute_scan_config(n_accepted);
    unsigned int nThreads_scan = scan_dim.first;
    unsigned int nBlocks_scan = scan_dim.second;

    assert(nBlocks_scan <= 1024 &&
           "nBlocks_scan larger than 1024 will cause invalid arguments in "
           "scan_block_offsets kernel");

    // Make buffers used in prefix sum calculation
    vecmem::data::vector_buffer<int> block_offsets_buffer{nBlocks_scan,
                                                          m_mr.main};
    m_copy.get().setup(block_offsets_buffer)->ignore();
    vecmem::data::vector_buffer<int> scanned_block_offsets_buffer{nBlocks_scan,
                                                                  m_mr.main};
    m_copy.get().setup(scanned_block_offsets_buffer)->ignore();

    // --- Parallel batch greedy (Tier 2a) buffers. Allocated unconditionally
    // so sizes are known up front; only populated when m_parallel_batch is
    // true. Total extra allocation is ~meas_count*sizeof(unsigned) bytes.
    vecmem::data::vector_buffer<unsigned int> claimed_by_buffer{meas_count,
                                                                m_mr.main};
    m_copy.get().setup(claimed_by_buffer)->ignore();
    vecmem::data::vector_buffer<unsigned int> batch_ids_buffer{n_tracks,
                                                               m_mr.main};
    m_copy.get().setup(batch_ids_buffer)->ignore();
    vecmem::unique_alloc_ptr<unsigned int> batch_size_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);

    // Snapshot of n_accepted taken by batch_prologue at the start of every
    // graph replay. PBG identify and apply read this snapshot rather than the
    // live n_accepted, eliminating an intra-grid race that otherwise causes
    // distinct blocks to disagree on the candidate window.
    vecmem::unique_alloc_ptr<unsigned int> n_acc_snapshot_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);

    // First-failure rank for the conflict-free prefix admission rule. Reset
    // by batch_prologue, lowered by batch_confirm via atomicMin, read by
    // apply_batch_removals to admit ranks [0, *first_fail).
    vecmem::unique_alloc_ptr<unsigned int> first_fail_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);

    // Per-outer-iteration batch size log (host-side buffer, populated after
    // each graph_replay burst).
    std::vector<unsigned int> batch_sizes_this_call;
    batch_sizes_this_call.reserve(256);

    // --- Tier 2c (explicit conflict graph) buffers and logs. Allocated only
    // when a graph algorithm is selected; otherwise the pointers stay unused.
    // Memory cost is O(n_tracks + max_edges) on top of the baseline.
    vecmem::data::vector_buffer<int> is_removed_buffer{n_tracks, m_mr.main};
    vecmem::data::vector_buffer<int> keep_flag_per_slot_buffer{n_tracks,
                                                               m_mr.main};
    vecmem::data::vector_buffer<int> keep_prefix_sums_buffer{n_tracks,
                                                             m_mr.main};
    vecmem::data::vector_buffer<unsigned int> mis_priority_buffer{n_tracks,
                                                                  m_mr.main};
    vecmem::data::vector_buffer<int> mis_active_buffer{n_tracks, m_mr.main};
    vecmem::data::vector_buffer<int> mis_state_buffer{n_tracks, m_mr.main};
    m_copy.get().setup(is_removed_buffer)->ignore();
    m_copy.get().setup(keep_flag_per_slot_buffer)->ignore();
    m_copy.get().setup(keep_prefix_sums_buffer)->ignore();
    m_copy.get().setup(mis_priority_buffer)->ignore();
    m_copy.get().setup(mis_active_buffer)->ignore();
    m_copy.get().setup(mis_state_buffer)->ignore();
    m_copy.get().memset(is_removed_buffer, 0)->ignore();

    // Upper bound on directed edges: two per accepted-track pair per unique
    // measurement. Tracks sharing multiple measurements contribute duplicate
    // edges; the MIS / coloring kernels tolerate them (each duplicate just
    // means one more neighbour-loop iteration). The bound is a one-time
    // overestimate; no incremental growth during the outer loop.
    std::size_t max_edges_ub = 0u;
    if (m_graph_algo != graph_algo_t::NONE) {
        for (auto n_m : unique_meas_counts) {
            if (n_m > 1) {
                max_edges_ub += static_cast<std::size_t>(n_m) *
                                static_cast<std::size_t>(n_m - 1);
            }
        }
        if (max_edges_ub == 0u) {
            max_edges_ub = 1u;  // avoid zero-size allocation
        }
    } else {
        max_edges_ub = 1u;
    }

    vecmem::data::vector_buffer<unsigned int> coo_src_buffer{
        static_cast<unsigned int>(max_edges_ub), m_mr.main};
    vecmem::data::vector_buffer<unsigned int> coo_dst_buffer{
        static_cast<unsigned int>(max_edges_ub), m_mr.main};
    vecmem::data::vector_buffer<unsigned int> row_ptr_buffer{n_tracks + 1u,
                                                             m_mr.main};
    m_copy.get().setup(coo_src_buffer)->ignore();
    m_copy.get().setup(coo_dst_buffer)->ignore();
    m_copy.get().setup(row_ptr_buffer)->ignore();

    vecmem::unique_alloc_ptr<unsigned int> edge_count_device =
        vecmem::make_unique_alloc<unsigned int>(m_mr.main);
    vecmem::unique_alloc_ptr<int> any_undecided_device =
        vecmem::make_unique_alloc<int>(m_mr.main);

    std::vector<unsigned int> graph_batches_this_call;
    std::vector<std::pair<unsigned int, unsigned int>> graph_sizes_this_call;
    if (m_graph_algo != graph_algo_t::NONE) {
        graph_batches_this_call.reserve(256);
        graph_sizes_this_call.reserve(256);
    }

    AR_NVTX_PUSH("gpu_ar_eviction_loop");
    // Start the iteration
    while (!terminate && n_accepted > 0) {
        nBlocks_adaptive =
            (n_accepted + nThreads_adaptive - 1) / nThreads_adaptive;

        scan_dim = compute_scan_config(n_accepted);
        nThreads_scan = scan_dim.first;
        nBlocks_scan = scan_dim.second;
        nBlocks_rearrange =
            (n_accepted + (nThreads_rearrange / kernels::nThreads_per_track) -
             1) /
            (nThreads_rearrange / kernels::nThreads_per_track);

        if (m_graph_algo != graph_algo_t::NONE) {
            // ----------------------------------------------------------------
            // Tier 2c: explicit conflict-graph removal path.
            //
            // Per-iteration structure:
            //   1. Zero per-iteration scratch counters.
            //   2. Build the undirected conflict graph over the currently
            //      accepted & still-shared tracks (COO, then sort to CSR).
            //   3. Initialise MIS/JP state from inverted_ids (priority) and
            //      n_shared (active mask).
            //   4. Either (LUBY_MIS) run Luby rounds until no UNDECIDED
            //      vertices remain, or (JP_COLOR) run a single round so the
            //      selected set is the round-1 local-max IS.
            //   5. apply_graph_removals propagates measurement bookkeeping
            //      and raises is_removed[t] for each IN_MIS track.
            //   6. Stage 1 compaction rewrites sorted_ids with the removed
            //      tracks dropped, then the standard rearrange / update_status
            //      path repositions survivors whose rel_shared has changed.
            //
            // Kernel launches are direct (no cudaGraph capture) because the
            // thrust calls for sort_by_key and upper_bound are stream-based
            // and the per-iteration structure changes between outer steps.
            // See docs/analysis/novelty_algs/conflict_graph_design.md.
            const unsigned int nt_vtx = m_warp_size * 2;
            const unsigned int nb_vtx = (n_tracks + nt_vtx - 1) / nt_vtx;

            // Clear per-iteration scratch. These fan-in through direct launches
            // rather than a captured graph, so the zero is just a memset.
            cudaMemsetAsync(batch_size_device.get(), 0, sizeof(unsigned int),
                            stream);
            cudaMemsetAsync(n_updated_tracks_device.get(), 0,
                            sizeof(unsigned int), stream);
            cudaMemsetAsync(edge_count_device.get(), 0, sizeof(unsigned int),
                            stream);

            // Refresh inverted_ids so it reflects the current sorted_ids
            // layout. The baseline fill_inverted_ids kernel early-returns
            // when n_updated_tracks == 0 (because in the baseline/PBG
            // pipeline it is only used to patch positions for in-place
            // rearrangement); in graph mode we need a full rebuild every
            // outer iteration, so we scatter via thrust directly.
            {
                auto ci_begin = thrust::counting_iterator<unsigned int>(0u);
                thrust::scatter(thrust_policy, ci_begin, ci_begin + n_accepted,
                                sorted_ids_buffer.ptr(),
                                inverted_ids_buffer.ptr());
            }

            // Build COO edge list. One block per unique measurement.
            {
                const unsigned int nt_coo = 128u;
                const std::size_t smem_bytes =
                    static_cast<std::size_t>(nt_coo) * sizeof(unsigned int);
                kernels::build_conflict_coo<<<meas_count, nt_coo, smem_bytes,
                                              stream>>>(
                    device::build_conflict_coo_payload{
                        .tracks_per_measurement_view =
                            tracks_per_measurement_buffer,
                        .track_status_per_measurement_view =
                            track_status_per_measurement_buffer,
                        .n_accepted_tracks_per_measurement_view =
                            n_accepted_tracks_per_measurement_buffer,
                        .coo_src_view = coo_src_buffer,
                        .coo_dst_view = coo_dst_buffer,
                        .edge_count = edge_count_device.get(),
                        .meas_count = meas_count});
            }

            // Read edge count to host. Only rank-0 thrust calls need it, but
            // getting it once lets us bound the sort size.
            unsigned int n_edges_host = 0u;
            cudaMemcpyAsync(&n_edges_host, edge_count_device.get(),
                            sizeof(unsigned int), cudaMemcpyDeviceToHost,
                            stream);
            m_stream.get().synchronize();

            if (n_edges_host > 0u) {
                // Sort COO pairs by src. Co-sort dst so that col_idx lines up.
                thrust::sort_by_key(thrust_policy, coo_src_buffer.ptr(),
                                    coo_src_buffer.ptr() + n_edges_host,
                                    coo_dst_buffer.ptr());

                // Build row_ptr[0..n_tracks]. row_ptr[v] = number of edges
                // with src < v (== lower_bound position of v in sorted src).
                auto ci_begin = thrust::counting_iterator<unsigned int>(0u);
                thrust::lower_bound(thrust_policy, coo_src_buffer.ptr(),
                                    coo_src_buffer.ptr() + n_edges_host,
                                    ci_begin, ci_begin + (n_tracks + 1u),
                                    row_ptr_buffer.ptr());
            } else {
                // No edges: row_ptr is all zeros. Still safe to launch MIS.
                cudaMemsetAsync(row_ptr_buffer.ptr(), 0,
                                sizeof(unsigned int) * (n_tracks + 1u), stream);
            }

            // Initialise MIS state from the current iteration.
            kernels::graph_mis_init<<<nb_vtx, nt_vtx, 0, stream>>>(
                device::graph_mis_init_payload{
                    .inverted_ids_view = inverted_ids_buffer,
                    .n_shared_view = n_shared_buffer,
                    .n_accepted = n_accepted_device.get(),
                    .priority_view = mis_priority_buffer,
                    .mis_active_view = mis_active_buffer,
                    .mis_state_view = mis_state_buffer,
                    .n_vertices = n_tracks});

            // MIS / JP rounds.
            {
                device::graph_mis_round_payload round_payload{
                    .mis_active_view = mis_active_buffer,
                    .mis_state_view = mis_state_buffer,
                    .priority_view = mis_priority_buffer,
                    .row_ptr_view = row_ptr_buffer,
                    .col_idx_view = coo_dst_buffer,
                    .n_vertices = n_tracks,
                    .any_undecided = any_undecided_device.get()};

                const unsigned int max_rounds =
                    (m_graph_algo == graph_algo_t::JP_COLOR) ? 1u : 32u;
                for (unsigned int r = 0u; r < max_rounds; ++r) {
                    cudaMemsetAsync(any_undecided_device.get(), 0, sizeof(int),
                                    stream);
                    kernels::graph_mis_propose<<<nb_vtx, nt_vtx, 0, stream>>>(
                        round_payload);
                    kernels::graph_mis_finalize<<<nb_vtx, nt_vtx, 0, stream>>>(
                        round_payload);

                    // Host-side convergence check. Synchronise so that the
                    // flag reflects this round's result.
                    int undecided_host = 0;
                    cudaMemcpyAsync(&undecided_host, any_undecided_device.get(),
                                    sizeof(int), cudaMemcpyDeviceToHost,
                                    stream);
                    m_stream.get().synchronize();
                    if (undecided_host == 0) {
                        break;
                    }
                }
            }

            // Apply removals: flip per-measurement bookkeeping, raise
            // is_removed / is_updated.
            kernels::apply_graph_removals<<<nb_vtx, nt_vtx, 0, stream>>>(
                device::apply_graph_removals_payload{
                    .meas_ids_view = meas_ids_buffer,
                    .n_meas_view = n_meas_buffer,
                    .meas_id_to_unique_id_view = meas_id_to_unique_id_buffer,
                    .tracks_per_measurement_view =
                        tracks_per_measurement_buffer,
                    .track_status_per_measurement_view =
                        track_status_per_measurement_buffer,
                    .n_accepted_tracks_per_measurement_view =
                        n_accepted_tracks_per_measurement_buffer,
                    .n_shared_view = n_shared_buffer,
                    .mis_state_view = mis_state_buffer,
                    .is_removed_view = is_removed_buffer,
                    .is_updated_view = is_updated_buffer,
                    .updated_tracks_view = updated_tracks_buffer,
                    .batch_size = batch_size_device.get(),
                    .n_updated_tracks = n_updated_tracks_device.get(),
                    .n_vertices = n_tracks});

            // Recompute rel_shared for survivors whose n_shared changed.
            kernels::update_rel_shared<<<(n_tracks + (m_warp_size * 2) - 1) /
                                             (m_warp_size * 2),
                                         m_warp_size * 2, 0, stream>>>(
                device::update_rel_shared_payload{
                    .terminate = terminate_device.get(),
                    .n_updated_tracks = n_updated_tracks_device.get(),
                    .updated_tracks_view = updated_tracks_buffer,
                    .n_shared_view = n_shared_buffer,
                    .n_meas_view = n_meas_buffer,
                    .rel_shared_view = rel_shared_buffer});

            // --- Stage 1 compaction: rewrite sorted_ids dropping removed
            //     tracks. keep_flag_per_slot[i] = !is_removed[sorted_ids[i]];
            //     the inclusive prefix sum gives each survivor its new slot,
            //     and compact_sorted_ids scatters into temp_sorted_ids.
            kernels::fill_keep_flags<<<nBlocks_adaptive, nThreads_adaptive, 0,
                                       stream>>>(
                device::fill_keep_flags_payload{
                    .sorted_ids_view = sorted_ids_buffer,
                    .n_accepted = n_accepted_device.get(),
                    .terminate = terminate_device.get(),
                    .is_removed_view = is_removed_buffer,
                    .keep_flag_per_slot_view = keep_flag_per_slot_buffer});

            thrust::inclusive_scan(thrust_policy,
                                   keep_flag_per_slot_buffer.ptr(),
                                   keep_flag_per_slot_buffer.ptr() + n_accepted,
                                   keep_prefix_sums_buffer.ptr());

            kernels::compact_sorted_ids<<<nBlocks_adaptive, nThreads_adaptive,
                                          0, stream>>>(
                device::compact_sorted_ids_payload{
                    .sorted_ids_view = sorted_ids_buffer,
                    .keep_flag_per_slot_view = keep_flag_per_slot_buffer,
                    .keep_prefix_sums_view = keep_prefix_sums_buffer,
                    .compacted_sorted_ids_view = temp_sorted_ids_buffer,
                    .n_accepted = n_accepted_device.get(),
                    .terminate = terminate_device.get()});

            // Copy compacted sorted_ids back. We copy at most the old count —
            // the tail positions past the new n_accepted are left untouched
            // but never read, since all downstream kernels gate on n_accepted.
            cudaMemcpyAsync(sorted_ids_buffer.ptr(),
                            temp_sorted_ids_buffer.ptr(),
                            sizeof(unsigned int) * n_accepted,
                            cudaMemcpyDeviceToDevice, stream);

            // Refresh n_accepted on host early — later launches depend on it
            // via the device-side value, so we only need it for the outer
            // loop condition and the nBlocks computation next iteration.
            m_stream.get().synchronize();
            cudaMemcpyAsync(&n_accepted, n_accepted_device.get(),
                            sizeof(unsigned int), cudaMemcpyDeviceToHost,
                            stream);
            m_stream.get().synchronize();

            // After a graph-algorithm batch the number of simultaneously
            // updated survivors can vastly exceed the assumptions baked into
            // the baseline insertion-sort pipeline (which caps the bitonic
            // sort_updated_tracks at 512 entries and relies on a
            // contiguous-tail removal shape). Instead of shoehorning those
            // kernels into graph mode, we globally re-sort sorted_ids with
            // the same comparator used at init. This costs O(n log n) per
            // outer iteration but is always correct.
            if (n_accepted > 0u) {
                thrust::sort(thrust_policy, sorted_ids_buffer.ptr(),
                             sorted_ids_buffer.ptr() + n_accepted, trk_comp);

                // Clear is_updated flags for all tracks that got flagged
                // during apply_graph_removals, so the next outer iteration
                // sees a fresh slate. We reuse updated_tracks_buffer (which
                // still contains the list of survivors whose n_shared moved
                // during apply_graph_removals) as the reset target.
                {
                    const unsigned int nt = nThreads_adaptive;
                    const unsigned int nb_upd = (n_tracks + nt - 1) / nt;
                    (void)nb_upd;
                }
                cudaMemsetAsync(is_updated_buffer.ptr(), 0,
                                sizeof(int) * n_tracks, stream);

                // Reset max_shared and recompute it from n_shared over the
                // surviving tracks. Baseline remove_tracks does this inside
                // a kernel; with thrust::sort we use thrust::max_element on
                // the n_shared values of still-accepted tracks. This is
                // slightly wasteful (iterates all n_tracks) but trivially
                // correct; the runtime is dominated by the sort above.
                cudaMemsetAsync(max_shared_device.get(), 0,
                                sizeof(unsigned int), stream);

                auto max_it =
                    thrust::max_element(thrust_policy, n_shared_buffer.ptr(),
                                        n_shared_buffer.ptr() + n_tracks);
                cudaMemcpyAsync(max_shared_device.get(), max_it,
                                sizeof(unsigned int), cudaMemcpyDeviceToDevice,
                                stream);
            }

            // Pull back iteration summary. Terminate when no track was
            // removed this iteration (MIS was empty / graph had no edges).
            unsigned int batch_host = 0u;
            cudaMemcpyAsync(&batch_host, batch_size_device.get(),
                            sizeof(unsigned int), cudaMemcpyDeviceToHost,
                            stream);
            unsigned int max_shared_host = 0u;
            cudaMemcpyAsync(&max_shared_host, max_shared_device.get(),
                            sizeof(unsigned int), cudaMemcpyDeviceToHost,
                            stream);
            m_stream.get().synchronize();

            if (m_graph_batch_log != nullptr) {
                graph_batches_this_call.push_back(batch_host);
            }
            if (m_graph_size_log != nullptr) {
                graph_sizes_this_call.emplace_back(n_accepted, n_edges_host);
            }

            if (batch_host == 0u || max_shared_host == 0u) {
                terminate = 1;
                int one = 1;
                cudaMemcpyAsync(terminate_device.get(), &one, sizeof(int),
                                cudaMemcpyHostToDevice, stream);
                m_stream.get().synchronize();
            }

            continue;
        }

        // Make a CUDA Graph. We use CUDA graph to minimize the overheads from
        // kernel launches
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        if (!m_parallel_batch) {
            // --- Baseline path: single-block batched removal with prefix
            // conflict-free rule. Unchanged from upstream.
            kernels::remove_tracks<<<1, 512, 0, stream>>>(
                device::remove_tracks_payload{
                    .sorted_ids_view = sorted_ids_buffer,
                    .n_accepted = n_accepted_device.get(),
                    .meas_ids_view = meas_ids_buffer,
                    .n_meas_view = n_meas_buffer,
                    .meas_id_to_unique_id_view = meas_id_to_unique_id_buffer,
                    .tracks_per_measurement_view =
                        tracks_per_measurement_buffer,
                    .track_status_per_measurement_view =
                        track_status_per_measurement_buffer,
                    .n_accepted_tracks_per_measurement_view =
                        n_accepted_tracks_per_measurement_buffer,
                    .n_shared_view = n_shared_buffer,
                    .rel_shared_view = rel_shared_buffer,
                    .n_removable_tracks = n_removable_tracks_device.get(),
                    .n_meas_to_remove = n_meas_to_remove_device.get(),
                    .terminate = terminate_device.get(),
                    .max_shared = max_shared_device.get(),
                    .n_updated_tracks = n_updated_tracks_device.get(),
                    .updated_tracks_view = updated_tracks_buffer,
                    .is_updated_view = is_updated_buffer,
                    .n_valid_threads = n_valid_threads_device.get(),
                    .track_count_view = track_count_buffer});
        } else {
            // --- Tier 2a: parallel batch greedy removal. Priority-CAS claim
            // on contested measurements, confirm + apply, recompute rel_shared
            // on the survivors. See docs/analysis/parallel_batch_greedy_design
            // .md Sec. 4.
            const unsigned int window =
                std::min(n_accepted, m_parallel_batch_window);

            // Reset claimed_by to UINT_MAX (0xff bytes) for this outer step.
            cudaMemsetAsync(claimed_by_buffer.ptr(), 0xff,
                            sizeof(unsigned int) * meas_count, stream);

            kernels::batch_prologue<<<1, 1, 0, stream>>>(
                device::batch_prologue_payload{
                    .terminate = terminate_device.get(),
                    .max_shared = max_shared_device.get(),
                    .n_updated_tracks = n_updated_tracks_device.get(),
                    .batch_size = batch_size_device.get(),
                    .n_accepted = n_accepted_device.get(),
                    .n_acc_snapshot = n_acc_snapshot_device.get(),
                    .first_fail = first_fail_device.get(),
                    .candidate_window_size = window});

            const unsigned int identify_threads = 256u;
            const unsigned int identify_blocks =
                (window + identify_threads - 1u) / identify_threads;
            kernels::batch_identify_removals<<<identify_blocks,
                                               identify_threads, 0, stream>>>(
                device::batch_identify_removals_payload{
                    .sorted_ids_view = sorted_ids_buffer,
                    .n_accepted = n_acc_snapshot_device.get(),
                    .meas_ids_view = meas_ids_buffer,
                    .meas_id_to_unique_id_view = meas_id_to_unique_id_buffer,
                    .n_accepted_tracks_per_measurement_view =
                        n_accepted_tracks_per_measurement_buffer,
                    .n_shared_view = n_shared_buffer,
                    .claimed_by_view = claimed_by_buffer,
                    .terminate = terminate_device.get(),
                    .candidate_window_size = window});

            // Confirm + first-fail computation. One thread per candidate.
            // Output is *first_fail = min rank that failed the conflict-free
            // prefix check (initialised to window by batch_prologue). apply
            // admits ranks [0, *first_fail) — never a non-contiguous set.
            const unsigned int confirm_threads = 256u;
            const unsigned int confirm_blocks =
                (window + confirm_threads - 1u) / confirm_threads;
            kernels::
                batch_confirm<<<confirm_blocks, confirm_threads, 0, stream>>>(
                    device::batch_confirm_payload{
                        .sorted_ids_view = sorted_ids_buffer,
                        .n_accepted = n_acc_snapshot_device.get(),
                        .meas_ids_view = meas_ids_buffer,
                        .meas_id_to_unique_id_view =
                            meas_id_to_unique_id_buffer,
                        .n_accepted_tracks_per_measurement_view =
                            n_accepted_tracks_per_measurement_buffer,
                        .n_shared_view = n_shared_buffer,
                        .claimed_by_view = claimed_by_buffer,
                        .terminate = terminate_device.get(),
                        .candidate_window_size = window,
                        .first_fail = first_fail_device.get()});

            const unsigned int apply_threads = 128u;
            const unsigned int apply_blocks =
                (window + apply_threads - 1u) / apply_threads;
            kernels::apply_batch_removals<<<apply_blocks, apply_threads, 0,
                                            stream>>>(
                device::apply_batch_removals_payload{
                    .sorted_ids_view = sorted_ids_buffer,
                    .n_accepted = n_acc_snapshot_device.get(),
                    .meas_ids_view = meas_ids_buffer,
                    .n_meas_view = n_meas_buffer,
                    .meas_id_to_unique_id_view = meas_id_to_unique_id_buffer,
                    .tracks_per_measurement_view =
                        tracks_per_measurement_buffer,
                    .track_status_per_measurement_view =
                        track_status_per_measurement_buffer,
                    .n_accepted_tracks_per_measurement_view =
                        n_accepted_tracks_per_measurement_buffer,
                    .n_shared_view = n_shared_buffer,
                    .rel_shared_view = rel_shared_buffer,
                    .claimed_by_view = claimed_by_buffer,
                    .batch_ids_view = batch_ids_buffer,
                    .batch_size = batch_size_device.get(),
                    .terminate = terminate_device.get(),
                    .candidate_window_size = window,
                    .n_updated_tracks = n_updated_tracks_device.get(),
                    .updated_tracks_view = updated_tracks_buffer,
                    .is_updated_view = is_updated_buffer,
                    .track_count_view = track_count_buffer,
                    .first_fail = first_fail_device.get()});

            // Commit the per-iteration batch by subtracting batch_size from
            // the live n_accepted. Must run after apply (so batch_size is
            // final) and before any downstream kernel that reads n_accepted
            // (sort_updated_tracks, fill_inverted_ids, update_status, ...).
            kernels::batch_commit<<<1, 1, 0, stream>>>(
                device::batch_commit_payload{
                    .terminate = terminate_device.get(),
                    .n_accepted = n_accepted_device.get(),
                    .batch_size = batch_size_device.get()});

            kernels::update_rel_shared<<<(n_tracks + (m_warp_size * 2) - 1) /
                                             (m_warp_size * 2),
                                         m_warp_size * 2, 0, stream>>>(
                device::update_rel_shared_payload{
                    .terminate = terminate_device.get(),
                    .n_updated_tracks = n_updated_tracks_device.get(),
                    .updated_tracks_view = updated_tracks_buffer,
                    .n_shared_view = n_shared_buffer,
                    .n_meas_view = n_meas_buffer,
                    .rel_shared_view = rel_shared_buffer});
        }

        // After the kernel "remove_tracks", sorted_ids_view is not sorted
        // anymore as the number of measurements of a few tracks might change.
        // We can consider using thrust::sort() like the following:
        /*
        cudaMemcpyAsync(&n_accepted, n_accepted_device.get(),
                        sizeof(unsigned int), cudaMemcpyDeviceToHost,
                        stream);
        thrust::sort(thrust_policy, sorted_ids_buffer.ptr(),
                     sorted_ids_buffer.ptr() + n_accepted,
                     trk_comp);
        */
        // However, thrust::sort (Radix sort) is not optimized for our case
        // where we only need to rearrange the indices of a few tracks whose
        // number of measurement changed. In such case, insertion sort would be
        // a good choice and the following seven kernels are collaborating each
        // other to do insertion sort

        // This kernel sort the tracks whose number of measurement changed
        // during remove_track kernel. The number of such tracks are very small
        // and we apply bitoncic sort here. The purpose is to make each of
        // updated tracks not interfere each other when we rearrange them during
        // the insertion sort
        kernels::sort_updated_tracks<<<1, 512, 0, stream>>>(
            device::sort_updated_tracks_payload{
                .rel_shared_view = rel_shared_buffer,
                .pvals_view = pvals_buffer,
                .terminate = terminate_device.get(),
                .n_updated_tracks = n_updated_tracks_device.get(),
                .updated_tracks_view = updated_tracks_buffer,
            });

        // Fill the inverted_ids vector which converts a track id to the index
        // of sorted ids, which is for the fast lookup
        kernels::fill_inverted_ids<<<nBlocks_adaptive, nThreads_adaptive, 0,
                                     stream>>>(
            device::fill_inverted_ids_payload{
                .sorted_ids_view = sorted_ids_buffer,
                .terminate = terminate_device.get(),
                .n_accepted = n_accepted_device.get(),
                .n_updated_tracks = n_updated_tracks_device.get(),
                .inverted_ids_view = inverted_ids_buffer,
            });

        // The three kernels (block_inclusive_scan, scan_block_offsets, and
        // add_block_offset) work together to compute the prefix sum of track
        // IDs, with respect to the number of updated tracks, based on the
        // indices of sorted_ids. The kernels are splitted as it is not
        // efficient to calculate this in a single kernel. This prefix sums are
        // used during the insertion sorting in rearrange_tracks to precisely
        // calculate the new index of updated tracks

        // Caculate the prefix sum of the number of updated tracks block-wisely.
        // block_offset is the last element of block-wise prefix sums, which is
        // used to get the real prefix sum later
        kernels::block_inclusive_scan<<<nBlocks_scan, nThreads_scan,
                                        nThreads_scan * sizeof(int), stream>>>(
            device::block_inclusive_scan_payload{
                .sorted_ids_view = sorted_ids_buffer,
                .terminate = terminate_device.get(),
                .n_accepted = n_accepted_device.get(),
                .n_updated_tracks = n_updated_tracks_device.get(),
                .is_updated_view = is_updated_buffer,
                .block_offsets_view = block_offsets_buffer,
                .prefix_sums_view = prefix_sums_buffer});

        // Calculate the scanned block offsets which is the prefix sum of block
        // offsets
        kernels::scan_block_offsets<<<1, nBlocks_scan,
                                      nBlocks_scan * sizeof(int), stream>>>(
            device::scan_block_offsets_payload{
                .terminate = terminate_device.get(),
                .n_accepted = n_accepted_device.get(),
                .n_updated_tracks = n_updated_tracks_device.get(),
                .block_offsets_view = block_offsets_buffer,
                .scanned_block_offsets_view = scanned_block_offsets_buffer});

        // To calculate the real prefix-sum, add the scanned block offsets to
        // block-wise prefix sums of the number of updated tracks.
        kernels::add_block_offset<<<nBlocks_scan, nThreads_scan, 0, stream>>>(
            device::add_block_offset_payload{
                .terminate = terminate_device.get(),
                .n_accepted = n_accepted_device.get(),
                .n_updated_tracks = n_updated_tracks_device.get(),
                .block_offsets_view = scanned_block_offsets_buffer,
                .prefix_sums_view = prefix_sums_buffer});

        // Apply the insertion sort algorithm to sorted_ids_view using the
        // sorted updated tracks and prefix sums. The sorted elements are stored
        // in temp_sorted_ids_view
        kernels::rearrange_tracks<<<nBlocks_rearrange, nThreads_rearrange, 0,
                                    stream>>>(device::rearrange_tracks_payload{
            .sorted_ids_view = sorted_ids_buffer,
            .inverted_ids_view = inverted_ids_buffer,
            .rel_shared_view = rel_shared_buffer,
            .pvals_view = pvals_buffer,
            .terminate = terminate_device.get(),
            .n_accepted = n_accepted_device.get(),
            .n_updated_tracks = n_updated_tracks_device.get(),
            .updated_tracks_view = updated_tracks_buffer,
            .is_updated_view = is_updated_buffer,
            .prefix_sums_view = prefix_sums_buffer,
            .temp_sorted_ids_view = temp_sorted_ids_buffer,
        });

        // Find the max shared number of measurements to decide whether to
        // terminate the process. Also Move the elements in temp_sorted_ids to
        // sorted_ids
        kernels::
            update_status<<<nBlocks_adaptive, nThreads_adaptive, 0, stream>>>(
                device::update_status_payload{
                    .terminate = terminate_device.get(),
                    .n_accepted = n_accepted_device.get(),
                    .n_updated_tracks = n_updated_tracks_device.get(),
                    .temp_sorted_ids_view = temp_sorted_ids_buffer,
                    .sorted_ids_view = sorted_ids_buffer,
                    .updated_tracks_view = updated_tracks_buffer,
                    .is_updated_view = is_updated_buffer,
                    .n_shared_view = n_shared_buffer,
                    .max_shared = max_shared_device.get()});

        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

        // Adaptive formula minimises CUDA Graph construction overhead.
        // Benchmark data shows graph construction (not launch count) dominates
        // cost. High n_it amortises construction; the exception is very small n
        // where too many no-op launches waste time when convergence is fast.
        const unsigned int n_it =
            m_adaptive_n_it
                ? (n_accepted < 500u
                       ? std::max(10u, std::min(50u, n_accepted / 5u))
                       : m_n_it_max)
                : m_n_it_max;
        for (unsigned int iter = 0; iter < n_it; iter++) {
            cudaGraphLaunch(graphExec, stream);
        }
        n_graph_launches += n_it;

        cudaMemcpyAsync(&terminate, terminate_device.get(), sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&n_accepted, n_accepted_device.get(),
                        sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);

        // Pull back the batch size observed on the last graph replay in this
        // outer step when PBG logging is requested. The value is the batch
        // admitted on the final replay only; ncu / nsys traces remain the
        // source of truth for per-replay detail.
        unsigned int last_batch = 0u;
        if (m_parallel_batch && m_batch_size_log != nullptr) {
            cudaMemcpyAsync(&last_batch, batch_size_device.get(),
                            sizeof(unsigned int), cudaMemcpyDeviceToHost,
                            stream);
        }
        m_stream.get().synchronize();
        if (m_parallel_batch && m_batch_size_log != nullptr) {
            batch_sizes_this_call.push_back(last_batch);
        }
    }
    AR_NVTX_POP();
    perf.mark(6, stream);

    // Publish the per-outer-iteration batch size log once the loop has
    // finished. Only written when PBG and logging are both enabled.
    if (m_parallel_batch && m_batch_size_log != nullptr) {
        m_batch_size_log->assign(batch_sizes_this_call.begin(),
                                 batch_sizes_this_call.end());
    }

    // Tier 2c graph-mode logs.
    if (m_graph_batch_log != nullptr) {
        m_graph_batch_log->assign(graph_batches_this_call.begin(),
                                  graph_batches_this_call.end());
    }
    if (m_graph_size_log != nullptr) {
        m_graph_size_log->assign(graph_sizes_this_call.begin(),
                                 graph_sizes_this_call.end());
    }

    cudaMemcpyAsync(&n_accepted, n_accepted_device.get(), sizeof(unsigned int),
                    cudaMemcpyDeviceToHost, stream);

    auto max_it =
        std::max_element(candidate_sizes.begin(), candidate_sizes.end());
    const unsigned int max_cands_size = *max_it;

    // Create resolved candidate buffer
    edm::track_container<default_algebra>::buffer res_track_candidates_buffer{
        {std::vector<std::size_t>(n_accepted, max_cands_size), m_mr.main,
         m_mr.host, vecmem::data::buffer_type::resizable},
        {},
        tracks_view.measurements};
    m_copy.get().setup(res_track_candidates_buffer.tracks)->ignore();

    AR_NVTX_PUSH("gpu_ar_output_copy");
    // Fill the output track candidates
    {
        if (n_accepted > 0) {
            kernels::fill_track_candidates<<<
                static_cast<unsigned int>((n_accepted + 63) / 64), 64, 0,
                stream>>>(device::fill_track_candidates_payload{
                .tracks_view = tracks_view,
                .n_accepted = n_accepted,
                .sorted_ids_view = sorted_ids_buffer,
                .res_tracks_view = res_track_candidates_buffer});
            TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

            m_stream.get().synchronize();
        }
    }
    AR_NVTX_POP();
    perf.mark(7, stream);

    if (m_profiling) {
        cudaStreamSynchronize(stream);
        m_last_profile.filter_setup_ms = perf.elapsed(0, 1);
        m_last_profile.unique_meas_ms = perf.elapsed(1, 2);
        m_last_profile.inverted_index_ms = perf.elapsed(2, 3);
        m_last_profile.shared_count_ms = perf.elapsed(3, 4);
        m_last_profile.initial_sort_ms = perf.elapsed(4, 5);
        m_last_profile.eviction_loop_ms = perf.elapsed(5, 6);
        m_last_profile.output_copy_ms = perf.elapsed(6, 7);
        m_last_profile.eviction_graph_launches = n_graph_launches;
    }

    return res_track_candidates_buffer;
}

}  // namespace traccc::cuda
