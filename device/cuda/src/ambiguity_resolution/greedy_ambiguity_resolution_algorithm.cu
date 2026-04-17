/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/utils.hpp"
#include "./kernels/add_block_offset.cuh"
#include "./kernels/block_inclusive_scan.cuh"
#include "./kernels/count_shared_measurements.cuh"
#include "./kernels/fill_inverted_ids.cuh"
#include "./kernels/fill_track_candidates.cuh"
#include "./kernels/fill_tracks_per_measurement.cuh"
#include "./kernels/fill_unique_meas_id_map.cuh"
#include "./kernels/fill_vectors.cuh"
#include "./kernels/rearrange_tracks.cuh"
#include "./kernels/remove_tracks.cuh"
#include "./kernels/scan_block_offsets.cuh"
#include "./kernels/sort_tracks_per_measurement.cuh"
#include "./kernels/sort_updated_tracks.cuh"
#include "./kernels/update_status.cuh"
#include "traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/definitions/math.hpp"

// NVTX range helpers — zero overhead when no profiler is attached.
#if __has_include("nvtx3/nvToolsExt.h")
#    include "nvtx3/nvToolsExt.h"
#    define AR_NVTX_PUSH(name) nvtxRangePushA(name)
#    define AR_NVTX_POP()      nvtxRangePop()
#else
#    define AR_NVTX_PUSH(name) ((void)0)
#    define AR_NVTX_POP()      ((void)0)
#endif

// Thrust include(s).
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <vector>
namespace {

/// Lightweight CUDA event set used for per-phase elapsed-time measurement.
/// All events are created and destroyed inside operator() when profiling is on.
struct PerfEvents {
    cudaEvent_t ev[8]{};
    bool active;

    PerfEvents(bool on, cudaStream_t s) : active(on) {
        if (!active) return;
        for (auto& e : ev) cudaEventCreate(&e);
        cudaEventRecord(ev[0], s);
    }

    ~PerfEvents() {
        if (!active) return;
        for (auto& e : ev) cudaEventDestroy(e);
    }

    void mark(int idx, cudaStream_t s) {
        if (active) cudaEventRecord(ev[idx], s);
    }

    float elapsed(int a, int b) const {
        float ms = 0.f;
        if (active) cudaEventElapsedTime(&ms, ev[a], ev[b]);
        return ms;
    }
};

struct eviction_launch_config {
    unsigned int nThreads_adaptive{0u};
    unsigned int nBlocks_adaptive{0u};
    unsigned int nThreads_rearrange{0u};
    unsigned int nBlocks_rearrange{0u};
    unsigned int nThreads_scan{0u};
    unsigned int nBlocks_scan{0u};
};

struct eviction_graph_nodes {
    cudaGraphNode_t fill_inverted_ids{nullptr};
    cudaGraphNode_t block_inclusive_scan{nullptr};
    cudaGraphNode_t scan_block_offsets{nullptr};
    cudaGraphNode_t add_block_offset{nullptr};
    cudaGraphNode_t rearrange_tracks{nullptr};
    cudaGraphNode_t update_status{nullptr};
};

struct graph_exec_holder {
    cudaGraph_t graph{nullptr};
    cudaGraphExec_t exec{nullptr};

    ~graph_exec_holder() {
        if (exec != nullptr) {
            cudaGraphExecDestroy(exec);
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
    }
};

void set_kernel_node_launch(cudaGraphExec_t graph_exec, cudaGraphNode_t node,
                            dim3 grid, dim3 block,
                            std::size_t shared_mem_bytes = 0u) {
    cudaKernelNodeParams params{};
    TRACCC_CUDA_ERROR_CHECK(cudaGraphKernelNodeGetParams(node, &params));
    params.gridDim = grid;
    params.blockDim = block;
    params.sharedMemBytes = static_cast<unsigned int>(shared_mem_bytes);
    TRACCC_CUDA_ERROR_CHECK(
        cudaGraphExecKernelNodeSetParams(graph_exec, node, &params));
}

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

    const edm::measurement_collection<default_algebra>::const_device
        measurements(tracks_view.measurements);

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
    unsigned int n_graph_instantiations = 0u;

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

    if (m_profiling) m_stream.get().synchronize();
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

    // Compute the threadblock dimensions for the eviction-loop kernels.
    auto compute_eviction_launch_config = [&](unsigned int accepted) {
        eviction_launch_config cfg{};
        cfg.nThreads_adaptive = m_warp_size;
        cfg.nBlocks_adaptive =
            (accepted + cfg.nThreads_adaptive - 1) / cfg.nThreads_adaptive;

        cfg.nThreads_rearrange = 1024;
        cfg.nBlocks_rearrange =
            (accepted +
             (cfg.nThreads_rearrange / kernels::nThreads_per_track) - 1) /
            (cfg.nThreads_rearrange / kernels::nThreads_per_track);

        cfg.nThreads_scan = m_warp_size * 4;
        cfg.nBlocks_scan =
            (accepted + cfg.nThreads_scan - 1) / cfg.nThreads_scan;

        while (cfg.nThreads_scan <= 1024) {
            if (cfg.nBlocks_scan > 1024) {
                cfg.nThreads_scan *= 2;
                cfg.nBlocks_scan =
                    (accepted + cfg.nThreads_scan - 1) / cfg.nThreads_scan;
            } else {
                break;
            }
        }

        return cfg;
    };

    auto launch_cfg = compute_eviction_launch_config(n_accepted);

    assert(launch_cfg.nBlocks_scan <= 1024 &&
           "nBlocks_scan larger than 1024 will cause invalid arguments in "
           "scan_block_offsets kernel");

    // Make buffers used in prefix sum calculation
    vecmem::data::vector_buffer<int> block_offsets_buffer{
        launch_cfg.nBlocks_scan,
                                                          m_mr.main};
    m_copy.get().setup(block_offsets_buffer)->ignore();
    vecmem::data::vector_buffer<int> scanned_block_offsets_buffer{
        launch_cfg.nBlocks_scan,
                                                                  m_mr.main};
    m_copy.get().setup(scanned_block_offsets_buffer)->ignore();

    auto capture_eviction_graph = [&](const eviction_launch_config& cfg,
                                      cudaGraph_t* graph_out,
                                      cudaGraphExec_t* graph_exec_out) {
        TRACCC_CUDA_ERROR_CHECK(
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        kernels::remove_tracks<<<1, 512, 0, stream>>>(
            device::remove_tracks_payload{
                .sorted_ids_view = sorted_ids_buffer,
                .n_accepted = n_accepted_device.get(),
                .meas_ids_view = meas_ids_buffer,
                .n_meas_view = n_meas_buffer,
                .meas_id_to_unique_id_view = meas_id_to_unique_id_buffer,
                .tracks_per_measurement_view = tracks_per_measurement_buffer,
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

        kernels::sort_updated_tracks<<<1, 512, 0, stream>>>(
            device::sort_updated_tracks_payload{
                .rel_shared_view = rel_shared_buffer,
                .pvals_view = pvals_buffer,
                .terminate = terminate_device.get(),
                .n_updated_tracks = n_updated_tracks_device.get(),
                .updated_tracks_view = updated_tracks_buffer,
            });

        kernels::fill_inverted_ids<<<cfg.nBlocks_adaptive,
                                     cfg.nThreads_adaptive, 0, stream>>>(
            device::fill_inverted_ids_payload{
                .sorted_ids_view = sorted_ids_buffer,
                .terminate = terminate_device.get(),
                .n_accepted = n_accepted_device.get(),
                .n_updated_tracks = n_updated_tracks_device.get(),
                .inverted_ids_view = inverted_ids_buffer,
            });

        kernels::block_inclusive_scan<<<cfg.nBlocks_scan, cfg.nThreads_scan,
                                        cfg.nThreads_scan * sizeof(int),
                                        stream>>>(
            device::block_inclusive_scan_payload{
                .sorted_ids_view = sorted_ids_buffer,
                .terminate = terminate_device.get(),
                .n_accepted = n_accepted_device.get(),
                .n_updated_tracks = n_updated_tracks_device.get(),
                .is_updated_view = is_updated_buffer,
                .block_offsets_view = block_offsets_buffer,
                .prefix_sums_view = prefix_sums_buffer});

        kernels::scan_block_offsets<<<1, cfg.nBlocks_scan,
                                      cfg.nBlocks_scan * sizeof(int), stream>>>(
            device::scan_block_offsets_payload{
                .terminate = terminate_device.get(),
                .n_accepted = n_accepted_device.get(),
                .n_updated_tracks = n_updated_tracks_device.get(),
                .block_offsets_view = block_offsets_buffer,
                .scanned_block_offsets_view = scanned_block_offsets_buffer});

        kernels::add_block_offset<<<cfg.nBlocks_scan, cfg.nThreads_scan, 0,
                                    stream>>>(
            device::add_block_offset_payload{
                .terminate = terminate_device.get(),
                .n_accepted = n_accepted_device.get(),
                .n_updated_tracks = n_updated_tracks_device.get(),
                .block_offsets_view = scanned_block_offsets_buffer,
                .prefix_sums_view = prefix_sums_buffer});

        kernels::rearrange_tracks<<<cfg.nBlocks_rearrange,
                                    cfg.nThreads_rearrange, 0, stream>>>(
            device::rearrange_tracks_payload{
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

        kernels::
            update_status<<<cfg.nBlocks_adaptive, cfg.nThreads_adaptive, 0,
                            stream>>>(device::update_status_payload{
                .terminate = terminate_device.get(),
                .n_accepted = n_accepted_device.get(),
                .n_updated_tracks = n_updated_tracks_device.get(),
                .temp_sorted_ids_view = temp_sorted_ids_buffer,
                .sorted_ids_view = sorted_ids_buffer,
                .updated_tracks_view = updated_tracks_buffer,
                .is_updated_view = is_updated_buffer,
                .n_shared_view = n_shared_buffer,
                .max_shared = max_shared_device.get()});

        TRACCC_CUDA_ERROR_CHECK(cudaStreamEndCapture(stream, graph_out));
        TRACCC_CUDA_ERROR_CHECK(
            cudaGraphInstantiate(graph_exec_out, *graph_out, nullptr, nullptr,
                                 0));
        ++n_graph_instantiations;
    };

    auto collect_eviction_graph_nodes = [&](cudaGraph_t graph) {
        std::size_t num_nodes = 0u;
        TRACCC_CUDA_ERROR_CHECK(cudaGraphGetNodes(graph, nullptr, &num_nodes));
        if (num_nodes != 8u) {
            throw std::runtime_error(
                "Unexpected eviction graph structure while collecting nodes");
        }

        std::vector<cudaGraphNode_t> nodes(num_nodes);
        TRACCC_CUDA_ERROR_CHECK(
            cudaGraphGetNodes(graph, nodes.data(), &num_nodes));

        for (cudaGraphNode_t node : nodes) {
            cudaGraphNodeType node_type{};
            TRACCC_CUDA_ERROR_CHECK(cudaGraphNodeGetType(node, &node_type));
            if (node_type != cudaGraphNodeTypeKernel) {
                throw std::runtime_error(
                    "Eviction graph contains a non-kernel node");
            }
        }

        // The captured graph is a fixed linear chain of eight kernels.
        return eviction_graph_nodes{
            .fill_inverted_ids = nodes[2],
            .block_inclusive_scan = nodes[3],
            .scan_block_offsets = nodes[4],
            .add_block_offset = nodes[5],
            .rearrange_tracks = nodes[6],
            .update_status = nodes[7],
        };
    };

    auto update_eviction_graph_launches = [&](cudaGraphExec_t graph_exec,
                                              const eviction_graph_nodes& nodes,
                                              const eviction_launch_config& cfg) {
        set_kernel_node_launch(graph_exec, nodes.fill_inverted_ids,
                               dim3(cfg.nBlocks_adaptive),
                               dim3(cfg.nThreads_adaptive));
        set_kernel_node_launch(graph_exec, nodes.block_inclusive_scan,
                               dim3(cfg.nBlocks_scan),
                               dim3(cfg.nThreads_scan),
                               cfg.nThreads_scan * sizeof(int));
        set_kernel_node_launch(graph_exec, nodes.scan_block_offsets,
                               dim3(1u), dim3(cfg.nBlocks_scan),
                               cfg.nBlocks_scan * sizeof(int));
        set_kernel_node_launch(graph_exec, nodes.add_block_offset,
                               dim3(cfg.nBlocks_scan),
                               dim3(cfg.nThreads_scan));
        set_kernel_node_launch(graph_exec, nodes.rearrange_tracks,
                               dim3(cfg.nBlocks_rearrange),
                               dim3(cfg.nThreads_rearrange));
        set_kernel_node_launch(graph_exec, nodes.update_status,
                               dim3(cfg.nBlocks_adaptive),
                               dim3(cfg.nThreads_adaptive));
    };

    AR_NVTX_PUSH("gpu_ar_eviction_loop");
    graph_exec_holder reused_graph;
    eviction_graph_nodes reused_nodes{};
    bool reused_graph_ready = false;

    // Start the iteration
    while (!terminate && n_accepted > 0) {
        launch_cfg = compute_eviction_launch_config(n_accepted);

        assert(launch_cfg.nBlocks_scan <= 1024 &&
               "nBlocks_scan larger than 1024 will cause invalid arguments "
               "in scan_block_offsets kernel");

        cudaGraphExec_t graph_exec = nullptr;

        if (m_reuse_eviction_graph) {
            if (!reused_graph_ready) {
                capture_eviction_graph(launch_cfg, &reused_graph.graph,
                                      &reused_graph.exec);
                reused_nodes = collect_eviction_graph_nodes(reused_graph.graph);
                reused_graph_ready = true;
            } else {
                update_eviction_graph_launches(reused_graph.exec, reused_nodes,
                                               launch_cfg);
            }
            graph_exec = reused_graph.exec;
        } else {
            graph_exec_holder loop_graph;
            capture_eviction_graph(launch_cfg, &loop_graph.graph,
                                  &loop_graph.exec);
            graph_exec = loop_graph.exec;

            // Adaptive formula minimises CUDA Graph construction overhead.
            // Benchmark data shows graph construction (not launch count)
            // dominates cost. High n_it amortises construction; the exception
            // is very small n where too many no-op launches waste time when
            // convergence is fast.
            const unsigned int n_it =
                m_adaptive_n_it
                    ? (n_accepted < 500u
                           ? std::max(10u, std::min(50u, n_accepted / 5u))
                           : m_n_it_max)
                    : m_n_it_max;
            for (unsigned int iter = 0; iter < n_it; iter++) {
                TRACCC_CUDA_ERROR_CHECK(cudaGraphLaunch(graph_exec, stream));
            }
            n_graph_launches += n_it;

            cudaMemcpyAsync(&terminate, terminate_device.get(), sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(&n_accepted, n_accepted_device.get(),
                            sizeof(unsigned int), cudaMemcpyDeviceToHost,
                            stream);
            m_stream.get().synchronize();
            continue;
        }

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
            TRACCC_CUDA_ERROR_CHECK(cudaGraphLaunch(graph_exec, stream));
        }
        n_graph_launches += n_it;

        cudaMemcpyAsync(&terminate, terminate_device.get(), sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&n_accepted, n_accepted_device.get(),
                        sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        m_stream.get().synchronize();
    }
    AR_NVTX_POP();
    perf.mark(6, stream);

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
        m_last_profile.filter_setup_ms          = perf.elapsed(0, 1);
        m_last_profile.unique_meas_ms           = perf.elapsed(1, 2);
        m_last_profile.inverted_index_ms        = perf.elapsed(2, 3);
        m_last_profile.shared_count_ms          = perf.elapsed(3, 4);
        m_last_profile.initial_sort_ms          = perf.elapsed(4, 5);
        m_last_profile.eviction_loop_ms         = perf.elapsed(5, 6);
        m_last_profile.output_copy_ms           = perf.elapsed(6, 7);
        m_last_profile.eviction_graph_launches  = n_graph_launches;
        m_last_profile.eviction_graph_instantiations = n_graph_instantiations;
    }

    return res_track_candidates_buffer;
}

}  // namespace traccc::cuda
