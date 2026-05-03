/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/cuda/utils/stream.hpp"

// Project include(s).
#include "traccc/ambiguity_resolution/ambiguity_resolution_config.hpp"
#include "traccc/edm/track_container.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <utility>
#include <vector>

namespace traccc::cuda {

/// Per-phase GPU timing data populated when set_profiling(true) is active.
struct gpu_profile_data_t {
    float filter_setup_ms = 0.f;
    float unique_meas_ms = 0.f;
    float inverted_index_ms = 0.f;
    float shared_count_ms = 0.f;
    float initial_sort_ms = 0.f;
    float eviction_loop_ms = 0.f;
    float output_copy_ms = 0.f;
    unsigned int eviction_graph_launches = 0u;
    unsigned int unique_meas_count = 0u;
};

/// Evicts tracks that seem to be duplicates or fakes. This algorithm takes a
/// greedy approach in the sense that it will remove the track which looks "most
/// duplicate/fake"
class greedy_ambiguity_resolution_algorithm
    : public algorithm<edm::track_container<default_algebra>::buffer(
          const edm::track_container<default_algebra>::const_view&)>,
      public messaging {

    public:
    using config_type = ambiguity_resolution_config;

    /// Constructor for the greedy ambiguity resolution algorithm
    ///
    /// @param cfg The configuration
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param str The CUDA stream to perform the operations in
    ///
    greedy_ambiguity_resolution_algorithm(
        const config_type& cfg, const traccc::memory_resource& mr,
        vecmem::copy& copy, stream& str,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Run the algorithm
    ///
    /// @param tracks the container view of found patterns
    /// @return the container without ambiguous tracks
    output_type operator()(
        const edm::track_container<default_algebra>::const_view& tracks)
        const override;

    /// Get configuration
    config_type& get_config() { return m_config; }

    /// Enable per-phase CUDA event timing for the next operator() call.
    void set_profiling(bool on) { m_profiling = on; }

    /// Returns profiling data from the most recent operator() call.
    /// Only valid after a call where set_profiling(true) was active.
    const gpu_profile_data_t& last_profile() const { return m_last_profile; }

    /// Override the maximum inner-loop iterations per outer eviction step.
    /// Default is 100. Lower values reduce overhead at small n_candidates.
    /// Calling this also disables the adaptive formula (see set_adaptive_n_it).
    void set_n_it_max(unsigned int n) { m_n_it_max = n; }

    /// When true (default), n_it per outer step is computed adaptively to
    /// minimise CUDA Graph construction overhead:
    ///   n_accepted >= 500: n_it = m_n_it_max  (maximise amortisation)
    ///   n_accepted <  500: n_it = clamp(n_accepted/5, 10, 50)
    /// Set to false to force a fixed m_n_it_max every outer step.
    void set_adaptive_n_it(bool on) { m_adaptive_n_it = on; }

    /// Algorithm selector for the explicit-conflict-graph removal path.
    ///   NONE — fall through to the baseline single-block remove_tracks
    ///          path. This is the default and the existing CUDA resolver
    ///          behaviour.
    ///   JP   — build the CSR conflict graph once per outer iteration and
    ///          run a single Jones-Plassmann coloring round with a
    ///          deterministic priority (lowest sorted-rank wins). The set
    ///          of color-0 vertices (those whose priority is locally
    ///          maximal among still-undecided neighbours) becomes the
    ///          batch removed in that iteration. JP is conflict-free by
    ///          construction: by definition, no two color-0 vertices
    ///          share an edge, so removing them in parallel is safe.
    ///   MIS  — Luby-style maximal independent set: same propose/finalize
    ///          kernel pair as JP but iterated up to 32 rounds per outer
    ///          iteration so the IN_MIS set is a true MIS (locally-worst
    ///          vertex of every neighbourhood, not just one colour class).
    ///          Slower per outer iteration than JP but converges in fewer
    ///          outer iterations on dense conflict graphs.
    enum class graph_algo_t { NONE, JP, MIS };
    void set_conflict_graph_mode(graph_algo_t a) { m_graph_algo = a; }
    graph_algo_t conflict_graph_mode() const { return m_graph_algo; }

    /// Per-outer-iteration number of tracks removed (color-0 set size).
    /// Hooked when conflict_graph_mode is not NONE. Caller owns storage;
    /// pass nullptr to disable. Default: nullptr.
    void set_graph_batch_log(std::vector<unsigned int>* out) {
        m_graph_batch_log = out;
    }

    /// Per-outer-iteration number of vertices / edges in the conflict graph
    /// that was built. Hooked when conflict_graph_mode is not NONE. Caller
    /// owns storage; pass nullptr to disable. Default: nullptr.
    void set_graph_size_log(
        std::vector<std::pair<unsigned int, unsigned int>>* out) {
        m_graph_size_log = out;
    }

    private:
    /// Algorithm configuration
    config_type m_config;
    /// The memory resource to use
    traccc::memory_resource m_mr;
    /// The copy object to use
    std::reference_wrapper<vecmem::copy> m_copy;
    /// The CUDA stream to use
    std::reference_wrapper<stream> m_stream;
    /// Warp size of the GPU being used
    unsigned int m_warp_size;
    mutable bool m_profiling{false};
    mutable gpu_profile_data_t m_last_profile{};
    unsigned int m_n_it_max{100u};
    bool m_adaptive_n_it{true};
    graph_algo_t m_graph_algo{graph_algo_t::NONE};
    mutable std::vector<unsigned int>* m_graph_batch_log{nullptr};
    mutable std::vector<std::pair<unsigned int, unsigned int>>*
        m_graph_size_log{nullptr};
};

}  // namespace traccc::cuda
