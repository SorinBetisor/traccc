/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/ambiguity_resolution/ambiguity_resolution_config.hpp"
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/track_container.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <algorithm>
#include <cstddef>
#include <set>
#include <vector>

using namespace traccc;

namespace {

void fill_measurements(edm::measurement_collection::host& measurements,
                       const measurement_id_type max_meas_id) {
    measurements.reserve(static_cast<std::size_t>(max_meas_id) + 1u);
    for (measurement_id_type i = 0; i <= max_meas_id; i++) {
        measurements.push_back({});
        measurements.at(measurements.size() - 1).identifier() = i;
    }
}

void fill_pattern(edm::track_container<default_algebra>::host& trk_cands,
                  const traccc::scalar pval,
                  const std::vector<measurement_id_type>& pattern) {
    trk_cands.tracks.resize(trk_cands.tracks.size() + 1u);
    trk_cands.tracks.pval().back() = pval;

    edm::measurement_collection::const_device measurements{
        trk_cands.measurements};
    for (const auto& meas_id : pattern) {
        const auto it = std::lower_bound(measurements.identifier().begin(),
                                         measurements.identifier().end(),
                                         meas_id);
        const auto idx = static_cast<measurement_id_type>(
            std::distance(measurements.identifier().begin(), it));
        trk_cands.tracks.constituent_links().back().push_back(
            {edm::track_constituent_link::measurement, idx});
    }
}

std::vector<measurement_id_type> get_pattern(
    const edm::track_container<default_algebra>::host& trk_cands,
    const std::size_t idx) {
    edm::measurement_collection::const_device measurements{
        trk_cands.measurements};
    std::vector<measurement_id_type> ids;
    const auto links = trk_cands.tracks.at(idx).constituent_links();
    for (const auto& [type, m_idx] : links) {
        (void)type;
        ids.push_back(measurements.at(m_idx).identifier());
    }
    return ids;
}

std::vector<measurement_id_type> get_pattern(
    const edm::track_container<default_algebra>::const_device& trk_cands,
    const std::size_t idx) {
    std::vector<measurement_id_type> ids;
    for (const auto& [type, m_idx] :
         trk_cands.tracks.constituent_links().at(idx)) {
        (void)type;
        ids.push_back(trk_cands.measurements.at(m_idx).identifier());
    }
    return ids;
}

std::set<std::vector<measurement_id_type>> sorted_track_set_host(
    const edm::track_container<default_algebra>::host& trk_cands) {
    std::set<std::vector<measurement_id_type>> out;
    for (std::size_t i = 0; i < trk_cands.tracks.size(); ++i) {
        auto p = get_pattern(trk_cands, i);
        std::sort(p.begin(), p.end());
        out.insert(std::move(p));
    }
    return out;
}

std::set<std::vector<measurement_id_type>> sorted_track_set_device(
    const edm::track_container<default_algebra>::const_device& dev) {
    std::set<std::vector<measurement_id_type>> out;
    for (std::size_t i = 0; i < dev.tracks.size(); ++i) {
        auto p = get_pattern(dev, i);
        std::sort(p.begin(), p.end());
        out.insert(std::move(p));
    }
    return out;
}

}  // namespace

// ----------------------------------------------------------------------------
// JP selection-identical to CPU on a tiny hand-rolled case.
//
// Two pairs of conflicting tracks (each pair shares one measurement) plus
// one isolated track. The CPU greedy resolver keeps the higher-pval track
// in each conflicting pair; the JP backend MUST produce the same
// selection. This nails down the "agree with CPU on a forced answer"
// property of the resolver-validity contract on a case where the answer
// is unique and the relative-shared rank is unambiguous.
//
// On larger/denser inputs JP is allowed to pick any valid maximal
// independent set; that property is exercised by the resolver-only
// benchmark harness (traccc_benchmark_resolver_cuda --enable-jp
// --determinism-runs=N) on real ODD/Fatras dumps, not in this unit test,
// because synthetic inputs with many ties on relative-shared can legally
// expose multiple valid MIS even though every individual selection is
// conflict-free.
// ----------------------------------------------------------------------------
TEST(CUDAJonesPlassmannResolverTests, SelectionIdenticalToCPU_Tiny) {

    vecmem::cuda::managed_memory_resource mng_mr;
    traccc::memory_resource mr{mng_mr};
    traccc::cuda::stream stream;
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    edm::measurement_collection::host measurements{mng_mr};
    fill_measurements(measurements, 19);

    edm::track_container<default_algebra>::host trk_cands{
        mng_mr, vecmem::get_data(measurements)};
    // Pair A: tracks 0 and 1 share measurement 5. Track 1 wins (higher pval).
    fill_pattern(trk_cands, 0.30f, {0, 1, 2, 3, 5});
    fill_pattern(trk_cands, 0.90f, {5, 6, 7, 8, 9});
    // Pair B: tracks 2 and 3 share measurement 12. Track 3 wins (higher pval).
    fill_pattern(trk_cands, 0.20f, {10, 11, 12, 13});
    fill_pattern(trk_cands, 0.85f, {12, 14, 15, 16, 17});
    // Track 4 is isolated and always survives.
    fill_pattern(trk_cands, 0.50f, {18, 19});

    // CPU reference.
    traccc::host::greedy_ambiguity_resolution_algorithm::config_type cfg;
    traccc::host::greedy_ambiguity_resolution_algorithm cpu_alg(cfg, mng_mr);
    auto cpu_res = cpu_alg(
        edm::track_container<default_algebra>::const_data(trk_cands));
    const auto cpu_set = sorted_track_set_host(cpu_res);

    // GPU JP backend.
    traccc::cuda::greedy_ambiguity_resolution_algorithm gpu_alg(cfg, mr, copy,
                                                                stream);
    gpu_alg.set_conflict_graph_mode(
        traccc::cuda::greedy_ambiguity_resolution_algorithm::graph_algo_t::JP);

    auto gpu_buf = gpu_alg(
        edm::track_container<default_algebra>::const_data(trk_cands));
    stream.synchronize();
    edm::track_container<default_algebra>::const_device gpu_dev{gpu_buf};
    const auto jp_set = sorted_track_set_device(gpu_dev);

    EXPECT_EQ(jp_set.size(), cpu_set.size())
        << "JP must keep the same number of tracks as CPU on this tiny case";
    EXPECT_EQ(jp_set, cpu_set)
        << "JP must select the exact same tracks as CPU on this tiny case";
}
