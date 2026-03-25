/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 *
 * Resolver-only benchmark harness for ambiguity resolution.
 */

#include "traccc/ambiguity_resolution/ambiguity_resolution_config.hpp"
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/edm/track_container.hpp"
#include "traccc/io/ambiguity_io.hpp"

#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef __linux__
#include <sys/resource.h>
#endif

namespace {

void fill_measurements(
    traccc::edm::measurement_collection<traccc::default_algebra>::host& m,
    traccc::measurement_id_type max_id) {
    m.reserve(max_id + 1);
    for (traccc::measurement_id_type i = 0; i <= max_id; i++) {
        m.push_back({});
        m.at(m.size() - 1).identifier() = i;
    }
}

void fill_pattern(
    traccc::edm::track_container<traccc::default_algebra>::host& tc,
    traccc::scalar pval,
    const std::vector<traccc::measurement_id_type>& pattern) {
    tc.tracks.resize(tc.tracks.size() + 1u);
    tc.tracks.pval().back() = pval;
    traccc::edm::measurement_collection<traccc::default_algebra>::const_device
        meas{tc.measurements};
    for (traccc::measurement_id_type meas_id : pattern) {
        auto it = std::lower_bound(meas.identifier().begin(),
                                   meas.identifier().end(), meas_id);
        auto idx = static_cast<traccc::measurement_id_type>(
            std::distance(meas.identifier().begin(), it));
        tc.tracks.constituent_links().back().push_back(
            {traccc::edm::track_constituent_link::measurement, idx});
    }
}

std::string compute_output_hash(
    const traccc::edm::track_container<traccc::default_algebra>::host& out) {
    std::vector<std::vector<traccc::measurement_id_type>> patterns;
    traccc::edm::measurement_collection<traccc::default_algebra>::const_device
        meas{out.measurements};
    for (std::size_t i = 0; i < out.tracks.size(); ++i) {
        std::vector<traccc::measurement_id_type> p;
        for (const auto& [type, idx] : out.tracks.at(i).constituent_links()) {
            if (type == traccc::edm::track_constituent_link::measurement) {
                p.push_back(meas.at(idx).identifier());
            }
        }
        std::sort(p.begin(), p.end());
        patterns.push_back(std::move(p));
    }
    std::sort(patterns.begin(), patterns.end());
    std::ostringstream oss;
    for (const auto& p : patterns) {
        for (auto id : p) oss << id << ",";
        oss << ";";
    }
    return std::to_string(std::hash<std::string>{}(oss.str()));
}

double get_peak_rss_mb() {
#ifdef __linux__
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        return static_cast<double>(ru.ru_maxrss) / 1024.0;
    }
#endif
    return 0.0;
}

/// Per-phase wall-clock breakdown for one resolver call.
struct PhaseTimingMs {
    double filter_setup    = 0;  ///< Allocate vectors, filter by min_meas, build meas_ids
    double unique_meas     = 0;  ///< Collect unique measurement IDs and sort them
    double inverted_index  = 0;  ///< Build tracks_per_measurement (inverted index)
    double shared_count    = 0;  ///< Count n_shared and rel_shared per track
    double initial_sort    = 0;  ///< Sort sorted_ids by (rel_shared, pval)
    double eviction_loop   = 0;  ///< Iterative worst-track removal + bookkeeping updates
    double output_copy     = 0;  ///< Copy accepted tracks into output container
    unsigned int n_iterations    = 0;  ///< Number of eviction iterations actually executed
    unsigned int n_shared_updates = 0; ///< Number of n_shared[tid] decrements in eviction
    std::size_t  unique_meas_count = 0; ///< Size of the unique measurement universe
    std::string  output_hash;
};

/// Reimplements the greedy resolver with per-phase std::chrono timing.
/// Output hash is computed for cross-checking against the reference resolver.
static PhaseTimingMs run_with_phase_timing(
    const traccc::edm::track_container<traccc::default_algebra>::const_view&
        track_view,
    const traccc::ambiguity_resolution_config& config,
    vecmem::memory_resource& mr) {

    using clk = std::chrono::high_resolution_clock;
    using ms  = std::chrono::duration<double, std::milli>;

    PhaseTimingMs result;

    const traccc::edm::track_container<traccc::default_algebra>::const_device
        input_tracks(track_view);
    const std::size_t n_tracks = input_tracks.tracks.size();

    traccc::edm::track_container<traccc::default_algebra>::host output{mr};
    output.measurements = track_view.measurements;

    if (n_tracks == 0) return result;

    // ------------------------------------------------------------------
    // Phase 1: filter_setup
    //   Allocate per-track vectors, filter by min_meas_per_track, build
    //   meas_ids / pvals / n_meas.
    // ------------------------------------------------------------------
    auto t0 = clk::now();

    std::vector<unsigned int> accepted_ids(n_tracks);
    std::iota(accepted_ids.begin(), accepted_ids.end(), 0u);
    std::vector<std::vector<std::size_t>> meas_ids(n_tracks);
    std::vector<traccc::scalar>           pvals(n_tracks);
    std::vector<std::size_t>              n_meas(n_tracks, 0u);

    for (unsigned int i = 0; i < n_tracks; ++i) {
        pvals[i] = input_tracks.tracks.at(i).pval();
        const auto links = input_tracks.tracks.at(i).constituent_links();
        const unsigned int n_cands = links.size();
        if (n_cands < config.min_meas_per_track) {
            const auto it =
                std::lower_bound(accepted_ids.begin(), accepted_ids.end(), i);
            if (it != accepted_ids.end() && *it == i)
                accepted_ids.erase(it);
        } else {
            meas_ids[i].reserve(n_cands);
            for (const auto [type, idx] : links)
                meas_ids[i].push_back(
                    input_tracks.measurements.at(idx).identifier());
            n_meas[i] = n_cands;
        }
    }

    auto t1 = clk::now();
    result.filter_setup = ms(t1 - t0).count();

    // ------------------------------------------------------------------
    // Phase 2: unique_meas
    //   Collect all unique measurement IDs from accepted tracks; sort
    //   into a dense vector for O(log N) binary-search lookups.
    // ------------------------------------------------------------------
    std::unordered_set<std::size_t> unique_set;
    for (const unsigned int i : accepted_ids)
        for (const std::size_t mid : meas_ids[i])
            unique_set.insert(mid);

    std::vector<std::size_t> unique_meas(unique_set.begin(), unique_set.end());
    std::sort(unique_meas.begin(), unique_meas.end());
    result.unique_meas_count = unique_meas.size();

    auto t2 = clk::now();
    result.unique_meas = ms(t2 - t1).count();

    // ------------------------------------------------------------------
    // Phase 3: inverted_index
    //   For every accepted track, map each of its measurements to the
    //   sorted per-measurement track list (tracks_per_measurement).
    //   Lower_bound lookups into unique_meas are O(log |unique_meas|).
    // ------------------------------------------------------------------
    std::vector<std::vector<unsigned int>> tracks_per_meas(unique_meas.size());
    for (const unsigned int i : accepted_ids) {
        std::unordered_set<std::size_t> dedup(meas_ids[i].begin(),
                                              meas_ids[i].end());
        for (const std::size_t mid : dedup) {
            const auto it =
                std::lower_bound(unique_meas.begin(), unique_meas.end(), mid);
            const std::size_t uidx =
                static_cast<std::size_t>(std::distance(unique_meas.begin(), it));
            tracks_per_meas[uidx].push_back(i);
        }
    }

    auto t3 = clk::now();
    result.inverted_index = ms(t3 - t2).count();

    // ------------------------------------------------------------------
    // Phase 4: shared_count
    //   For every accepted track, count how many of its measurements are
    //   shared with at least one other track.  Computes rel_shared.
    // ------------------------------------------------------------------
    std::vector<unsigned int>  n_shared(n_tracks, 0u);
    std::vector<traccc::scalar> rel_shared(n_tracks, 0.f);

    for (const unsigned int i : accepted_ids) {
        for (const std::size_t mid : meas_ids[i]) {
            const auto it =
                std::lower_bound(unique_meas.begin(), unique_meas.end(), mid);
            const std::size_t uidx =
                static_cast<std::size_t>(std::distance(unique_meas.begin(), it));
            if (tracks_per_meas[uidx].size() > 1u)
                n_shared[i]++;
        }
        rel_shared[i] = static_cast<traccc::scalar>(n_shared[i]) /
                        static_cast<traccc::scalar>(n_meas[i]);
    }

    auto t4 = clk::now();
    result.shared_count = ms(t4 - t3).count();

    // ------------------------------------------------------------------
    // Phase 5: initial_sort
    //   Sort sorted_ids ascending by (rel_shared, -pval) so the worst
    //   track sits at the back.
    // ------------------------------------------------------------------
    std::vector<unsigned int> sorted_ids = accepted_ids;
    auto track_cmp = [&rel_shared, &pvals](unsigned int a, unsigned int b) {
        if (rel_shared[a] != rel_shared[b]) return rel_shared[a] < rel_shared[b];
        return pvals[a] > pvals[b];
    };
    std::sort(sorted_ids.begin(), sorted_ids.end(), track_cmp);

    auto t5 = clk::now();
    result.initial_sort = ms(t5 - t4).count();

    // ------------------------------------------------------------------
    // Phase 6: eviction_loop
    //   Iteratively remove the worst track and update bookkeeping.
    //   Hot sub-operations:
    //     - max_shared scan:          O(n_accepted) per iteration
    //     - lower_bound lookups:      O(log |unique_meas|) per measurement
    //     - tracks_per_meas erase:    O(|tracks_per_meas[uidx]|)
    //     - sorted_ids find+reinsert: O(n_accepted) per updated track  ← likely hotspot
    // ------------------------------------------------------------------
    for (unsigned int iter = 0; iter < config.max_iterations; ++iter) {
        if (accepted_ids.empty()) break;

        unsigned int max_shared = 0u;
        for (const unsigned int i : accepted_ids)
            if (n_shared[i] > max_shared) max_shared = n_shared[i];
        if (max_shared < config.max_shared_meas) break;

        result.n_iterations++;

        const unsigned int worst = sorted_ids.back();
        const auto ia =
            std::lower_bound(accepted_ids.begin(), accepted_ids.end(), worst);
        if (ia != accepted_ids.end() && *ia == worst)
            accepted_ids.erase(ia);
        sorted_ids.pop_back();

        std::unordered_set<std::size_t> seen;
        std::vector<std::size_t> to_remove;
        for (const std::size_t mid : meas_ids[worst])
            if (seen.insert(mid).second)
                to_remove.push_back(mid);

        for (const std::size_t mid : to_remove) {
            const auto it =
                std::lower_bound(unique_meas.begin(), unique_meas.end(), mid);
            if (it == unique_meas.end()) continue;
            const std::size_t uidx =
                static_cast<std::size_t>(std::distance(unique_meas.begin(), it));

            auto& tracks = tracks_per_meas[uidx];
            if (tracks.empty()) continue;
            const auto it2 =
                std::lower_bound(tracks.begin(), tracks.end(), worst);
            if (it2 != tracks.end() && *it2 == worst)
                tracks.erase(it2);

            if (tracks.size() == 1u) {
                const unsigned int tid = tracks[0];
                n_shared[tid] -= static_cast<unsigned int>(
                    std::count(meas_ids[tid].begin(), meas_ids[tid].end(), mid));
                rel_shared[tid] =
                    static_cast<traccc::scalar>(n_shared[tid]) /
                    static_cast<traccc::scalar>(n_meas[tid]);

                // O(n_accepted) linear scan to reposition tid in sorted_ids
                const auto it3 =
                    std::find(sorted_ids.begin(), sorted_ids.end(), tid);
                if (it3 != sorted_ids.end()) {
                    const auto it4 = std::lower_bound(
                        sorted_ids.begin(), it3, tid, track_cmp);
                    if (it3 != it4) {
                        sorted_ids.erase(it3);
                        sorted_ids.insert(it4, tid);
                    }
                }
                result.n_shared_updates++;
            }
        }
    }

    auto t6 = clk::now();
    result.eviction_loop = ms(t6 - t5).count();

    // ------------------------------------------------------------------
    // Phase 7: output_copy
    //   Copy accepted tracks into the host output container.
    // ------------------------------------------------------------------
    output.tracks.reserve(accepted_ids.size());
    for (const unsigned int i : accepted_ids) {
        const auto tcand = input_tracks.tracks.at(i);
        output.tracks.push_back(
            {tcand.fit_outcome(), tcand.params(), tcand.ndf(), tcand.chi2(),
             tcand.pval(), tcand.nholes(),
             {tcand.constituent_links().begin(),
              tcand.constituent_links().end()}});
    }

    auto t7 = clk::now();
    result.output_copy = ms(t7 - t6).count();

    result.output_hash = compute_output_hash(output);
    return result;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string input_dump;
    bool synthetic = false;
    std::size_t n_candidates = 10000;
    std::string conflict_density = "med";
    std::string backend = "cpu";
    std::size_t repeats = 10;
    std::size_t warmup = 3;
    bool profile = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--input-dump=") == 0) {
            input_dump = arg.substr(13);
        } else if (arg == "--synthetic") {
            synthetic = true;
        } else if (arg.find("--n-candidates=") == 0) {
            n_candidates = std::stoull(arg.substr(15));
        } else if (arg.find("--conflict-density=") == 0) {
            conflict_density = arg.substr(19);
        } else if (arg.find("--backend=") == 0) {
            backend = arg.substr(10);
        } else if (arg.find("--repeats=") == 0) {
            repeats = std::stoull(arg.substr(10));
        } else if (arg.find("--warmup=") == 0) {
            warmup = std::stoull(arg.substr(9));
        } else if (arg == "--profile") {
            profile = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "benchmark_resolver: Resolver-only benchmark\n"
                      << "  --input-dump=<path>   Load from JSON dump\n"
                      << "  --synthetic           Generate synthetic data\n"
                      << "  --n-candidates=N      For synthetic (default 10000)\n"
                      << "  --conflict-density=   low|med|high (default med)\n"
                      << "  --backend=cpu|gpu     (default cpu)\n"
                      << "  --repeats=N           (default 10)\n"
                      << "  --warmup=N            (default 3)\n"
                      << "  --profile             Per-phase timing breakdown (1 warmup + 1 pass)\n";
            return 0;
        }
    }

    if (backend != "cpu") {
        std::cerr << "Only --backend=cpu is supported in this build\n";
        return 1;
    }

    vecmem::host_memory_resource host_mr;
    traccc::ambiguity_resolution_config config;
    traccc::edm::track_container<traccc::default_algebra>::host track_candidates{
        host_mr};
    traccc::edm::track_container<traccc::default_algebra>::host* input_tracks =
        &track_candidates;

    std::optional<traccc::io::ambiguity_input_data> dump_data;
    std::optional<traccc::edm::measurement_collection<
        traccc::default_algebra>::host>
        synthetic_measurements;
    if (!input_dump.empty()) {
        dump_data = traccc::io::read_ambiguity_input(input_dump, host_mr);
        input_tracks = &dump_data->tracks;
        config = dump_data->config;
    } else if (synthetic) {
        traccc::measurement_id_type max_meas_id = 10000;
        std::array<std::size_t, 2> trk_len_range = {3, 10};
        if (conflict_density == "low") {
            max_meas_id = 50000;
            trk_len_range = {3, 10};
        } else if (conflict_density == "high") {
            max_meas_id = 500;
            trk_len_range = {5, 15};
        }

        synthetic_measurements.emplace(host_mr);
        fill_measurements(*synthetic_measurements, max_meas_id);
        track_candidates.measurements =
            vecmem::get_data(*synthetic_measurements);

        std::mt19937 gen(42);
        std::uniform_int_distribution<std::size_t> len_dist(trk_len_range[0],
                                                            trk_len_range[1]);
        std::uniform_int_distribution<traccc::measurement_id_type> id_dist(
            0, max_meas_id);
        std::uniform_real_distribution<traccc::scalar> pval_dist(0.0f, 1.0f);

        for (std::size_t i = 0; i < n_candidates; ++i) {
            std::size_t len = len_dist(gen);
            traccc::scalar pval = pval_dist(gen);
            std::vector<traccc::measurement_id_type> pattern;
            while (pattern.size() < len) {
                auto id = id_dist(gen);
                if (std::find(pattern.begin(), pattern.end(), id) ==
                    pattern.end()) {
                    pattern.push_back(id);
                }
            }
            std::sort(pattern.begin(), pattern.end());
            fill_pattern(track_candidates, pval, pattern);
        }
    } else {
        std::cerr << "Use --input-dump=<path> or --synthetic\n";
        return 1;
    }

    traccc::host::greedy_ambiguity_resolution_algorithm resolver(config,
                                                                  host_mr);

    for (std::size_t w = 0; w < warmup; ++w) {
        resolver(traccc::edm::track_container<
                 traccc::default_algebra>::const_data(*input_tracks));
    }

    std::vector<double> times_ms;
    std::string first_hash;
    std::size_t n_selected = 0;

    for (std::size_t r = 0; r < repeats; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = resolver(
            traccc::edm::track_container<
                traccc::default_algebra>::const_data(*input_tracks));
        auto end = std::chrono::high_resolution_clock::now();
        double ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        times_ms.push_back(ms);
        n_selected = result.tracks.size();
        std::string h = compute_output_hash(result);
        if (r == 0) {
            first_hash = h;
        } else if (h != first_hash) {
            std::cerr << "WARNING: non-deterministic output at repeat " << r
                      << "\n";
        }
    }

    double mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) /
                     static_cast<double>(repeats);
    double sum_sq = 0;
    for (double t : times_ms) sum_sq += (t - mean_ms) * (t - mean_ms);
    double std_ms = std::sqrt(sum_sq / static_cast<double>(repeats));
    std::nth_element(times_ms.begin(), times_ms.begin() + repeats / 2,
                    times_ms.end());
    double median_ms = times_ms[repeats / 2];
    std::size_t p95_idx =
        static_cast<std::size_t>(static_cast<double>(repeats) * 0.95);
    if (p95_idx >= repeats) p95_idx = repeats - 1;
    std::nth_element(times_ms.begin(),
                    times_ms.begin() + static_cast<std::ptrdiff_t>(p95_idx),
                    times_ms.end());
    double p95_ms = times_ms[p95_idx];

    std::size_t n_input = input_tracks->tracks.size();
    double peak_mb = get_peak_rss_mb();

    std::cout << "n_candidates=" << n_input << " n_selected=" << n_selected
              << " n_removed=" << (n_input - n_selected) << "\n"
              << "time_ms_mean=" << mean_ms << " time_ms_std=" << std_ms
              << " time_ms_median=" << median_ms << " time_ms_p95=" << p95_ms
              << "\n"
              << "events_per_sec=" << (1000.0 / mean_ms) << "\n"
              << "peak_memory_mb=" << peak_mb << "\n"
              << "output_hash=" << first_hash << "\n";

    if (profile) {
        // one warmup so caches are warm before the single profiling pass
        resolver(traccc::edm::track_container<
                 traccc::default_algebra>::const_data(*input_tracks));

        const auto prof = run_with_phase_timing(
            traccc::edm::track_container<
                traccc::default_algebra>::const_data(*input_tracks),
            config, host_mr);

        const bool hash_ok = (prof.output_hash == first_hash);
        if (!hash_ok)
            std::cerr << "WARNING: profiling path produced different hash\n";

        std::cout << "profile_filter_setup_ms="   << prof.filter_setup    << "\n"
                  << "profile_unique_meas_ms="    << prof.unique_meas     << "\n"
                  << "profile_inverted_index_ms=" << prof.inverted_index  << "\n"
                  << "profile_shared_count_ms="   << prof.shared_count    << "\n"
                  << "profile_initial_sort_ms="   << prof.initial_sort    << "\n"
                  << "profile_eviction_loop_ms="  << prof.eviction_loop   << "\n"
                  << "profile_output_copy_ms="    << prof.output_copy     << "\n"
                  << "profile_eviction_iterations="  << prof.n_iterations      << "\n"
                  << "profile_eviction_shared_updates=" << prof.n_shared_updates << "\n"
                  << "profile_unique_meas_count=" << prof.unique_meas_count << "\n"
                  << "profile_hash_match="        << (hash_ok ? "true" : "false") << "\n";
    }

    return 0;
}
