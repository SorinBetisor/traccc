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

}  // namespace

int main(int argc, char* argv[]) {
    std::string input_dump;
    bool synthetic = false;
    std::size_t n_candidates = 10000;
    std::string conflict_density = "med";
    std::string backend = "cpu";
    std::size_t repeats = 10;
    std::size_t warmup = 3;

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
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "benchmark_resolver: Resolver-only benchmark\n"
                      << "  --input-dump=<path>   Load from JSON dump\n"
                      << "  --synthetic           Generate synthetic data\n"
                      << "  --n-candidates=N      For synthetic (default 10000)\n"
                      << "  --conflict-density=   low|med|high (default med)\n"
                      << "  --backend=cpu|gpu     (default cpu)\n"
                      << "  --repeats=N           (default 10)\n"
                      << "  --warmup=N            (default 3)\n";
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
    return 0;
}
