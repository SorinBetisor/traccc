/** TRACCC library, part of the ACTS project (R&D line)
 *
 * Mozilla Public License Version 2.0
 *
 * GPU resolver benchmark harness.  Matches the output format of
 * benchmark_resolver.cpp (CPU) so results can be compared directly.  Extra
 * fields: backend=gpu, time_h2d_ms, time_d2h_ms, cpu_hash, gpu_hash,
 * hash_match.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>
#include <vector>

#include "traccc/ambiguity_resolution/ambiguity_resolution_config.hpp"
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/edm/track_container.hpp"
#include "traccc/io/ambiguity_io.hpp"
#include "traccc/utils/memory_resource.hpp"

#ifdef __linux__
#include <sys/resource.h>
#endif

namespace {

void fill_measurements(traccc::edm::measurement_collection::host& m,
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
    traccc::edm::measurement_collection::const_device meas{tc.measurements};
    for (traccc::measurement_id_type meas_id : pattern) {
        auto it = std::lower_bound(meas.identifier().begin(),
                                   meas.identifier().end(), meas_id);
        auto idx = static_cast<traccc::measurement_id_type>(
            std::distance(meas.identifier().begin(), it));
        tc.tracks.constituent_links().back().push_back(
            {traccc::edm::track_constituent_link::measurement, idx});
    }
}

/// Stable, portable FNV-1a 64-bit hash of a string.
/// std::hash<std::string> is explicitly non-portable across compilers and
/// standard library versions. FNV-1a is deterministic, widely documented,
/// and produces the same output on any conforming C++17 implementation.
std::uint64_t fnv1a_64(const std::string& s) {
    constexpr std::uint64_t FNV_OFFSET = 14695981039346656037ULL;
    constexpr std::uint64_t FNV_PRIME  = 1099511628211ULL;
    std::uint64_t h = FNV_OFFSET;
    for (char raw : s) {
        h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(raw));
        h *= FNV_PRIME;
    }
    return h;
}

/// Compute a stable, portable hash of the selected-track measurement patterns.
/// The hash identifies the **selection** (set of tracks by sorted
/// measurement-id pattern), not the raw output bytes. Two runs producing the
/// same selection but in different order will have the same hash.
/// Works on a host track_container (CPU reference or after D2H).
std::string compute_hash_host(
    const traccc::edm::track_container<traccc::default_algebra>::host& out) {
    std::vector<std::vector<traccc::measurement_id_type>> patterns;
    traccc::edm::measurement_collection::const_device meas{out.measurements};
    for (std::size_t i = 0; i < out.tracks.size(); ++i) {
        std::vector<traccc::measurement_id_type> p;
        for (const auto& [type, idx] : out.tracks.at(i).constituent_links()) {
            if (type == traccc::edm::track_constituent_link::measurement)
                p.push_back(meas.at(idx).identifier());
        }
        std::sort(p.begin(), p.end());
        patterns.push_back(std::move(p));
    }
    std::sort(patterns.begin(), patterns.end());
    std::ostringstream oss;
    for (const auto& p : patterns) {
        for (auto id : p)
            oss << id << ",";
        oss << ";";
    }
    return std::to_string(fnv1a_64(oss.str()));
}

/// Compute hash from a track_container buffer that has been copied to host
/// memory. The measurements are referenced via the original host collection.
std::string compute_hash_buffer(
    const traccc::edm::track_container<traccc::default_algebra>::buffer& buf) {
    traccc::edm::track_container<traccc::default_algebra>::const_device dev{
        buf};
    const traccc::edm::measurement_collection::const_device meas{
        buf.measurements};
    std::vector<std::vector<traccc::measurement_id_type>> patterns;
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(dev.tracks.size());
         ++i) {
        std::vector<traccc::measurement_id_type> p;
        for (const auto& [type, idx] : dev.tracks.at(i).constituent_links()) {
            if (type == traccc::edm::track_constituent_link::measurement)
                p.push_back(meas.at(idx).identifier());
        }
        std::sort(p.begin(), p.end());
        patterns.push_back(std::move(p));
    }
    std::sort(patterns.begin(), patterns.end());
    std::ostringstream oss;
    for (const auto& p : patterns) {
        for (auto id : p)
            oss << id << ",";
        oss << ";";
    }
    return std::to_string(fnv1a_64(oss.str()));
}

/// Extract the set of sorted measurement-id patterns that identify the
/// selected tracks. Two tracks are considered "the same selection" iff they
/// have the same sorted measurement pattern.
std::set<std::vector<traccc::measurement_id_type>> extract_patterns_host(
    const traccc::edm::track_container<traccc::default_algebra>::host& out) {
    std::set<std::vector<traccc::measurement_id_type>> patterns;
    traccc::edm::measurement_collection::const_device meas{out.measurements};
    for (std::size_t i = 0; i < out.tracks.size(); ++i) {
        std::vector<traccc::measurement_id_type> p;
        for (const auto& [type, idx] : out.tracks.at(i).constituent_links()) {
            if (type == traccc::edm::track_constituent_link::measurement)
                p.push_back(meas.at(idx).identifier());
        }
        std::sort(p.begin(), p.end());
        patterns.insert(std::move(p));
    }
    return patterns;
}

std::set<std::vector<traccc::measurement_id_type>> extract_patterns_buffer(
    const traccc::edm::track_container<traccc::default_algebra>::buffer& buf) {
    std::set<std::vector<traccc::measurement_id_type>> patterns;
    traccc::edm::track_container<traccc::default_algebra>::const_device dev{
        buf};
    const traccc::edm::measurement_collection::const_device meas{
        buf.measurements};
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(dev.tracks.size());
         ++i) {
        std::vector<traccc::measurement_id_type> p;
        for (const auto& [type, idx] : dev.tracks.at(i).constituent_links()) {
            if (type == traccc::edm::track_constituent_link::measurement)
                p.push_back(meas.at(idx).identifier());
        }
        std::sort(p.begin(), p.end());
        patterns.insert(std::move(p));
    }
    return patterns;
}

/// Track-selection overlap: |gpu ∩ cpu| / |cpu|. A quality proxy that works
/// when the GPU output is not byte-identical to the CPU reference (as is the
/// case for parallel batch greedy by design).
double track_selection_overlap(
    const std::set<std::vector<traccc::measurement_id_type>>& gpu,
    const std::set<std::vector<traccc::measurement_id_type>>& cpu) {
    if (cpu.empty()) {
        return 1.0;
    }
    std::size_t inter = 0;
    for (const auto& p : gpu) {
        if (cpu.count(p) > 0) {
            ++inter;
        }
    }
    return static_cast<double>(inter) / static_cast<double>(cpu.size());
}

/// Post-resolve duplicate rate = (# measurement slots that are shared by >= 2
/// accepted tracks) / (total # distinct measurements used by accepted tracks).
/// Range [0, 1]; lower is cleaner (fewer remaining ambiguities). Computed
/// identically for CPU and GPU so the two are directly comparable.
double duplicate_rate(
    const std::set<std::vector<traccc::measurement_id_type>>& patterns) {
    std::map<traccc::measurement_id_type, std::size_t> count;
    for (const auto& p : patterns) {
        for (auto id : p) {
            ++count[id];
        }
    }
    if (count.empty()) {
        return 0.0;
    }
    std::size_t shared = 0;
    for (const auto& [_, n] : count) {
        if (n >= 2) {
            ++shared;
        }
    }
    return static_cast<double>(shared) / static_cast<double>(count.size());
}

double get_peak_rss_mb() {
#ifdef __linux__
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0)
        return static_cast<double>(ru.ru_maxrss) / 1024.0;
#endif
    return 0.0;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string input_dump;
    bool synthetic = false;
    std::size_t n_candidates = 10000;
    std::string conflict_density = "med";
    std::size_t repeats = 10;
    std::size_t warmup = 3;
    bool profile_mode = false;
    unsigned int n_it_max = 100u;
    bool adaptive_n_it = true;
    bool run_graph_jp = false;
    std::string log_graph_sizes_path;
    std::string log_graph_batches_path;
    std::size_t determinism_runs = 0;
    std::string truth_file;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--input-dump=") == 0)
            input_dump = arg.substr(13);
        else if (arg == "--synthetic")
            synthetic = true;
        else if (arg.find("--n-candidates=") == 0)
            n_candidates = std::stoull(arg.substr(15));
        else if (arg.find("--conflict-density=") == 0)
            conflict_density = arg.substr(19);
        else if (arg.find("--repeats=") == 0)
            repeats = std::stoull(arg.substr(10));
        else if (arg.find("--warmup=") == 0)
            warmup = std::stoull(arg.substr(9));
        else if (arg == "--profile")
            profile_mode = true;
        else if (arg.find("--n-it=") == 0) {
            n_it_max = static_cast<unsigned int>(std::stoull(arg.substr(7)));
            adaptive_n_it = false;
        } else if (arg == "--enable-jp" || arg == "--conflict-graph=jp") {
            run_graph_jp = true;
        } else if (arg.find("--log-graph-sizes=") == 0) {
            log_graph_sizes_path = arg.substr(18);
        } else if (arg.find("--log-graph-batches=") == 0) {
            log_graph_batches_path = arg.substr(20);
        } else if (arg.find("--determinism-runs=") == 0) {
            determinism_runs = std::stoull(arg.substr(19));
        } else if (arg.find("--truth-file=") == 0) {
            truth_file = arg.substr(13);
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "benchmark_resolver_cuda: GPU greedy ambiguity resolver "
                   "benchmark\n"
                << "  --input-dump=<path>   Load pre-frozen JSON dump\n"
                << "  --synthetic           Generate synthetic input\n"
                << "  --n-candidates=N      Track candidates for synthetic "
                   "mode (default 10000)\n"
                << "  --conflict-density=   low|med|high (default med)\n"
                << "  --repeats=N           Timed iterations (default 10)\n"
                << "  --warmup=N            Warmup iterations (default 3)\n"
                << "  --n-it=N              Fix eviction inner-loop iterations "
                   "to N (disables adaptive)\n"
                << "  --profile             Run one extra call with per-phase "
                   "CUDA event timing\n"
                << "  --enable-jp           Also run the explicit-conflict-"
                   "graph Jones-Plassmann backend "
                   "(equivalent to --conflict-graph=jp)\n"
                << "  --log-graph-batches=<path.csv> Write per-outer-iteration "
                   "batch sizes (JP backend)\n"
                << "  --log-graph-sizes=<path.csv>  Write per-outer-iteration "
                   "|V|,|E| (JP backend)\n"
                << "  --determinism-runs=N  Run each enabled GPU backend N "
                   "extra times and assert selection-identical output on every "
                   "run (validates determinism; default 0 = disabled)\n"
                << "  --truth-file=<path>   TSV file with per-track truth "
                   "association: one line per input track, columns: "
                   "track_idx<TAB>particle_id (-1 = fake/unmatched). "
                   "Enables efficiency and fake-rate reporting.\n"
                   "  Truth file format: generated alongside Fatras dumps "
                   "from particles.csv + truth_hits.csv using the companion "
                   "script scripts/make_truth_file.py\n"
                << "\nAdaptive n_it (default, no --n-it):\n"
                << "  n_it per outer step = max(1, min(100, n_accepted/50))\n"
                << "  Gives n=87 -> n_it~1, n=1000 -> n_it~10, n=10000 -> "
                   "n_it=100\n"
                << "\n"
                << "Output fields (same as benchmark_resolver + GPU extras):\n"
                << "  backend, n_candidates, n_selected, n_removed\n"
                << "  time_ms_{mean,std,median,p95}  -- GPU resolver only (no "
                   "transfer)\n"
                << "  events_per_sec, peak_memory_mb\n"
                << "  output_hash (== cpu_hash when correct)\n"
                << "  time_h2d_ms, time_d2h_ms, cpu_hash, gpu_hash, "
                   "hash_match\n";
            return 0;
        }
    }

    // ------------------------------------------------------------------
    // Build host input  (same logic + same seed as benchmark_resolver.cpp)
    // ------------------------------------------------------------------
    vecmem::host_memory_resource host_mr;
    traccc::ambiguity_resolution_config config;
    traccc::edm::track_container<traccc::default_algebra>::host
        track_candidates{host_mr};
    traccc::edm::track_container<traccc::default_algebra>::host* input_tracks =
        &track_candidates;

    std::optional<traccc::io::ambiguity_input_data> dump_data;
    std::optional<traccc::edm::measurement_collection::host>
        synthetic_measurements;

    if (!input_dump.empty()) {
        dump_data = traccc::io::read_ambiguity_input(input_dump, host_mr);
        input_tracks = &dump_data->tracks;
        config = dump_data->config;
        // The GPU resolver requires measurement IDs to form a dense contiguous
        // range [0, n_meas-1]. Real physics dumps have sparse detector-geometry
        // IDs. Reassign identifiers sequentially; constituent links already
        // store collection indices (not raw IDs), so they remain valid.
        for (traccc::measurement_id_type i = 0;
             i < static_cast<traccc::measurement_id_type>(
                     dump_data->measurements.size());
             ++i) {
            dump_data->measurements.at(i).identifier() = i;
        }
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
                    pattern.end())
                    pattern.push_back(id);
            }
            std::sort(pattern.begin(), pattern.end());
            fill_pattern(track_candidates, pval, pattern);
        }
    } else {
        std::cerr << "Use --input-dump=<path> or --synthetic\n";
        return 1;
    }

    // Pointer to host measurement collection (valid for lifetime of benchmark)
    traccc::edm::measurement_collection::host* meas_host_ptr =
        synthetic_measurements ? &(*synthetic_measurements)
                               : &dump_data->measurements;

    const std::size_t n_input = input_tracks->tracks.size();

    // ------------------------------------------------------------------
    // CPU reference pass  (1 warmup + 1 timed, for hash comparison)
    // ------------------------------------------------------------------
    traccc::host::greedy_ambiguity_resolution_algorithm cpu_resolver(config,
                                                                     host_mr);
    cpu_resolver(
        traccc::edm::track_container<traccc::default_algebra>::const_data(
            *input_tracks));
    auto cpu_result = cpu_resolver(
        traccc::edm::track_container<traccc::default_algebra>::const_data(
            *input_tracks));
    const std::string cpu_hash = compute_hash_host(cpu_result);
    const std::size_t n_selected_cpu = cpu_result.tracks.size();

    // ------------------------------------------------------------------
    // CUDA resources
    // ------------------------------------------------------------------
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &host_mr};
    traccc::cuda::stream stream;
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    traccc::cuda::greedy_ambiguity_resolution_algorithm gpu_resolver(
        config, mr, copy, stream);
    gpu_resolver.set_n_it_max(n_it_max);
    gpu_resolver.set_adaptive_n_it(adaptive_n_it);

    // CPU reference patterns for overlap computation.
    const auto cpu_patterns = extract_patterns_host(cpu_result);
    const double cpu_dup_rate = duplicate_rate(cpu_patterns);

    // ------------------------------------------------------------------
    // H2D transfer (timed — single pass, before warmup)
    // ------------------------------------------------------------------
    using clk = std::chrono::high_resolution_clock;
    using ms_dur = std::chrono::duration<double, std::milli>;

    auto h2d_t0 = clk::now();

    auto meas_device_buf =
        copy.to(vecmem::get_data(*meas_host_ptr), device_mr, &host_mr,
                vecmem::copy::type::host_to_device);
    auto tracks_device_buf =
        copy.to(vecmem::get_data(input_tracks->tracks), device_mr, &host_mr,
                vecmem::copy::type::host_to_device);
    stream.synchronize();

    double h2d_ms = ms_dur(clk::now() - h2d_t0).count();

    traccc::edm::track_container<traccc::default_algebra>::buffer device_input{
        std::move(tracks_device_buf), {}, meas_device_buf};

    // ------------------------------------------------------------------
    // Per-backend run helper. Captures warmup + timed loop + D2H + metrics.
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    // Optional truth association (--truth-file).
    // File format: one tab-separated line per input track (in order of
    // track index as written by write_ambiguity_input / synthetic gen):
    //   <track_idx>\t<particle_id>
    // particle_id == -1 means the track is a fake (no truth particle).
    // ------------------------------------------------------------------
    std::vector<long long> truth_particle_ids;  // index = track_idx
    bool has_truth = false;
    if (!truth_file.empty()) {
        std::ifstream tf(truth_file);
        if (!tf) {
            std::cerr << "WARNING: cannot open truth file " << truth_file
                      << " — truth metrics disabled\n";
        } else {
            std::string line;
            while (std::getline(tf, line)) {
                if (line.empty() || line[0] == '#') {
                    continue;
                }
                std::istringstream iss(line);
                std::size_t idx;
                long long pid;
                if (iss >> idx >> pid) {
                    if (idx >= truth_particle_ids.size()) {
                        truth_particle_ids.resize(idx + 1, -2LL);
                    }
                    truth_particle_ids[idx] = pid;
                }
            }
            has_truth = true;
        }
    }

    struct run_metrics {
        std::string label;
        double mean_ms = 0.0, std_ms = 0.0, median_ms = 0.0, p95_ms = 0.0;
        double d2h_ms = 0.0;
        std::size_t n_selected = 0;
        std::string gpu_hash;
        bool hash_match = false;
        double track_overlap_vs_cpu = 0.0;
        double duplicate_rate_post = 0.0;
        // Truth-based metrics (populated only when --truth-file is provided).
        double selection_efficiency = -1.0;  // |selected matched| / |all matched|
        double fake_rate = -1.0;             // |selected fakes| / |selected|
        std::vector<unsigned int> batch_sizes;
        std::vector<std::pair<unsigned int, unsigned int>> graph_sizes;
    };

    using graph_algo_t =
        traccc::cuda::greedy_ambiguity_resolution_algorithm::graph_algo_t;

    auto run_one =
        [&](const std::string& label, graph_algo_t graph_algo,
            std::vector<unsigned int>* batch_log,
            std::vector<std::pair<unsigned int, unsigned int>>* graph_size_log)
        -> run_metrics {
        gpu_resolver.set_conflict_graph_mode(graph_algo);
        gpu_resolver.set_graph_batch_log(
            graph_algo != graph_algo_t::NONE ? batch_log : nullptr);
        gpu_resolver.set_graph_size_log(
            graph_algo != graph_algo_t::NONE ? graph_size_log : nullptr);
        (void)label;

        for (std::size_t w = 0; w < warmup; ++w) {
            gpu_resolver(device_input);
            stream.synchronize();
        }

        std::vector<double> times_ms;
        times_ms.reserve(repeats);
        for (std::size_t r = 0; r < repeats; ++r) {
            auto t0 = clk::now();
            gpu_resolver(device_input);
            stream.synchronize();
            times_ms.push_back(ms_dur(clk::now() - t0).count());
        }

        auto check_result_buf = gpu_resolver(device_input);
        stream.synchronize();

        auto d2h_t0 = clk::now();
        traccc::edm::track_container<traccc::default_algebra>::buffer
            result_host_buf{copy.to(check_result_buf.tracks, host_mr, nullptr,
                                    vecmem::copy::type::device_to_host),
                            {},
                            vecmem::get_data(*meas_host_ptr)};
        stream.synchronize();
        double d2h_ms_local = ms_dur(clk::now() - d2h_t0).count();

        run_metrics m;
        m.label = label;

        m.gpu_hash = compute_hash_buffer(result_host_buf);
        m.n_selected =
            traccc::edm::track_container<traccc::default_algebra>::const_device{
                result_host_buf}
                .tracks.size();
        m.d2h_ms = d2h_ms_local;

        m.mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) /
                    static_cast<double>(repeats);
        double sum_sq = 0;
        for (double t : times_ms)
            sum_sq += (t - m.mean_ms) * (t - m.mean_ms);
        m.std_ms = std::sqrt(sum_sq / static_cast<double>(repeats));

        std::vector<double> sorted_times = times_ms;
        std::nth_element(
            sorted_times.begin(),
            sorted_times.begin() + static_cast<std::ptrdiff_t>(repeats / 2),
            sorted_times.end());
        m.median_ms = sorted_times[repeats / 2];

        std::size_t p95_idx =
            static_cast<std::size_t>(static_cast<double>(repeats) * 0.95);
        if (p95_idx >= repeats)
            p95_idx = repeats - 1;
        std::nth_element(
            sorted_times.begin(),
            sorted_times.begin() + static_cast<std::ptrdiff_t>(p95_idx),
            sorted_times.end());
        m.p95_ms = sorted_times[p95_idx];

        m.hash_match = (m.gpu_hash == cpu_hash);

        auto gpu_patterns = extract_patterns_buffer(result_host_buf);
        m.track_overlap_vs_cpu =
            track_selection_overlap(gpu_patterns, cpu_patterns);
        m.duplicate_rate_post = duplicate_rate(gpu_patterns);

        // Truth-based metrics: requires truth_particle_ids to be populated.
        // Each track is identified by its sorted measurement-id pattern.
        // We map patterns to truth status using the input track order.
        if (has_truth && !truth_particle_ids.empty()) {
            // Build a pattern -> truth_pid map from the input tracks.
            std::map<std::vector<traccc::measurement_id_type>, long long>
                pattern_to_pid;
            traccc::edm::measurement_collection::const_device in_meas{
                input_tracks->measurements};
            for (std::size_t ti = 0; ti < input_tracks->tracks.size(); ++ti) {
                std::vector<traccc::measurement_id_type> p;
                for (const auto& [type, idx] :
                     input_tracks->tracks.at(ti).constituent_links()) {
                    if (type == traccc::edm::track_constituent_link::measurement)
                        p.push_back(in_meas.at(idx).identifier());
                }
                std::sort(p.begin(), p.end());
                long long pid = (ti < truth_particle_ids.size())
                                    ? truth_particle_ids[ti]
                                    : -1LL;
                pattern_to_pid[p] = pid;
            }

            // Count matched (particle_id >= 0) and fake (particle_id == -1)
            // tracks in the input and in the selected set.
            std::size_t n_input_matched = 0;
            for (const auto& [pat, pid] : pattern_to_pid) {
                if (pid >= 0) ++n_input_matched;
            }

            std::size_t n_selected_matched = 0, n_selected_fake = 0;
            for (const auto& pat : gpu_patterns) {
                auto it = pattern_to_pid.find(pat);
                long long pid = (it != pattern_to_pid.end()) ? it->second : -1LL;
                if (pid >= 0)
                    ++n_selected_matched;
                else
                    ++n_selected_fake;
            }

            m.selection_efficiency =
                n_input_matched > 0
                    ? static_cast<double>(n_selected_matched) /
                          static_cast<double>(n_input_matched)
                    : 1.0;
            m.fake_rate =
                m.n_selected > 0
                    ? static_cast<double>(n_selected_fake) /
                          static_cast<double>(m.n_selected)
                    : 0.0;
        }

        if (batch_log != nullptr) {
            m.batch_sizes = *batch_log;
        }
        if (graph_size_log != nullptr) {
            m.graph_sizes = *graph_size_log;
        }
        return m;
    };

    // Baseline (always run).
    const auto baseline =
        run_one("baseline", graph_algo_t::NONE, nullptr, nullptr);

    // Optional Jones-Plassmann conflict-graph run.
    std::optional<run_metrics> graph_jp_run;
    if (run_graph_jp) {
        std::vector<unsigned int> jp_batches;
        std::vector<std::pair<unsigned int, unsigned int>> jp_sizes;
        graph_jp_run =
            run_one("graph_jp", graph_algo_t::JP, &jp_batches, &jp_sizes);
    }

    // ------------------------------------------------------------------
    // Determinism check (optional, --determinism-runs=N)
    // Runs each enabled backend N additional times on the same frozen input
    // and asserts that every run produces a selection-identical output.
    // ------------------------------------------------------------------
    struct determinism_result {
        std::string label;
        std::size_t n_runs  = 0;
        std::size_t n_pass  = 0;
        std::size_t n_fail  = 0;
    };
    std::vector<determinism_result> det_results;

    if (determinism_runs > 0) {
        auto check_determinism =
            [&](const std::string& label, const std::string& reference_hash,
                graph_algo_t graph_algo) -> determinism_result {
            determinism_result dr;
            dr.label = label;
            dr.n_runs = determinism_runs;
            gpu_resolver.set_conflict_graph_mode(graph_algo);
            gpu_resolver.set_graph_batch_log(nullptr);
            gpu_resolver.set_graph_size_log(nullptr);

            for (std::size_t r = 0; r < determinism_runs; ++r) {
                auto det_buf = gpu_resolver(device_input);
                stream.synchronize();
                traccc::edm::track_container<
                    traccc::default_algebra>::buffer det_host{
                    copy.to(det_buf.tracks, host_mr, nullptr,
                            vecmem::copy::type::device_to_host),
                    {},
                    vecmem::get_data(*meas_host_ptr)};
                stream.synchronize();
                const std::string run_hash = compute_hash_buffer(det_host);
                if (run_hash == reference_hash) {
                    ++dr.n_pass;
                } else {
                    ++dr.n_fail;
                    std::cerr << "DETERMINISM FAIL: " << label << " run " << r
                              << " hash=" << run_hash
                              << " expected=" << reference_hash << "\n";
                }
            }
            return dr;
        };

        det_results.push_back(
            check_determinism("baseline", baseline.gpu_hash, graph_algo_t::NONE));
        if (graph_jp_run) {
            det_results.push_back(check_determinism(
                "graph_jp", graph_jp_run->gpu_hash, graph_algo_t::JP));
        }
    }

    double peak_mb = get_peak_rss_mb();

    auto dump_backend_metrics = [&](const run_metrics& m,
                                    const std::string& prefix) {
        std::cout << prefix << "label=" << m.label << "\n"
                  << prefix << "n_selected=" << m.n_selected
                  << " n_removed=" << (n_input - m.n_selected) << "\n"
                  << prefix << "time_ms_mean=" << m.mean_ms
                  << " time_ms_std=" << m.std_ms
                  << " time_ms_median=" << m.median_ms
                  << " time_ms_p95=" << m.p95_ms << "\n"
                  << prefix << "events_per_sec=" << (1000.0 / m.mean_ms) << "\n"
                  << prefix << "time_d2h_ms=" << m.d2h_ms << "\n"
                  << prefix << "output_hash=" << m.gpu_hash << "\n"
                  << prefix
                  << "hash_match=" << (m.hash_match ? "true" : "false") << "\n"
                  << prefix << "track_overlap_vs_cpu=" << m.track_overlap_vs_cpu
                  << "\n"
                  << prefix << "duplicate_rate_post=" << m.duplicate_rate_post
                  << "\n";
        if (m.selection_efficiency >= 0.0) {
            std::cout << prefix
                      << "selection_efficiency=" << m.selection_efficiency << "\n"
                      << prefix << "fake_rate=" << m.fake_rate << "\n";
        }
    };

    std::cout << "backend=gpu\n"
              << "n_it_max=" << n_it_max << "\n"
              << "adaptive_n_it=" << (adaptive_n_it ? "true" : "false") << "\n"
              << "n_candidates=" << n_input
              << " n_selected_cpu=" << n_selected_cpu << "\n"
              << "time_h2d_ms=" << h2d_ms << "\n"
              << "peak_memory_mb=" << peak_mb << "\n"
              << "cpu_hash=" << cpu_hash << "\n"
              << "cpu_duplicate_rate=" << cpu_dup_rate << "\n";

    dump_backend_metrics(baseline, "baseline_");

    auto dump_graph_metrics = [&](const run_metrics& m,
                                  const std::string& prefix) {
        dump_backend_metrics(m, prefix);
        std::cout << prefix << "n_outer_iterations=" << m.batch_sizes.size()
                  << "\n";
        if (!m.batch_sizes.empty()) {
            double sum_b = 0.0;
            unsigned int max_b = 0;
            for (auto b : m.batch_sizes) {
                sum_b += b;
                if (b > max_b)
                    max_b = b;
            }
            std::cout << prefix << "avg_batch_size="
                      << (sum_b / static_cast<double>(m.batch_sizes.size()))
                      << "\n"
                      << prefix << "max_batch_size=" << max_b << "\n";
        }
        if (!m.graph_sizes.empty()) {
            double sum_v = 0.0, sum_e = 0.0;
            unsigned int max_v = 0u, max_e = 0u;
            for (auto& ve : m.graph_sizes) {
                sum_v += ve.first;
                sum_e += ve.second;
                if (ve.first > max_v)
                    max_v = ve.first;
                if (ve.second > max_e)
                    max_e = ve.second;
            }
            const double n_iter = static_cast<double>(m.graph_sizes.size());
            std::cout << prefix << "avg_vertices=" << (sum_v / n_iter) << "\n"
                      << prefix << "avg_edges=" << (sum_e / n_iter) << "\n"
                      << prefix << "max_vertices=" << max_v << "\n"
                      << prefix << "max_edges=" << max_e << "\n";
        }
    };

    if (graph_jp_run) {
        dump_graph_metrics(*graph_jp_run, "graph_jp_");
        if (!log_graph_batches_path.empty()) {
            std::ofstream f(log_graph_batches_path + ".jp.csv");
            f << "outer_iter,batch_size\n";
            for (std::size_t i = 0; i < graph_jp_run->batch_sizes.size(); ++i) {
                f << i << "," << graph_jp_run->batch_sizes[i] << "\n";
            }
        }
        if (!log_graph_sizes_path.empty()) {
            std::ofstream f(log_graph_sizes_path + ".jp.csv");
            f << "outer_iter,n_vertices,n_edges\n";
            for (std::size_t i = 0; i < graph_jp_run->graph_sizes.size(); ++i) {
                f << i << "," << graph_jp_run->graph_sizes[i].first << ","
                  << graph_jp_run->graph_sizes[i].second << "\n";
            }
        }
    }

    // Legacy single-backend field names preserved for downstream scripts that
    // parse the pre-PBG format. They always reflect the baseline run.
    std::cout << "n_selected=" << baseline.n_selected
              << " n_removed=" << (n_input - baseline.n_selected) << "\n"
              << "time_ms_mean=" << baseline.mean_ms
              << " time_ms_std=" << baseline.std_ms
              << " time_ms_median=" << baseline.median_ms
              << " time_ms_p95=" << baseline.p95_ms << "\n"
              << "events_per_sec=" << (1000.0 / baseline.mean_ms) << "\n"
              << "output_hash=" << baseline.gpu_hash << "\n"
              << "time_d2h_ms=" << baseline.d2h_ms << "\n"
              << "gpu_hash=" << baseline.gpu_hash << "\n"
              << "hash_match=" << (baseline.hash_match ? "true" : "false")
              << "\n";

    // Determinism summary output.
    if (!det_results.empty()) {
        std::cout << "determinism_runs=" << determinism_runs << "\n";
        bool all_pass = true;
        for (const auto& dr : det_results) {
            std::cout << "det_" << dr.label << "_pass=" << dr.n_pass
                      << " det_" << dr.label << "_fail=" << dr.n_fail << "\n";
            if (dr.n_fail > 0) {
                all_pass = false;
            }
        }
        std::cout << "determinism_all_pass=" << (all_pass ? "true" : "false")
                  << "\n";
        if (!all_pass) {
            std::cerr << "DETERMINISM: some runs produced non-identical "
                         "selections — see det_*_fail fields above.\n";
        }
    }

    if (!baseline.hash_match)
        std::cerr
            << "WARNING: GPU baseline hash does not match CPU reference\n";
    if (baseline.n_selected != n_selected_cpu)
        std::cerr << "WARNING: GPU baseline selected " << baseline.n_selected
                  << " tracks but CPU selected " << n_selected_cpu << "\n";
    // Keep shims so the rest of the original output block (profile section)
    // below still compiles.
    const bool hash_match = baseline.hash_match;
    const std::string gpu_hash = baseline.gpu_hash;
    (void)hash_match;
    (void)gpu_hash;

    if (profile_mode) {
        gpu_resolver.set_profiling(true);
        gpu_resolver(device_input);
        stream.synchronize();
        gpu_resolver.set_profiling(false);

        const traccc::cuda::gpu_profile_data_t& pd =
            gpu_resolver.last_profile();
        std::cout << "profile_filter_setup_ms=" << pd.filter_setup_ms << "\n"
                  << "profile_unique_meas_ms=" << pd.unique_meas_ms << "\n"
                  << "profile_inverted_index_ms=" << pd.inverted_index_ms
                  << "\n"
                  << "profile_shared_count_ms=" << pd.shared_count_ms << "\n"
                  << "profile_initial_sort_ms=" << pd.initial_sort_ms << "\n"
                  << "profile_eviction_loop_ms=" << pd.eviction_loop_ms << "\n"
                  << "profile_output_copy_ms=" << pd.output_copy_ms << "\n"
                  << "profile_eviction_graph_launches="
                  << pd.eviction_graph_launches << "\n"
                  << "profile_unique_meas_count=" << pd.unique_meas_count
                  << "\n"
                  << "profile_hash_match=" << (hash_match ? "true" : "false")
                  << "\n";
    }

    return 0;
}
