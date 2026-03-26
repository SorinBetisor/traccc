/**
 * GPU resolver benchmark harness.
 * Matches the output format of benchmark_resolver.cpp (CPU) so results can be
 * compared directly.  Extra fields: backend=gpu, time_h2d_ms, time_d2h_ms,
 * cpu_hash, gpu_hash, hash_match.
 */

#include "traccc/ambiguity_resolution/ambiguity_resolution_config.hpp"
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/edm/track_container.hpp"
#include "traccc/io/ambiguity_io.hpp"
#include "traccc/utils/memory_resource.hpp"

#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
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

/// Compute a deterministic hash of the selected-track measurement patterns.
/// Works on a host track_container (CPU reference or after D2H).
std::string compute_hash_host(
    const traccc::edm::track_container<traccc::default_algebra>::host& out) {
    std::vector<std::vector<traccc::measurement_id_type>> patterns;
    traccc::edm::measurement_collection<traccc::default_algebra>::const_device
        meas{out.measurements};
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
        for (auto id : p) oss << id << ",";
        oss << ";";
    }
    return std::to_string(std::hash<std::string>{}(oss.str()));
}

/// Compute hash from a track_container buffer that has been copied to host
/// memory. The measurements are referenced via the original host collection.
std::string compute_hash_buffer(
    const traccc::edm::track_container<traccc::default_algebra>::buffer& buf) {
    traccc::edm::track_container<traccc::default_algebra>::const_device dev{
        buf};
    const traccc::edm::measurement_collection<
        traccc::default_algebra>::const_device meas{buf.measurements};
    std::vector<std::vector<traccc::measurement_id_type>> patterns;
    for (std::uint32_t i = 0;
         i < static_cast<std::uint32_t>(dev.tracks.size()); ++i) {
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
        for (auto id : p) oss << id << ",";
        oss << ";";
    }
    return std::to_string(std::hash<std::string>{}(oss.str()));
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
    bool synthetic       = false;
    std::size_t n_candidates   = 10000;
    std::string conflict_density = "med";
    std::size_t repeats  = 10;
    std::size_t warmup   = 3;

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
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "benchmark_resolver_cuda: GPU greedy ambiguity resolver benchmark\n"
                << "  --input-dump=<path>   Load pre-frozen JSON dump\n"
                << "  --synthetic           Generate synthetic input\n"
                << "  --n-candidates=N      Track candidates for synthetic mode (default 10000)\n"
                << "  --conflict-density=   low|med|high (default med)\n"
                << "  --repeats=N           Timed iterations (default 10)\n"
                << "  --warmup=N            Warmup iterations (default 3)\n"
                << "\n"
                << "Output fields (same as benchmark_resolver + GPU extras):\n"
                << "  backend, n_candidates, n_selected, n_removed\n"
                << "  time_ms_{mean,std,median,p95}  -- GPU resolver only (no transfer)\n"
                << "  events_per_sec, peak_memory_mb\n"
                << "  output_hash (== cpu_hash when correct)\n"
                << "  time_h2d_ms, time_d2h_ms, cpu_hash, gpu_hash, hash_match\n";
            return 0;
        }
    }

    // ------------------------------------------------------------------
    // Build host input  (same logic + same seed as benchmark_resolver.cpp)
    // ------------------------------------------------------------------
    vecmem::host_memory_resource host_mr;
    traccc::ambiguity_resolution_config config;
    traccc::edm::track_container<traccc::default_algebra>::host track_candidates{
        host_mr};
    traccc::edm::track_container<traccc::default_algebra>::host* input_tracks =
        &track_candidates;

    std::optional<traccc::io::ambiguity_input_data> dump_data;
    std::optional<
        traccc::edm::measurement_collection<traccc::default_algebra>::host>
        synthetic_measurements;

    if (!input_dump.empty()) {
        dump_data = traccc::io::read_ambiguity_input(input_dump, host_mr);
        input_tracks = &dump_data->tracks;
        config       = dump_data->config;
    } else if (synthetic) {
        traccc::measurement_id_type max_meas_id = 10000;
        std::array<std::size_t, 2> trk_len_range = {3, 10};
        if (conflict_density == "low") {
            max_meas_id   = 50000;
            trk_len_range = {3, 10};
        } else if (conflict_density == "high") {
            max_meas_id   = 500;
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
    traccc::edm::measurement_collection<traccc::default_algebra>::host*
        meas_host_ptr = synthetic_measurements
                            ? &(*synthetic_measurements)
                            : &dump_data->measurements;

    const std::size_t n_input = input_tracks->tracks.size();

    // ------------------------------------------------------------------
    // CPU reference pass  (1 warmup + 1 timed, for hash comparison)
    // ------------------------------------------------------------------
    traccc::host::greedy_ambiguity_resolution_algorithm cpu_resolver(config,
                                                                     host_mr);
    cpu_resolver(traccc::edm::track_container<
                 traccc::default_algebra>::const_data(*input_tracks));
    auto cpu_result = cpu_resolver(traccc::edm::track_container<
                                   traccc::default_algebra>::const_data(*input_tracks));
    const std::string cpu_hash     = compute_hash_host(cpu_result);
    const std::size_t n_selected_cpu = cpu_result.tracks.size();

    // ------------------------------------------------------------------
    // CUDA resources
    // ------------------------------------------------------------------
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &host_mr};
    traccc::cuda::stream stream;
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    traccc::cuda::greedy_ambiguity_resolution_algorithm gpu_resolver(config, mr,
                                                                     copy,
                                                                     stream);

    // ------------------------------------------------------------------
    // H2D transfer (timed — single pass, before warmup)
    // ------------------------------------------------------------------
    using clk   = std::chrono::high_resolution_clock;
    using ms_dur = std::chrono::duration<double, std::milli>;

    auto h2d_t0 = clk::now();

    auto meas_device_buf = copy.to(vecmem::get_data(*meas_host_ptr), device_mr,
                                   &host_mr,
                                   vecmem::copy::type::host_to_device);
    auto tracks_device_buf =
        copy.to(vecmem::get_data(input_tracks->tracks), device_mr, &host_mr,
                vecmem::copy::type::host_to_device);
    stream.synchronize();

    double h2d_ms = ms_dur(clk::now() - h2d_t0).count();

    traccc::edm::track_container<traccc::default_algebra>::buffer device_input{
        std::move(tracks_device_buf), {}, meas_device_buf};

    // ------------------------------------------------------------------
    // Warmup (not timed)
    // ------------------------------------------------------------------
    for (std::size_t w = 0; w < warmup; ++w) {
        gpu_resolver(device_input);
        stream.synchronize();
    }

    // ------------------------------------------------------------------
    // Timed loop  — resolver + synchronize only, no transfers
    // ------------------------------------------------------------------
    std::vector<double> times_ms;
    times_ms.reserve(repeats);

    for (std::size_t r = 0; r < repeats; ++r) {
        auto t0 = clk::now();
        gpu_resolver(device_input);
        stream.synchronize();
        times_ms.push_back(ms_dur(clk::now() - t0).count());
    }

    // ------------------------------------------------------------------
    // Post-timing: one extra GPU call, then D2H for correctness check
    // ------------------------------------------------------------------
    auto check_result_buf = gpu_resolver(device_input);
    stream.synchronize();

    auto d2h_t0 = clk::now();

    traccc::edm::track_container<traccc::default_algebra>::buffer result_host_buf{
        copy.to(check_result_buf.tracks, host_mr, nullptr,
                vecmem::copy::type::device_to_host),
        {},
        vecmem::get_data(*meas_host_ptr)};
    stream.synchronize();

    double d2h_ms = ms_dur(clk::now() - d2h_t0).count();

    const std::string gpu_hash = compute_hash_buffer(result_host_buf);
    const std::size_t n_selected_gpu =
        traccc::edm::track_container<traccc::default_algebra>::const_device{
            result_host_buf}
            .tracks.size();

    // ------------------------------------------------------------------
    // Statistics
    // ------------------------------------------------------------------
    double mean_ms =
        std::accumulate(times_ms.begin(), times_ms.end(), 0.0) /
        static_cast<double>(repeats);
    double sum_sq = 0;
    for (double t : times_ms) sum_sq += (t - mean_ms) * (t - mean_ms);
    double std_ms  = std::sqrt(sum_sq / static_cast<double>(repeats));

    std::vector<double> sorted_times = times_ms;
    std::nth_element(sorted_times.begin(),
                     sorted_times.begin() + static_cast<std::ptrdiff_t>(repeats / 2),
                     sorted_times.end());
    double median_ms = sorted_times[repeats / 2];

    std::size_t p95_idx =
        static_cast<std::size_t>(static_cast<double>(repeats) * 0.95);
    if (p95_idx >= repeats) p95_idx = repeats - 1;
    std::nth_element(sorted_times.begin(),
                     sorted_times.begin() + static_cast<std::ptrdiff_t>(p95_idx),
                     sorted_times.end());
    double p95_ms = sorted_times[p95_idx];

    double peak_mb       = get_peak_rss_mb();
    bool   hash_match    = (gpu_hash == cpu_hash);

    if (!hash_match)
        std::cerr << "WARNING: GPU output hash does not match CPU reference\n";
    if (n_selected_gpu != n_selected_cpu)
        std::cerr << "WARNING: GPU selected " << n_selected_gpu
                  << " tracks but CPU selected " << n_selected_cpu << "\n";

    // ------------------------------------------------------------------
    // Output  (same key names as benchmark_resolver.cpp where applicable)
    // ------------------------------------------------------------------
    std::cout << "backend=gpu\n"
              << "n_candidates=" << n_input
              << " n_selected=" << n_selected_gpu
              << " n_removed=" << (n_input - n_selected_gpu) << "\n"
              << "time_ms_mean="   << mean_ms
              << " time_ms_std="   << std_ms
              << " time_ms_median=" << median_ms
              << " time_ms_p95="   << p95_ms  << "\n"
              << "events_per_sec=" << (1000.0 / mean_ms) << "\n"
              << "peak_memory_mb=" << peak_mb << "\n"
              << "output_hash="    << gpu_hash << "\n"
              << "time_h2d_ms="    << h2d_ms   << "\n"
              << "time_d2h_ms="    << d2h_ms   << "\n"
              << "cpu_hash="       << cpu_hash  << "\n"
              << "gpu_hash="       << gpu_hash  << "\n"
              << "hash_match="     << (hash_match ? "true" : "false") << "\n";

    return 0;
}
