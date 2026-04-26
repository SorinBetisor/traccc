/** TRACCC library, part of the ACTS project (R&D line)
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/io/ambiguity_io.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vecmem/utils/copy.hpp>
#include <vector>

#include "traccc/io/utils.hpp"

namespace traccc::io {

namespace {

void write_ambiguity_input_impl(
    const std::string& path,
    const edm::track_container<default_algebra>::const_device& tracks,
    const ambiguity_resolution_config& config) {

    using size_type =
        edm::track_collection<default_algebra>::const_device::size_type;

    nlohmann::json j;
    j["config"]["min_meas_per_track"] = config.min_meas_per_track;
    j["config"]["max_iterations"] = config.max_iterations;
    j["config"]["max_shared_meas"] = config.max_shared_meas;

    std::set<measurement_id_type> unique_ids;
    using size_type =
        edm::track_collection<default_algebra>::const_device::size_type;
    for (size_type i = 0; i < tracks.tracks.size(); ++i) {
        const auto& tc = tracks.tracks.at(i);
        for (const auto& [type, idx] : tc.constituent_links()) {
            if (type == edm::track_constituent_link::measurement) {
                unique_ids.insert(tracks.measurements.at(idx).identifier());
            }
        }
    }
    j["measurements"] = nlohmann::json::array();
    for (measurement_id_type id : unique_ids) {
        j["measurements"].push_back({{"identifier", id}});
    }

    std::unordered_map<measurement_id_type, std::size_t> id_to_idx;
    std::size_t idx = 0;
    for (measurement_id_type id : unique_ids) {
        id_to_idx[id] = idx++;
    }

    j["tracks"] = nlohmann::json::array();
    for (size_type i = 0; i < tracks.tracks.size(); ++i) {
        const auto& tc = tracks.tracks.at(i);
        nlohmann::json track_json;
        track_json["pval"] = tc.pval();
        track_json["measurement_ids"] = nlohmann::json::array();
        for (const auto& [type, meas_idx] : tc.constituent_links()) {
            if (type == edm::track_constituent_link::measurement) {
                track_json["measurement_ids"].push_back(
                    tracks.measurements.at(meas_idx).identifier());
            }
        }
        j["tracks"].push_back(track_json);
    }

    std::ofstream f(path);
    if (!f) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    f << j.dump(2);
}

}  // namespace

void write_ambiguity_input(
    std::string_view path,
    const edm::track_container<default_algebra>::const_view& tracks_view,
    const ambiguity_resolution_config& config) {

    const edm::track_container<default_algebra>::const_device tracks{
        tracks_view};
    write_ambiguity_input_impl(std::string(path), tracks, config);
}

ambiguity_input_data read_ambiguity_input(std::string_view path,
                                          vecmem::memory_resource& mr) {

    std::string path_str(path);
    std::ifstream f(path_str);
    if (!f) {
        throw std::runtime_error("Failed to open file for reading: " +
                                 path_str);
    }
    nlohmann::json j = nlohmann::json::parse(f);

    ambiguity_input_data result{mr};

    result.config.min_meas_per_track =
        j["config"].value("min_meas_per_track", 3u);
    result.config.max_iterations =
        j["config"].value("max_iterations", 0xFFFFFFFFu);
    result.config.max_shared_meas = j["config"].value("max_shared_meas", 1u);

    result.measurements.reserve(j["measurements"].size());
    for (const auto& m : j["measurements"]) {
        const auto idx_back = result.measurements.size();
        result.measurements.resize(idx_back + 1u);
        edm::measurement meas = result.measurements.at(idx_back);
        meas.identifier() = m["identifier"].get<measurement_id_type>();
    }

    std::unordered_map<measurement_id_type, std::size_t> id_to_idx;
    edm::measurement_collection::const_device meas_dev{
        vecmem::get_data(result.measurements)};
    for (std::size_t i = 0; i < result.measurements.size(); ++i) {
        id_to_idx[meas_dev.at(static_cast<unsigned int>(i)).identifier()] = i;
    }

    result.tracks.measurements = vecmem::get_data(result.measurements);

    for (const auto& t : j["tracks"]) {
        scalar pval = t["pval"].get<scalar>();
        result.tracks.tracks.resize(result.tracks.tracks.size() + 1u);
        result.tracks.tracks.pval().back() = pval;
        for (const auto& mid : t["measurement_ids"]) {
            measurement_id_type id = mid.get<measurement_id_type>();
            auto it = id_to_idx.find(id);
            if (it != id_to_idx.end()) {
                result.tracks.tracks.constituent_links().back().push_back(
                    {edm::track_constituent_link::measurement,
                     static_cast<unsigned int>(it->second)});
            }
        }
    }

    // Remap identifiers to dense [0, N-1].
    //
    // The dump file preserves each measurement's original global identifier
    // (e.g. from a Fatras / Geant4 simulation) so that downstream tools can
    // cross-reference the original event. The GPU greedy resolver, however,
    // requires identifiers to form a contiguous [0, n_measurements - 1]
    // range because it uses them as direct indices into a per-measurement
    // working array (see device/cuda/src/ambiguity_resolution/
    // greedy_ambiguity_resolution_algorithm.cu: the
    // "max measurement id should be equal to (the number of measurements
    // - 1)" invariant). The conflict-graph structure only depends on which
    // tracks share which measurement, not on the absolute identifier
    // value, so remapping to dense indices here preserves the resolver
    // semantics while satisfying the invariant. The constituent_links
    // above already reference dense indices via id_to_idx, so we only
    // need to overwrite the per-measurement identifier field.
    edm::measurement_collection::device meas_dev_mut{
        vecmem::get_data(result.measurements)};
    for (std::size_t i = 0; i < result.measurements.size(); ++i) {
        edm::measurement m = meas_dev_mut.at(static_cast<unsigned int>(i));
        m.identifier() = static_cast<measurement_id_type>(i);
    }

    return result;
}

}  // namespace traccc::io
