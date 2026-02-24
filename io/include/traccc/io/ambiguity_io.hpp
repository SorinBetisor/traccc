/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/ambiguity_resolution/ambiguity_resolution_config.hpp"
#include "traccc/edm/track_container.hpp"

#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

#include <string_view>

namespace traccc::io {

void write_ambiguity_input(
    std::string_view path,
    const edm::track_container<default_algebra>::const_view& tracks,
    const ambiguity_resolution_config& config);

inline void write_ambiguity_input(
    std::string_view path,
    const edm::track_container<default_algebra>::const_data& tracks,
    const ambiguity_resolution_config& config) {
    write_ambiguity_input(path,
                         edm::track_container<default_algebra>::const_view{
                             tracks},
                         config);
}

struct ambiguity_input_data {
    edm::measurement_collection<default_algebra>::host measurements;
    edm::track_container<default_algebra>::host tracks;
    ambiguity_resolution_config config;

    explicit ambiguity_input_data(vecmem::memory_resource& mr)
        : measurements(mr), tracks(mr, vecmem::get_data(measurements)) {}
};

ambiguity_input_data read_ambiguity_input(std::string_view path,
                                          vecmem::memory_resource& mr);

}  // namespace traccc::io
