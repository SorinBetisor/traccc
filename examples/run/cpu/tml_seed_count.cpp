/** Minimal TML seeder: reads spacepoints from a tml_full event directory,
 *  runs traccc seeding, and prints seed count. Stops before
 * track_params_estimation so it works without a valid detray detector geometry
 * for the TML format.
 */

#include <cstdlib>
#include <iostream>
#include <string>
#include <vecmem/memory/host_memory_resource.hpp>

#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/seeding/detail/seed_finding.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_binning.hpp"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: tml_seed_count <data_dir> <n_events>\n"
                  << "  e.g. tml_seed_count tml_full/ttbar_mu200 10\n";
        return EXIT_FAILURE;
    }

    const std::string data_dir = argv[1];
    const std::size_t n_events = static_cast<std::size_t>(std::atoi(argv[2]));

    vecmem::host_memory_resource host_mr;

    traccc::seedfinder_config sf_cfg;
    sf_cfg.zMin = -1186.f * traccc::unit<float>::mm;
    sf_cfg.zMax = 1186.f * traccc::unit<float>::mm;
    sf_cfg.cotThetaMax = 7.40627f;
    sf_cfg.deltaRMin = 1.f * traccc::unit<float>::mm;
    sf_cfg.deltaRMax = 60.f * traccc::unit<float>::mm;
    sf_cfg.sigmaScattering = 1.0f;
    sf_cfg.deltaZMax = 1000000.f * traccc::unit<float>::mm;

    const traccc::seedfilter_config filt_cfg;
    const traccc::spacepoint_grid_config grid_cfg(sf_cfg);

    traccc::host::details::spacepoint_binning sb{sf_cfg, grid_cfg, host_mr};
    traccc::host::details::seed_finding seed_find{sf_cfg, filt_cfg, host_mr};

    uint64_t total_sp = 0;
    uint64_t total_seeds = 0;

    std::cout << "event,n_spacepoints,n_seeds\n";

    for (std::size_t ev = 0; ev < n_events; ++ev) {
        traccc::edm::spacepoint_collection::host sp{host_mr};
        traccc::edm::measurement_collection<traccc::default_algebra>::host meas{
            host_mr};

        traccc::io::read_spacepoints(sp, meas, ev, data_dir, nullptr, nullptr,
                                     traccc::data_format::csv);

        auto internal_sp = sb(vecmem::get_data(sp));
        auto seeds = seed_find(vecmem::get_data(sp), internal_sp);

        std::cout << ev << "," << sp.size() << "," << seeds.size() << "\n";
        total_sp += sp.size();
        total_seeds += seeds.size();
    }

    std::cout << "\nmean spacepoints/event : " << total_sp / n_events << "\n"
              << "mean seeds/event       : " << total_seeds / n_events << "\n";

    return EXIT_SUCCESS;
}
