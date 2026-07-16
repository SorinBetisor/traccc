/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/ambiguity_resolution/ambiguity_resolution_config.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/track_container.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <functional>

namespace traccc::host {

/// Jones–Plassmann graph-colouring ambiguity resolver (CPU).
///
/// Each outer iteration builds a conflict graph over the currently accepted
/// tracks (two tracks conflict if they share at least one measurement) and
/// applies one round of JP colouring to select a batch of tracks for removal.
/// A track enters the independent set — and is removed — if its sort priority
/// (position in the rel_shared / pval ordering, worst = highest) exceeds that
/// of every conflicting neighbour, and it has at least one such neighbour.
/// Its surviving neighbours are marked REMOVED_NEIGHBOR and skipped for this
/// round; they participate again in the next outer iteration.
///
/// The algorithm converges to the same result as greedy on real FATRAS /
/// Geant4 geometries and is strictly faster at high pile-up, because it
/// removes O(batch) tracks per outer iteration instead of one.
///
/// Interface is identical to greedy_ambiguity_resolution_algorithm so the
/// two can be swapped without changing call sites.
class jp_ambiguity_resolution_algorithm
    : public algorithm<edm::track_container<default_algebra>::host(
          const edm::track_container<default_algebra>::const_view&)>,
      public messaging {

    public:
    using config_type = ambiguity_resolution_config;

    jp_ambiguity_resolution_algorithm(
        const config_type& cfg, vecmem::memory_resource& mr,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone())
        : messaging(std::move(logger)), m_config{cfg}, m_mr{mr} {}

    output_type operator()(
        const edm::track_container<default_algebra>::const_view& tracks)
        const override;

    config_type& get_config() { return m_config; }

    private:
    config_type m_config;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace traccc::host
