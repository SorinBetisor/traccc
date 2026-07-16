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

/// Score-based ambiguity resolver (CPU) — traccc port of ACTS's
/// ScoreBasedAmbiguityResolution.
///
/// One-pass algorithm (in contrast to greedy's iterative removal):
///   1. Filter tracks below `min_meas_per_track`.
///   2. Score every surviving track:
///        score = w_meas * n_meas
///              - w_holes * nholes
///              - w_chi2  * (chi2 / max(1, ndf))
///        + pval as monotone tiebreaker (higher = better).
///   3. Sort by score descending; walk the list and accept a track iff its
///      addition would not push any of its measurements over
///      `max_shared_meas_per_measurement` accepted tracks.
///   4. Apply a final `min_score` cut on accepted tracks.
///
/// This mirrors ACTS's ScoreBased algorithm (Core/Acts/AmbiguityResolution/
/// ScoreBasedAmbiguityResolution.{hpp,ipp}) reduced to the fields that
/// traccc's v2 ambiguity dump carries (chi2, ndf, nholes, n_measurements).
/// Detector-region-specific weights and eta-binned cuts from the ACTS
/// production version are omitted because our dumps do not carry per-hit
/// region info; adding them is a straightforward extension once regions are
/// serialised.
///
/// Interface matches greedy / JP so it can be swapped without touching
/// call sites.
class score_based_ambiguity_resolution_algorithm
    : public algorithm<edm::track_container<default_algebra>::host(
          const edm::track_container<default_algebra>::const_view&)>,
      public messaging {

    public:
    struct config_type {
        // Inherited AR contract
        unsigned int min_meas_per_track = 3;

        // ScoreBased weights (chosen to reproduce ACTS defaults where
        // possible; overridable per benchmark).
        traccc::scalar w_meas = 1.0f;      ///< positive weight per measurement
        traccc::scalar w_holes = 3.0f;     ///< penalty per hole
        traccc::scalar w_chi2 = 0.5f;      ///< penalty per chi2/ndf unit
        traccc::scalar min_score = 0.0f;   ///< reject tracks scoring below this
        traccc::scalar max_chi2_ndf = 100.0f;

        // Sharing constraint: how many accepted tracks may claim a single
        // measurement. ACTS default = 1 (i.e. no sharing), matching greedy's
        // max_shared_meas = 1 case.
        unsigned int max_shared_tracks_per_measurement = 1;
    };

    score_based_ambiguity_resolution_algorithm(
        const config_type& cfg, vecmem::memory_resource& mr,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone())
        : messaging(std::move(logger)), m_config{cfg}, m_mr{mr} {}

    /// Constructor that adapts an ambiguity_resolution_config (the shared one
    /// used by greedy / JP) into the ScoreBased config, so the benchmark
    /// harness can build all three algorithms from the same dump-config.
    score_based_ambiguity_resolution_algorithm(
        const ambiguity_resolution_config& shared_cfg,
        vecmem::memory_resource& mr,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone())
        : messaging(std::move(logger)), m_mr{mr} {
        m_config.min_meas_per_track = shared_cfg.min_meas_per_track;
        m_config.max_shared_tracks_per_measurement = shared_cfg.max_shared_meas;
    }

    output_type operator()(
        const edm::track_container<default_algebra>::const_view& tracks)
        const override;

    config_type& get_config() { return m_config; }

    private:
    config_type m_config;
    std::reference_wrapper<vecmem::memory_resource> m_mr;
};

}  // namespace traccc::host
