/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/ambiguity_resolution/score_based_ambiguity_resolution_algorithm.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace traccc::host {

auto score_based_ambiguity_resolution_algorithm::operator()(
    const edm::track_container<default_algebra>::const_view& track_view) const
    -> output_type {

    const edm::track_container<default_algebra>::const_device input_tracks(
        track_view);

    const std::size_t n_tracks = input_tracks.tracks.size();

    edm::track_container<default_algebra>::host output{m_mr.get()};
    output.measurements = track_view.measurements;

    if (n_tracks == 0) {
        return output;
    }

    // ------------------------------------------------------------------ //
    // Phase 1 — score every track                                         //
    // ------------------------------------------------------------------ //

    struct scored {
        unsigned int track_idx;
        traccc::scalar score;
        std::vector<std::size_t> meas_ids;
    };

    std::vector<scored> candidates;
    candidates.reserve(n_tracks);

    for (unsigned int i = 0; i < n_tracks; ++i) {
        const auto tc = input_tracks.tracks.at(i);
        const auto links = tc.constituent_links();
        const unsigned int n_meas = links.size();

        if (n_meas < m_config.min_meas_per_track) {
            continue;
        }

        const traccc::scalar ndf   = tc.ndf();
        const traccc::scalar chi2  = tc.chi2();
        const traccc::scalar pval  = tc.pval();
        const unsigned int nholes  = tc.nholes();

        // chi2/ndf hard cut. In enriched (v2) dumps this suppresses badly
        // fitted tracks; in v1 dumps ndf==0 so we protect the divide and skip
        // the cut in that case.
        const traccc::scalar chi2_per_ndf =
            (ndf > 0.f) ? (chi2 / ndf) : 0.f;
        if (ndf > 0.f && chi2_per_ndf > m_config.max_chi2_ndf) {
            continue;
        }

        // Weighted score. pval is added as a small monotone tiebreaker so
        // two tracks with identical hit / hole / chi2 stats still sort
        // deterministically.
        const traccc::scalar score =
              m_config.w_meas  * static_cast<traccc::scalar>(n_meas)
            - m_config.w_holes * static_cast<traccc::scalar>(nholes)
            - m_config.w_chi2  * chi2_per_ndf
            + 1e-3f * pval;

        if (score < m_config.min_score) {
            continue;
        }

        std::vector<std::size_t> mids;
        mids.reserve(n_meas);
        for (const auto& [type, meas_idx] : links) {
            assert(type == edm::track_constituent_link::measurement);
            mids.push_back(input_tracks.measurements.at(meas_idx).identifier());
        }
        candidates.push_back({i, score, std::move(mids)});
    }

    // ------------------------------------------------------------------ //
    // Phase 2 — sort by score (descending) and accept greedily under      //
    //           the shared-hits constraint                                //
    // ------------------------------------------------------------------ //

    std::sort(candidates.begin(), candidates.end(),
              [](const scored& a, const scored& b) {
                  return a.score > b.score;
              });

    // meas_used_count[measurement_id] = # of accepted tracks touching it.
    std::unordered_map<std::size_t, unsigned int> meas_used_count;
    meas_used_count.reserve(candidates.size() * 8);

    std::vector<unsigned int> accepted_ids;
    accepted_ids.reserve(candidates.size());

    for (const auto& c : candidates) {
        // A track is accepted iff none of its measurements would exceed
        // max_shared_tracks_per_measurement after inclusion.
        bool ok = true;
        for (std::size_t mid : c.meas_ids) {
            auto it = meas_used_count.find(mid);
            const unsigned int cur = (it == meas_used_count.end()) ? 0u
                                                                   : it->second;
            if (cur >= m_config.max_shared_tracks_per_measurement) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }
        for (std::size_t mid : c.meas_ids) {
            ++meas_used_count[mid];
        }
        accepted_ids.push_back(c.track_idx);
    }

    // ------------------------------------------------------------------ //
    // Phase 3 — build output (same shape as greedy / JP)                  //
    // ------------------------------------------------------------------ //

    // Restore ascending track_idx order so downstream tools that assume it
    // (e.g. hash comparison) match greedy/JP semantics.
    std::sort(accepted_ids.begin(), accepted_ids.end());

    output.tracks.reserve(accepted_ids.size());
    for (const unsigned int i : accepted_ids) {
        const edm::track tcand = input_tracks.tracks.at(i);
        output.tracks.push_back({tcand.fit_outcome(),
                                 tcand.params(),
                                 tcand.ndf(),
                                 tcand.chi2(),
                                 tcand.pval(),
                                 tcand.nholes(),
                                 {tcand.constituent_links().begin(),
                                  tcand.constituent_links().end()}});
    }

    return output;
}

}  // namespace traccc::host
