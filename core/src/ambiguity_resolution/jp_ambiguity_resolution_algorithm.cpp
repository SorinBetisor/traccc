/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/ambiguity_resolution/jp_ambiguity_resolution_algorithm.hpp"

#include <algorithm>
#include <cassert>
#include <unordered_set>
#include <vector>

namespace traccc::host {

// JP MIS state codes — mirrors device::graph_mis_state on CPU.
namespace {
constexpr int UNDECIDED        = 0;
constexpr int IN_MIS           = 1;
constexpr int REMOVED_NEIGHBOR = 2;
}  // namespace

auto jp_ambiguity_resolution_algorithm::operator()(
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

    assert(m_config.max_shared_meas > 0u);

    // ------------------------------------------------------------------ //
    // Phase 1 — identical setup to greedy_ambiguity_resolution_algorithm  //
    // ------------------------------------------------------------------ //

    std::vector<unsigned int> accepted_ids(n_tracks);
    std::iota(accepted_ids.begin(), accepted_ids.end(), 0);

    std::vector<std::vector<std::size_t>> meas_ids(n_tracks);
    std::vector<traccc::scalar> pvals(n_tracks);
    std::vector<std::size_t> n_meas(n_tracks, 0);

    for (unsigned int i = 0; i < n_tracks; i++) {
        pvals[i] = input_tracks.tracks.at(i).pval();

        const vecmem::device_vector<const edm::track_constituent_link>
            links = input_tracks.tracks.at(i).constituent_links();
        const unsigned int n_cands = links.size();

        if (n_cands < m_config.min_meas_per_track) {
            const auto it =
                std::lower_bound(accepted_ids.begin(), accepted_ids.end(), i);
            assert(it != accepted_ids.end() && *it == i);
            accepted_ids.erase(it);
        } else {
            meas_ids[i].reserve(n_cands);
            for (const auto [type, idx] : links) {
                assert(type == edm::track_constituent_link::measurement);
                meas_ids[i].push_back(
                    input_tracks.measurements.at(idx).identifier());
            }
            n_meas[i] = n_cands;
        }
    }

    // Sorted unique measurement IDs (needed for tracks_per_measurement lookup).
    std::unordered_set<std::size_t> unique_meas_set;
    for (const unsigned int i : accepted_ids) {
        for (const auto [type, idx] :
             input_tracks.tracks.at(i).constituent_links()) {
            assert(type == edm::track_constituent_link::measurement);
            unique_meas_set.insert(
                input_tracks.measurements.at(idx).identifier());
        }
    }
    std::vector<std::size_t> unique_meas(unique_meas_set.begin(),
                                         unique_meas_set.end());
    std::sort(unique_meas.begin(), unique_meas.end());

    // Inverted index: measurement ID → list of accepted track indices.
    std::vector<std::vector<unsigned int>> tracks_per_measurement(
        unique_meas.size());
    for (const unsigned int i : accepted_ids) {
        std::unordered_set<std::size_t> dedup(meas_ids[i].begin(),
                                              meas_ids[i].end());
        for (const std::size_t mid : dedup) {
            const auto it =
                std::lower_bound(unique_meas.begin(), unique_meas.end(), mid);
            assert(it != unique_meas.end());
            tracks_per_measurement[std::distance(unique_meas.begin(), it)]
                .push_back(i);
        }
    }

    // n_shared[i] = # of this track's measurements that are shared with >= 1
    // other accepted track.
    std::vector<unsigned int> n_shared(n_tracks, 0);
    for (const unsigned int i : accepted_ids) {
        for (const std::size_t mid : meas_ids[i]) {
            const auto it =
                std::lower_bound(unique_meas.begin(), unique_meas.end(), mid);
            assert(it != unique_meas.end());
            const std::size_t idx =
                static_cast<std::size_t>(std::distance(unique_meas.begin(), it));
            if (tracks_per_measurement[idx].size() > 1) {
                n_shared[i]++;
            }
        }
    }

    std::vector<traccc::scalar> rel_shared(n_tracks, 0.f);
    for (const unsigned int i : accepted_ids) {
        rel_shared[i] = static_cast<traccc::scalar>(n_shared[i]) /
                        static_cast<traccc::scalar>(n_meas[i]);
    }

    // sorted_ids[0] = best track, sorted_ids.back() = worst track.
    // "Worst" = highest rel_shared; ties broken by lowest pval.
    std::vector<unsigned int> sorted_ids = accepted_ids;
    auto track_comparator = [&rel_shared, &pvals](unsigned int a,
                                                   unsigned int b) {
        if (rel_shared[a] != rel_shared[b]) {
            return rel_shared[a] < rel_shared[b];
        }
        return pvals[a] > pvals[b];
    };
    std::sort(sorted_ids.begin(), sorted_ids.end(), track_comparator);

    // ------------------------------------------------------------------ //
    // Phase 2 — JP outer loop                                             //
    // ------------------------------------------------------------------ //

    for (unsigned int iter = 0; iter < m_config.max_iterations; iter++) {
        if (accepted_ids.empty()) {
            break;
        }

        // Termination: no remaining track exceeds the shared-meas threshold.
        unsigned int max_shared = 0;
        for (const unsigned int i : accepted_ids) {
            if (n_shared[i] > max_shared) {
                max_shared = n_shared[i];
            }
        }
        if (max_shared < m_config.max_shared_meas) {
            break;
        }

        // Priority = position in sorted_ids.  Worst track has the highest
        // position index → highest priority to be removed.  This matches the
        // GPU implementation's inverted_ids convention.
        const std::size_t n_sorted = sorted_ids.size();
        std::vector<unsigned int> inverted_ids(n_tracks,
                                               static_cast<unsigned int>(-1));
        for (unsigned int k = 0; k < n_sorted; k++) {
            inverted_ids[sorted_ids[k]] = k;
        }

        // ---- JP PROPOSE ----
        // Track v enters IN_MIS if it is active (n_shared > 0) AND its
        // priority is strictly greater than every active neighbour's priority.
        // All comparisons use the state at the start of this round (parallel
        // semantics: no track's result influences another track's check).
        std::vector<int> state(n_tracks, UNDECIDED);

        for (const unsigned int v : accepted_ids) {
            if (n_shared[v] == 0) {
                continue;  // inactive — no conflicts this iteration
            }

            const unsigned int prio_v = inverted_ids[v];
            bool is_local_max        = true;
            bool has_active_neighbor = false;

            // Walk neighbours via the shared-measurement inverted index.
            // De-duplicate so each neighbour u is checked exactly once.
            std::unordered_set<unsigned int> visited;
            for (const std::size_t mid : meas_ids[v]) {
                const auto it = std::lower_bound(unique_meas.begin(),
                                                 unique_meas.end(), mid);
                if (it == unique_meas.end()) {
                    continue;
                }
                const std::size_t meas_idx = static_cast<std::size_t>(
                    std::distance(unique_meas.begin(), it));

                for (const unsigned int u : tracks_per_measurement[meas_idx]) {
                    if (u == v || !visited.insert(u).second) {
                        continue;
                    }
                    if (n_shared[u] == 0) {
                        continue;  // inactive neighbour
                    }
                    has_active_neighbor = true;
                    if (inverted_ids[u] >= prio_v) {
                        is_local_max = false;
                        break;
                    }
                }
                if (!is_local_max) {
                    break;
                }
            }

            if (is_local_max && has_active_neighbor) {
                state[v] = IN_MIS;
            }
        }

        // ---- JP FINALIZE ----
        // Mark undecided active neighbours of IN_MIS tracks as
        // REMOVED_NEIGHBOR.  They are not removed — they stay accepted but
        // are excluded from the MIS for this outer iteration, and participate
        // again in the next one.
        for (const unsigned int v : accepted_ids) {
            if (state[v] != UNDECIDED || n_shared[v] == 0) {
                continue;
            }
            for (const std::size_t mid : meas_ids[v]) {
                const auto it = std::lower_bound(unique_meas.begin(),
                                                 unique_meas.end(), mid);
                if (it == unique_meas.end()) {
                    continue;
                }
                const std::size_t meas_idx = static_cast<std::size_t>(
                    std::distance(unique_meas.begin(), it));
                for (const unsigned int u : tracks_per_measurement[meas_idx]) {
                    if (u != v && state[u] == IN_MIS) {
                        state[v] = REMOVED_NEIGHBOR;
                        goto next_track;
                    }
                }
            }
            next_track:;
        }

        // ---- APPLY BATCH REMOVAL ----
        // Collect IN_MIS tracks.
        std::vector<unsigned int> to_remove;
        for (const unsigned int v : accepted_ids) {
            if (state[v] == IN_MIS) {
                to_remove.push_back(v);
            }
        }

        if (to_remove.empty()) {
            break;  // no progress — should not happen with unique priorities
        }

        const std::unordered_set<unsigned int> remove_set(to_remove.begin(),
                                                           to_remove.end());

        // Collect all measurements touched by removed tracks.
        std::unordered_set<std::size_t> affected_meas;
        for (const unsigned int r : to_remove) {
            for (const std::size_t mid : meas_ids[r]) {
                affected_meas.insert(mid);
            }
        }

        // Remove all IN_MIS entries from tracks_per_measurement at once.
        for (const std::size_t mid : affected_meas) {
            const auto it = std::lower_bound(unique_meas.begin(),
                                             unique_meas.end(), mid);
            if (it == unique_meas.end()) {
                continue;
            }
            const std::size_t meas_idx = static_cast<std::size_t>(
                std::distance(unique_meas.begin(), it));
            auto& tracks = tracks_per_measurement[meas_idx];
            tracks.erase(
                std::remove_if(tracks.begin(), tracks.end(),
                               [&](unsigned int t) {
                                   return remove_set.count(t) != 0u;
                               }),
                tracks.end());
        }

        // Recompute n_shared and rel_shared for surviving tracks that are
        // still in any affected measurement's list.
        std::unordered_set<unsigned int> survivors_to_update;
        for (const std::size_t mid : affected_meas) {
            const auto it = std::lower_bound(unique_meas.begin(),
                                             unique_meas.end(), mid);
            if (it == unique_meas.end()) {
                continue;
            }
            const std::size_t meas_idx = static_cast<std::size_t>(
                std::distance(unique_meas.begin(), it));
            for (const unsigned int t : tracks_per_measurement[meas_idx]) {
                survivors_to_update.insert(t);
            }
        }
        for (const unsigned int t : survivors_to_update) {
            n_shared[t] = 0;
            for (const std::size_t mid : meas_ids[t]) {
                const auto it = std::lower_bound(unique_meas.begin(),
                                                 unique_meas.end(), mid);
                if (it == unique_meas.end()) {
                    continue;
                }
                const std::size_t meas_idx = static_cast<std::size_t>(
                    std::distance(unique_meas.begin(), it));
                if (tracks_per_measurement[meas_idx].size() > 1) {
                    n_shared[t]++;
                }
            }
            rel_shared[t] = static_cast<traccc::scalar>(n_shared[t]) /
                            static_cast<traccc::scalar>(n_meas[t]);
        }

        // Drop removed tracks from accepted_ids and sorted_ids.
        accepted_ids.erase(
            std::remove_if(accepted_ids.begin(), accepted_ids.end(),
                           [&](unsigned int i) {
                               return remove_set.count(i) != 0u;
                           }),
            accepted_ids.end());
        sorted_ids.erase(
            std::remove_if(sorted_ids.begin(), sorted_ids.end(),
                           [&](unsigned int i) {
                               return remove_set.count(i) != 0u;
                           }),
            sorted_ids.end());

        // Re-sort survivors after batch update.
        std::sort(sorted_ids.begin(), sorted_ids.end(), track_comparator);
    }

    // ------------------------------------------------------------------ //
    // Phase 3 — build output (identical to greedy)                        //
    // ------------------------------------------------------------------ //

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
