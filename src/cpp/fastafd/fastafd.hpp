// FastAFD (FDM, ICDE 2024): approximate FD mining via dataset transformation,
// clustering-based search-space reduction, and covariance-guided traversal.
//
// Adapted from the prototype FastAFD.hpp. Changes:
//   - wrapped in namespace fastafd (the prototype shared globals with its
//     DFD.hpp; ours conflict with ../pfminer/dfd.hpp)
//   - support_threshold / error_threshold macros -> runtime members (support_threshold /
//     error_threshold), PROPOGATE kept at the paper value (3 layers)
//   - in-memory input via ReaderAFD instead of file paths
//   - __builtin_popcountll -> portable popcount64 (common.hpp)
//   - dead code removed: cluster(), mergeClusters(), clusterCentroid(),
//     calculateEuclideanDistance(), swap(), quickSort() (unreferenced in the
//     prototype; the live clustering path is build_search_subspaces())
#ifndef FastAFD_H
#define FastAFD_H

#include <iostream>
#include <queue>
#include <vector>
#include <string>
#include <stack>
#include <unordered_map>
#include <algorithm>
#include <set>
#include <cassert>
#include <cmath>
#include <limits>
#include <cstdint>
#include <random>
#include "common.hpp"
#include "reader_afd.hpp"


namespace fastafd {

class FastAFD {
public:
    // Mining thresholds (paper defaults; were #defines in the prototype):
    // support and error are pair-based ratios over the comparison matrix.
    double support_threshold = 0.2;
    double error_threshold = 0.03;

    FastAFD() = default;

    void set_data(ReaderAFD& r) {
        data = std::move(r.data);
        nrow = data.size();
        nrowk = r.nrow;
        ncol = r.ncol;
        tabu_for_unique_cols = 0;
        tabu_for_unuse_cols = 0;
        NodeSet.resize(bitmap.at(ncol));
        set_part_map.reserve(1500);
        cov_order_map.reserve(1500);
        build_column_bitsets();
    }


    void compute_cov() {
        cov.assign(ncol, std::vector<double>(ncol, 0.0));
        if (nrow == 0 || ncol == 0) return;

        std::vector<double> col_sum(ncol, 0.0);
        std::vector<std::vector<double>> cross_sum(ncol, std::vector<double>(ncol, 0.0));

        for (int row = 0; row < nrow; ++row) {
            for (int i = 0; i < ncol; ++i) {
                const double xi = data[row][i];
                col_sum[i] += xi;
                for (int k = i; k < ncol; ++k) {
                    cross_sum[i][k] += xi * data[row][k];
                }
            }
        }

        const double inv_nrow = 1.0 / nrow;
        for (int i = 0; i < ncol; ++i) {
            const double mean_i = col_sum[i] * inv_nrow;
            for (int k = i; k < ncol; ++k) {
                const double value = cross_sum[i][k] * inv_nrow - mean_i * col_sum[k] * inv_nrow;
                cov[i][k] = value;
                cov[k][i] = value;
            }
        }
    }

    void output(std::ostream& out) {
        for (auto& fd : FD) {
            std::string result;
            for (auto x : fd) {
                result += std::to_string(x);
                result += " ";
            }
            result.replace(result.rfind(' '), 1, "");
            result.replace(result.rfind(' '), 1, " -> ");
            out << result << std::endl;
        }
    }

    void extraction() {
        compute_cov();
        for (int i = 0; i < ncol; i++) {
            set_part_map[bitmap.at(i)] = std::move(one_column_partition(i));
            double num = double(set_part_map[bitmap.at(i)].size()) / nrow;
            if (num < support_threshold)
            {
                tabu_for_unique_cols |= bitmap.at(i);
            }
            else {
                non_unique_cols.push_back(i);
            }
        }

        build_search_subspaces();
        for (auto can : non_unique_cols) {
            current_rhs = can;
			findLHSs();
			for (NodeIndex j : minDeps) {
				auto fd = getColIndexVector(j);
				for (auto& x : fd) x += 1;
				fd.push_back(can + 1);
				FD.emplace_back(fd);
			}
			for (auto& n : NodeSet) {
				if (n.isVisited) {
					n.isVisited = false;
					n.isCandidateMinDep = true;
					n.isCandidateMaxNonDep = true;
					n.isDep = false;
					n.isNonDep = false;
					n.isMinDep = false;
					n.isMaxNonDep = false;
					n.isErrorNonDep = false;
				}
			}
			minDeps.clear();
			maxNonDeps.clear();
			tabu_for_unuse_cols = 0;
			select_cols.clear();
            ErrorNonDeps.clear();
            cov_order_map.clear();
		}
		std::sort(FD.begin(), FD.end(), [](const std::vector<int>& lhs, const std::vector<int>& rhs) -> bool {
			auto iter1 = lhs.begin();
			auto iter2 = rhs.begin();
			while (iter1 != lhs.end() && iter2 != rhs.end()) {
				if (*iter1 < *iter2) return true;
				if (*iter1 > *iter2) return false;
				++iter1;
				++iter2;
			}
			return(iter2 != rhs.end());
			});
    }

	std::vector<std::vector<int>>& getFD() {
		return FD;
	}

    private:
        std::vector<std::vector<int>> data;
        int nrow;
        int ncol;
        int nrowk;//
        int tabu_for_unique_cols;
        int tabu_for_unuse_cols;
        int current_rhs;
        std::vector<std::vector<int>> FD;
        std::vector<int> non_unique_cols;
        std::vector<int> select_cols;
        std::vector<std::vector<double>> cov;
        std::vector<Node> NodeSet;
        std::vector<NodeIndex> minDeps;
        std::vector<NodeIndex> maxNonDeps;
        std::vector<NodeIndex> ErrorNonDeps;
        std::stack<NodeIndex> trace;

        std::unordered_map<int, std::vector<ColIndex>> set_part_map;//
        std::unordered_map<NodeIndex, std::vector<double>> cov_order_map;//
        std::vector<NodeIndex> search_space_masks;
        std::vector<std::vector<uint64_t>> col_bits;
        int bit_words;

        std::vector<std::vector<int>> build_full_comparison_matrix(const std::vector<std::vector<int>>& rows, int min_compare_) {
            std::vector<std::vector<int>> matrix;
            if (rows.empty() || ncol == 0) return matrix;
            matrix.reserve(rows.size() * std::min<size_t>(rows.size(), 100));
            for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
                for (int j = i + 1; j < static_cast<int>(rows.size()); ++j) {
                    std::vector<int> comparison;
                    comparison.reserve(ncol);
                    int match = 0;
                    for (int col = 0; col < ncol; ++col) {
                        int same = rows[i][col] == rows[j][col] ? 1 : 0;
                        comparison.push_back(same);
                        match += same;
                    }
                    if (match >= min_compare_) {
                        matrix.push_back(std::move(comparison));
                    }
                }
            }
            return matrix;
        }

        void build_column_bitsets() {
            bit_words = (nrow + 63) / 64;
            col_bits.assign(ncol, std::vector<uint64_t>(bit_words, 0));
            for (int row = 0; row < nrow; ++row) {
                uint64_t bit = uint64_t(1) << (row & 63);
                int word = row >> 6;
                for (int col = 0; col < ncol; ++col) {
                    if (data[row][col] > 0) {
                        col_bits[col][word] |= bit;
                    }
                }
            }
        }

        int popcount_bits(const std::vector<uint64_t>& bits) const {
            int count = 0;
            for (uint64_t word : bits) {
                count += static_cast<int>(popcount64(word));
            }
            return count;
        }

        std::vector<uint64_t> node_bits(NodeIndex nodeID) const {
            std::vector<uint64_t> bits(bit_words, ~uint64_t(0));
            if ((nrow & 63) != 0 && bit_words > 0) {
                bits.back() = (uint64_t(1) << (nrow & 63)) - 1;
            }
            for (int col = 0; col < ncol; ++col) {
                if ((nodeID & bitmap.at(col)) == 0) continue;
                for (int word = 0; word < bit_words; ++word) {
                    bits[word] &= col_bits[col][word];
                }
            }
            return bits;
        }

        int intersection_count(const std::vector<uint64_t>& lhs_bits, int rhs) const {
            int count = 0;
            for (int word = 0; word < bit_words; ++word) {
                count += static_cast<int>(__builtin_popcountll(lhs_bits[word] & col_bits[rhs][word]));
            }
            return count;
        }

        double centroid_distance(const std::vector<double>& lhs, const std::vector<double>& rhs) {
            double distance = 0.0;
            for (int i = 0; i < ncol; ++i) {
                double diff = lhs[i] - rhs[i];
                distance += diff * diff;
            }
            return std::sqrt(distance);
        }

        void merge_cluster_pair(std::vector<int>& lhs, const std::vector<int>& rhs, std::vector<double>& lhs_centroid, const std::vector<double>& rhs_centroid) {
            int lhs_size = static_cast<int>(lhs.size());
            int rhs_size = static_cast<int>(rhs.size());
            for (int i = 0; i < ncol; ++i) {
                lhs_centroid[i] = (lhs_centroid[i] * lhs_size + rhs_centroid[i] * rhs_size) / (lhs_size + rhs_size);
            }
            lhs.insert(lhs.end(), rhs.begin(), rhs.end());
        }

        void build_search_subspaces() {
            search_space_masks.assign(ncol, 0);
            if (nrow == 0 || ncol == 0) return;

            const int sample_limit = 200;
            int sample_size = std::min(sample_limit, nrow);
            int target_clusters = std::min(7, std::max(1, sample_size));
            std::vector<std::vector<int>> clusters;
            std::vector<std::vector<double>> centroids;
            std::vector<int> sampled_rows;
            clusters.reserve(sample_size);
            centroids.reserve(sample_size);
            sampled_rows.reserve(sample_size);

            for (int row = 0; row < nrow && static_cast<int>(sampled_rows.size()) < sample_size; ++row) {
                sampled_rows.push_back(row);
            }
            std::mt19937 rng(1);
            for (int row = sample_size; row < nrow; ++row) {
                std::uniform_int_distribution<int> dist(0, row);
                int slot = dist(rng);
                if (slot < sample_size) {
                    sampled_rows[slot] = row;
                }
            }

            for (int row : sampled_rows) {
                clusters.push_back(std::vector<int>(1, row));
                centroids.push_back(std::vector<double>(ncol, 0.0));
                for (int col = 0; col < ncol; ++col) {
                    centroids.back()[col] = data[row][col];
                }
            }

            while (static_cast<int>(clusters.size()) > target_clusters) {
                int best_i = 0;
                int best_j = 1;
                double best_distance = std::numeric_limits<double>::max();
                for (int i = 0; i < static_cast<int>(clusters.size()); ++i) {
                    for (int j = i + 1; j < static_cast<int>(clusters.size()); ++j) {
                        double distance = centroid_distance(centroids[i], centroids[j]);
                        if (distance < best_distance) {
                            best_distance = distance;
                            best_i = i;
                            best_j = j;
                        }
                    }
                }
                merge_cluster_pair(clusters[best_i], clusters[best_j], centroids[best_i], centroids[best_j]);
                clusters.erase(clusters.begin() + best_j);
                centroids.erase(centroids.begin() + best_j);
            }

            int cluster_min_support = std::max(1, static_cast<int>(std::ceil(support_threshold * sample_size)));
            for (auto& cluster_rows : clusters) {
                if (static_cast<int>(cluster_rows.size()) < cluster_min_support) continue;

                std::vector<int> attr_count(ncol, 0);
                for (int row : cluster_rows) {
                    for (int col = 0; col < ncol; ++col) {
                        if (data[row][col] > 0) {
                            ++attr_count[col];
                        }
                    }
                }

                NodeIndex correlated_set = 0;
                for (int col = 0; col < ncol; ++col) {
                    if (attr_count[col] >= cluster_min_support) {
                        correlated_set |= bitmap.at(col);
                    }
                }
                if (correlated_set == 0) continue;

                for (int rhs = 0; rhs < ncol; ++rhs) {
                    if (correlated_set & bitmap.at(rhs)) {
                        search_space_masks[rhs] |= correlated_set;
                    }
                }
            }

            NodeIndex full_mask = bitmap.at(ncol) - 1;
            for (int rhs = 0; rhs < ncol; ++rhs) {
                if (search_space_masks[rhs] == 0) {
                    search_space_masks[rhs] = full_mask;
                }
                search_space_masks[rhs] &= ~bitmap.at(rhs);
            }
        }
        void subset(NodeIndex n, std::vector<NodeIndex>&S, int depth = -1) {
            if (depth == -1) {
                depth = 32;
            }
            if (depth == 0 || n == 0)
                return;
            NodeIndex nn = n;
            while (nn != 0) {
                NodeIndex firstOne = nn & ~(nn - 1);
                if ((n & ~firstOne) == 0) break;
                nn = nn & ~firstOne;
                S.push_back(n & ~firstOne);
                subset(n & ~firstOne, S, depth - 1);
            }
        }

        void superset(int n, std::vector<NodeIndex>&S, NodeIndex tabu, NodeIndex mask, int depth = -1) { // optimization: just use 1 layer propogate now
            if (depth == -1) {
                depth = 32;
            }
            if (depth == 0 || (n | tabu) == mask) return;
            NodeIndex nn = mask & ~n;
            while (nn != 0) {
                NodeIndex firstOne = nn & ~(nn - 1);
                nn = nn & ~firstOne;
                if ((firstOne & tabu) == 0) {
                    S.push_back(n | firstOne);
                    superset(n | firstOne, S, tabu, mask, depth - 1);
                }
            }
        }

        std::vector<int> getColIndexVector(NodeIndex nodeID) {
            std::vector<int> V;
            for (int i = 0; i < bitmap.size(); ++i) {
                if (nodeID & bitmap.at(i)) {
                    V.push_back(i);
                }
            }
            return V;
        }

        void minimize(std::vector<NodeIndex>&newSeeds, std::vector<NodeIndex>&seeds) {
            seeds.clear();
            while (!newSeeds.empty()) {
                auto x = *newSeeds.begin();
                newSeeds.erase(newSeeds.begin());
                bool reserve_it = true;
                for (auto iter = newSeeds.begin(); iter != newSeeds.end();) {
                    if ((*iter & x) == x) {
                        iter = newSeeds.erase(iter);
                    }
                    else if ((*iter & x) == *iter) {
                        reserve_it = false;
                        break;
                    }
                    else {
                        ++iter;
                    }
                }
                if (reserve_it) {
                    seeds.push_back(x);
                }
            }
            newSeeds.clear();
        }

        int node_attr_count(NodeIndex nodeID) const {
            int count = 0;
            for (int col = 0; col < ncol; ++col) {
                if (nodeID & bitmap.at(col)) ++count;
            }
            return count;
        }

        bool has_min_dep_subset(NodeIndex nodeID) const {
            for (auto dep : minDeps) {
                if ((nodeID & dep) == dep) return true;
            }
            return false;
        }

        std::vector<NodeIndex> min_dep_subsets(NodeIndex nodeID) const {
            std::vector<NodeIndex> subsets;
            for (auto dep : minDeps) {
                if ((nodeID & dep) == dep) {
                    subsets.push_back(dep);
                }
            }
            return subsets;
        }

        void minimize_masks(std::vector<NodeIndex>& masks) const {
            std::sort(masks.begin(), masks.end());
            masks.erase(std::unique(masks.begin(), masks.end()), masks.end());
            for (auto iter = masks.begin(); iter != masks.end();) {
                bool redundant = false;
                for (auto other : masks) {
                    if (other == *iter) continue;
                    if (((*iter) & other) == other) {
                        redundant = true;
                        break;
                    }
                }
                if (redundant) {
                    iter = masks.erase(iter);
                }
                else {
                    ++iter;
                }
            }
        }

        std::vector<NodeIndex> minimal_hitting_sets(const std::vector<NodeIndex>& sets) const {
            std::vector<NodeIndex> hitting_sets(1, 0);
            for (auto set_mask : sets) {
                std::vector<NodeIndex> next_sets;
                for (auto hit : hitting_sets) {
                    if ((hit & set_mask) != 0) {
                        next_sets.push_back(hit);
                        continue;
                    }
                    for (int col = 0; col < ncol; ++col) {
                        if ((set_mask & bitmap.at(col)) != 0) {
                            next_sets.push_back(hit | bitmap.at(col));
                        }
                    }
                }
                minimize_masks(next_sets);
                hitting_sets.swap(next_sets);
            }
            return hitting_sets;
        }

        void add_min_dep(NodeIndex nodeID) {
            for (auto dep : minDeps) {
                if ((nodeID & dep) == dep) return;
            }
            for (auto iter = minDeps.begin(); iter != minDeps.end();) {
                if (((*iter) & nodeID) == nodeID) {
                    iter = minDeps.erase(iter);
                }
                else {
                    ++iter;
                }
            }
            minDeps.push_back(nodeID);
        }

        void validate_candidate(NodeIndex nodeID) {
            auto& node = NodeSet.at(nodeID);
            if (!node.isVisited && !inferCategory(nodeID)) {
                computePartition(nodeID);
            }
        }

        double candidate_cov_score(NodeIndex nodeID) {
            build_cov_map(nodeID);
            return cov_order_map[nodeID][0];
        }

        std::vector<NodeIndex> sorted_by_cov(std::vector<NodeIndex> candidates) {
            std::sort(candidates.begin(), candidates.end(), [this](NodeIndex lhs, NodeIndex rhs) {
                double lhs_score = candidate_cov_score(lhs);
                double rhs_score = candidate_cov_score(rhs);
                if (lhs_score == rhs_score) return lhs < rhs;
                return lhs_score > rhs_score;
            });
            return candidates;
        }

        std::vector<NodeIndex> search_upward(NodeIndex start, NodeIndex candidate_mask, double lower_bound) {
            std::vector<NodeIndex> local_min_deps;
            std::vector<NodeIndex> frontier;
            std::set<NodeIndex> queued;
            frontier.push_back(start);
            queued.insert(start);
            while (!frontier.empty()) {
                frontier = sorted_by_cov(frontier);
                NodeIndex nodeID = frontier.front();
                frontier.erase(frontier.begin());
                if (candidate_cov_score(nodeID) < lower_bound) continue;
                if (has_min_dep_subset(nodeID)) continue;
                validate_candidate(nodeID);
                auto& node = NodeSet.at(nodeID);
                if (node.isDep) {
                    local_min_deps.push_back(nodeID);
                    continue;
                }
                if (node.isErrorNonDep) continue;

                NodeIndex remaining = candidate_mask & ~nodeID;
                for (int col = 0; col < ncol; ++col) {
                    if ((remaining & bitmap.at(col)) == 0) continue;
                    NodeIndex next = nodeID | bitmap.at(col);
                    if (queued.find(next) != queued.end()) continue;
                    if (has_min_dep_subset(next)) continue;
                    if (candidate_cov_score(next) < lower_bound) continue;
                    frontier.push_back(next);
                    queued.insert(next);
                }
            }
            return local_min_deps;
        }

        NodeIndex search_down_recursive(NodeIndex nodeID, double lower_bound) {
            if (node_attr_count(nodeID) <= 1) return 0;
            std::vector<NodeIndex> generalizations;
            for (int col = 0; col < ncol; ++col) {
                if ((nodeID & bitmap.at(col)) == 0) continue;
                NodeIndex next = nodeID & ~bitmap.at(col);
                if (next == 0 || has_min_dep_subset(next)) continue;
                if (candidate_cov_score(next) < lower_bound) continue;
                generalizations.push_back(next);
            }
            generalizations = sorted_by_cov(generalizations);
            for (auto next : generalizations) {
                validate_candidate(next);
                auto& node = NodeSet.at(next);
                if (!node.isDep) break;
                NodeIndex smaller = search_down_recursive(next, lower_bound);
                return smaller == 0 ? next : smaller;
            }
            return 0;
        }

        void search_down_phase(std::vector<NodeIndex>& local_min_deps, double lower_bound) {
            std::set<NodeIndex> queued(local_min_deps.begin(), local_min_deps.end());
            local_min_deps = sorted_by_cov(local_min_deps);
            while (!local_min_deps.empty()) {
                NodeIndex candidate = local_min_deps.front();
                local_min_deps.erase(local_min_deps.begin());
                std::vector<NodeIndex> pruners = min_dep_subsets(candidate);
                if (pruners.empty()) {
                    NodeIndex minimal = search_down_recursive(candidate, lower_bound);
                    if (minimal == 0) minimal = candidate;
                    add_min_dep(minimal);
                }
                else {
                    std::vector<NodeIndex> hitting_sets = minimal_hitting_sets(pruners);
                    for (auto hit : hitting_sets) {
                        NodeIndex next = candidate & ~hit;
                        if (next == 0 || queued.find(next) != queued.end()) continue;
                        validate_candidate(next);
                        if (NodeSet.at(next).isDep) {
                            local_min_deps.push_back(next);
                            queued.insert(next);
                        }
                    }
                }
                local_min_deps = sorted_by_cov(local_min_deps);
            }
        }

        void findLHSs() {
            if (non_unique_cols.size() < 2) return;
            double rhs_support = double(set_part_map[bitmap.at(current_rhs)].size()) / nrow;
            double lower_bound = (1 - rhs_support) * support_threshold - error_threshold;
            NodeIndex candidate_mask = 0;
            std::vector<NodeIndex> start_nodes;

            for (auto col : non_unique_cols) {
                if (col == current_rhs) continue;
                NodeIndex single = bitmap.at(col);
                bool sample_related = search_space_masks.empty() ||
                    (search_space_masks[current_rhs] & single) != 0;
                double score = cov[col][current_rhs] + (sample_related ? 1.0 : 0.0);
                cov_order_map[single].clear();
                cov_order_map[single].push_back(score);
                cov_order_map[single].push_back(1.0);

                if (cov[col][current_rhs] >= lower_bound) {
                    candidate_mask |= single;
                    start_nodes.push_back(single);
                }
                else {
                    tabu_for_unuse_cols |= single;
                }
            }

            start_nodes = sorted_by_cov(start_nodes);
            std::vector<NodeIndex> local_min_deps;
            for (auto start : start_nodes) {
                if (has_min_dep_subset(start)) continue;
                std::vector<NodeIndex> found = search_upward(start, candidate_mask, lower_bound);
                local_min_deps.insert(local_min_deps.end(), found.begin(), found.end());
            }

            search_down_phase(local_min_deps, lower_bound);
        }

        NodeIndex pickNextNode(NodeIndex nodeID) {
            auto& node = NodeSet.at(nodeID);
            if (node.isCandidateMinDep) {
                std::vector<NodeIndex> S;
                subset(nodeID, S, 1);
                for (auto iter = S.begin(); iter != S.end();) {
                    if (NodeSet.at(*iter).isVisited) {
                        if (NodeSet.at(*iter).isNonDep) {
                            iter = S.erase(iter);
                        }
                        else if(NodeSet.at(*iter).isDep){
                            node.isCandidateMinDep = false;
                            S.clear();
                            break;
                        }
                    }
                    else {
                        ++iter;
                    }
                }
                if (S.empty()) {
                    if (node.isCandidateMinDep) {
                        node.isMinDep = true;
                        node.isCandidateMinDep = false;
                        minDeps.push_back(nodeID);
                    }
                }
                else {
                    NodeIndex nextNode = selectmax(S);
                    trace.push(nodeID);
                    return nextNode;
                }
            }
            else if (node.isCandidateMaxNonDep) {
                std::vector<NodeIndex> S;
                superset(nodeID, S, tabu_for_unique_cols | bitmap.at(current_rhs) | tabu_for_unuse_cols, (1 << ncol) - 1, 1); // Not sure whether I understand it correctly
                for (auto iter = S.begin(); iter != S.end();) {
                    if (NodeSet.at(*iter).isVisited) {
                        if (NodeSet.at(*iter).isDep || NodeSet.at(*iter).isErrorNonDep) {
                            iter = S.erase(iter);
                        }
                        else if (NodeSet.at(*iter).isNonDep) {
                            node.isCandidateMaxNonDep = false;
                            break;
                            S.clear();
                        }
                    }
                    else {
                        ++iter;
                    }
                }
                if (S.empty()) {
                    if (node.isCandidateMaxNonDep) {
                        node.isMaxNonDep = true;
                        node.isCandidateMaxNonDep = false;
                        maxNonDeps.push_back(nodeID);
                    }
                }
                else {
                    NodeIndex nextNode = selectmax(S);
                    trace.push(nodeID);
                    return nextNode;
                }
            }
            NodeIndex idx = 0;
            while (!trace.empty()) {
                idx = trace.top();
                if (NodeSet.at(idx).isVisited && !NodeSet.at(idx).isCandidateMaxNonDep && !NodeSet.at(idx).isCandidateMinDep) {
                    trace.pop();
                }
                else {
                    break;
                }
            }
            if (trace.empty())
                return -1;
            trace.pop();
            return idx;
        }

      
        ColIndex selectmax (std::vector<NodeIndex> S) {
            NodeIndex nextNode = 0;
			double max_score = -1.000;
			for (auto iter = S.begin(); iter != S.end(); iter++) {
                build_cov_map(*iter);
				if (cov_order_map[*iter][0] > max_score) {
                    max_score = cov_order_map[*iter][0];
					nextNode = *iter;
				}
			}
			return nextNode;
        }
        

        std::vector<NodeIndex> generateNextSeeds() {
            std::vector<NodeIndex> seeds;
            std::vector<NodeIndex> newSeeds;
            for (auto maxNonDep : maxNonDeps) {
                NodeIndex complement = (bitmap.at(ncol) - 1) & (~maxNonDep) & ~(tabu_for_unique_cols | bitmap.at(current_rhs) | tabu_for_unuse_cols);
                if (seeds.empty()) {
                    for (ColIndex i = 0; i < ncol; ++i) {
                        if (complement & bitmap.at(i)) {
                            seeds.push_back(bitmap.at(i));
                        }
                    }
                }
                else {
                    for (auto dep : seeds) {
                        for (ColIndex i = 0; i < ncol; ++i) {
                            if (complement & bitmap.at(i)) {
                                newSeeds.push_back(dep | bitmap.at(i));
                            }
                        }
                    }
                    minimize(newSeeds, seeds);
                }
            }

            for (auto dep : minDeps) {
                auto it = std::find(seeds.begin(), seeds.end(), dep);
                if (it != seeds.end()) {
                    seeds.erase(it);
                }
            }

            for (auto errornondep : ErrorNonDeps) {
                auto it = std::find(seeds.begin(), seeds.end(), errornondep);
                if (it != seeds.end()) {
                    seeds.erase(it);
                }
            }

            return seeds;
        }

        void build_set(NodeIndex nodeID) {
            if (set_part_map.find(nodeID) != set_part_map.end()) return;
            std::vector<NodeIndex> S;
            subset(nodeID, S, 1);
            int childNodeID[2] = { -1, -1 };
            int idx = 0;
            for (auto s = S.begin(); s != S.end(); ++s) {
                if (set_part_map.find(*s) != set_part_map.end()) {
                    childNodeID[idx++] = *s;
                    if (idx == 2) break;
                }
            }
            if (idx == 2) {
                multiply_partitions(set_part_map[childNodeID[0]], set_part_map[childNodeID[1]], set_part_map[nodeID]);
            }
            else if (idx == 1) {
                multiply_partitions(set_part_map[nodeID & ~childNodeID[0]], set_part_map[childNodeID[0]], set_part_map[nodeID]);
            }
            else {
                auto lhs = getColIndexVector(nodeID);//
                auto e = bitmap.at(lhs.back());
                lhs.erase(lhs.end() - 1);
                for (auto l : lhs) {
                    auto x = bitmap.at(l);
                    if (set_part_map.find(e | x) == set_part_map.end()) {
                        multiply_partitions(set_part_map[e], set_part_map[x], set_part_map[e | x]);
                    }
                    e = e | x;
                }
            }
        }

        void build_cov_map(NodeIndex nodeID) {
            if (cov_order_map.find(nodeID) != cov_order_map.end()) return;
            std::vector<NodeIndex> S;
            subset(nodeID, S, 1);
            NodeIndex childNodeID[2] = { -1, -1 };
            std::vector<NodeIndex> reminS;
            int idx = 0;
            for (auto s = S.begin(); s != S.end(); ++s) {
                if (cov_order_map.find(*s) != cov_order_map.end() && cov_order_map[*s][1] == 1.0) {
                    childNodeID[idx++] = *s;
                    if (idx == 2) break;
                }
                else if (cov_order_map.find(*s) != cov_order_map.end()) {
                    reminS.push_back(*s);
                }
            }
            if (idx == 2) {
                cov_compute(childNodeID[0], childNodeID[1], cov_order_map[nodeID]);
            }
            else if (idx == 1) {
                if (reminS.size() > 0) {
                    cov_compute(reminS[0], childNodeID[0], cov_order_map[nodeID]);
                }
                else
                {
                    cov_compute(nodeID & ~childNodeID[0], childNodeID[0], cov_order_map[nodeID]);
                }
            }
            else
            {
                if (reminS.size() == 2) {
                    cov_compute(reminS[0], reminS[1], cov_order_map[nodeID]);
                }
                else if (reminS.size() == 1) {
                    cov_compute(nodeID & ~reminS[0], reminS[0], cov_order_map[nodeID]);
                }
                else {
                    auto lhs = getColIndexVector(nodeID);
                    auto e = bitmap.at(lhs.back());
                    lhs.erase(lhs.end() - 1);
                    for (auto l : lhs) {
                        auto x = bitmap.at(l);
                        if (cov_order_map.find(e | x) == cov_order_map.end()) {
                            cov_compute(e, x, cov_order_map[e | x]);
                        }
                        e = e | x;
                    }
                }
            }
        }

        void cov_compute(NodeIndex childid0, NodeIndex childid1, std::vector<double>&buf)
        {
            buf.reserve(2);
            NodeIndex n = 0;
            NodeIndex ii = 0;
            NodeIndex jj = 0;
            n = childid0 | childid1;
            ii = ~childid0 & n;
            jj = ~childid1 & n;
            int r = -1;
            int l = -1;
            for (ColIndex i = 0; i < bitmap.size(); ++i) {
                if (ii & bitmap.at(i)) {
                    r = i;
                }
                if (jj & bitmap.at(i)) {
                    l = i;
                }
            }
            double x = cov_order_map[childid0][0] + cov_order_map[childid1][0] + cov[l][r];
            double b = x / 3;
            buf.push_back(b);
            buf.push_back(0.0);
        }

        bool inferCategory(NodeIndex nodeID) {
            std::vector<NodeIndex> S;
            auto& node = NodeSet.at(nodeID);
            subset(nodeID, S, 1);
            for (auto s : S) {
                if (NodeSet.at(s).isVisited && NodeSet.at(s).isDep) {
                    node.isVisited = true;
                    node.isDep = true;
                    node.isCandidateMinDep = false;
                    node.isCandidateMaxNonDep = false;
                    return true;
                }
                if (NodeSet.at(s).isVisited && NodeSet.at(s).isErrorNonDep) {       
                    node.isVisited = true;
                    node.isErrorNonDep = true;
                    node.isCandidateMaxNonDep = false;
                    node.isCandidateMinDep = false;
                    return true;
                }
            }
            S.clear();
            superset(nodeID, S, tabu_for_unique_cols | bitmap.at(current_rhs) | tabu_for_unuse_cols, (1 << ncol) - 1, 1);
            for (auto s : S) {
                if (NodeSet.at(s).isVisited && NodeSet.at(s).isNonDep) {
                    node.isVisited = true;
                    node.isNonDep = true;
                    node.isCandidateMaxNonDep = false;
                    node.isCandidateMinDep = false;
                    return true;
                }
            }
            return false;
        }

        void computePartition(NodeIndex nodeID) {
            auto& node = NodeSet.at(nodeID);
            node.isVisited = true;
            std::vector<uint64_t> lhs_bits = node_bits(nodeID);
            double a = popcount_bits(lhs_bits);
            double b = intersection_count(lhs_bits, current_rhs);
            double num = b / nrow;
            double ka = double(set_part_map[bitmap.at(current_rhs)].size()) / nrow;
            double error = (a - b) / nrow;
            build_cov_map(nodeID);
            cov_order_map[nodeID][0] = (1.0 - ka) * a / nrow - error;
            cov_order_map[nodeID][1] = 1.0;
            if (error <= error_threshold && num >= support_threshold) {
                node.isDep = true;
                node.isCandidateMinDep = true;
                node.isNonDep = false;
                node.isCandidateMaxNonDep = false;
                node.isErrorNonDep = false;
            }
            else if (num < support_threshold) {
                node.isDep = false;
                node.isCandidateMinDep = false;
                node.isNonDep = false;
                node.isCandidateMaxNonDep = false;
                node.isErrorNonDep = true;
                ErrorNonDeps.push_back(nodeID);
            }
            else {
                node.isDep = false;
                node.isCandidateMinDep = false;
                node.isNonDep = true;
                node.isCandidateMaxNonDep = true;
                node.isErrorNonDep = false;
            }

#if 1  // PROPOGATE (paper: 3-layer propagation)
            if (node.isDep) { 
                std::vector<NodeIndex> S;
                superset(nodeID, S, bitmap.at(current_rhs) | tabu_for_unique_cols | tabu_for_unuse_cols, (1 << ncol) - 1, 3 /* PROPOGATELAYER */);
                for (int s : S) {
                    auto& node = NodeSet.at(s);
                    node.isVisited = true;
                    node.isDep = true;
                    node.isCandidateMinDep = false;
                    node.isCandidateMaxNonDep = false;
                }
            }
            else if (node.isErrorNonDep) {
                std::vector<NodeIndex> S;
                superset(nodeID, S, bitmap.at(current_rhs) | tabu_for_unique_cols | tabu_for_unuse_cols, (1 << ncol) - 1, 3 /* PROPOGATELAYER */);
                for (int s : S) {
                    auto& node = NodeSet.at(s);
                    node.isVisited = true;
                    node.isErrorNonDep = true;
                    node.isCandidateMinDep = false;
                    node.isCandidateMaxNonDep = false;
                }
            }
            else {
                std::vector<NodeIndex> S;
                subset(nodeID, S, 3 /* PROPOGATELAYER */);
                for (int s : S) {
                    auto& node = NodeSet.at(s);
                    node.isVisited = true;
                    node.isNonDep = true;
                    node.isCandidateMinDep = false;
                    node.isCandidateMaxNonDep = false;
                }
            }
#endif
        }

        inline std::vector<ColIndex> one_column_partition(int col) {//ֻ��0,1����cluster.����1
            std::vector<ColIndex> ret;
            for (int ridx = 0; ridx < data.size(); ++ridx) {
                auto cat = data[ridx][col];
                if (cat > 0) {
                    ret.push_back(ridx);
                }
            }
            return ret;
        }

        inline void multiply_partitions(std::vector<ColIndex>&lhs, std::vector<ColIndex>&rhs, std::vector<ColIndex>&buf) {
            buf.resize(lhs.size());
            std::fill(buf.begin(), buf.end(), 0);
            auto iter = set_intersection(begin(lhs), end(lhs), begin(rhs), end(rhs), buf.begin());
            buf.resize(iter - buf.begin());
        }
    };


}  // namespace fastafd

#endif  // FastAFD_H
