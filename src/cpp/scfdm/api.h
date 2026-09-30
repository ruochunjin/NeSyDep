// In-memory API for the SCFDM CFD mining kernel.
//
// Adapted from the prototype's main.cpp (code-for-CFD, parallel/SCFDM_all):
// the MPI/input.txt driver is replaced by a plain function that takes a
// table in memory and returns structured CFDs. The mining algorithms
// themselves are untouched.
#ifndef SCFDM_API_H_
#define SCFDM_API_H_

#include <string>
#include <tuple>
#include <vector>

namespace scfdm {

// One mined CFD in plain data:
//   (lhs_attributes, rhs_attribute, lhs_patterns, rhs_pattern)
// Pattern "_" marks a variable (wildcard) position. LHS attributes and
// patterns are position-aligned.
using CFDTuple =
    std::tuple<std::vector<std::string>, std::string, std::vector<std::string>, std::string>;

// strategy must be one of the prototype's ten names, e.g.:
//   "FD-First-DFS-dfs" (default, fastest), "Integrated-BFS" (CTane), ...
// constant_only filters out CFDs with any wildcard pattern afterwards
// (the dedicated constant-only miner from SCFDM_part is merged this way).
std::vector<CFDTuple> mine_cfds(
    const std::vector<std::string>& columns,
    const std::vector<std::vector<std::string>>& rows,
    int support,
    double confidence,
    int max_lhs,
    const std::string& strategy,
    bool constant_only);

}  // namespace scfdm

#endif  // SCFDM_API_H_
