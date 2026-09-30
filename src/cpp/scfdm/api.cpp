#include "api.h"

#include <stdexcept>

#include "algorithms/cfddiscovery.h"
#include "data/database.h"
#include "data/databasereader.h"

namespace scfdm {

namespace {

// Builds a Database from in-memory rows. Database::setAttributes/addRow/
// translateToken are protected, so (like DatabaseReader) we subclass.
class InMemoryReader : private Database {
public:
    static Database fromRows(const std::vector<std::string>& columns,
                             const std::vector<std::vector<std::string>>& rows) {
        InMemoryReader db;
        db.setAttributes(columns);
        for (const auto& row : rows) {
            if (row.size() != columns.size()) {
                throw std::invalid_argument("row width does not match columns");
            }
            Transaction t;
            t.reserve(row.size());
            for (size_t j = 0; j < row.size(); ++j) {
                t.push_back(db.translateToken(static_cast<int>(j), row[j]));
            }
            db.addRow(t);
        }
        return db;
    }
};

void run_strategy(CFDDiscovery& cfdd, const std::string& strategy,
                  int supp, int max_size, double conf) {
    if (strategy == "FD-First-DFS-dfs") {
        cfdd.fdsFirstDFS(supp, max_size, SUBSTRATEGY::DFS, conf);
    } else if (strategy == "FD-First-DFS-bfs") {
        cfdd.fdsFirstDFS(supp, max_size, SUBSTRATEGY::BFS, conf);
    } else if (strategy == "FD-First-BFS-dfs") {
        cfdd.fdsFirstBFS(supp, max_size, SUBSTRATEGY::DFS, conf);
    } else if (strategy == "FD-First-BFS-bfs") {
        cfdd.fdsFirstBFS(supp, max_size, SUBSTRATEGY::BFS, conf);
    } else if (strategy == "Itemset-First-DFS-dfs") {
        cfdd.itemsetsFirstDFS(supp, max_size, SUBSTRATEGY::DFS, conf);
    } else if (strategy == "Itemset-First-DFS-bfs") {
        cfdd.itemsetsFirstDFS(supp, max_size, SUBSTRATEGY::BFS, conf);
    } else if (strategy == "Itemset-First-BFS-dfs") {
        cfdd.itemsetsFirstBFS(supp, max_size, SUBSTRATEGY::DFS, conf);
    } else if (strategy == "Itemset-First-BFS-bfs") {
        cfdd.itemsetsFirstBFS(supp, max_size, SUBSTRATEGY::BFS, conf);
    } else if (strategy == "Integrated-DFS") {
        cfdd.integratedDFS(supp, max_size, conf);
    } else if (strategy == "Integrated-BFS") {  // CTane
        cfdd.ctane(supp, max_size, conf);
    } else {
        throw std::invalid_argument("unknown CFD strategy: " + strategy);
    }
}

CFDTuple convert(const CFD& c, const Database& db) {
    const Itemset& lhs = c.first;
    int rhs = c.second;

    std::vector<std::string> lhs_attrs;
    std::vector<std::string> lhs_pats;
    for (uint ix = 0; ix < lhs.size(); ++ix) {
        int item = lhs[ix];
        if (item == 0) continue;  // attribute not part of the pattern
        // Mirror Output::printItemset's attribute derivation.
        std::string attr =
            item < 0 ? db.getAttrName(-1 - item) : db.getAttrName(db.getAttrIndex(item));
        lhs_attrs.push_back(attr);
        lhs_pats.push_back(item < 0 ? "_" : db.getValue(item));
    }
    std::string rhs_attr = rhs < 0 ? db.getAttrName(-1 - rhs)
                                   : db.getAttrName(db.getAttrIndex(rhs));
    std::string rhs_pat = rhs < 0 ? "_" : db.getValue(rhs);
    return {lhs_attrs, rhs_attr, lhs_pats, rhs_pat};
}

bool is_constant(const CFDTuple& c) {
    if (std::get<3>(c) == "_") return false;
    for (const auto& p : std::get<2>(c)) {
        if (p == "_") return false;
    }
    return true;
}

}  // namespace

std::vector<CFDTuple> mine_cfds(
    const std::vector<std::string>& columns,
    const std::vector<std::vector<std::string>>& rows,
    int support,
    double confidence,
    int max_lhs,
    const std::string& strategy,
    bool constant_only) {
    Database db = InMemoryReader::fromRows(columns, rows);
    CFDDiscovery cfdd(db);
    run_strategy(cfdd, strategy, support, max_lhs, confidence);

    std::vector<CFDTuple> out;
    for (const auto& c : cfdd.getCFDs()) {
        CFDTuple converted = convert(c, db);
        if (constant_only && !is_constant(converted)) continue;
        out.push_back(std::move(converted));
    }
    return out;
}

}  // namespace scfdm
