// Adapted from the BSFD prototype (code-for-BSFD, algorithmsClassic/DFD.hpp).
// Changes: SUPPORT/CONFIDENCE are runtime members instead of #defines, and an
// in-memory constructor replaces file-only input.
#ifndef DFD_H
#define DFD_H

#include <string>
#include <vector>
#include <stack>
#include <unordered_map>
#include <algorithm>
#include <set>
#include <cassert>
#include <cmath>
#include "../common/reader.hpp"

#define PROPOGATE
#define PROPOGATELAYER 3
//#define RANDOM

typedef long long NodeIndex;
typedef long long ColIndex;


const std::vector<ColIndex> bitmap({ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776 });//40
struct Node {
    bool isVisited = false;
    bool isDep = false;
    bool isMinDep = false;
    bool isCandidateMinDep = true;
    bool isNonDep = false;
    bool isMaxNonDep = false;
    bool isCandidateMaxNonDep = true;
    bool isErrorNonDep = false;
    double conf = 0;
    double supp = 0;
    bool isFlexiable = 0;
    int expansionCount = 0;

};

class DFD {
public:
    // Mining thresholds (paper defaults; were #defines in the prototype).
    double support_threshold = 0.10;
    double confidence_threshold = 0.90;

    DFD() = default;

    // Load an in-memory table (columns + hash-coded rows) prepared by Reader.
    void set_data(Reader& r) {
        data = std::move(r.data);
        nrow = r.nrow;
        ncol = r.ncol;
        T.resize(nrow);
        attributes = std::move(r.attributes);
        tabu_for_unique_cols = 0;
        NodeSet.resize(bitmap.at(ncol));
        set_part_map.reserve(1500);
    }

    void output(std::ostream& out) {
        for (auto& fd : FD) {
            std::string result;
            for (auto x : fd) {
                // result += std::to_string(x);
                // result += " ";
                result += attributes[x-1];
                result += " ";
            }
            result.replace(result.rfind(' '), 1, "");
            result.replace(result.rfind(' '), 1, " -> ");
            out << result << std::endl;
        }
    }
    void get_deps(std::set<std::string>& allFD) {
        std::sort(FD.begin(), FD.end(), [](const std::vector<int>& lhs, const std::vector<int>& rhs) -> bool {
        auto iter1 = lhs.begin();
        auto iter2 = rhs.begin();
        while (iter1 != lhs.end() && iter2 != rhs.end()) {
            if (*iter1 < *iter2) return true;
            if (*iter1 > *iter2) return false;
            ++iter1;
            ++iter2;
        }
        return (iter2 != rhs.end());
        });
        for (auto& fd : FD) {
            std::string result;
            for (auto x : fd) {
                // result += std::to_string(x);
                // result += " ";
                result += attributes[x-1];
                result += " ";
            }
            result.replace(result.rfind(' '), 1, "");
            result.replace(result.rfind(' '), 1, " -> ");
            std::cout << result << std::endl;
            allFD.insert(result);
        }
    }

    void extraction() {
        for (int i = 0; i < ncol; ++i) {
            set_part_map[bitmap.at(i)] = std::move(one_column_partition(i));//pli 
            double num = calculate_support(set_part_map[bitmap.at(i)]);
            //std::cout<<"column: "<<i+1<<" support: "<<num<<std::endl;
            if (num < support_threshold) {
                //tabu_for_unique_cols是一个二进制的数，可以标记哪些列是不能当作RHS
                tabu_for_unique_cols |= bitmap.at(i);
                
            }
            else {
                non_unique_cols.push_back(i);
            }
        }
        for (int i = 0; i < ncol; ++i) {
            current_rhs = i;
            //std::cout<<"current rhs: "<<i<<std::endl;
            findLHSs();
            printf(" ");
            for (NodeIndex j : minDeps) {
                auto fd = getColIndexVector(j);
                bool trivial = false;
                for (auto& x : fd) {
                    if (x == current_rhs){
                        trivial = true;
                        break;
                    }
                }
                if (trivial) continue;    

                for (auto& x : fd) x += 1;
                fd.push_back(i + 1);
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
                    n.isFlexiable = false;
                    n.conf = 0;
                    n.expansionCount = 0;
                }
            }
            minDeps.clear();
            maxNonDeps.clear();
            ErrorNonDeps.clear();
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

    const std::vector<std::string>& getAttributes() const {
        return attributes;
    }

private:
    std::vector<std::vector<int>> data;
    int nrow;
    int ncol;
    ColIndex tabu_for_unique_cols;
    int current_rhs;
    std::vector<std::vector<int>> FD;
    std::vector<int> non_unique_cols;
    std::vector<Node> NodeSet;
    std::vector<NodeIndex> minDeps;
    std::vector<NodeIndex> maxNonDeps;
    std::vector<NodeIndex> ErrorNonDeps;
    std::stack<NodeIndex> trace;
    std::vector<std::string> attributes;
    using Partition = std::vector<std::vector<int>>;
    std::vector<int> T;
    Partition counter;
    std::unordered_map<ColIndex, Partition> set_part_map;
    Partition S;

    int getRandom(std::set<int>& S) {
        int idx = rand() % S.size(); // not equal prob but acceptable
        auto iter = S.begin();
        while (idx--) {
            ++iter;
        }
        return *iter;
    }

    void subset(NodeIndex n, std::set<NodeIndex>& S, int depth = -1) {
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
            S.insert(n & ~firstOne);
            subset(n & ~firstOne, S, depth - 1);
        }
    }

    void superset(NodeIndex n, std::set<NodeIndex>& S, NodeIndex tabu, NodeIndex mask, int depth = -1) { 
        if (depth == -1) {
            depth = 32;
        }
        if (depth == 0 || (n | tabu) == mask) return;
        NodeIndex nn = mask & ~n;
        while (nn != 0) {
            NodeIndex firstOne = nn & ~(nn - 1);
            nn = nn & ~firstOne;
            if ((firstOne & tabu) == 0) {
                S.insert(n | firstOne);
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

    void minimize(std::set<NodeIndex>& newSeeds, std::set<NodeIndex>& seeds) {
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
                seeds.insert(x);
            }
        }
        newSeeds.clear();
    }

    std::string getNodeIDAttr(NodeIndex nodeID) {
        std::vector<int> lhs = getColIndexVector(nodeID);
        std::string lhs_str = "";
        for (int i = 0; i < lhs.size(); ++i) {
            lhs_str += attributes[lhs[i]] + " ";
        }
        return lhs_str;
    }


    void findLHSs() {
        if (non_unique_cols.size() < 2) return;
        std::set<NodeIndex> seeds;
        for (auto col : non_unique_cols) {
            if (col == current_rhs) continue;//0,1,2,3,4
            seeds.insert(bitmap.at(col));//
        }
        while (!seeds.empty()) {
            while (!seeds.empty()) {
#ifdef RANDOM
                NodeIndex nodeID = getRandom(seeds);
#else
                NodeIndex nodeID = *seeds.begin();
                seeds.erase(nodeID);
#endif
                while (nodeID != -1) {
                    auto& node = NodeSet.at(nodeID);
                    if (node.isVisited) {
                        if (node.isCandidateMinDep || node.isCandidateMaxNonDep) {//or
                            if (node.isDep) {
                                if (node.isMinDep && node.conf >= confidence_threshold) {
                                    node.isCandidateMinDep = false;
                                    minDeps.push_back(nodeID);
                                }
                            }
                            else {
                                if (node.isMaxNonDep) {
                                    node.isCandidateMaxNonDep = false;
                                    maxNonDeps.push_back(nodeID);
                                }
                            }
                        }
                    }
                    else {
                        //inferCategory:visited or not
                        if (!inferCategory(nodeID)) {
                            computePartition(nodeID);
                        }
                    }
                    nodeID = pickNextNode(nodeID);
                }
            }
            seeds = generateNextSeeds();
        }
    }

    NodeIndex pickNextNode(NodeIndex nodeID) {
        auto& node = NodeSet.at(nodeID);
        std::string lhs_str = getNodeIDAttr(nodeID);

        if (node.isCandidateMinDep) {
            //std::cout<<"pick next node for minCandinode======================"<<lhs_str<<std::endl;
            std::set<NodeIndex> S;
            subset(nodeID, S, 1);
            for (auto iter = S.begin(); iter != S.end();) {
                if (NodeSet.at(*iter).isVisited) {
                    //changed====================== add the latter judge
                    if (NodeSet.at(*iter).isNonDep || NodeSet.at(*iter).isErrorNonDep || NodeSet.at(*iter).isFlexiable) {
                        std::string lhs_str = getNodeIDAttr(*iter);
                        // std::cout<<"nondep node: "<<lhs_str<<std::endl;
                        iter = S.erase(iter);
                    }
                    // else if(NodeSet.at(*iter).isFlexiable){
                    //     // node.isCandidateMinDep = false;
                    //     S.clear();
                    //     break;
                    // }
                    else {
                        // ++iter;
                        //==============delete the following and add the above
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
#ifdef RANDOM
                NodeIndex nextNode = getRandom(S);
#else
                NodeIndex nextNode = *S.begin(); 
#endif
                trace.push(nodeID);
                return nextNode;
            }
        }
        else if (node.isCandidateMaxNonDep) {
            //std::cout<<"pick next node for maxCandinode****************************"<<lhs_str<<std::endl;
            std::set<NodeIndex> S;
            superset(nodeID, S, tabu_for_unique_cols | bitmap.at(current_rhs), (1 << ncol) - 1, 1); // Not sure whether I understand it correctly
            for (auto iter = S.begin(); iter != S.end();) {
                if (NodeSet.at(*iter).isVisited) {
                    if (NodeSet.at(*iter).isDep || NodeSet.at(*iter).isErrorNonDep) {
                    // if (NodeSet.at(*iter).isDep) {
                        iter = S.erase(iter);
                    }
                    else if (NodeSet.at(*iter).isNonDep) {
                        node.isCandidateMaxNonDep = false;
                        S.clear();
                        break;
                    }
                }
                else {
                    ++iter;
                }
            }
            if (S.empty()) {
                // std::cout<<"maxCandinode is dep node"<<std::endl;
                if (node.isCandidateMaxNonDep) {
                    node.isMaxNonDep = true;
                    node.isCandidateMaxNonDep = false;
                    maxNonDeps.push_back(nodeID);
                }
            }
            else {
#ifdef RANDOM
                NodeIndex nextNode = getRandom(S);
#else
//这里的S就是后面的候选集
                NodeIndex nextNode = *S.begin();
#endif
                trace.push(nodeID);
                return nextNode;
            }
        }
        else if (node.isFlexiable){
            node.expansionCount++;
            //std::cout<<"pick next node for flexnode======================"<<lhs_str<< "expansionCount: "<<node.expansionCount<<std::endl;
            const int MAX_EXPANSION_ATTEMPTS = 3;
            if (node.expansionCount >= MAX_EXPANSION_ATTEMPTS){
                node.isFlexiable = false;
                node.isNonDep = true;
                // return -1;
            }
            std::set<NodeIndex> S_sub, S_sup;
            subset(nodeID, S_sub, 1);
            superset(nodeID, S_sup, tabu_for_unique_cols | bitmap.at(current_rhs), (1 << ncol) - 1, 1);

            for (auto iter = S_sub.begin(); iter != S_sub.end();) {
                
                if (NodeSet.at(*iter).isVisited) {
                    //changed====================== add the latter judge
                    if (NodeSet.at(*iter).isNonDep || NodeSet.at(*iter).isErrorNonDep || NodeSet.at(*iter).isFlexiable) {
                        std::string lhs_str = getNodeIDAttr(*iter);
                        // std::cout<<"nondep node: "<<lhs_str<<std::endl;
                        iter = S_sub.erase(iter);
                    }
                    else {
                        ++iter;
                        //==============delete the following and add the above
                        // node.isCandidateMinDep = false;
                        // S.clear();
                        // break;
                    }
                }
                else {
                    ++iter;
                }
            }
            if (S_sub.empty()) {
                if (node.isCandidateMinDep) {
                    node.isMinDep = true;
                    node.isCandidateMinDep = false;
                    minDeps.push_back(nodeID);
                }
            }
            else {
#ifdef RANDOM
                NodeIndex nextNode = getRandom(S);
#else
                NodeIndex nextNode = *S_sub.begin(); 
#endif
                trace.push(nodeID);
                return nextNode;
            }

            for (auto iter = S_sup.begin(); iter != S_sup.end();) {
                //if(attributes[current_rhs] == "I"){
                //    std::string super = getNodeIDAttr(*iter);
                //    std::cout<<"super set for flexnode: "<<lhs_str<<" "<<super<<std::endl;
                //}
                
                if (NodeSet.at(*iter).isVisited) {
                    if (NodeSet.at(*iter).isDep || NodeSet.at(*iter).isErrorNonDep || NodeSet.at(*iter).isFlexiable) {
                    // if (NodeSet.at(*iter).isDep) {
                        iter = S_sup.erase(iter);
                    }
                    else if (NodeSet.at(*iter).isNonDep) {
                        node.isCandidateMaxNonDep = false;
                        S_sup.clear();
                        break;
                    }
                }
                else {
                    ++iter;
                }
            }
            if (S_sup.empty()) {
                // std::cout<<"maxCandinode is dep node"<<std::endl;
                if (node.isCandidateMaxNonDep) {
                    node.isMaxNonDep = true;
                    node.isCandidateMaxNonDep = false;
                    maxNonDeps.push_back(nodeID);
                }
            }
            else {
#ifdef RANDOM
                NodeIndex nextNode = getRandom(S);
#else
//这里的S就是后面的候选集
                NodeIndex nextNode = *S_sup.begin();
#endif
                trace.push(nodeID);
                return nextNode;
            }
            node.isFlexiable = false;
            node.isNonDep = true;
            // return -1;
        }
        NodeIndex idx;
        while (!trace.empty()) {
            idx = trace.top();
            if (NodeSet.at(idx).isVisited && !NodeSet.at(idx).isCandidateMaxNonDep && !NodeSet.at(idx).isCandidateMinDep && \
            !NodeSet.at(idx).isFlexiable) {
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

    std::set<NodeIndex> generateNextSeeds() {
        std::set<NodeIndex> seeds;
        std::set<NodeIndex> newSeeds;
        for (auto maxNonDep : maxNonDeps) {
            NodeIndex complement = (bitmap.at(ncol) - 1) & (~maxNonDep) & ~(tabu_for_unique_cols | bitmap.at(current_rhs));//差集 a& (~b)
            if (seeds.empty()) {
                for (int i = 0; i < ncol; ++i) {
                    if (complement & bitmap.at(i)) {
                        seeds.insert(bitmap.at(i));
                    }
                }
            }
            else {
                for (auto dep : seeds) {
                    for (int i = 0; i < ncol; ++i) {
                        if (complement & bitmap.at(i)) {
                            newSeeds.insert(dep | bitmap.at(i));
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
        std::set<NodeIndex> S;
        subset(nodeID, S, 1);
        NodeIndex childNodeID[2] = { 1, 1 };
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
            auto lhs = getColIndexVector(nodeID);
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

    double calculate_error(Partition& lhs, Partition& l_rhs) {
        double l_error = 0.000;
        double lr_error = 0.000;
        double error = 0.000;
        for (int cidx = 0; cidx < lhs.size(); cidx++) {
            auto& p = lhs[cidx];
            l_error = l_error + (std::pow(p.size(), 2) - p.size());
        }
        for (int cidx = 0; cidx < l_rhs.size(); cidx++) {
            auto& p = l_rhs[cidx];
            lr_error = lr_error + (std::pow(p.size(), 2) - p.size());
        }
        error = (l_error - lr_error) / (std::pow(nrow, 2) - nrow);
        return error;
    }

    double calculate_support(Partition& l_rhs) {
        double sup = 0.000;
        for (int cidx = 0; cidx < l_rhs.size(); cidx++) {
            auto& p = l_rhs[cidx];
            sup = sup + (std::pow(p.size(), 2) - p.size());
            // sup = sup + p.size();
        }
        return sup / (std::pow(nrow, 2) - nrow);
        // return sup / nrow;
    }

    double calculate_confidence(Partition& lhs, Partition& l_rhs) {
        double l_conf = 0.000;
        double lr_conf = 0.000;
        double confidence = 0.000;
        // std::cout<<lhs.size()<<" "<<l_rhs.size()<<std::endl;

        for (int cidx = 0; cidx < lhs.size(); cidx++) {
            auto& p = lhs[cidx];
            l_conf = l_conf + (std::pow(p.size(), 2) - p.size());
        }
        for (int cidx = 0; cidx < l_rhs.size(); cidx++) {
            auto& p = l_rhs[cidx];
            lr_conf = lr_conf + (std::pow(p.size(), 2) - p.size());
        }
        confidence = lr_conf / l_conf;
        return confidence;
    }

    std::pair<double, double> calculate_pertuple_prob(Partition& lhs, Partition& l_rhs) {
        double numerator = 0.000;
        double denominator = 0.000;
        std::pair<double, double> criterions = std::make_pair(0.000, 0.000);
        for(const auto& VX : lhs){
            std::unordered_map<int, int> VA_count;
            int max_count_VA = 0;
            for (const auto& VXA : l_rhs) {
            // 检查 VX 和 VXA 是否有交集（是否属于同一个 VX）
                if (std::any_of(VX.begin(), VX.end(), [&](int tuple_id) {
                        return std::find(VXA.begin(), VXA.end(), tuple_id) != VXA.end();
                    })) {
                    // VXA 中的 VA 是 VXA 的最后一个元素
                    // int VA = VXA.back();
                    // VA_count[VA]++;
                    int cur_VXA = VXA.size();
                    max_count_VA = std::max(max_count_VA, cur_VXA);
                }
                
                // for(const auto& [VA, count] : VA_count){
                    
                // }
            
            }
            numerator += max_count_VA;
            denominator += VX.size();
        }
        
        criterions.first = numerator / denominator;
        criterions.second = numerator/ nrow;
        return criterions;
    }



    bool inferCategory(NodeIndex nodeID) {
        std::set<NodeIndex> S;
        auto& node = NodeSet.at(nodeID);
        subset(nodeID, S, 1);
        for (auto s : S) {
            if (NodeSet.at(s).isVisited && NodeSet.at(s).isDep) {
                node.isVisited = true;
                node.isDep = true;
                //changed
                node.isCandidateMinDep = false;
                node.isCandidateMaxNonDep = false;
                return true;
            }
            // if (NodeSet.at(s).isVisited && NodeSet.at(s).isErrorNonDep) {  
            //     node.isVisited = true;
            //     node.isErrorNonDep = true;
            //     node.isCandidateMaxNonDep = false;
            //     node.isCandidateMinDep = false;
            //     return true;
            // }
        }
        S.clear();
        
        superset(nodeID, S, tabu_for_unique_cols | bitmap.at(current_rhs), (1 << ncol) - 1, 1);
        for (auto s : S) {
            if (NodeSet.at(s).isVisited && NodeSet.at(s).isNonDep) {
                // std::vector<int> lhs = getColIndexVector(nodeID);
                // std::string lhs_str = "";
                // for (int i = 0; i < lhs.size(); ++i) {
                //     lhs_str += attributes[lhs[i]] + " ";
                // }
                // std::cout<<"sub non-FD:"<<lhs_str<<std::endl;
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

        double t_error;
        build_set(nodeID);
        build_set(nodeID | bitmap.at(current_rhs));
        double num;
        double conf;
        std::vector<int> lhs = getColIndexVector(nodeID);
        std::string lhs_str = "";
        for (int i = 0; i < lhs.size(); ++i) {
            lhs_str += attributes[lhs[i]] + " ";
        }
        num = calculate_support(set_part_map[nodeID | bitmap.at(current_rhs)]);
        t_error = calculate_error(set_part_map[nodeID], set_part_map[nodeID | bitmap.at(current_rhs)]);
        conf = calculate_confidence(set_part_map[nodeID], set_part_map[nodeID | bitmap.at(current_rhs)]);
        bool isFD = set_part_map[nodeID].size() == set_part_map[nodeID | bitmap.at(current_rhs)].size();
        // double tmp_error_bound = set_part_map[nodeID].size() * (1 - CONFIDENCE)/ nrow;
        // std::pair<double, double> criterions = calculate_pertuple_prob(set_part_map[nodeID], set_part_map[nodeID | bitmap.at(current_rhs)]);
        // if (t_error <= tmp_error_bound && num >= SUPPORT) {
        // if (t_error <= MAX_ERROR && num >= SUPPORT) {
        // std::cout<<num<<" "<<conf<<std::endl;
        // if(num >= SUPPORT && 1-t_error >= CONFIDENCE) {
        // std::cout<<criterions.second<<" "<<criterions.first<<std::endl;
        // if (criterions.first >= CONFIDENCE && criterions.second >= SUPPORT) {
        //if (attributes[current_rhs] == "I"){
        //    std::cout<<"FD: "<<lhs_str<<" -> "<<attributes[current_rhs]<<"current supp:"<<num<<"current confidence:"<<conf<<std::endl;
        //}
        
        node.conf = conf;
        node.supp = num;
        if (num >= support_threshold) {
            if (conf >= confidence_threshold) {
                node.isDep = true;
                node.isCandidateMinDep = true;
                node.isNonDep = false;
                node.isCandidateMaxNonDep = false;
                node.isErrorNonDep = false;
                node.isFlexiable = false;
            }
            else{
                node.isDep = false;
                node.isCandidateMaxNonDep = false;
                node.isCandidateMinDep = false;
                node.isNonDep = false;
                node.isErrorNonDep = false;
                node.isFlexiable = true;
            }
        }
        // else if (num < support_threshold) {  
        else {
            node.isDep = false;
            node.isCandidateMinDep = false;
            node.isNonDep = false;
            node.isCandidateMaxNonDep = false;
            node.isFlexiable = false;
            node.isErrorNonDep = true;
            ErrorNonDeps.push_back(nodeID);
        }
        // else {
        //     node.isDep = false;
        //     node.isCandidateMinDep = false;
        //     node.isNonDep = true;
        //     node.isCandidateMaxNonDep = true;
        //     node.isErrorNonDep = false;
        // }

#ifdef PROPOGATE
       if (node.isDep) { 
            std::set<NodeIndex> S;
            superset(nodeID, S, bitmap.at(current_rhs) | tabu_for_unique_cols, (1 << ncol) - 1, PROPOGATELAYER);

            for (NodeIndex s : S) {
                auto& node = NodeSet.at(s);
                node.isVisited = true;
                node.isDep = true;
                node.isCandidateMinDep = false;
                node.isCandidateMaxNonDep = false;
            }
        }
        else if (node.isErrorNonDep) {
            std::set<NodeIndex> S;
            superset(nodeID, S, bitmap.at(current_rhs) | tabu_for_unique_cols, (1 << ncol) - 1, PROPOGATELAYER);//找超集，，，(1 << ncol) - 1表示所有元素列都在的全集
            for (NodeIndex s : S) {
                auto& node = NodeSet.at(s);
                node.isVisited = true;
                node.isErrorNonDep = true;
                node.isCandidateMinDep = false;
                node.isCandidateMaxNonDep = false;
            }
        }
        // else {
        //     std::set<NodeIndex> S;
        //     subset(nodeID, S, PROPOGATELAYER);
        //     for (NodeIndex s : S) {
        //         auto& node = NodeSet.at(s);
        //         node.isVisited = true;
        //         node.isNonDep = true;//
        //         node.isCandidateMinDep = false;
        //         node.isCandidateMaxNonDep = false;
        //     }
        // }
#endif
    }

    inline Partition one_column_partition(int col) {
        Partition tmp;
        for (int ridx = 0; ridx < data.size(); ++ridx) {
            auto cat = data[ridx][col];
            if (cat >= tmp.size()) {
                tmp.emplace_back();
            }
            tmp[cat].push_back(ridx);
        }
        Partition ret;
        for (int i = 0; i < tmp.size(); ++i) {
            if (tmp[i].size() > 1) {
                ret.emplace_back(tmp[i]);
            }
        }
        return ret;
    }

    inline void multiply_partitions(Partition& lhs, Partition& rhs, Partition& buf) {
        buf.reserve(150);
        std::fill(T.begin(), T.end(), -1);
        for (int cidx = 0; cidx < lhs.size(); ++cidx) {
            auto& p = lhs[cidx];
            for (int i = 0; i < p.size(); ++i) {
                T[p[i]] = cidx;
            }
        }
        while (S.size() < lhs.size()) {
            S.emplace_back();
        }
        for (int cidx = 0; cidx < rhs.size(); ++cidx) {
            auto& p = rhs[cidx];
            for (int i = 0; i < p.size(); ++i) {
                auto ridx = p[i];
                if (T[ridx] != -1) {
                    S[T[ridx]].push_back(ridx);
                }
            }
            for (int i = 0; i < p.size(); ++i) {
                auto ridx = p[i];
                if (T[ridx] != -1) {
                    if (S[T[ridx]].size() > 1) {
                        buf.emplace_back();
                        buf[buf.size() - 1] = S[T[ridx]];
                    }
                    S[T[ridx]].clear();
                }
            }
        }
    }
};

#endif // DFD_H
