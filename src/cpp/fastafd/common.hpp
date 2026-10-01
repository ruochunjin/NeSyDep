// Shared definitions for the FastAFD kernel (FDM, ICDE 2024), extracted from
// the prototype's DFD.hpp. Kept in a dedicated namespace to avoid colliding
// with the TANE/DFD kernels in ../pfminer.
#ifndef FASTAFD_COMMON_H_
#define FASTAFD_COMMON_H_

#include <cstdint>
#include <vector>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace fastafd {

typedef long long NodeIndex;
typedef long long ColIndex;

// 2^k table (40 entries -> at most 40 columns per mining call).
const std::vector<ColIndex> bitmap({1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
    8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608,
    16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648,
    4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472,
    274877906944, 549755813888, 1099511627776});

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

// Portable popcount (the prototype used the GCC/Clang builtin).
inline int popcount64(uint64_t w) {
#ifdef _MSC_VER
    return static_cast<int>(__popcnt64(w));
#else
    return __builtin_popcountll(w);
#endif
}

}  // namespace fastafd

#endif  // FASTAFD_COMMON_H_
