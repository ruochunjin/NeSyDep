// pybind11 bindings for the FD mining kernels (TANE, DFD, PFMiner).
//
// PFMiner in the prototype is a *framework* (MPI distributes sub-tables; each
// sub-table is mined with TANE/DFD). Here the sub-table distribution lives in
// Python (ProcessPoolExecutor), so the native entry points mine one in-memory
// sub-table per call.
//
// All functions take (columns, rows, support, confidence, max_lhs):
//   columns    -- attribute names
//   rows       -- string cells, one inner vector per tuple
//   support    -- absolute minimum support in ROWS; converted here to the
//                 pair-based ratio the kernels use internally
//   confidence -- minimum confidence in (0, 1]
//   max_lhs    -- maximum LHS size, 0 = unlimited
// and return a vector of (lhs_columns, rhs_column) pairs.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <utility>
#include <vector>

#include "../common/reader.hpp"
#include "../pfminer/tane.hpp"
#include "../pfminer/dfd.hpp"

namespace py = pybind11;

using FDPairs = std::vector<std::pair<std::vector<std::string>, std::string>>;

namespace {

double support_ratio(int support_rows, int nrow) {
    // The kernels measure pair-support sup / (n^2 - n); convert an absolute
    // row support S to the equivalent ratio S*(S-1) / (n*(n-1)).
    if (support_rows <= 1 || nrow <= 1) return 0.0;
    double n = static_cast<double>(nrow);
    double s = static_cast<double>(support_rows);
    return (s * (s - 1.0)) / (n * (n - 1.0));
}

FDPairs run_tane(const std::vector<std::string>& columns,
                 const std::vector<std::vector<std::string>>& rows,
                 int support, double confidence, int max_lhs) {
    Reader r;
    r.read_from_rows(columns, rows);

    TANE t;
    t.support_threshold = support_ratio(support, r.nrow);
    t.confidence_threshold = confidence;
    t.set_data(r);
    t.run();

    FDPairs out;
    out.reserve(t.FD.size());
    for (auto& item : t.FD) {
        auto lhs_idx = decode_to_vector(item.first);
        if (max_lhs > 0 && static_cast<int>(lhs_idx.size()) > max_lhs) continue;
        std::vector<std::string> lhs;
        lhs.reserve(lhs_idx.size());
        for (int i : lhs_idx) lhs.push_back(t.attributes[i]);
        out.emplace_back(std::move(lhs), t.attributes[item.second]);
    }
    return out;
}

FDPairs run_dfd(const std::vector<std::string>& columns,
                const std::vector<std::vector<std::string>>& rows,
                int support, double confidence, int max_lhs) {
    Reader r;
    r.read_from_rows(columns, rows);

    DFD d;
    d.support_threshold = support_ratio(support, r.nrow);
    d.confidence_threshold = confidence;
    d.set_data(r);
    d.extraction();

    FDPairs out;
    out.reserve(d.FD.size());
    for (auto& fd : d.FD) {
        // DFD stores 1-based column indices; the last entry is the RHS.
        if (max_lhs > 0 && static_cast<int>(fd.size()) - 1 > max_lhs) continue;
        std::vector<std::string> lhs;
        for (size_t j = 0; j + 1 < fd.size(); ++j) lhs.push_back(d.attributes[fd[j] - 1]);
        out.emplace_back(std::move(lhs), d.attributes[fd.back() - 1]);
    }
    return out;
}

}  // namespace

void bind_fd(py::module_& m) {
    m.def("tane_mine", &run_tane,
          py::arg("columns"), py::arg("rows"), py::arg("support"),
          py::arg("confidence"), py::arg("max_lhs"),
          py::call_guard<py::gil_scoped_release>(),
          "Mine FDs with TANE on one in-memory sub-table.");
    m.def("dfd_mine", &run_dfd,
          py::arg("columns"), py::arg("rows"), py::arg("support"),
          py::arg("confidence"), py::arg("max_lhs"),
          py::call_guard<py::gil_scoped_release>(),
          "Mine FDs with DFD on one in-memory sub-table.");
    // PFMiner = per-sub-table mining inside the parallel framework; the
    // framework lives in Python, so the kernel is TANE.
    m.def("pfminer_mine", &run_tane,
          py::arg("columns"), py::arg("rows"), py::arg("support"),
          py::arg("confidence"), py::arg("max_lhs"),
          py::call_guard<py::gil_scoped_release>(),
          "PFMiner per-sub-table FD kernel (TANE inside the parallel framework).");
}
