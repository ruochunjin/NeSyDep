// pybind11 bindings for the SCFDM CFD mining kernel.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../scfdm/api.h"

namespace py = pybind11;

void bind_cfd(py::module_& m) {
    m.def("cfd_mine", &scfdm::mine_cfds,
          py::arg("columns"), py::arg("rows"), py::arg("support"),
          py::arg("confidence"), py::arg("max_lhs"), py::arg("strategy"),
          py::arg("constant_only"),
          py::call_guard<py::gil_scoped_release>(),
          "Mine CFDs on one in-memory sub-table. Returns a list of "
          "(lhs_attributes, rhs_attribute, lhs_patterns, rhs_pattern) tuples.");
}
