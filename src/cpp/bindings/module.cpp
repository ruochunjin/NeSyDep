// pybind11 module entry point for the nesydep native mining kernels.
#include <pybind11/pybind11.h>

void bind_fd(pybind11::module_&);
void bind_cfd(pybind11::module_&);

PYBIND11_MODULE(_core, m) {
    m.doc() = "NeSyDep native mining kernels (TANE/DFD/PFMiner, SCFDM/CTane)";
    bind_fd(m);
    bind_cfd(m);
}
