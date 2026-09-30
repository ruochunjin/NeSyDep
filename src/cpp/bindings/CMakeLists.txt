# pybind11 extension module: nesydep._core
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core
    module.cpp
    fd_bindings.cpp
    cfd_bindings.cpp
)
target_link_libraries(_core PRIVATE nesydep_fd nesydep_scfdm)

if(NESYDEP_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(nesydep_scfdm PUBLIC OpenMP::OpenMP_CXX)
endif()

install(TARGETS _core LIBRARY DESTINATION nesydep RUNTIME DESTINATION nesydep)
