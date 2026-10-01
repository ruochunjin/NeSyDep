cmake_minimum_required(VERSION 3.18)
project(nesydep LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NESYDEP_ENABLE_OPENMP "Enable OpenMP intra-subtable parallelism" ON)
option(NESYDEP_ENABLE_MPI "Build the standalone MPI miner executable (not shipped in wheels)" OFF)
option(NESYDEP_BUILD_PYTHON "Build the pybind11 _core extension" ON)

# Position-independent code so the static engine libs can be linked into the
# shared pybind11 module.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Static CRT on MSVC: the shipped wheel must not require users to install the
# Visual C++ redistributable.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(NESYDEP_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(NOT OpenMP_CXX_FOUND)
        message(WARNING "OpenMP not found; mining kernels will run single-threaded.")
    endif()
endif()

add_subdirectory(src/cpp/common)
add_subdirectory(src/cpp/pfminer)
add_subdirectory(src/cpp/scfdm)

if(NESYDEP_BUILD_PYTHON)
    add_subdirectory(src/cpp/bindings)
endif()

if(NESYDEP_ENABLE_MPI)
    find_package(MPI REQUIRED)
    # Standalone MPI executable for cluster use; intentionally not part of the
    # Python wheel. Entry point lives in src/cpp/mpi_cli (added in Phase 1+).
endif()
