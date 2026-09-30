# SCFDM CFD mining kernel (library form; MPI/Boost/LLVM from the prototype's
# build are intentionally gone — the core never used them).
add_library(nesydep_scfdm STATIC
    api.cpp
    algorithms/cfddiscovery.cpp
    algorithms/minernode.cpp
    algorithms/partitiontable.cpp
    data/cfd.cpp
    data/database.cpp
    data/databasereader.cpp
    data/types.cpp
    util/setutil.cpp
    util/stringutil.cpp
)
target_include_directories(nesydep_scfdm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
