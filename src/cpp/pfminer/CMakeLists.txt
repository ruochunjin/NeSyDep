# FD mining kernels: TANE / DFD (header-only) + PFMiner framework glue.
add_library(nesydep_fd INTERFACE)
target_include_directories(nesydep_fd INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nesydep_fd INTERFACE nesydep_common)
