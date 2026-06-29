cmake_minimum_required(VERSION 4.2)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(matrix_transpose LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA SM architecture")

add_executable(matrix_transpose transpose_bench.cu)
target_compile_options(matrix_transpose PRIVATE -O3 -std=c++23)

add_custom_target(
  run
  COMMAND $<TARGET_FILE:matrix_transpose>
  DEPENDS matrix_transpose
  VERBATIM
)
