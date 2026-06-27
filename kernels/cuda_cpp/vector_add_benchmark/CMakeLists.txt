cmake_minimum_required(VERSION 4.2)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(vector_add_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA SM architecture")

add_executable(vector_add_bench vector_add_bench.cu)
target_compile_options(vector_add_bench PRIVATE -O3 -std=c++23)

add_custom_target(
  run
  COMMAND $<TARGET_FILE:vector_add_bench>
  DEPENDS vector_add_bench
  VERBATIM
)
