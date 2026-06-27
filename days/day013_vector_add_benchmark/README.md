# Day 013: Vector Add Benchmark

Date: 2026-06-27

## Topic

Vector Add Benchmark: CPU vs CUDA, correctness check, and timing.

## Goal

Complete a minimal CUDA performance experiment for vector add with:

- CPU reference implementation
- CUDA kernel implementation
- host-to-device timing
- kernel timing
- device-to-host timing
- correctness check

## Implementation

- Fixed input size to `N = 1024`
- Kept the CPU reference in `vector_add_cpu`
- Added `vector_add_kernel`
- Used `CUDA_CHECK` for CUDA API calls
- Measured kernel time with `cudaEvent`
- Measured transfer time with `std::chrono`
- Printed one summary line in table format

## Run

Build:

```bash
make -C kernels/cuda_cpp/vector_add_benchmark
```

Run:

```bash
make -C kernels/cuda_cpp/vector_add_benchmark run
```

## Result

The code compiles successfully. Runtime execution depends on a CUDA-capable GPU in the local environment.

## Summary

This experiment now covers the full basic loop:

1. prepare host data
2. allocate device memory
3. copy input to GPU
4. launch kernel
5. copy output back
6. compare CPU and GPU results
7. print timing data

## Next Step

Extend the same benchmark to multiple sizes, then compare how `CPU_ms`, `GPU_kernel_ms`, `H2D_ms`, and `D2H_ms` change with `N`.
