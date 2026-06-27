#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err__));                                \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

void vector_add_cpu(
  const std::vector<float>& a,
  const std::vector<float>& b,
  std::vector<float>& c
) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    c[i] = a[i] + b[i];
  }
}

__global__ void vector_add_kernel(const float* a,
                                  const float* b,
                                  float* c,
                                  int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

int main() {
  std::cout << "Vector Add Benchmark\n";
  std::cout << "N\tCPU_ms\tGPU_kernel_ms\tH2D_ms\tD2H_ms\tcorrect\n";
  constexpr std::size_t N = 1024;

  std::vector<float> a(N);
  std::vector<float> b(N);
  std::vector<float> c_cpu(N);
  std::vector<float> c_gpu(N);

  for (std::size_t i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = i * 2;
  }

  const auto cpu_start = std::chrono::high_resolution_clock::now();
  vector_add_cpu(a, b, c_cpu);
  const auto cpu_end = std::chrono::high_resolution_clock::now();
  const double cpu_ms =
      std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  const std::size_t bytes = N * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  const auto h2d_start = std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
  const auto h2d_end = std::chrono::high_resolution_clock::now();
  const double h2d_ms =
      std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();

  const int threads_per_block = 256;
  const int blocks_per_grid =
    static_cast<int>((N + threads_per_block - 1) / threads_per_block);

  cudaEvent_t kernel_start;
  cudaEvent_t kernel_stop;
  CUDA_CHECK(cudaEventCreate(&kernel_start));
  CUDA_CHECK(cudaEventCreate(&kernel_stop));

  CUDA_CHECK(cudaEventRecord(kernel_start));
  vector_add_kernel<<<blocks_per_grid, threads_per_block>>>(
      d_a, d_b, d_c, static_cast<int>(N));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(kernel_stop));
  CUDA_CHECK(cudaEventSynchronize(kernel_stop));

  float gpu_kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_kernel_ms, kernel_start, kernel_stop));

  const auto d2h_start = std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaMemcpy(c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  const auto d2h_end = std::chrono::high_resolution_clock::now();
  const double d2h_ms =
      std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();

  bool correct = true;
  for (std::size_t i = 0; i < N; ++i) {
    if (std::abs(c_cpu[i] - c_gpu[i]) > 1e-5f) {
      correct = false;
      break;
    }
  }

  CUDA_CHECK(cudaEventDestroy(kernel_start));
  CUDA_CHECK(cudaEventDestroy(kernel_stop));

  std::cout << N << '\t'
            << cpu_ms << '\t'
            << gpu_kernel_ms << '\t'
            << h2d_ms << '\t'
            << d2h_ms << '\t'
            << (correct ? "true" : "false") << '\n';

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return 0;
}
