#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
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

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;

__global__ void transpose_naive(const float *in, float *out, int width,
                                int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    out[x * height + y] = in[y * width + x];
  }
}

__global__ void transpose_shared_32x32(const float *in, float *out, int width,
                                       int height) {
  __shared__ float tile[kTileDim][kTileDim];

  int x = blockIdx.x * kTileDim + threadIdx.x;
  int y = blockIdx.y * kTileDim + threadIdx.y;

  for (int j = 0; j < kTileDim; j += kBlockRows) {
    if (x < width && y + j < height) {
      tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
    }
  }

  __syncthreads();

  x = blockIdx.y * kTileDim + threadIdx.x;
  y = blockIdx.x * kTileDim + threadIdx.y;

  for (int j = 0; j < kTileDim; j += kBlockRows) {
    if (x < height && y + j < width) {
      out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

__global__ void transpose_shared_32x33(const float *in, float *out, int width,
                                       int height) {
  __shared__ float tile[kTileDim][kTileDim + 1];

  int x = blockIdx.x * kTileDim + threadIdx.x;
  int y = blockIdx.y * kTileDim + threadIdx.y;

  for (int j = 0; j < kTileDim; j += kBlockRows) {
    if (x < width && y + j < height) {
      tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
    }
  }

  __syncthreads();

  x = blockIdx.y * kTileDim + threadIdx.x;
  y = blockIdx.x * kTileDim + threadIdx.y;

  for (int j = 0; j < kTileDim; j += kBlockRows) {
    if (x < height && y + j < width) {
      out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

void transpose_cpu(const std::vector<float> &in, std::vector<float> &out,
                   int width, int height) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      out[x * height + y] = in[y * width + x];
    }
  }
}

bool check_result(const std::vector<float> &got,
                  const std::vector<float> &expected) {
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::fabs(got[i] - expected[i]) > 1e-5f) {
      std::cerr << "Mismatch at " << i << ": got " << got[i]
                << ", expected " << expected[i] << "\n";
      return false;
    }
  }
  return true;
}

using KernelFn = void (*)(const float *, float *, int, int);

float run_kernel(const std::string &name, KernelFn kernel, const float *d_in,
                 float *d_out, int width, int height, dim3 block, dim3 grid,
                 int iterations) {
  for (int i = 0; i < 5; ++i) {
    kernel<<<grid, block>>>(d_in, d_out, width, height);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    kernel<<<grid, block>>>(d_in, d_out, width, height);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return elapsed_ms / static_cast<float>(iterations);
}

void benchmark_case(int width, int height, int iterations) {
  std::size_t count = static_cast<std::size_t>(width) * height;
  std::size_t bytes = count * sizeof(float);

  std::vector<float> h_in(count);
  std::vector<float> h_out(count);
  std::vector<float> h_expected(count);

  for (std::size_t i = 0; i < count; ++i) {
    h_in[i] = static_cast<float>(static_cast<int>(i % 1024) - 512) * 0.25f;
  }
  transpose_cpu(h_in, h_expected, width, height);

  float *d_in = nullptr;
  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  struct KernelCase {
    const char *name;
    KernelFn fn;
    dim3 block;
    dim3 grid;
  };

  const dim3 shared_block(kTileDim, kBlockRows);
  const dim3 shared_grid((width + kTileDim - 1) / kTileDim,
                         (height + kTileDim - 1) / kTileDim);
  const dim3 naive_block(32, 32);
  const dim3 naive_grid((width + 31) / 32, (height + 31) / 32);

  const KernelCase kernels[] = {
      {"naive", transpose_naive, naive_block, naive_grid},
      {"shared_32x32", transpose_shared_32x32, shared_block, shared_grid},
      {"shared_32x33", transpose_shared_32x33, shared_block, shared_grid},
  };

  std::cout << "\nMatrix: " << width << " x " << height
            << ", iterations: " << iterations << "\n";
  std::cout << std::left << std::setw(16) << "kernel" << std::right
            << std::setw(14) << "avg_ms" << std::setw(18) << "GB/s"
            << std::setw(14) << "correct" << "\n";

  for (const auto &entry : kernels) {
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    float avg_ms = run_kernel(entry.name, entry.fn, d_in, d_out, width, height,
                              entry.block, entry.grid, iterations);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    bool ok = check_result(h_out, h_expected);
    double moved_bytes = static_cast<double>(bytes) * 2.0;
    double gb_per_s = moved_bytes / (avg_ms / 1000.0) / 1.0e9;

    std::cout << std::left << std::setw(16) << entry.name << std::right
              << std::setw(14) << std::fixed << std::setprecision(4) << avg_ms
              << std::setw(18) << std::fixed << std::setprecision(2)
              << gb_per_s << std::setw(14) << (ok ? "yes" : "no") << "\n";

    if (!ok) {
      CUDA_CHECK(cudaFree(d_in));
      CUDA_CHECK(cudaFree(d_out));
      std::exit(EXIT_FAILURE);
    }
  }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
}

int main(int argc, char **argv) {
  int iterations = 100;
  if (argc >= 2) {
    iterations = std::max(1, std::atoi(argv[1]));
  }

  int device = 0;
  CUDA_CHECK(cudaSetDevice(device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::cout << "Device: " << prop.name << "\n";

  benchmark_case(1024, 1024, iterations);
  benchmark_case(2048, 2048, iterations);
  benchmark_case(4096, 4096, iterations);

  return 0;
}
