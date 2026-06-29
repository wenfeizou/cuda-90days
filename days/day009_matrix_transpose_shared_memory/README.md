# Day 009: Matrix Transpose Shared Memory

Date: 2026-06-15

## 今日目标

今天把 Day 006 的 shared memory 和 Day 008 的 bank conflict 概念落到一个可运行实验中：实现 matrix transpose 的 naive 版本、shared memory `32x32` 版本、shared memory `32x33` padding 版本，并为后续 benchmark / profiling 做准备。

## 实验位置

```text
kernels/cuda_cpp/matrix_transpose/
├── CMakeLists.txt
├── .gitignore
├── .vscode/
└── transpose_bench.cu
```

## Kernel 版本

### 1. `transpose_naive`

直接从 global memory 读取：

```cpp
out[x * height + y] = in[y * width + x];
```

这个版本逻辑最简单，但 transpose 的读写方向会导致其中一侧 global memory 访问不连续，通常性能较差。

### 2. `transpose_shared_32x32`

使用 shared memory tile：

```cpp
__shared__ float tile[32][32];
```

思路是先把一块连续数据读入 shared memory，再交换 block 坐标写回 global memory，从而改善 global memory 访问模式。

这个版本的问题是：按列读取 shared memory 时，`tile[threadIdx.x][threadIdx.y + j]` 在 row-major 布局下可能出现 stride 32 访问，容易产生 shared memory bank conflict。

### 3. `transpose_shared_32x33`

使用 padding：

```cpp
__shared__ float tile[32][33];
```

每行多 1 个 padding 元素，让按列访问的 stride 从 32 变成 33。按照简化映射：

```text
bank_id = index % 32
```

stride 33 会让一个 warp 的访问分散到不同 bank，从而减少 bank conflict。

## 构建命令

```bash
cd kernels/cuda_cpp/matrix_transpose
cmake -S . -B build -G Ninja
cmake --build build
```

当前环境编译结果：

```text
cmake --build build
# nvcc -O3 -std=c++23 -arch=native ...
# nvcc warning : Cannot find valid GPU for '-arch=native', default arch is used
```

结论：代码可以通过 `nvcc` 编译。

## 运行命令

```bash
cd kernels/cuda_cpp/matrix_transpose
cmake --build build --target run
# 或
./build/matrix_transpose 50
```

当前环境运行结果：

```text
CUDA error transpose_bench.cu:210: no CUDA-capable device is detected
```

结论：当前 shell 环境没有检测到可用 CUDA GPU，因此今天没有产生真实性能数据。

## Benchmark 设计

程序默认测试：

```text
1024 x 1024
2048 x 2048
4096 x 4096
```

每个 kernel 输出：

```text
kernel
avg_ms
GB/s
correct
```

带宽估算使用：

```text
effective_bandwidth = 2 * matrix_bytes / elapsed_time
```

因为 transpose 每个元素至少读一次、写一次。

## Correctness 设计

程序会在 host 侧计算 CPU reference：

```cpp
out[x * height + y] = in[y * width + x];
```

每个 CUDA kernel 运行后把结果拷回 host，逐元素与 CPU reference 对比。任何 mismatch 都会打印位置和数值，并让程序失败退出。

## 今日理解

- matrix transpose 是观察 global memory coalescing、shared memory tile、bank conflict 的经典实验。
- naive transpose 容易因为读写方向交换，让 global memory 的一侧访问不连续。
- shared memory tile 可以把 global memory 的读写组织得更连续。
- `tile[32][32]` 虽然用了 shared memory，但按列访问 tile 时容易出现 stride 32 的 bank conflict。
- `tile[32][33]` 通过 padding 改变 stride，减少 shared memory bank conflict。
- correctness check 和 benchmark 应该放进同一个最小可运行程序，避免只写 kernel 没有验证。

## 待补 benchmark

在有 CUDA GPU 的环境上运行：

```bash
cd kernels/cuda_cpp/matrix_transpose
rm -rf build
cmake -S . -B build -G Ninja
cmake --build build
./build/matrix_transpose 100
```

然后记录类似表格：

| Matrix | Kernel | avg_ms | GB/s | Correct |
| --- | --- | ---: | ---: | --- |
| 1024 x 1024 | naive | TBD | TBD | TBD |
| 1024 x 1024 | shared_32x32 | TBD | TBD | TBD |
| 1024 x 1024 | shared_32x33 | TBD | TBD | TBD |
| 2048 x 2048 | naive | TBD | TBD | TBD |
| 2048 x 2048 | shared_32x32 | TBD | TBD | TBD |
| 2048 x 2048 | shared_32x33 | TBD | TBD | TBD |
| 4096 x 4096 | naive | TBD | TBD | TBD |
| 4096 x 4096 | shared_32x32 | TBD | TBD | TBD |
| 4096 x 4096 | shared_32x33 | TBD | TBD | TBD |

## 下一步

- 在有 GPU 的环境运行 benchmark，补全表格。
- 用 Nsight Compute 观察 shared memory bank conflict 相关指标。
- 继续实现 copy kernel，对比 transpose 和纯 copy 的有效带宽上限。
- 后续尝试把同类实验迁移到 Rust / cuda-oxide。
