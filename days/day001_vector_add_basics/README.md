# Day 001: Vector Add Basics

Date: 2026-06-05

## 今日目标

今天已经搭建好 CUDA 运行环境，并成功运行第一个算子。学习目标保持轻量，通过 10 个交互式问答理解 CUDA vector add 程序中的基础概念。

## 10 个概念问题

### 1. thread / block / grid

**Question:** 在 CUDA 里，`thread`、`block`、`grid` 分别代表什么？它们之间是什么关系？

**Explanation:** CUDA kernel 启动后，会产生大量并行执行单元。理解这些执行单元的层级，是理解 kernel launch 的基础。

**Correct Answer:** thread 是执行 kernel 代码的一份并行实例；block 是一组 thread；grid 是一次 kernel launch 启动出来的全部 block。层级关系是 `grid -> block -> thread`。

### 2. 全局线程索引

**Question:** 解释下面代码中的 `blockIdx.x`、`blockDim.x`、`threadIdx.x` 和 `idx`。

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

**Explanation:** 每个 thread 都需要知道自己负责处理哪个数据元素。这个表达式把 block 编号和 block 内 thread 编号组合成一个全局线性索引。

**Correct Answer:** `blockIdx.x` 是当前 block 在 x 维度上的索引；`blockDim.x` 是每个 block 在 x 维度上的 thread 数量；`threadIdx.x` 是当前 thread 在 block 内 x 维度上的索引；`idx` 是一维场景下常用的全局 thread 索引。

### 3. 边界判断

**Question:** 为什么 kernel 里经常写 `if (idx < n)`？如果不写会发生什么？

```cpp
if (idx < n) {
    c[idx] = a[idx] + b[idx];
}
```

**Explanation:** CUDA 程序通常会向上取整启动 thread 数，保证覆盖所有数据元素。因此总 thread 数可能大于真实数据长度。

**Correct Answer:** `if (idx < n)` 用来避免 thread 访问越界内存。如果不写，超出 `n` 的 thread 可能读越界、写越界、产生错误结果，甚至触发 illegal memory access 或程序崩溃。

### 4. host / device

**Question:** 解释 `host`、`device`、`host memory`、`device memory`。

**Explanation:** CUDA 程序通常由 CPU 负责调度，由 GPU 负责执行并行 kernel。两端通常拥有不同的内存空间。

**Correct Answer:** host 是 CPU 端，负责准备数据、分配 GPU 内存、发起 kernel；device 是 GPU 端，负责执行 kernel；host memory 是 CPU 使用的内存；device memory 是 GPU 使用的显存。

### 5. cudaMalloc

**Question:** 为什么 CUDA 程序里通常要先用 `cudaMalloc(&d_a, size)`，而不是直接把 CPU 上的普通数组 `h_a` 传给 kernel 使用？

**Explanation:** 普通 CPU 数组位于 host memory，而 kernel 在 GPU 上运行，需要访问 device memory 中的地址。

**Correct Answer:** 需要用 `cudaMalloc` 在 device memory 中分配内存，得到 GPU 可访问的指针 `d_a`。然后再用 `cudaMemcpy` 把 host memory 中的 `h_a` 拷贝到 device memory 中的 `d_a`。

### 6. HostToDevice 拷贝

**Question:** 解释下面这行代码中的 `d_a`、`h_a`、`size` 和 `cudaMemcpyHostToDevice`。

```cpp
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
```

**Explanation:** `cudaMemcpy` 需要明确源地址、目标地址、拷贝大小和拷贝方向。

**Correct Answer:** `d_a` 是 device memory 中的目标地址；`h_a` 是 host memory 中的源地址；`size` 是拷贝的字节数；`cudaMemcpyHostToDevice` 表示数据方向是 host memory 到 device memory，也就是 `h_a -> d_a`。

### 7. DeviceToHost 拷贝

**Question:** kernel 计算完结果在 `d_c` 中，想把结果拿回 CPU 端的 `h_c`，应该补全什么方向？

```cpp
cudaMemcpy(h_c, d_c, size, ???);
```

**Explanation:** kernel 计算结果在 device memory 中，CPU 端要检查或使用结果，需要把数据拷回 host memory。

**Correct Answer:** 使用 `cudaMemcpyDeviceToHost`。完整写法是：

```cpp
cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
```

数据方向是 device memory 到 host memory，也就是 `d_c -> h_c`。

### 8. Kernel launch 配置

**Question:** 解释下面 kernel launch 中的 `numBlocks`、`blockSize` 和 `<<<numBlocks, blockSize>>>`。

```cpp
vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
```

**Explanation:** `<<<...>>>` 是 CUDA 的 kernel launch 配置语法，用来指定启动多少并行执行资源。

**Correct Answer:** `numBlocks` 是启动的 block 数量；`blockSize` 是每个 block 中的 thread 数量，不是字节数；`<<<numBlocks, blockSize>>>` 表示启动 `numBlocks` 个 block，每个 block 中有 `blockSize` 个 thread。

### 9. numBlocks 向上取整

**Question:** 为什么经常这样计算 `numBlocks`，而不是直接写 `n / blockSize`？

```cpp
int numBlocks = (n + blockSize - 1) / blockSize;
```

请用 `n = 1000`、`blockSize = 256` 举例解释。

**Explanation:** 如果 `n` 不能被 `blockSize` 整除，直接使用整数除法会向下取整，导致部分数据没有 thread 处理。

**Correct Answer:** 这个公式用于向上取整。以 `n = 1000`、`blockSize = 256` 为例，`1000 / 256 = 3` 余 `232`。如果只启动 3 个 block，只有 `3 * 256 = 768` 个 thread，无法覆盖全部数据。向上取整得到 4 个 block，总 thread 数是 1024，多出来的 thread 通过 `if (idx < n)` 避免越界。

### 10. Vector add 程序流程

**Question:** 将完整 CUDA vector add 程序的步骤按正确顺序排列，并简单解释。

```text
A. cudaMemcpy 把结果从 device 拷回 host
B. 在 host 上准备输入数据
C. cudaMalloc 在 device 上分配内存
D. 启动 kernel
E. cudaMemcpy 把输入从 host 拷到 device
F. cudaFree 释放 device memory
```

**Explanation:** CUDA 程序通常先在 CPU 端准备数据，再把数据传到 GPU，启动 kernel 计算，最后把结果拿回 CPU 并释放资源。

**Correct Answer:** `B -> C -> E -> D -> A -> F`。完整流程是：host 准备数据，device 分配内存，把输入从 host 拷到 device，启动 kernel，结果从 device 拷回 host，最后释放 device memory。

## 今日总结

今天已经理解：

- `grid -> block -> thread` 的层级关系
- thread 全局索引 `idx` 的计算方式
- `idx < n` 的边界保护
- host / device 的区别
- host memory / device memory 的区别
- `cudaMalloc` / `cudaMemcpy` / `cudaFree` 的基本流程
- kernel launch 中 `numBlocks` 和 `blockSize` 的含义
- 使用向上取整计算 block 数

## 易错点

- `kernel` 是运行在 GPU 上、由大量 thread 并行执行的函数入口。
- `blockSize` 是每个 block 中的 thread 数量，不是字节数。
- `cudaMemcpy` 的 `size` 通常是字节数，不是元素个数。

## 下一步

- 为什么 GPU 版本不一定比 CPU 快
- `cudaGetLastError` 和 `cudaDeviceSynchronize`
- global memory 访问
- Nsight / `nvidia-smi` 的最小使用
