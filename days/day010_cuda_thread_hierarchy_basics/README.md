# Day 010: CUDA Thread Hierarchy Basics

Date: 2026-06-24

## 今日目标

通过 10 个交互式问答，理解 CUDA 中 `grid -> block -> thread` 的线程层次结构，以及一维数组处理时常用的全局线程索引计算方式。

## 10 个概念问题

### 1. 为什么 CUDA 要使用 grid / block / thread 层次结构？

**Question:** 在 CUDA 中，一个 kernel 启动时通常会创建很多线程。为什么这些线程不是简单地放在一个“大列表”里，而是要分成 `grid -> block -> thread` 这样的层次？

**Explanation:** CUDA 需要把大量线程映射到 GPU 硬件上执行。`block` 是调度和资源分配的重要单位，和 SM、寄存器、shared memory 等资源密切相关。

**Correct Answer:** `grid -> block -> thread` 的结构既方便硬件调度和资源分配，也方便程序员表达数据划分。一次 kernel launch 产生一个 grid，grid 中包含多个 block，每个 block 中包含多个 thread。

### 2. 一维全局线程索引

**Question:** 下面这个公式表示什么？

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

**Explanation:** `threadIdx.x` 只表示线程在当前 block 内的局部编号。要得到整个 grid 中唯一的一维编号，需要结合 block 的编号和每个 block 的线程数。

**Correct Answer:** 这个公式计算当前线程在整个 grid 中的一维全局线程编号。`blockIdx.x * blockDim.x` 表示前面所有 block 已经占用的线程位置，`threadIdx.x` 表示当前线程在本 block 内的偏移。

### 3. 计算需要启动多少个 block

**Question:** 如果要用 CUDA 处理一个长度为 `N = 1000` 的数组，并且每个 block 有 `256` 个线程，通常需要启动多少个 block？为什么？

**Explanation:** block 数通常需要向上取整，保证所有数组元素都有线程覆盖。

**Correct Answer:** 需要 4 个 block，因为 `4 * 256 = 1024`，可以覆盖 1000 个元素。常用写法是：

```cpp
int blocks = (N + blockSize - 1) / blockSize;
```

### 4. 为什么需要边界判断？

**Question:** 多启动了一些线程时，kernel 里面为什么通常要写这样的判断？

```cpp
if (idx < N) {
    out[idx] = in[idx] * 2;
}
```

**Explanation:** 向上取整启动线程后，总线程数可能大于数组长度。例如 1000 个元素配 256 个线程/block，会启动 1024 个线程，多出 24 个线程。

**Correct Answer:** `if (idx < N)` 用于防止数组访问越界。没有这个判断时，多出来的线程可能访问 `in[N]`、`out[N]` 之后的非法位置，导致错误结果或 illegal memory access。

### 5. blockDim.x 和 threadIdx.x 的区别

**Question:** 在 CUDA 中，`blockDim.x` 表示什么？它和 `threadIdx.x` 有什么区别？

**Explanation:** `blockDim.x` 和 `threadIdx.x` 都和 block 内线程有关，但一个表示数量，一个表示当前线程的编号。

**Correct Answer:** `blockDim.x` 表示每个 block 在 x 方向有多少个 thread；`threadIdx.x` 表示当前线程在本 block 内 x 方向的编号。例如 `blockDim.x = 256` 时，`threadIdx.x` 的范围通常是 `0 ~ 255`。

### 6. kernel launch 配置

**Question:** 如果 kernel 启动参数是：

```cpp
myKernel<<<4, 256>>>();
```

这里的 `4` 和 `256` 分别表示什么？

**Explanation:** `<<<...>>>` 是 CUDA kernel launch 配置语法。一维配置中，第一个参数表示 grid 的 block 数，第二个参数表示每个 block 的 thread 数。

**Correct Answer:** `4` 表示启动 4 个 block，也就是 `gridDim.x = 4`；`256` 表示每个 block 有 256 个 thread，也就是 `blockDim.x = 256`。总线程数是 `4 * 256 = 1024`。

### 7. 只使用 threadIdx.x 会有什么问题？

**Question:** 如果写：

```cpp
int idx = threadIdx.x;
```

而不是：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

当有多个 block 时，会发生什么问题？

**Explanation:** `threadIdx.x` 只在当前 block 内唯一。多个 block 中都会出现相同的 `threadIdx.x` 值。

**Correct Answer:** 多个 block 会重复访问相同的数组位置。例如每个 block 都有 `threadIdx.x = 0 ~ 255`，只用 `threadIdx.x` 会导致多个 block 都处理 `0 ~ 255` 这些元素，后面的数据被漏处理。

### 8. block 的线程数为什么不能无限大？

**Question:** CUDA 中一个 block 通常会被调度到一个 SM 上执行。为什么 block 的线程数不能无限大？可以从硬件资源角度回答。

**Explanation:** `SM` 指 Streaming Multiprocessor，不是 shared memory。SM 是 GPU 上执行线程 block 的计算单元。

**Correct Answer:** block 的线程数不能无限大，因为 SM 上的硬件资源有限，包括寄存器、shared memory、最大线程数、最大常驻 block 数等。一个 block 本身也有硬件上限，常见最大值是 1024 个线程/block，具体取决于 GPU 架构。

### 9. 为什么 block size 常选 128 / 256 / 512？

**Question:** 为什么 CUDA 程序里经常选择 `128`、`256`、`512` 这样的 block size，而不是随便选一个比如 `300`？

**Explanation:** CUDA 的线程调度基本单位是 warp。一个 warp 通常包含 32 个线程。

**Correct Answer:** `128`、`256`、`512` 都是 32 的倍数，分别对应 4、8、16 个 warp，更贴合 GPU 的执行方式，也更容易获得较好的资源利用率。`300` 不是不能用，但不是 32 的倍数，最后一个 warp 只有部分线程有效。

### 10. 四个常用内置变量的含义

**Question:** 用自己的话说，`threadIdx.x`、`blockIdx.x`、`blockDim.x`、`gridDim.x` 分别表示什么？

**Explanation:** 这四个变量是理解一维 CUDA kernel 索引计算的基础。

**Correct Answer:**

```cpp
threadIdx.x  // 当前线程在 block 内 x 方向的编号
blockIdx.x   // 当前 block 在 grid 内 x 方向的编号
blockDim.x   // 每个 block 在 x 方向有多少个线程
gridDim.x    // grid 在 x 方向有多少个 block
```

## 今日总结

今天已经理解：

- CUDA 线程组织结构是 `grid -> block -> thread`
- `block` 是重要的调度和资源分配单位
- 一维全局线程编号公式是 `blockIdx.x * blockDim.x + threadIdx.x`
- 处理数组时，block 数通常使用向上取整
- 多启动的线程需要用 `if (idx < N)` 防止越界
- `threadIdx.x` 是 block 内局部编号，不能直接当全局数组索引
- `SM` 是 Streaming Multiprocessor，不是 shared memory
- 常见 block size 选 `128`、`256`、`512`，主要因为它们是 32 的倍数，适合 warp 调度

## 易错点

- `blockDim.x` 表示一个 block 里的线程数，不是 grid 里的 block 数。
- `gridDim.x` 才表示 grid 里的 block 数。
- block size 常选 32 的倍数，原因和 warp 有关，不是因为字节是 8 位。

## 下一步

- CUDA memory 基础
- global memory / shared memory / register 的区别
- shared memory 与 SM 的关系
- warp 和 block size 对性能的影响
