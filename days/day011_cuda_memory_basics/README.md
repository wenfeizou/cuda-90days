# Day 011: CUDA Memory Basics

Date: 2026-06-25

## 今日目标

通过 10 个交互式问答，理解 CUDA memory 的基础概念，重点区分 `host memory`、`device/global memory`、`shared memory` 和 `register` 的位置、访问范围、速度特点和典型用途。

## 10 个概念问题

### 1. host memory 和 device memory

**Question:** 在 CUDA 程序里，`host memory` 和 `device memory` 分别指什么？为什么 CPU 上的普通数组通常不能直接给 GPU kernel 使用？

**Explanation:** CUDA 程序通常由 CPU 端准备数据、GPU 端执行 kernel。两端通常有不同的内存空间，需要显式拷贝数据。

**Correct Answer:** `host memory` 是 CPU 侧内存，`device memory` 是 GPU 侧内存。CPU 普通数组位于 host memory 中，而 GPU kernel 通常访问 device memory，因此需要用 `cudaMemcpy` 把数据从 host memory 拷贝到 device memory。

### 2. cudaMalloc 分配出来的指针

**Question:** `cudaMalloc(&d_a, size)` 分配出来的 `d_a` 在哪里？CPU 代码和 GPU kernel 分别如何使用这个指针？

**Explanation:** CPU 代码可以持有 device pointer 的指针值，但不能像普通 host pointer 一样直接解引用它。

**Correct Answer:** `d_a` 指向 GPU 侧 device memory。CPU 代码可以保存、传递 `d_a`，并把它传给 `cudaMemcpy` 或 kernel launch，但不能直接用 `d_a[0]` 访问其内容。GPU kernel 可以直接使用 `d_a` 读写 device memory。

### 3. global memory 和 device memory 的关系

**Question:** 在 CUDA 中，`global memory` 是什么？它和 `device memory` 是完全一样的吗？

**Explanation:** `device memory` 是较宽泛的 GPU 侧内存说法，`global memory` 是其中最常用的一类内存空间。

**Correct Answer:** `global memory` 是 GPU 显存中的主要内存空间，所有 thread、所有 block 都可以访问。`cudaMalloc` 分配出来的内存通常可以理解为 global memory。`device memory` 是更宽泛的说法，可能泛指 GPU 侧内存。

### 4. shared memory 和 global memory 的区别

**Question:** `shared memory` 是什么？它和 `global memory` 在访问范围、速度、生命周期上有什么区别？

**Explanation:** shared memory 是 block 内线程协作的重要工具，常用于缓存 global memory 中的一小块数据。

**Correct Answer:** shared memory 是每个 block 独有的一块片上存储，同一个 block 内的 thread 可以共享。global memory 可被所有 block/thread 访问，容量大但延迟高；shared memory 只在 block 内可见，容量小但速度快。global memory 在 kernel 结束后仍存在，直到 `cudaFree`；shared memory 随 block 开始而存在，随 block 结束而消失。

### 5. block 之间为什么不能直接用 shared memory 通信？

**Question:** 为什么不同 block 之间不能直接通过 shared memory 交换数据？如果两个 block 要交换结果，通常应该通过什么内存？

**Explanation:** shared memory 是 per-block 的，每个 block 都有自己的 shared memory 实例。

**Correct Answer:** 不同 block 的 shared memory 相互独立，不能直接访问彼此的数据。跨 block 交换结果通常通过 global memory 完成。普通 CUDA 中同一个 kernel 内没有隐式全 grid 同步，常见做法是让一个 kernel 写 global memory，再通过 kernel 边界作为同步点，让后续 kernel 读取。

### 6. register 的用途和访问范围

**Question:** `register` 在 CUDA 里通常存放什么？它和 `shared memory` 的访问范围有什么区别？

**Explanation:** register 是线程私有的最快存储，通常由编译器为局部变量和临时值分配。

**Correct Answer:** register 通常存放局部变量、临时计算结果、频繁使用的小标量。register 属于单个 thread，其他 thread 不能访问；shared memory 属于一个 block，同一个 block 内的 thread 可以共同访问。

### 7. 普通局部变量在哪里？

**Question:** 如果一个 kernel 里定义了普通局部变量：

```cpp
float sum = 0.0f;
```

你觉得 `sum` 通常会放在哪里？它是每个 thread 一份，还是整个 block 一份？

**Explanation:** 普通局部标量变量通常是 thread-private 的。每个 thread 执行 kernel 代码时都有自己的局部变量实例。

**Correct Answer:** `sum` 通常会放在 register 中，并且是每个 thread 一份，不是整个 block 一份。如果一个 block 有 256 个 thread，那么通常有 256 份彼此独立的 `sum`。

### 8. `__shared__` 数组的作用

**Question:** 如果一个 kernel 里写：

```cpp
__shared__ float tile[32][32];
```

这个 `tile` 是每个 thread 一份，还是每个 block 一份？它通常用来解决什么问题？

**Explanation:** `__shared__` 显式声明 shared memory，常用于 block 内数据复用和协作计算。

**Correct Answer:** `tile` 是每个 block 一份，block 内所有 thread 共享这一份。它通常用于缓存 global memory 中的一小块数据，让 block 内线程快速读写、复用数据，或完成 transpose、reduction、stencil 等 block 内协作计算。

### 9. shared memory 和 `__syncthreads()`

**Question:** 为什么 shared memory 通常需要配合 `__syncthreads()` 使用？如果一个 thread 还没把数据写入 shared memory，另一个 thread 就去读，会有什么问题？

**Explanation:** shared memory 经常用于 thread 之间交换数据，因此需要保证写入完成后再读取。

**Correct Answer:** `__syncthreads()` 是 block 内屏障同步。同一个 block 内所有 thread 都到达这个点之后，才会继续往下执行。如果没有同步，某些 thread 可能还没写完 shared memory，其他 thread 就已经读取，可能读到旧值、未初始化值或不完整数据。

### 10. 几种 memory 的整体总结

**Question:** 请用自己的话总结下面几种 memory：

```text
register
shared memory
global memory
host memory
```

分别在哪里、谁能访问、速度大致如何？

**Explanation:** 理解 memory 层级的关键，是同时看位置、访问范围、速度和生命周期。

**Correct Answer:**

```text
register
位置：GPU SM 内的寄存器资源
访问者：单个 thread 私有
速度：最快
用途：局部变量、临时计算结果

shared memory
位置：GPU SM 上的片上共享存储
访问者：同一个 block 内的 thread
速度：很快，但要注意 bank conflict
用途：block 内协作、缓存 tile、数据复用

global memory
位置：GPU 显存/device memory
访问者：所有 thread、所有 block
速度：较慢，延迟高
用途：大数组、kernel 输入输出、跨 block 数据交换

host memory
位置：CPU 内存
访问者：CPU 代码
速度：对 GPU 来说不能像 device memory 那样直接高效访问
用途：CPU 端准备数据、接收 GPU 计算结果
```

## 今日总结

今天已经理解：

- `host memory` 是 CPU 侧内存，`device memory` 是 GPU 侧内存。
- `cudaMalloc` 分配的是 GPU 侧 device/global memory。
- CPU 可以保存 device pointer，但不能像普通数组一样直接解引用。
- `global memory` 是所有 block/thread 都能访问的 GPU 显存，容量大但延迟高。
- `shared memory` 是每个 block 一份，block 内 thread 共享，速度快但容量有限。
- `register` 是每个 thread 私有，通常保存局部变量和临时计算结果。
- 普通局部变量通常每个 thread 一份，不是整个 block 一份。
- shared memory 常常需要配合 `__syncthreads()` 避免读写顺序问题。
- block 间通信通常通过 global memory，并经常需要 kernel 边界作为同步点。

## 易错点

- `host memory` 是 CPU 侧内存，不是 GPU 侧内存。
- shared memory 不是从 global memory 分出来的，它是 SM 上的片上共享存储。
- register 是 thread-private；shared memory 是 block-private。
- 普通局部变量不是整个 block 一份，而是每个 thread 一份。

## 下一步

- global memory coalescing
- 连续访问显存为什么更快
- warp 和 memory transaction 的关系
- shared memory bank conflict 的实践观察
