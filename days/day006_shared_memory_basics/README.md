# Day 006: Shared Memory Basics

Date: 2026-06-11

## 今日目标

今天继续 CUDA 主线，学习 shared memory 的基础直觉。重点理解 shared memory 和 global memory 的区别、block 内作用范围、`__syncthreads()` 的意义，以及什么时候 shared memory 值得使用。

## 10 个概念问题

### 1. shared memory 是什么

**Question:** 在 CUDA 里，`shared memory` 通常指什么？请从 GPU/CPU 侧、速度、共享范围、是否适合长期保存大量数据几个角度解释。

**Explanation:** shared memory 是 GPU 内存层级中的高速临时存储，常用于 block 内 thread 协作。

**Correct Answer:** shared memory 在 GPU 侧，通常比 global memory 更快，只在一个 block 内的 thread 之间共享，不适合长期保存大量数据。可以记成：shared memory 是 GPU 上 block 内 thread 共享的高速临时内存。

### 2. 为什么 shared memory 更快

**Question:** 为什么 shared memory 通常比 global memory 快？可以从位置、容量、访问延迟、是否更靠近计算单元几个角度解释。

**Explanation:** GPU 内存层级中，离计算单元越近、容量通常越小，但访问延迟更低。

**Correct Answer:** shared memory 更靠近 SM 上的计算单元，访问延迟低、带宽高，但容量小。global memory 在显存里，容量大但距离更远、延迟更高。

### 3. shared memory 的 block 作用范围

**Question:** 为什么 shared memory 的作用范围通常说是一个 block 内，而不是整个 grid 内？为什么不同 block 不能直接通过 shared memory 互相交换数据？

**Explanation:** 每个 block 都有自己独立的一份 shared memory。block 可能被调度到不同 SM 上执行，执行顺序也不保证。

**Correct Answer:** shared memory 是 block-local 的。block 0、block 1、block 2 各自有独立的 shared memory，彼此不能直接访问。不同 block 之间如果要通信，通常需要通过 global memory、多个 kernel launch、atomic 或 cooperative groups 等机制。

### 4. `__shared__ float tile[256]`

**Question:** 下面这个声明是什么意思？

```cpp
__shared__ float tile[256];
```

请解释 `__shared__`、`tile`、`256`、哪些 thread 可以访问它，以及生命周期到什么时候。

**Explanation:** `__shared__` 是 CUDA 中声明 shared memory 变量的关键字。

**Correct Answer:** `__shared__` 表示变量存放在 shared memory；`float` 表示元素类型；`tile` 是数组名；`256` 表示数组有 256 个 float 元素。同一个 block 内的所有 thread 都可以访问 `tile`。它通常在当前 block 执行期间存在，block 执行结束后内容不再保留。

### 5. 为什么先加载到 shared memory

**Question:** 如果多个 thread 都会读同一段 global memory 数据，为什么可以考虑先把这段数据加载到 shared memory？

**Explanation:** global memory 访问慢，shared memory 访问快。如果数据会被 block 内多个 thread 重复使用，把它缓存到 shared memory 可能减少重复 global memory 访问。

**Correct Answer:** 如果同一段 global memory 数据会被 block 内多个 thread 多次使用，可以先从 global memory 读一次到 shared memory，后续 thread 从 shared memory 读。这样用更快的 block-local cache 替代重复的慢速 global memory 访问。

### 6. `__syncthreads()`

**Question:** 使用 shared memory 时，为什么经常会看到：

```cpp
__syncthreads();
```

它的作用是什么？

**Explanation:** 典型场景是一些 thread 先把数据从 global memory 写入 shared memory，其他 thread 随后要读取 shared memory。读取前必须确保写入完成。

**Correct Answer:** `__syncthreads()` 是 block 内 thread 的同步屏障。它保证同一个 block 内所有 thread 都执行到这里后，大家才会继续往后执行。它常用于确保 shared memory 数据已经被整个 block 加载完成，再进行读取。

### 7. 没有复用时是否适合 shared memory

**Question:** 下面场景适不适合用 shared memory？为什么？

```text
每个 thread 只读取 a[idx] 一次，
计算 c[idx] = a[idx] + 1，
之后再也不会复用 a[idx]。
```

**Explanation:** shared memory 不是免费优化。它会引入加载代码、同步开销、容量占用和复杂度。

**Correct Answer:** 不适合。因为每个数据只读一次，没有 block 内复用。此时把数据从 global memory 搬到 shared memory 再读，可能比直接从 global memory 读到 register 更复杂，甚至更慢。

### 8. matrix transpose 为什么常用 shared memory

**Question:** 为什么 matrix transpose 常常会用 shared memory 优化？

```text
input[row][col] -> output[col][row]
```

这里可能出现什么 global memory 访问问题？

**Explanation:** transpose 的关键问题不只是复用，而是读写访问模式可能一边连续、一边跨步，导致 coalescing 差。

**Correct Answer:** matrix transpose 直接实现时，可能读 input 连续，但写 output 跨步，或反过来。shared memory 可以先把一个 tile 连续读入 shared memory，在 shared memory 中交换行列，再把结果连续写回 global memory，从而改善 global memory access pattern，提高 coalescing。

### 9. shared memory 使用过多的影响

**Question:** shared memory 容量有限。如果一个 kernel 每个 block 使用了很多 shared memory，可能会对并发执行产生什么影响？

**Explanation:** 每个 SM 的 shared memory 总量有限。每个 block 占用越多 shared memory，同一个 SM 上能同时放下的 block 通常越少。

**Correct Answer:** 每个 block 使用很多 shared memory 时，一个 SM 上能同时驻留的 block 数量通常会减少。这可能导致 occupancy 下降，可并发执行的 warps 变少，隐藏 memory latency 的能力下降，性能可能变差。

### 10. shared memory 什么时候值得用

**Question:** 用一句话总结：什么时候 shared memory 值得用？尽量使用 `global memory`、`shared memory`、`block`、`reuse`、`__syncthreads` 这些词。

**Explanation:** shared memory 的价值来自 block 内复用，但它也有加载和同步成本。

**Correct Answer:** 当一个 block 内的多个 thread 会重复使用同一批 global memory 数据，并且通过 shared memory + `__syncthreads()` 的开销小于重复访问 global memory 的成本时，shared memory 值得使用。

## 今日总结

今天已经理解：

- shared memory 是 GPU 上 block 内 thread 共享的高速临时内存
- shared memory 通常比 global memory 更快，但容量小
- shared memory 更靠近 SM 上的计算单元，访问延迟更低
- 每个 block 有自己独立的一份 shared memory
- 不同 block 不能直接通过 shared memory 通信
- `__shared__` 用于声明 shared memory 变量
- shared memory 的生命周期通常是 block 执行期间
- shared memory 适合缓存会被 block 内多个 thread 重复使用的数据
- `__syncthreads()` 是 block 内同步屏障
- 没有数据复用时，使用 shared memory 可能反而不划算
- matrix transpose 常用 shared memory 改善 global memory 访问模式
- 每个 block 使用过多 shared memory 可能降低 SM 上的并发 block 数和 occupancy
- SM 是 Streaming Multiprocessor，是 GPU 中执行 block 的计算单元集群

## 易错点

- shared memory 不是更快的 global memory，而是 block 内 thread 协作复用数据的高速临时工作区。
- shared memory 不能跨 block 直接共享。
- `__syncthreads()` 只能同步同一个 block 内的 thread。
- 没有复用的数据不一定值得放进 shared memory。
- shared memory 使用过多可能降低 occupancy。

## 下一步

- 学习 shared memory bank conflict
- 写一个 matrix transpose naive vs shared memory 版本
- 观察 shared memory 使用量对 occupancy 的影响
- 继续完善 `docs/glossary.md` 中的 CUDA 内存层级概念
