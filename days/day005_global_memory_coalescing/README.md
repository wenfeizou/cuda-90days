# Day 005: Global Memory and Coalescing

Date: 2026-06-10

## 今日目标

今天回到 CUDA 主线，学习 global memory 与 memory coalescing 的基础直觉。重点不是背硬件细节，而是理解为什么 GPU 喜欢相邻 thread 以规则、连续的方式访问 global memory。

## 10 个概念问题

### 1. global memory 是什么

**Question:** 在 CUDA 里，`global memory` 通常指什么？请从 CPU/GPU 侧、是否属于显存、kernel thread 能否访问、host 普通指针能否直接使用这几个角度解释。

**Explanation:** CUDA 中的 host memory 和 device global memory 是两个不同的内存空间。理解它们的位置和访问权限，是后续理解性能的前提。

**Correct Answer:** `global memory` 通常指 GPU device 侧的全局内存，通常是显存的一部分。kernel 里的 thread 可以访问 global memory。host 端普通指针不能直接当作 global memory 指针使用；通过 `cudaMalloc` 得到的 `d_a` 这类 device pointer 才指向 device global memory。

### 2. 为什么要拷贝到 global memory

**Question:** 为什么 CUDA 程序里经常要这样做？

```cpp
cudaMalloc(&d_a, size);
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
```

请用 `host memory`、`global memory`、`kernel thread` 解释。

**Explanation:** CPU 侧准备的数据通常在 host memory 中，而 GPU kernel thread 需要访问 device global memory。

**Correct Answer:** `cudaMalloc(&d_a, size)` 在 device global memory 中为 `d_a` 分配空间；`cudaMemcpyHostToDevice` 把 host memory 中的 `h_a` 数据拷贝到 device global memory 中的 `d_a`；kernel thread 随后读写 `d_a` 对应的显存数据。

### 3. global memory 为什么慢

**Question:** 为什么说 global memory 访问通常比较“慢”？这里的“慢”主要是相对于哪些内存或存储位置而言？

**Explanation:** GPU 内部有不同层级的存储。global memory 容量大，但访问延迟相对高。

**Correct Answer:** global memory 通常比 register 和 shared memory 慢很多。大致直觉是：register 最快、thread 私有；shared memory 很快、block 内共享；global memory 容量大、所有 thread 可访问，但访问延迟高。

### 4. vector add 的连续访问

**Question:** 在 vector add 里，如果每个 thread 做：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
c[idx] = a[idx] + b[idx];
```

相邻 thread 通常会访问什么样的内存地址？

**Explanation:** 一维 vector add 中，全局索引 `idx` 通常随 thread 连续增长。

**Correct Answer:** 通常是：

```text
thread 0 -> a[0]
thread 1 -> a[1]
thread 2 -> a[2]
thread 3 -> a[3]
```

也就是相邻 thread 访问相邻内存地址。这种模式对 GPU 友好，因为更容易产生 memory coalescing。

### 5. memory coalescing

**Question:** `memory coalescing` 可以粗略理解成什么？为什么相邻 thread 访问相邻 global memory 地址通常更高效？

**Explanation:** GPU 会以 memory transaction 的形式访问 global memory。如果一组 thread 的访问整齐连续，硬件更容易把它们合并。

**Correct Answer:** `memory coalescing` 可以理解为 GPU 把同一组相邻 thread 对连续 global memory 地址的访问，合并成更少的 memory transaction。这样可以减少内存事务数量，提高有效带宽利用率，并减少等待 global memory 的时间。

### 6. 连续访问 vs 跨步访问

**Question:** 下面两种访问模式，哪一种更可能 coalescing 友好？

```cpp
// A
int idx = blockIdx.x * blockDim.x + threadIdx.x;
x = a[idx];

// B
int idx = blockIdx.x * blockDim.x + threadIdx.x;
x = a[idx * 16];
```

**Explanation:** 是否 coalescing 友好，关键看相邻 thread 是否访问相邻地址。

**Correct Answer:** A 更 coalescing 友好。A 中相邻 thread 访问 `a[0]`、`a[1]`、`a[2]` 等连续地址；B 中相邻 thread 访问 `a[0]`、`a[16]`、`a[32]` 等跨步地址，硬件更难把访问合并成少量 memory transaction。

### 7. stride access 为什么慢

**Question:** 为什么 `a[idx * 16]` 这种 stride access 可能会让 global memory 访问变慢？

**Explanation:** `stride access` 可以理解为跨步访问或步长访问，也就是相邻访问之间有固定间隔，而不是连续访问。

**Correct Answer:** `a[idx * 16]` 会让相邻 thread 访问 `a[0]`、`a[16]`、`a[32]` 这类不相邻地址。硬件难以把这些访问合并成少量 memory transaction，可能需要更多内存事务，每次事务的有效数据利用率更低，因此 global memory bandwidth 利用率下降。

### 8. vector add 为什么是 memory bandwidth-bound

**Question:** 为什么 vector add 通常是一个 `memory bandwidth-bound` 的 kernel？

```cpp
c[idx] = a[idx] + b[idx];
```

它的计算量和内存读写量哪个更突出？

**Explanation:** 判断一个 kernel 的瓶颈时，要看它主要花时间在计算上，还是花在读写内存上。

**Correct Answer:** vector add 每个元素通常只有 1 次加法，但需要读 `a[idx]`、读 `b[idx]`、写 `c[idx]`。计算量很少，global memory 读写量更突出，所以性能通常受限于 global memory bandwidth。

### 9. bandwidth-bound 的优化重点

**Question:** 如果一个 kernel 是 `memory bandwidth-bound`，优化重点应该更偏向哪一类？

```text
A. 减少不必要的 global memory 访问，提高 memory coalescing
B. 增加更多复杂数学计算，让 GPU 更忙
```

**Explanation:** 瓶颈在哪里，优化重点就应该优先放在哪里。vector add 这类简单 kernel 的瓶颈通常不是算力。

**Correct Answer:** 选 A。对 memory bandwidth-bound kernel，优化重点通常是减少不必要的 global memory 访问，让访问更连续、更 coalesced，提高有效带宽利用率，并避免重复读写。

### 10. 今天的核心直觉

**Question:** 用一句话总结：为什么 GPU 喜欢相邻 thread 访问相邻 global memory 地址？请尽量使用 `coalescing`、`memory transaction`、`bandwidth`、`global memory` 这些词。

**Explanation:** 这里的关键不是 global memory 使用量变少，也不是内存申请释放变少，而是 global memory 访问请求更容易被硬件合并。

**Correct Answer:** GPU 喜欢相邻 thread 访问相邻 global memory 地址，因为这种模式更容易 coalescing，能减少 memory transaction 数量，提高 global memory bandwidth 利用率。

## 今日总结

今天已经理解：

- global memory 是 GPU device 侧的全局内存，通常是显存的一部分
- `cudaMalloc` 分配的是 device global memory
- host memory 中的数据要通过 `cudaMemcpyHostToDevice` 拷到 global memory
- kernel thread 可以访问 global memory
- global memory 比 register / shared memory 慢，但容量大
- vector add 中相邻 thread 通常访问相邻数组元素
- memory coalescing 是把相邻 thread 的连续访问合并成更少的 memory transaction
- `a[idx]` 比 `a[idx * 16]` 更 coalescing-friendly
- stride access 是跨步访问，会降低 coalescing 效果
- vector add 通常是 memory bandwidth-bound
- memory bandwidth-bound kernel 的优化重点是减少不必要 global memory 访问，提高 coalescing 和有效带宽

## 易错点

- global memory 在 GPU device 侧，不在 CPU 侧。
- host 普通指针不能直接当作 device global memory 指针使用。
- stride access 是跨步访问，不是连续访问。
- coalescing 优化的是 memory transaction 和带宽利用率，不是减少 `cudaMalloc` / `cudaFree` 次数。
- global memory 使用量不一定变少，但访问效率可以变高。

## 下一步

- 对比连续访问和 stride access 的 benchmark
- 记录不同 stride 下的 kernel time
- 后续学习 shared memory 基础
- 后续使用 Nsight Compute 观察 global memory load/store 指标
