# Day 008: Shared Memory Bank Conflict

Date: 2026-06-12

## 今日目标

今天继续 CUDA 主线，学习 shared memory bank conflict 的基础直觉。重点理解 warp、shared memory bank、连续访问、stride 访问、padding，以及为什么 matrix transpose 常用 `tile[32][33]`。

## 10 个概念问题

### 1. shared memory bank 是什么

**Question:** 在 CUDA 里，`shared memory bank` 大概是什么？请从 shared memory 为什么会被分成多个 bank、一个 warp 里的多个 thread 同时访问 shared memory 时什么情况下比较快、什么情况下可能发生 bank conflict 几个角度解释。

**Explanation:** shared memory 内部不是一整块单通道内存，而是分成多个可以并行访问的小通道，通常叫 banks，中文可以理解为存储体。

**Correct Answer:** shared memory bank 是 shared memory 内部的存储体。一个 warp 的 32 个 thread 如果访问分布在不同 bank 上的数据，硬件可以并行服务，速度快。如果多个 thread 同时访问同一个 bank 的不同地址，就可能发生 bank conflict，访问会被拆成多次执行。

### 2. 为什么 shared memory 也可能变慢

**Question:** 为什么 shared memory 虽然很快，但仍然可能因为访问模式变慢？请尽量使用 `warp`、`bank`、`bank conflict`、`并行访问`。

**Explanation:** shared memory 的速度优势依赖于访问能否被多个 bank 并行服务。

**Correct Answer:** 一个 warp 中的多个 thread 同时访问 shared memory 时，如果它们访问不同 bank，通常可以并行访问。如果多个 thread 访问同一个 bank 的不同地址，就会发生 bank conflict，硬件需要串行化或拆分访问，因此变慢。

### 3. 连续 float 访问为什么理想

**Question:** 假设 shared memory 有 32 个 bank，`float` 是 4 bytes。下面这种访问通常为什么比较理想？

```cpp
__shared__ float s[32];

int tid = threadIdx.x;
float x = s[tid];
```

**Explanation:** 常见情况下，连续的 `float` 元素会按顺序映射到不同 bank。

**Correct Answer:** 一个 warp 有 32 个 thread，`thread 0` 到 `thread 31` 分别访问 `s[0]` 到 `s[31]`。这些连续地址通常映射到 bank 0 到 bank 31，因此可以并行访问，通常不会产生 bank conflict。

### 4. stride 32 为什么容易冲突

**Question:** 下面这个访问模式为什么容易产生 bank conflict？

```cpp
__shared__ float s[32 * 32];

int tid = threadIdx.x;
float x = s[tid * 32];
```

**Explanation:** 对 `float` 数组可以用简化规则 `bank_id = index % 32` 理解 bank 映射。

**Correct Answer:** `thread 0` 访问 `s[0]`，`thread 1` 访问 `s[32]`，`thread 2` 访问 `s[64]`。这些 index 对 32 取模都等于 0，所以 32 个 thread 都访问 bank 0 的不同地址，容易产生严重 bank conflict。

### 5. stride 33 为什么减少冲突

**Question:** 如果把访问改成下面这样，为什么冲突会明显减少？

```cpp
__shared__ float s[32 * 33];

int tid = threadIdx.x;
float x = s[tid * 33];
```

**Explanation:** stride 从 32 改为 33 后，bank 映射会和 32 个 bank 错开。

**Correct Answer:** `thread 0` 访问 `s[0]`，映射到 bank 0；`thread 1` 访问 `s[33]`，映射到 bank 1；`thread 2` 访问 `s[66]`，映射到 bank 2。一个 warp 的 32 个 thread 会分散到 bank 0 到 bank 31，因此冲突明显减少。

### 6. `tile[32][33]` 的意义

**Question:** 为什么 matrix transpose 里经常看到这种写法？

```cpp
__shared__ float tile[32][33];
```

它和下面这种写法相比，主要是在避免什么问题？

```cpp
__shared__ float tile[32][32];
```

**Explanation:** `tile[32][33]` 是 padding 技巧，每行多 1 个元素，用来改变按列访问时的 stride。

**Correct Answer:** `tile[32][32]` 每行正好 32 个 `float`，按列访问时相邻 thread 的地址间隔是 32，容易落到同一个 bank。`tile[32][33]` 每行多 1 个 padding 元素，按列访问时 stride 变成 33，bank 映射会错开，从而减少 bank conflict。

### 7. row-major 下哪个访问更容易冲突

**Question:** 下面两种 shared memory 访问，哪个更容易 bank conflict？为什么？

```cpp
// A
float x = tile[threadIdx.x][0];

// B
float x = tile[0][threadIdx.x];
```

假设：

```cpp
__shared__ float tile[32][32];
```

**Explanation:** CUDA C/C++ 二维数组通常是 row-major，`tile[row][col]` 的线性 index 是 `row * 32 + col`。

**Correct Answer:** A 更容易 bank conflict。`tile[threadIdx.x][0]` 的 index 是 `tid * 32`，stride 是 32，容易全部映射到同一个 bank。B 是连续访问 `tile[0][0..31]`，通常分散到 bank 0 到 bank 31。

### 8. `tile[32][33]` 如何改善按列访问

**Question:** 如果把声明改成：

```cpp
__shared__ float tile[32][33];
```

那么下面这个访问为什么会比刚才好很多？

```cpp
float x = tile[threadIdx.x][0];
```

请从线性 index 公式解释：

```text
index = row * 33 + col
bank_id = index % 32
```

**Explanation:** padding 让相邻行的同一列不再相隔 32 个 `float`，而是相隔 33 个 `float`。

**Correct Answer:** 对 `tile[threadIdx.x][0]`，`row 0` 的 index 是 0，映射到 bank 0；`row 1` 的 index 是 33，映射到 bank 1；`row 2` 的 index 是 66，映射到 bank 2。一个 warp 的访问会分散到不同 bank，因此比 `tile[32][32]` 更少冲突。

### 9. padding 的代价

**Question:** padding 虽然能减少 bank conflict，但它有没有代价？比如 `tile[32][32]` 和 `tile[32][33]` 相比，后者多用了什么资源？这种代价通常大不大？什么时候可能需要注意？

**Explanation:** padding 是用额外 shared memory 换取更好的访问模式。

**Correct Answer:** `tile[32][33]` 每行多 1 个 `float`，总共多 32 个 `float`。这些元素通常是 padding，不存真正的矩阵数据。代价是多占用一点 shared memory，通常不大；但如果 kernel shared memory 用量本来就很高，可能降低 occupancy。

### 10. 今日核心直觉

**Question:** 为什么 shared memory bank conflict 会让程序变慢？为什么 padding 可以缓解它？请尽量使用 `warp`、`bank` / `存储体`、`stride`、`padding`、`并行访问`。

**Explanation:** bank conflict 的本质是一个 warp 的访问没有被很好地分散到多个 bank。

**Correct Answer:** 一个 warp 内多个 thread 如果因为 stride 访问落到同一个 bank 的不同地址，硬件通常需要串行化或拆分访问，无法充分并行访问，所以程序变慢。padding 可以改变 stride，让访问分散到不同 bank，从而缓解 bank conflict。

## 今日总结

今天已经理解：

- `warp` 通常是 32 个 thread 的硬件执行单位
- `bank` 可以翻译为 shared memory 的“存储体”
- shared memory 被分成多个 bank，是为了支持并行访问
- 连续访问 `s[threadIdx.x]` 通常会分散到不同 bank
- `s[threadIdx.x * 32]` 这种 stride 32 访问容易让所有 thread 撞到同一个 bank
- `tile[32][33]` 是 padding 技巧，用额外 shared memory 改变 stride
- 在 row-major 布局里，`tile[row][col]` 的线性 index 可以按 `row * width + col` 理解
- matrix transpose 常用 padding 避免按列访问 shared memory 时的 bank conflict

## 易错点

- 不是“访问同一个 bank”一定冲突；如果访问同一个 bank 的同一个地址，可能 broadcast。
- `tile[32][33]` 不是只多 1 个元素，而是每行多 1 个，总共多 32 个 `float`。
- padding 有代价，会多占 shared memory；通常很小，但 shared memory 用量很高时可能影响 occupancy。

## 下一步

- 写一个 matrix transpose naive vs shared memory padding 版本
- 对比 `tile[32][32]` 和 `tile[32][33]` 的性能差异
- 用 profiler 观察 shared memory bank conflict 相关指标
