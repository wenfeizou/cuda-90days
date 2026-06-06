# Day 002: CUDA Synchronization, Error Checking, and Timing

Date: 2026-06-06

## 今日目标

今天继续轻量学习 CUDA 基础概念，重点理解 kernel launch 的异步特性、错误检查、同步点，以及为什么 GPU 程序计时和性能判断不能只按普通 CPU 程序的直觉来做。

## 10 个概念问题

### 1. Kernel launch 是否会等待 GPU 完成

**Question:** CUDA kernel launch 通常是异步的。下面代码中，CPU 执行到 kernel launch 后，会不会一定等 GPU 把 `vectorAdd` 全部算完，再继续执行 `printf`？

```cpp
vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
printf("kernel launched\n");
```

**Explanation:** `kernel launch` 可以理解为 CPU 向 GPU 提交一个并行任务，而不是 CPU 像普通函数调用那样自己进入函数并等待执行完成。

**Correct Answer:** 不会一定等待。默认情况下，kernel launch 通常是异步的，CPU 提交任务后会继续往下执行。`kernel launch` 中文可理解为“启动核函数”或“启动一个 GPU 核函数”。

### 2. `printf` 是否说明 kernel 已完成

**Question:** 下面代码中，如果 `printf("after launch\n")` 被打印出来，能不能说明 `vectorAdd` 已经正确执行完成？

```cpp
vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
printf("after launch\n");
```

**Explanation:** CPU 打印这行日志，只说明 CPU 已经执行到 launch 后面的语句，并不代表 GPU 已经完成前面提交的工作。

**Correct Answer:** 不能。`printf` 被执行只能说明 CPU 已经提交 kernel launch 并继续往后执行，不能证明 GPU 已完成 kernel，也不能证明 kernel 结果正确或没有运行时错误。

### 3. `cudaDeviceSynchronize`

**Question:** `cudaDeviceSynchronize()` 的作用是什么？它会让 CPU 等待什么？为什么调试 CUDA 程序时经常需要它？

```cpp
cudaDeviceSynchronize();
```

**Explanation:** kernel launch 是异步的，因此 CPU 端需要一个同步点来等待 GPU 完成前面提交的任务。

**Correct Answer:** `cudaDeviceSynchronize()` 会让 CPU 等待当前 device 前面提交的任务完成。它常用于调试，因为 kernel 运行时错误经常会在这里暴露。它不会自动把结果拷回 CPU；结果如果在 `d_c` 中，仍需要 `cudaMemcpyDeviceToHost` 拷回 `h_c`。

### 4. kernel 内部错误为什么不一定立刻报

**Question:** 如果 kernel 中发生非法内存访问，比如某个 thread 写了 `c[n + 100]`，为什么 CPU 端不一定会在 kernel launch 这一行立刻报错？

```cpp
vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
```

**Explanation:** kernel launch 这一行主要负责提交任务，而不是等待 GPU 完整执行任务。kernel 内部运行时错误往往要到后续同步点才暴露。

**Correct Answer:** 因为 kernel 在 GPU 端异步执行，CPU 端提交 launch 后会继续往后执行。非法内存访问等运行时错误可能会在 `cudaDeviceSynchronize()`、`cudaMemcpy()` 等同步点才被发现。

### 5. `cudaGetLastError`

**Question:** `cudaGetLastError()` 通常更适合检查什么？

```text
A. kernel launch 配置或启动阶段的错误
B. kernel 内部运行完成后的所有计算结果是否正确
```

**Explanation:** `cudaGetLastError()` 的名字容易误导。它检查 CUDA runtime 记录的最近一次错误，但不能证明 kernel 的计算结果正确。

**Correct Answer:** 选 A。`cudaGetLastError()` 常用于 kernel launch 后检查启动阶段错误，比如配置非法、参数错误、launch 失败等。计算结果是否正确需要把结果拷回 host 后做 correctness check。

### 6. launch error 与 runtime error

**Question:** 下面两段检查分别主要检查什么？

```cpp
vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

cudaError_t err = cudaGetLastError();
```

和：

```cpp
cudaError_t err = cudaDeviceSynchronize();
```

**Explanation:** CUDA 调试时要区分 kernel 是否成功启动，以及 kernel 运行期间是否出错。

**Correct Answer:** `cudaGetLastError()` 检查最近一次 CUDA runtime 错误，常用于检查 kernel launch error。`cudaDeviceSynchronize()` 等待 GPU 前面提交的工作完成，常用于暴露 kernel runtime error。

### 7. `cudaMemcpy` 为什么可能暴露前面 kernel 的错误

**Question:** 为什么有时候下面这行 `cudaMemcpy` 也会暴露前面 kernel 的错误？

```cpp
cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
```

**Explanation:** 对于默认 stream 中的同步拷贝，`cudaMemcpyDeviceToHost` 通常需要等待前面提交到同一 stream 的 kernel 完成，才能把结果从 device 拷回 host。

**Correct Answer:** 因为 `cudaMemcpy` 可能成为同步点。前面 kernel 中发生的异步错误，例如 illegal memory access，可能不会在 launch 行立刻出现，而是在后续 `cudaMemcpy` 或 `cudaDeviceSynchronize` 时暴露。

### 8. 普通 CPU 计时为什么不准

**Question:** 为什么下面这种普通 CPU 计时方式通常不能准确测量 kernel 真正执行耗时？

```cpp
auto start = std::chrono::high_resolution_clock::now();

vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

auto end = std::chrono::high_resolution_clock::now();
```

**Explanation:** 由于 kernel launch 通常是异步的，CPU 记录 `end` 时，GPU kernel 可能还没有真正执行完成。

**Correct Answer:** 这段代码测到的主要是 CPU 提交 kernel launch 的开销，而不是 GPU 执行 kernel 的耗时。如果用 CPU 计时，至少要在 kernel 后加 `cudaDeviceSynchronize()`；更常见的 GPU kernel 计时方式是使用 CUDA events。

### 9. CUDA event 计时顺序

**Question:** CUDA event 计时的大致流程是什么？请按顺序排列：

```text
A. cudaEventRecord(stop)
B. cudaEventCreate(start / stop)
C. cudaEventElapsedTime(...)
D. kernel launch
E. cudaEventRecord(start)
F. cudaEventSynchronize(stop)
```

**Explanation:** CUDA event 计时需要在 GPU 工作前后分别记录 event，并等待 stop event 完成后再计算 elapsed time。

**Correct Answer:** 正确顺序是 `B -> E -> D -> A -> F -> C`。

```cpp
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

### 10. 小数据量 GPU 为什么可能更慢

**Question:** 为什么“小数据量”的 vector add，GPU 版本可能比 CPU 版本还慢？至少说出两个原因。

**Explanation:** GPU 擅长大规模并行任务，但启动 GPU 工作、传输数据和同步都有额外开销。小数据量时，这些固定开销可能超过并行计算收益。

**Correct Answer:** 常见原因包括：kernel launch 有固定开销；host/device 数据拷贝有开销；小数据量并行度不够，GPU 资源用不满；CPU cache 对简单 vector add 很快；如果加入 `cudaDeviceSynchronize()`，同步等待也会带来额外开销。

## 今日总结

今天已经理解：

- kernel launch 通常是异步的
- `printf` 出现在 launch 后面，不代表 kernel 已经完成
- `cudaDeviceSynchronize()` 等 GPU 完成，但不负责拷回结果
- kernel 内部错误可能延迟到同步点暴露
- `cudaGetLastError()` 常用于检查 launch error
- `cudaDeviceSynchronize()` 常用于暴露 runtime error
- `cudaMemcpy` 也可能暴露前面 kernel 的异步错误
- 普通 CPU 计时不加同步时测不到真实 kernel 耗时
- CUDA event 计时顺序是 `B -> E -> D -> A -> F -> C`
- 小数据量 GPU 不一定比 CPU 快

## 易错点

- `kernel launch` 不是普通同步函数调用，而是 CPU 向 GPU 提交异步任务。
- `cudaDeviceSynchronize()` 不会自动把 device memory 中的结果拷回 host memory。
- `cudaGetLastError()` 不能证明 kernel 计算结果正确。
- 报错位置可能在 `cudaMemcpy` 或同步点，但真正错误可能发生在前面的 kernel 中。

## 下一步

- 最小 CUDA 错误检查宏
- `cudaGetErrorString`
- CUDA stream 的基本概念
- global memory 访问与 coalescing 入门
