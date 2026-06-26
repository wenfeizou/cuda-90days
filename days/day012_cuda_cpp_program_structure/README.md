# Day 012: CUDA C++ Program Structure

Date: 2026-06-26

## 今日目标

结合 C++ / Python 学习背景，通过 10 个交互式问答理解一个最小 CUDA C++ 程序的结构：host code 在 CPU 上负责准备数据、分配内存、发起 kernel；device code 在 GPU 上并行执行计算。

## 10 个概念问题

### 1. `main()` 和 `__global__` kernel 分别运行在哪里？

**Question:** 在一个 CUDA C++ 程序里，下面两段代码分别运行在哪里？

```cpp
int main() {
    // ...
}
```

和：

```cpp
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    // ...
}
```

请说明哪一段在 CPU 上运行，哪一段在 GPU 上运行。

**Explanation:** CUDA 程序通常同时包含 host code 和 device code。理解它们分别在哪里执行，是理解后续内存分配、数据拷贝和 kernel launch 的基础。

**Correct Answer:** `main()` 是 host code，运行在 CPU 上；`__global__` 修饰的 `vectorAdd` 是 kernel，由 CPU 端发起 launch，但函数体在 GPU 上由大量 thread 并行执行。

### 2. kernel launch 调用的组成

**Question:** 下面这个调用是什么意思？

```cpp
vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
```

请分别解释：

```text
vectorAdd
<<<numBlocks, blockSize>>>
d_a, d_b, d_c, n
```

**Explanation:** CUDA kernel launch 看起来像函数调用，但中间的 `<<<...>>>` 是 CUDA 特有的 launch 配置，不是普通函数参数。

**Correct Answer:** `vectorAdd` 是 kernel 函数名；`<<<numBlocks, blockSize>>>` 是 kernel launch 配置，表示启动多少个 block、每个 block 有多少个 thread；`d_a, d_b, d_c, n` 是传给 kernel 函数体的实参，其中 `d_a/d_b/d_c` 通常是 device pointer，`n` 是元素数量。

### 3. `__global__` 的含义

**Question:** 在 CUDA C++ 里，为什么 kernel 函数通常要加这个修饰符？

```cpp
__global__
```

例如：

```cpp
__global__ void vectorAdd(float* a, float* b, float* c, int n)
```

`__global__` 表示这个函数由谁调用、在哪里执行？

**Explanation:** CUDA 使用函数修饰符区分函数的调用位置和执行位置。`__global__` 是 kernel 函数最常见的修饰符。

**Correct Answer:** `__global__` 表示函数由 host 端调用，并在 device 端执行。也就是 CPU 通过 `kernel<<<...>>>(...)` 发起 launch，GPU 执行 kernel 函数体。

### 4. 为什么同一份 kernel 代码能处理不同元素？

**Question:** 在下面这段 kernel 中：

```cpp
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

为什么每个 thread 都会执行同一份代码，但最终处理的是不同的数组元素？

**Explanation:** CUDA kernel 是 many threads execute the same code 的模型。不同 thread 的差异主要来自内置索引变量。

**Correct Answer:** 每个 thread 都执行同一个 `vectorAdd` 函数体，但每个 thread 的 `blockIdx.x` 和 `threadIdx.x` 不同，所以 `idx = blockIdx.x * blockDim.x + threadIdx.x` 的结果不同。不同的 `idx` 对应不同数组元素。

### 5. 最小 CUDA C++ 程序步骤顺序

**Question:** 在 CUDA C++ 程序里，下面这些步骤通常应该按什么顺序出现？

```text
A. cudaMemcpy 把输入从 host 拷到 device
B. cudaMalloc 在 device 上分配内存
C. 在 host 上准备输入数据
D. 启动 kernel
E. cudaMemcpy 把结果从 device 拷回 host
F. cudaFree 释放 device memory
```

请给出顺序，并简单说明原因。

**Explanation:** CUDA host 端流程有清晰的数据依赖：先准备和分配，再拷贝输入，再计算，再拷贝输出，最后释放资源。

**Correct Answer:** 正确顺序是 `C -> B -> A -> D -> E -> F`，也就是：

```text
prepare -> allocate -> copy in -> compute -> copy out -> cleanup
```

### 6. `new` 和 `cudaMalloc` 的区别

**Question:** 下面两行代码有什么区别？

```cpp
float* h_a = new float[n];
```

和：

```cpp
cudaMalloc(&d_a, n * sizeof(float));
```

请从“在哪里分配内存”和“谁能直接访问”两个角度回答。

**Explanation:** CUDA 程序经常同时存在 host pointer 和 device pointer。区分它们能避免把 CPU 内存和 GPU 内存混用。

**Correct Answer:** `new float[n]` 在 host memory / CPU 内存中分配 `n` 个 `float`，CPU 代码可以直接访问。`cudaMalloc` 在 device/global memory / GPU 显存中分配 `n * sizeof(float)` 字节，GPU kernel 可以直接访问；CPU 代码通常不能像普通数组一样直接解引用 device pointer。

### 7. `cudaMemcpy` 为什么要指定方向？

**Question:** `cudaMemcpy` 里为什么需要指定方向？比如：

```cpp
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
```

如果方向写反了，可能会发生什么？

**Explanation:** `cudaMemcpy` 的源地址和目标地址可能属于不同内存空间，必须明确数据方向。

**Correct Answer:** `cudaMemcpy` 的参数顺序是 `cudaMemcpy(dst, src, size, direction)`。方向用于说明数据是在 host 和 device 之间如何移动。如果方向写反，可能 API 返回错误，也可能导致结果错误，甚至用错误数据覆盖目标缓冲区。

### 8. kernel launch 默认同步还是异步？

**Question:** CUDA kernel launch 默认是同步还是异步的？也就是说，CPU 执行到：

```cpp
vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
```

之后，会等 GPU kernel 完全执行完再继续下一行吗？还是通常会继续往下执行？

**Explanation:** CUDA host code 和 device work 的执行时间线并不完全同步。理解异步 launch 对调试和性能都很重要。

**Correct Answer:** kernel launch 对 host 来说通常是异步的。CPU 发起 kernel 后通常马上继续执行下一行，不会自动等待 GPU kernel 完成。常见同步点包括 `cudaDeviceSynchronize()` 和同步的 device-to-host `cudaMemcpy`。

### 9. `cudaGetLastError()` 和 `cudaDeviceSynchronize()`

**Question:** 为什么 kernel launch 后经常要写这两句？

```cpp
cudaGetLastError();
cudaDeviceSynchronize();
```

它们分别主要检查什么问题？

**Explanation:** kernel launch 可能有立即暴露的 launch 错误，也可能有执行过程中才暴露的 runtime 错误。

**Correct Answer:** `cudaGetLastError()` 主要检查 kernel launch 是否立刻失败，例如 launch 配置非法、参数错误等。`cudaDeviceSynchronize()` 等待 GPU 前面提交的任务完成，并可能暴露 kernel 执行过程中的错误，例如 illegal memory access。

### 10. 最小 CUDA C++ host 端流程

**Question:** 请用自己的话总结一个最小 CUDA C++ 程序的 host 端流程，从准备数据到释放资源，大概有哪些步骤？

**Explanation:** 这是把今天所有知识串起来的主线。CUDA C++ 程序通常由 CPU 端完成流程编排，由 GPU 端完成并行计算。

**Correct Answer:** 一个最小 CUDA C++ 程序的 host 端流程通常是：

```text
prepare -> allocate -> copy in -> compute -> copy out -> cleanup
```

对应代码层面：

```cpp
// 1. host 准备数据
float* h_a = new float[n];

// 2. device 分配内存
cudaMalloc(&d_a, size);

// 3. host -> device
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

// 4. 启动 kernel
vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

// 5. device -> host
cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

// 6. 释放资源
cudaFree(d_a);
delete[] h_a;
```

## 今日总结

今天已经理解：

- `main()` 是 host code，在 CPU 上运行。
- `__global__` kernel 由 CPU 发起，在 GPU 上执行。
- `<<<numBlocks, blockSize>>>` 是 kernel launch 配置，不是普通函数参数。
- kernel 实参如 `d_a, d_b, d_c, n` 才是传入函数体的数据。
- 每个 thread 执行同一份 kernel 代码，但通过不同 `idx` 处理不同元素。
- `new float[n]` 分配 host memory。
- `cudaMalloc` 分配 device/global memory。
- `cudaMemcpy(dst, src, size, direction)` 需要明确方向。
- kernel launch 默认对 CPU 异步。
- `cudaGetLastError()` 检查 launch 错误。
- `cudaDeviceSynchronize()` 等待 kernel 完成并暴露运行期错误。

## 易错点

- `<<<...>>>` 不是传给 kernel 函数体的普通参数，而是 launch 配置。
- `n` 是元素数量，不是最大索引；有效索引是 `0 ~ n - 1`。
- `cudaGetLastError()` 不负责等待 kernel 完成。
- CPU 不能像访问普通数组一样直接访问 `cudaMalloc` 得到的 device pointer。
- kernel launch 后 CPU 通常继续往下执行，除非遇到同步点。

## 下一步

- 写一个完整 `vectorAdd.cu`
- 练习从 C++ host code 到 CUDA kernel 的完整编译和运行
- 增加 CUDA error checking helper
- 对比 Python for loop、C++ for loop 和 CUDA kernel 的执行模型
