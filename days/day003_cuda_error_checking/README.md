# Day 003: CUDA Error Checking

Date: 2026-06-09

## 今日目标

今天继续轻量学习 CUDA 基础概念，重点把前两天的 kernel launch、同步和错误暴露机制落成最小工程习惯：检查 CUDA API 返回值，使用 `CUDA_CHECK` 宏，并在调试阶段正确检查 kernel launch error 和 runtime error。

## 10 个概念问题

### 1. `cudaError_t` / `cudaSuccess`

**Question:** 在 CUDA Runtime API 里，很多函数都会返回一个值，比如：

```cpp
cudaError_t err = cudaMalloc(&d_a, size);
```

请解释 `cudaError_t`、`cudaSuccess`、`err != cudaSuccess` 分别是什么意思。

**Explanation:** CUDA Runtime API 通常通过返回值报告调用是否成功。理解这个返回值，是写错误检查逻辑的基础。

**Correct Answer:** `cudaError_t` 是 CUDA Runtime API 返回的错误类型；`cudaSuccess` 表示调用成功、没有错误；`err != cudaSuccess` 表示返回值不是成功状态，也就是 CUDA API 调用失败。

### 2. 为什么要检查 `cudaMalloc`

**Question:** 下面这段代码从“错误检查”的角度看有什么问题？

```cpp
float* d_a;
cudaMalloc(&d_a, size);

vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
```

**Explanation:** `cudaMalloc` 可能失败。如果失败后继续使用 `d_a`，后面的 kernel 可能触发更难定位的问题。

**Correct Answer:** 这段代码没有检查 `cudaMalloc` 的返回值。应该保存并检查 `cudaError_t`：

```cpp
float* d_a = nullptr;

cudaError_t err = cudaMalloc(&d_a, size);
if (err != cudaSuccess) {
    printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
    return;
}
```

### 3. `cudaGetErrorString`

**Question:** `cudaGetErrorString(err)` 的作用是什么？为什么比只打印错误码更有用？

```cpp
printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
```

**Explanation:** `cudaError_t` 是错误枚举值。调试时只看到数字或枚举名，通常不如可读文本直接。

**Correct Answer:** `cudaGetErrorString(err)` 会把 CUDA 错误转换成可读字符串，比如 `out of memory`、`invalid argument`、`invalid device pointer`。它能让错误信息更容易理解和定位。

### 4. 为什么需要 `CUDA_CHECK`

**Question:** 如果每次都写下面这几行很重复，为什么很多 CUDA 程序会写一个 `CUDA_CHECK(...)` 宏？

```cpp
cudaError_t err = cudaMalloc(&d_a, size);
if (err != cudaSuccess) {
    printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
    return;
}
```

**Explanation:** CUDA API 调用很多，如果每次手写错误检查，代码会啰嗦，并且容易漏掉某一次检查。

**Correct Answer:** `CUDA_CHECK(...)` 用来封装重复的 CUDA API 错误检查逻辑，减少样板代码，避免漏检查，并且可以统一打印文件名、行号和错误字符串。

### 5. `do { ... } while (0)` 宏写法

**Question:** 为什么 `CUDA_CHECK` 宏里经常写成这种形式？

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            ... \
        } \
    } while (0)
```

为什么不用简单的多行宏展开？

**Explanation:** 这是 C/C++ 多行宏的常见安全写法。宏展开后如果不像一条语句，容易破坏 `if/else` 等控制流结构。

**Correct Answer:** `do { ... } while (0)` 让多行宏在语法上表现得像一条普通语句，可以安全地写在 `if/else` 等上下文里，并且调用处能稳定地以分号结尾。

### 6. kernel launch 能不能直接放进 `CUDA_CHECK`

**Question:** 下面这种写法能不能直接检查 kernel 内部运行错误？这段代码本身有什么问题？

```cpp
CUDA_CHECK(vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n));
```

**Explanation:** `CUDA_CHECK` 包装的是会返回 `cudaError_t` 的 CUDA Runtime API，而 kernel launch 语法不是普通的返回 `cudaError_t` 的函数调用。

**Correct Answer:** 不能这样写。kernel launch 不能直接塞进 `CUDA_CHECK(...)`。正确做法是先启动 kernel，再检查 launch error 和 runtime error：

```cpp
vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

### 7. `cudaGetLastError` 与 `cudaDeviceSynchronize`

**Question:** 下面两行分别检查什么？为什么调试阶段通常两行都写？

```cpp
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

**Explanation:** kernel launch 可能在启动阶段失败，也可能启动成功但运行过程中失败。调试时需要区分这两类错误。

**Correct Answer:** `CUDA_CHECK(cudaGetLastError())` 通常检查 kernel launch error，例如配置非法或启动失败；`CUDA_CHECK(cudaDeviceSynchronize())` 会等待 GPU 执行完成，并暴露 kernel runtime error，例如 illegal memory access。调试阶段两行都写，可以更快判断错误发生在 launch 阶段还是运行阶段。

### 8. benchmark 中为什么不能乱加同步

**Question:** 为什么在 benchmark / 性能测试代码里，不能随便在每个 kernel 后面都加：

```cpp
cudaDeviceSynchronize();
```

它可能会怎样影响性能测量？

**Explanation:** `cudaDeviceSynchronize()` 不只是在出错时才有影响。无论有没有错误，它都会让 CPU 等 GPU 前面提交的工作完成。

**Correct Answer:** 在 benchmark 中随便加 `cudaDeviceSynchronize()` 会强行打断 GPU 异步执行，破坏 kernel 之间可能的重叠或流水，把额外同步等待时间算进测量里，使结果不能代表真实 pipeline 性能。调试阶段同步有价值，性能测试阶段应谨慎放在明确的测量边界。

### 9. 最小 `CUDA_CHECK` 宏条件

**Question:** 补全一个最小 CUDA API 错误检查宏的核心逻辑：

```cpp
#define CUDA_CHECK(call)                         \
    do {                                         \
        cudaError_t err = (call);                \
        if (__________) {                        \
            fprintf(stderr, "CUDA error: %s\n",  \
                    cudaGetErrorString(err));    \
            exit(1);                             \
        }                                        \
    } while (0)
```

空白处应该写什么？为什么？

**Explanation:** CUDA API 返回 `cudaSuccess` 表示成功，因此判断失败时要检查返回值是否不是 `cudaSuccess`。

**Correct Answer:** 空白处应写：

```cpp
err != cudaSuccess
```

完整含义是：如果 CUDA API 返回值不是 `cudaSuccess`，就说明调用失败，需要打印错误并停止程序或进行错误处理。

### 10. 调试阶段 kernel 调用模板

**Question:** 请按顺序写出一个“调试阶段”的最小 CUDA kernel 调用模板，包含：

```text
1. kernel launch
2. 检查 launch error
3. 同步并检查 runtime error
```

**Explanation:** 调试 CUDA kernel 时，既要检查启动是否成功，也要让运行时错误尽早暴露。kernel 参数通常直接传 device pointer，例如 `d_a`，而不是 `&d_a`。

**Correct Answer:** 最小调试模板是：

```cpp
vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

CUDA_CHECK(cudaGetLastError());       // launch error
CUDA_CHECK(cudaDeviceSynchronize());  // runtime error, debug only
```

`cudaGetLastError()` 和 `cudaDeviceSynchronize()` 都是返回 `cudaError_t` 的 CUDA Runtime API，因此可以被 `CUDA_CHECK(...)` 包装。

## 今日总结

今天已经理解：

- `cudaError_t` 是 CUDA Runtime API 的错误返回类型
- `cudaSuccess` 表示成功
- `err != cudaSuccess` 表示出错
- CUDA API 返回值要检查
- `cudaGetErrorString(err)` 用于输出可读错误信息
- `CUDA_CHECK` 宏用于封装重复错误检查
- `do { ... } while (0)` 是多行宏的安全写法
- kernel launch 不能直接塞进 `CUDA_CHECK`
- kernel launch 后用 `cudaGetLastError()` 检查 launch error
- 调试时用 `cudaDeviceSynchronize()` 暴露 runtime error
- benchmark 中不能随便加 synchronize

## 易错点

- `err != cudaSuccess` 表示出错，不是成功。
- `CUDA_CHECK` 是 host 端 C/C++ 宏，不是 device 端逻辑。
- kernel launch 语法不是返回 `cudaError_t` 的普通函数调用。
- device pointer 参数通常传 `d_a`，不是 `&d_a`。
- `cudaDeviceSynchronize()` 对调试有帮助，但会影响 benchmark 语义。

## 下一步

- 把 `CUDA_CHECK` 宏加入第一个 vector add 程序
- 对每个 CUDA API 调用加错误检查
- 在调试版本中加入 kernel 后的 launch/runtime 检查
- 后续学习 global memory 访问与 coalescing 入门
