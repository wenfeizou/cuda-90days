# Glossary

本文件用于沉淀 CUDA、GPU、LLM inference 和 AI Infra 学习中的稳定概念。每日学习记录放在 `days/`，这里保存后续会反复查阅的术语解释。

## CUDA / GPU

### Kernel

CUDA kernel 是运行在 GPU 上、由大量 thread 并行执行的函数入口。CPU 端通过 kernel launch 启动它，例如：

```cpp
vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
```

kernel 不是普通 CPU 同步函数调用。默认情况下，CPU 提交 kernel 后通常会继续往下执行。

### Kernel Launch

kernel launch 指 CPU 向 GPU 提交并启动一个 kernel 任务。中文可以叫“启动核函数”或“kernel 启动”。

它更像：

```text
CPU 向 GPU 提交异步任务
```

而不是：

```text
CPU 自己进入函数并等待执行完
```

### Thread / Block / Grid

CUDA 的并行层级是：

```text
grid -> block -> thread
```

- thread: 执行 kernel 代码的一份并行实例
- block: 一组 thread
- grid: 一次 kernel launch 启动出来的全部 block

### Global Thread Index

一维 kernel 中常见的全局索引计算：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

- `blockIdx.x`: 当前 block 在 x 维度上的索引
- `blockDim.x`: 每个 block 在 x 维度上的 thread 数量
- `threadIdx.x`: 当前 thread 在 block 内 x 维度上的索引
- `idx`: 一维场景下常用的全局 thread 索引

### Boundary Check

CUDA kernel 中常见：

```cpp
if (idx < n) {
    c[idx] = a[idx] + b[idx];
}
```

这是为了避免多出来的 thread 访问越界内存。因为 block 数通常向上取整，总 thread 数可能大于真实数据长度。

### Host / Device

- host: CPU 端，负责准备数据、分配 GPU 内存、发起 kernel
- device: GPU 端，负责执行 kernel
- host memory: CPU 使用的内存
- device memory: GPU 使用的显存

### Global Memory

CUDA 中的 global memory 通常指 GPU device 侧的全局内存，通常是显存的一部分。

通过：

```cpp
cudaMalloc(&d_a, size);
```

分配得到的 `d_a` 指向 device global memory。kernel thread 可以访问 global memory，但 host 普通指针不能直接当作 device global memory 指针使用。

### cudaMalloc

`cudaMalloc` 用于在 device memory / global memory 中分配空间：

```cpp
float* d_a = nullptr;
cudaMalloc(&d_a, size);
```

常见命名习惯：

```text
h_a = host 上的 a
d_a = device 上的 a
```

### cudaMemcpy

`cudaMemcpy` 用于在 host memory 和 device memory 之间拷贝数据。

Host 到 device：

```cpp
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
```

Device 到 host：

```cpp
cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
```

`size` 通常是字节数，不是元素个数。

### cudaDeviceSynchronize

`cudaDeviceSynchronize()` 会让 CPU 等待当前 device 前面提交的任务完成。

它常用于调试，因为 kernel 运行时错误经常会在这里暴露。它不会自动把 device memory 中的结果拷回 host memory。

### cudaGetLastError

`cudaGetLastError()` 用于检查 CUDA runtime 记录的最近一次错误，常用于 kernel launch 后检查 launch error。

典型调试模板：

```cpp
vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

CUDA_CHECK(cudaGetLastError());       // launch error
CUDA_CHECK(cudaDeviceSynchronize());  // runtime error, debug only
```

### cudaGetErrorString

`cudaGetErrorString(err)` 会把 `cudaError_t` 转换成可读字符串，例如：

```text
out of memory
invalid argument
invalid device pointer
```

### cudaError_t / cudaSuccess

`cudaError_t` 是 CUDA Runtime API 返回的错误类型。

```text
cudaSuccess        -> 成功
err == cudaSuccess -> 没错
err != cudaSuccess -> 有错
```

### CUDA_CHECK

`CUDA_CHECK` 是常见的 host 端 C/C++ 宏，用于封装重复的 CUDA API 错误检查逻辑。

示例：

```cpp
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__,                       \
                    cudaGetErrorString(err));                 \
            exit(1);                                          \
        }                                                     \
    } while (0)
```

`do { ... } while (0)` 让多行宏在语法上表现得像一条普通语句。

### CUDA Event Timing

CUDA event 可以用于测量 GPU kernel 时间。基本顺序：

```text
cudaEventCreate(start / stop)
cudaEventRecord(start)
kernel launch
cudaEventRecord(stop)
cudaEventSynchronize(stop)
cudaEventElapsedTime(...)
```

也就是：

```text
B -> E -> D -> A -> F -> C
```

### Register

register 是 GPU 中最快的存储之一，通常是 thread 私有的。访问速度很快，但容量有限。

### Shared Memory

shared memory 是 block 内 thread 可共享的高速片上内存。它通常比 global memory 快很多，但容量有限。

### Memory Bandwidth

memory bandwidth 是单位时间内内存能搬运多少数据。中文叫内存带宽。

在 GPU 中，它可以理解为：

```text
GPU 从 global memory 读取 / 写入数据的速度上限
```

常见单位是 GB/s 或 TB/s。

直觉类比：

```text
显存 = 仓库
GPU core = 工人
memory bandwidth = 仓库到工人之间的运输能力
```

如果 kernel 主要时间花在搬数据，而不是计算上，它通常是 memory bandwidth-bound。

### Memory Bandwidth-bound

memory bandwidth-bound 表示 kernel 的性能主要受内存带宽限制，而不是受计算能力限制。

例如 vector add：

```cpp
c[idx] = a[idx] + b[idx];
```

每个元素只有 1 次加法，但需要读 `a[idx]`、读 `b[idx]`、写 `c[idx]`，因此通常受 global memory bandwidth 限制。

### Memory Transaction

memory transaction 可以理解为 GPU 访问 global memory 时的一次内存事务。coalescing 好时，多个相邻 thread 的访问可以合并成更少的 memory transaction。

### Memory Coalescing

memory coalescing 指 GPU 把同一组相邻 thread 对连续 global memory 地址的访问，合并成更少的 memory transaction。

核心收益：

```text
减少 memory transaction 数量
提高 global memory bandwidth 利用率
减少等待 global memory 的时间
```

一句话：

```text
GPU 喜欢相邻 thread 访问相邻 global memory 地址。
```

### Stride Access

stride access 中文可以叫跨步访问或步长访问。它指相邻访问之间有固定间隔，而不是连续访问。

连续访问：

```cpp
a[idx]
```

相邻 thread 访问：

```text
a[0], a[1], a[2], a[3]
```

跨步访问：

```cpp
a[idx * 16]
```

相邻 thread 访问：

```text
a[0], a[16], a[32], a[48]
```

stride access 通常会降低 memory coalescing 效果。

## LLM Inference / SGLang

### LLM

LLM 是 Large Language Model，大语言模型。它主要处理文本输入和文本输出。

### VLM

VLM 是 Vision-Language Model，视觉语言模型。它可以处理图像 / 视频 + 文本输入，并输出文本。

### SGLang

SGLang 是面向 LLM / VLM 的高性能推理与 serving 框架。

它负责：

```text
接收请求
调度请求
管理 KV cache
组织 prefill / decode
提供 API 服务
提升吞吐与延迟表现
```

SGLang 是上层推理 runtime / serving 系统，CUDA kernel 是底层计算执行单元。

### Offline Inference

offline inference 是离线批量推理。它通常一次性跑一批 prompt 并获得结果，更关注总吞吐。

### Online Serving

online serving 是在线推理服务。它通常部署为 API 服务，等待用户请求，更关注延迟、并发、稳定性和请求调度。

### Prefill

prefill 阶段处理用户输入的 prompt，把 prompt tokens 一次性送进模型，并建立初始 KV cache。

它通常影响 TTFT。

### Decode

decode 阶段基于已有 KV cache，一个 token 一个 token 地生成输出。

它通常影响 TPOT 和输出流畅度。

### Attention

attention 中文通常翻译为注意力。它可以粗略理解为：

```text
模型在处理当前 token 时，应该更关注上下文里的哪些 token
```

更工程化地说：

```text
当前 token 根据 Query 去和历史 token 的 Key 做匹配，再从对应 Value 里汇总信息。
```

attention 本质是 token 之间的信息路由 / 加权聚合机制。

### Query / Key / Value

在 attention 中：

- Query: 当前 token 想找什么信息
- Key: 历史 token 提供什么索引 / 特征
- Value: 历史 token 携带的内容表示

当前 Query 和历史 Key 算相似度，再用相似度权重加权汇总 Value。

### KV Cache

KV cache 缓存的是 Transformer attention 中已经计算过的历史 token 的 Key / Value 表示。

它不是“当前 token 到下一个 token 的映射”。它更像：

```text
模型对历史上下文的中间计算记忆
```

decode 阶段特别依赖 KV cache，因为新 token 可以复用历史 K/V，避免重复计算所有历史 token。

### Scheduler

LLM serving scheduler 决定哪些请求在什么时候一起跑。

它负责在这些目标之间做权衡：

```text
GPU 显存
计算资源
请求延迟
系统吞吐
KV cache 管理
```

### TTFT

TTFT 是 Time To First Token，表示从请求发出到收到第一个输出 token 的时间。

它通常受排队时间、prefill 时间、prompt 长度、调度策略和模型大小影响。

### TPOT

TPOT 是 Time Per Output Token，表示生成阶段平均每个输出 token 花费的时间。

它更反映 decode 阶段速度。

### Radix

radix 在 RadixAttention 语境里可以理解为 radix tree / prefix tree，也就是用公共前缀组织字符串或 token 序列的数据结构。

直觉：

```text
radix = prefix-sharing data structure
```

### RadixAttention

RadixAttention 是 SGLang 中围绕共享前缀和 KV cache 复用的机制。

它的核心想法是：

```text
把 prompt/token 序列按共享前缀组织起来，让相同前缀对应的 KV cache 可以被复用。
```

常见收益：

```text
减少重复 prefill
降低 TTFT
提升吞吐
节省部分计算资源
```

### Prefix Cache

prefix cache 用于缓存和复用相同前缀的计算结果，尤其是对应的 KV cache。

适用场景：

```text
相同 system prompt
相同工具说明
相同文档上下文
相同 few-shot examples
```

一句话：

```text
不要重复理解已经理解过的上下文。
```
