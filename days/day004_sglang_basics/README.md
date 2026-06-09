# Day 004: SGLang Basics

Date: 2026-06-09

## 今日目标

今天穿插学习 SGLang，目标不是深入源码，而是建立 LLM serving runtime 的基础概念：SGLang 的定位、offline inference / online serving、prefill / decode、KV cache、scheduler、TTFT / TPOT，以及它和 CUDA / GPU 优化的关系。

## 10 个概念问题

### 1. SGLang 的定位

**Question:** SGLang 大致属于下面哪一类工具？

```text
A. CUDA kernel 编译器
B. Python 数据分析库
C. LLM / VLM 推理与 serving 框架
D. 操作系统调度器
```

SGLang 主要解决什么问题？

**Explanation:** SGLang 位于模型和服务系统之间，不是底层 CUDA 编译器，也不是通用数据分析库。

**Correct Answer:** 选 C。SGLang 是面向 LLM / VLM 的推理与 serving 框架，主要用于接收请求、调度请求、管理 KV cache、组织 prefill / decode、提供 API 服务，并提升吞吐和延迟表现。

### 2. offline inference / online serving

**Question:** 在 LLM serving 里，`offline inference` 和 `online serving` 有什么区别？

**Explanation:** 两者都可以跑模型推理，但目标不同。离线推理更偏批处理，在线 serving 更偏服务化和请求调度。

**Correct Answer:** `offline inference` 是一次性离线跑一批 prompt 并获得结果，更关注总吞吐；`online serving` 是部署 API 服务等待用户请求，更关注延迟、并发、稳定性和请求调度。

### 3. prefill / decode

**Question:** LLM 推理中经常分成 `prefill` 和 `decode` 两个阶段。它们分别主要做什么？

**Explanation:** prompt 输入和输出生成不是同一个计算形态。prompt 通常先被整体处理，随后模型逐 token 生成答案。

**Correct Answer:** `prefill` 阶段处理用户输入的 prompt，把 prompt tokens 一次性送进模型，并建立初始 KV cache；`decode` 阶段基于已有 KV cache，一个 token 一个 token 地生成输出。

### 4. 为什么区分 prefill 和 decode

**Question:** 为什么 LLM serving 系统要特别区分 `prefill` 和 `decode`？它们在计算特征上有什么不同？

**Explanation:** `prefill` 和 `decode` 的输入规模、并行度和性能瓶颈不同，因此 runtime 的调度策略也不同。

**Correct Answer:** `prefill` 输入 token 多，一次性处理 prompt，并行度较高，更像大批量矩阵计算，通常影响首 token 延迟；`decode` 每次生成 1 个 token，需要反复调用模型，每一步依赖上一步结果，并行度相对低，通常影响输出速度和 TPOT。

### 5. KV cache 是什么

**Question:** LLM 推理里的 `KV cache` 是用来缓存什么的？为什么 decode 阶段特别依赖它？

**Explanation:** Transformer attention 中每个 token 都会产生 Key / Value 表示。历史 token 的 K/V 如果每一步都重新计算，会非常浪费。

**Correct Answer:** KV cache 缓存的是 Transformer attention 中已经计算过的历史 token 的 Key / Value 表示。decode 阶段每次生成新 token 时，可以复用历史 K/V，只计算新 token 的 K/V，从而避免重复计算历史上下文。

### 6. KV cache 为什么占显存

**Question:** 为什么 KV cache 会占用大量显存？可以从 batch size、sequence length、number of layers、hidden size / heads 等角度解释。

**Explanation:** KV cache 不是单个小对象，而是每层、每个请求、每个历史 token 都要保存的中间状态。

**Correct Answer:** KV cache 显存占用会随 batch size、sequence length、layer 数、hidden size / attention heads 和 dtype 增长。LLM serving 中显存不只被模型权重占用，KV cache 也可能成为主要显存压力来源。

### 7. 为什么需要 scheduler

**Question:** SGLang 这类 serving runtime 为什么需要 scheduler？为什么不简单地“来一个请求，立刻单独跑一个请求”？

**Explanation:** 在线推理服务中，请求数量、长度和阶段都不同，GPU 显存和计算资源有限，需要在吞吐和延迟之间做权衡。

**Correct Answer:** scheduler 决定哪些请求在什么时候一起跑。它可以通过 batching 提高吞吐，协调 prefill / decode，管理 KV cache 显存，并在请求延迟、GPU 利用率和系统吞吐之间做决策。

### 8. TTFT / TPOT

**Question:** LLM serving 里常见两个指标：`TTFT` 和 `TPOT`。它们分别是什么意思？

**Explanation:** 在线生成式服务不仅要关注总耗时，还要关注用户什么时候看到第一个 token，以及后续 token 输出是否流畅。

**Correct Answer:** `TTFT` 是 Time To First Token，表示从请求发出到收到第一个输出 token 的时间；`TPOT` 是 Time Per Output Token，表示 decode 阶段平均每个输出 token 花费的时间。TTFT 影响用户觉得“有没有开始响应”，TPOT 影响输出流畅度。

### 9. RadixAttention / prefix cache

**Question:** SGLang 提到的 RadixAttention / prefix cache 这类机制，大致想解决什么问题？为什么复用相同前缀的计算结果有价值？

**Explanation:** 很多请求可能共享相同前缀，例如 system prompt、工具说明、文档上下文或 few-shot examples。如果每个请求都从头 prefill，会浪费计算。

**Correct Answer:** RadixAttention / prefix cache 的核心目标是复用相同前缀对应的 KV cache，减少重复 prefill，降低 TTFT，提升吞吐，并节省部分计算资源。直觉上就是：不要重复理解已经理解过的上下文。

### 10. SGLang 和 CUDA / GPU 优化的关系

**Question:** 从 CUDA / GPU 学习角度看，为什么要了解 SGLang？SGLang serving runtime 和底层 CUDA kernel / GPU 性能优化有什么关系？

**Explanation:** CUDA kernel 优化和 serving runtime 优化处在同一条端到端推理链路的不同层。只看单个 kernel，可能不知道优化对真实推理服务是否有效。

**Correct Answer:** CUDA kernel 决定单个底层算子怎么跑，SGLang runtime 决定请求如何进入 GPU、如何 batch、如何使用 KV cache、如何调度 prefill / decode。kernel 优化解决局部算子效率，runtime 优化解决端到端推理效率。学习 SGLang 是为了理解 CUDA / GPU 优化最终落在 LLM serving 链路中的位置。

## 今日总结

今天已经理解：

- SGLang 是 LLM / VLM 推理与 serving 框架
- VLM 是视觉语言模型，支持图像/视频 + 文本输入
- offline inference 偏批处理，online serving 偏在线请求服务
- prefill 处理输入 prompt，并建立初始 KV cache
- decode 基于 KV cache 逐 token 生成输出
- prefill 和 decode 的计算特征不同，调度策略也不同
- KV cache 缓存的是 Transformer attention 中历史 token 的 Key / Value
- KV cache 显存占用会随 batch size、sequence length、layer 数和 hidden size 增长
- scheduler 用于协调请求、batching、显存和延迟/吞吐权衡
- TTFT 是首 token 延迟，TPOT 是每输出 token 时间
- RadixAttention / prefix cache 主要用于复用相同前缀的 KV cache
- SGLang runtime 和 CUDA kernel / GPU 优化是端到端推理链路中的上下层关系

## 易错点

- SGLang 不是 CUDA kernel 编译器，而是上层推理 runtime / serving 系统。
- KV cache 不是“当前 token 到下一个 token 的映射”，而是历史 token 的 Key / Value 中间状态。
- prefill 和 decode 不是同一种计算形态，不能用同一种性能直觉理解。
- scheduler 不只是排队，而是在显存、计算资源、吞吐和延迟之间做决策。

## 下一步

- 继续 CUDA global memory 与 coalescing 基础
- 后续轻量跑通 SGLang 的最小 serving demo
- 记录一次 DeepSeek 模型在 SGLang 中的启动参数、显存占用和 TTFT / TPOT
