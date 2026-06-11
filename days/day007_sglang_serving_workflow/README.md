# Day 007: SGLang Serving Workflow

Date: 2026-06-11

## 今日目标

今天继续学习 SGLang，重点理解一个请求进入 LLM serving runtime 后的大致流程，以及 scheduler、batching、KV cache、TTFT / TPOT、throughput / latency 如何共同影响在线推理服务。

## 10 个概念问题

### 1. 请求进入 serving runtime 的顺序

**Question:** 一个用户请求进入 SGLang 这类 LLM serving 系统后，下面这些步骤大致应该按什么顺序发生？

```text
A. 模型逐 token 生成输出
B. 请求进入 server
C. tokenizer 把文本变成 tokens
D. prefill 处理 prompt 并建立 KV cache
E. scheduler 决定请求什么时候运行、是否和其他请求 batch
```

**Explanation:** 模型不能直接处理原始文本，需要先 tokenization。scheduler 通常需要知道 token 数量、资源需求等信息，才能做排队和 batching 决策。

**Correct Answer:** 更合理的顺序是：

```text
B -> C -> E -> D -> A
```

也就是：请求进入 server，tokenizer 把文本变成 tokens，scheduler 决定何时运行和如何 batch，prefill 处理 prompt 并建立 KV cache，最后 decode 阶段逐 token 生成输出。

### 2. scheduler 为什么需要 token 数量

**Question:** 为什么 scheduler 需要知道 prompt 的 token 数量？请从 prefill 成本、KV cache 显存、batching、TTFT 几个角度解释。

**Explanation:** prompt token 数量不是普通元数据，而是 serving 调度的重要资源估计依据。

**Correct Answer:** prompt token 数量影响 prefill 计算成本、KV cache 显存占用、batching 策略和 TTFT。prompt 越长，prefill 通常越重，需要保存的 K/V 越多，首 token 延迟通常也越长。

### 3. 短 prompt 与长 prompt 的压力差异

**Question:** 假设有两个请求：

```text
Request A: prompt 长度 20 tokens
Request B: prompt 长度 8000 tokens
```

它们对 serving 系统的压力有什么不同？

**Explanation:** 长 prompt 不只是单个请求慢，也会影响显存占用和其他请求调度。

**Correct Answer:** Request A 的 prefill 很轻，KV cache 小，TTFT 通常短，比较容易和其他请求一起调度。Request B 的 prefill 很重，KV cache 大，TTFT 通常长，可能占用更多显存和 prefill 计算资源，scheduler 可能需要限制、排队或单独处理。

### 4. prefill-heavy / decode-heavy

**Question:** 什么样的请求更 prefill-heavy？什么样的请求更 decode-heavy？

**Explanation:** prefill-heavy 和 decode-heavy 的判断重点不同：一个看输入侧，一个看输出侧。

**Correct Answer:** prompt 很长、输入 token 很多、包含图像或长文档上下文的请求更 prefill-heavy。需要生成很多输出 tokens 的请求更 decode-heavy。图片 / 多模态输入通常更可能增加输入处理和 prefill 侧成本，不一定代表 decode-heavy。

### 5. 长输出请求为什么持续占用资源

**Question:** 为什么长输出请求，也就是 decode-heavy 请求，会持续占用 serving 系统资源？请从逐 token 生成、KV cache、scheduler、TPOT 角度解释。

**Explanation:** decode 不是一次算完，而是一个 token 一个 token 地生成，每一步都需要模型继续运行。

**Correct Answer:** decode-heavy 请求需要持续逐 token 生成，每生成一个 token 都要运行一次 decode step。整个过程中 KV cache 不能释放，scheduler 需要反复把它放进 decode batch。TPOT 越高，每个 token 花费越久，资源占用时间越长。

### 6. batching 的吞吐和延迟权衡

**Question:** 为什么 batching 能提高吞吐？为什么 batching 也可能增加某些请求的延迟？

**Explanation:** batching 可以更充分利用 GPU 并行能力，但等待和混合不同请求也会带来延迟代价。

**Correct Answer:** batching 把多个请求一起跑，可以提高 GPU 利用率和系统吞吐。但请求可能要等待凑 batch，短请求可能被长请求拖慢，prefill-heavy 请求可能让同 batch 的请求首 token 更慢，因此 TTFT 或其他 latency 指标可能上升。

### 7. throughput / latency

**Question:** LLM serving 里常说 `throughput` 和 `latency`。这两个指标有什么区别？

**Explanation:** 一个看单位时间处理多少工作，一个看单个请求花了多久。

**Correct Answer:** throughput 是单位时间内系统处理了多少工作，例如 requests/s、tokens/s、output tokens/s。latency 是单个请求从开始到完成或到某个阶段花了多久，例如 TTFT、TPOT、end-to-end latency。大 batch 可能提高 throughput，但也可能增加某些请求 latency。

### 8. KV cache 是资源管理问题

**Question:** 为什么 KV cache 不只是性能优化，也是 serving 系统的资源管理问题？

**Explanation:** KV cache 不是减少显存使用，而是用显存保存历史 token 的 K/V，以减少重复计算。

**Correct Answer:** KV cache 是用显存换计算。并发请求越多、上下文越长、输出越长，KV cache 显存压力越大。scheduler 需要决定哪些请求能进入、哪些要等待、哪些 cache 可以复用或释放，因此 KV cache 也是显存和请求调度的资源管理问题。

### 9. throughput 可能牺牲哪些 latency

**Question:** 如果一个系统追求更高 throughput，可能会牺牲哪些 latency 指标？为了提高 tokens/s，可能会让什么变慢？

**Explanation:** serving 系统经常在 GPU 利用率和用户等待时间之间做权衡。

**Correct Answer:** 追求更高 throughput 可能牺牲 TTFT、TPOT、end-to-end latency 或 tail latency。例如等待更多请求凑 batch、优先填满 GPU，可能让某些请求更晚开始 prefill 或更晚拿到第一个 token，从而增加 TTFT。

### 10. serving runtime 的核心直觉

**Question:** 为什么 serving runtime 不只是“把模型跑起来”，而是一个调度和资源管理系统？请尽量使用 scheduler、batching、KV cache、TTFT / TPOT、throughput / latency。

**Explanation:** 在线 LLM serving 的难点不只是单次模型计算，还包括不同长度、不同阶段、不同输出需求的请求如何共享有限 GPU 资源。

**Correct Answer:** SGLang 这类 serving runtime 通过 scheduler 管理 batching、KV cache 和 GPU 资源，在 throughput / latency 之间做权衡，并优化 TTFT、TPOT 等用户可感知指标。它本质上是一个面向在线推理的调度和资源管理系统。

## 今日总结

今天已经理解：

- 请求进入 serving 系统后，通常经历 server -> tokenizer -> scheduler -> prefill -> decode
- scheduler 需要知道 prompt token 数量，因为它影响 prefill 成本、KV cache、batching 和 TTFT
- 长 prompt 请求更 prefill-heavy
- 长输出请求更 decode-heavy
- 图片 / 多模态输入更可能增加输入处理和 prefill 侧成本
- decode-heavy 请求会持续占用 KV cache 和调度名额
- batching 能提高 GPU 利用率和 throughput，但可能增加 TTFT
- throughput 关注单位时间处理多少工作
- latency 关注单个请求等待和完成耗时
- KV cache 是用显存换计算，不是减少显存使用
- KV cache 也是 serving 系统的资源管理问题
- 追求更高 throughput 可能牺牲 TTFT、TPOT、end-to-end latency 或 tail latency
- serving runtime 本质是调度和资源管理系统

## 易错点

- tokenizer 通常发生在 prefill 之前，模型处理的是 tokens，不是原始字符串。
- KV cache 通常增加显存占用，用来减少重复计算。
- 图片 / 多模态输入更偏输入侧成本，不一定代表 decode-heavy。
- batching 不是越大越好，它会影响延迟和公平性。
- throughput 和 latency 经常需要权衡，不能只看 tokens/s。

## 下一步

- 在 `docs/glossary.md` 中补充 throughput、latency、tail latency、tokenizer、batching
- 后续跑一个 SGLang 最小 serving demo
- 记录 TTFT / TPOT / throughput 的简单 benchmark
