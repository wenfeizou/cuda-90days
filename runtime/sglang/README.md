# SGLang

SGLang 是本仓库的重点 LLM serving runtime 学习对象。

## 学习目标

- 能独立安装、启动模型服务并调用 OpenAI-compatible API
- 能完成吞吐、延迟、显存占用 benchmark
- 理解 runtime、scheduler、KV cache 与 RadixAttention
- 理解 structured generation 与复杂推理流程表达
- 能与 PyTorch baseline、Candle 和自研 kernel 实验进行对照

## 建议文档

- `install.md`: 安装、CUDA/PyTorch 版本兼容
- `serving.md`: server 启动、API 调用、常用参数
- `benchmark.md`: qps、latency、tokens/s、显存占用
- `internals.md`: scheduler、KV cache、RadixAttention、attention backend
- `notes.md`: 问题记录与排查
