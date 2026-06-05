# Inference Runtime

本目录记录成熟 LLM inference runtime 的使用、benchmark 与内部机制分析。

这些工具不是本仓库的主线实现对象，而是用于对照自研 CUDA/Rust kernel 与 AI Infra 系统设计。为避免学习范围发散，本仓库当前只保留 SGLang。

## 定位

- `sglang/`: 重点学习对象，关注 runtime、scheduler、KV cache、RadixAttention 与复杂推理流程表达

## 记录模板

每个 runtime 建议记录：

- 安装与版本兼容
- 模型启动命令
- API 调用方式
- benchmark 方法
- 显存、吞吐、延迟数据
- scheduler、KV cache、attention backend 等内部机制
- 与自研 kernel 或其他 runtime 的差异
