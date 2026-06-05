# Candle

Candle 是本仓库 Rust-native 推理框架学习主线。

## 学习目标

- 理解 Candle tensor、device、dtype 与 CUDA backend
- 完成模型加载与基础推理流程
- 理解 custom op / custom kernel 接入方式
- 与 `cuda-oxide` kernel、PyTorch baseline 和 LLM runtime 做对照

## 建议文档

- `tensor_basics.md`: tensor、device、dtype
- `model_loading.md`: 模型加载与权重格式
- `cuda_backend.md`: CUDA backend 使用与限制
- `custom_op.md`: custom op / custom kernel 接入
- `inference_pipeline.md`: Rust-native 推理流程
