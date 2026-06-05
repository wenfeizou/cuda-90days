# Frameworks

本目录记录模型执行框架层的学习与实验，连接底层 CUDA/Rust kernel 和上层 LLM serving runtime。

## 学习优先级

```text
PyTorch baseline -> Candle
```

## 定位

- `pytorch/`: baseline、correctness oracle、CUDA extension 学习入口
- `candle/`: Rust-native 推理框架主线

## 关注范围

- tensor abstraction
- CUDA backend
- model loading
- custom op / custom kernel
- inference pipeline
- 与 `kernels/`、`runtime/`、`benchmarks/` 的对照关系
