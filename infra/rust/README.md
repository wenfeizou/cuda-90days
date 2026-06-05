# Rust 支撑能力

Rust 是本仓库的核心支撑语言之一，目标是服务于 `cuda-oxide`、GPU kernel 编写和 AI 推理基础设施。

## 关注范围

- ownership、borrowing 与 GPU 资源生命周期建模
- `unsafe` 边界设计
- FFI 与 CUDA runtime / driver API 交互
- `cuda-oxide` 实验记录
- `tokio` 异步调度与推理服务 runtime
- 错误处理、资源释放与内存安全约束

## 非目标

本目录不作为通用 Rust 入门教程，所有笔记应尽量落到 GPU 编程、系统调度或推理引擎实现上。
