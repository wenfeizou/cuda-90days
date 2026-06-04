# 🚀 cuda-90days

> **90-day AI Infra engineering: Writing GPU kernels natively via Rust (cuda-oxide) & CUDA C++, building memory-safe, high-concurrency systems for production-grade LLM inference.**

本仓库用于记录从系统级开发向 **AI 基础设施（AI Infrastructure）与高性能推理引擎** 转型过程中的全部底层实验。

* **核心任务**：学习 `CUDA C++` 开发（战术防御与体系结构理解），同时着重攻坚利用纯 Rust 编写原生 GPU 核函数（基于 `cuda-oxide` 体系），打通从网络高并发调度到 bare-metal 算子加速的全链路闭环，设计出面向工业级大模型推理的高性能系统。

---

## 💻 实验环境 (Infrastructure Setup)
* **OS**: Ubuntu 26.04 LTS
* **CUDA Toolkit**: 13.3
* **Rustc**: 1.98+
* **Core Crates**: `cuda-oxide`, `tokio`, `candle-core`
* **Profile Tool**: NVIDIA Nsight Systems / Nsight Compute

---

## 🗓️ 90天硬核攻坚路线图 (Roadmap)
