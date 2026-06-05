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

本仓库主线是：

```text
CUDA kernel development -> Rust GPU programming -> LLM inference infrastructure
```

Linux、C++、Rust 与 Python 在本仓库中作为 CUDA Kernel 开发与 AI 推理基础设施的支撑能力层，而不是独立的通用学习路线。

---

## 📁 仓库结构 (Repository Layout)

```text
cuda-90days/
├── README.md
├── days/                  # 90 天每日实验记录
├── kernels/
│   ├── cuda_cpp/          # CUDA C++ kernel 实验
│   └── cuda_oxide/        # Rust / cuda-oxide kernel 实验
├── frameworks/
│   ├── pytorch/           # baseline、correctness oracle、CUDA extension 入口
│   └── candle/            # Rust-native 推理框架主线
├── runtime/
│   └── sglang/            # 重点学习的 LLM serving runtime
├── infra/
│   ├── linux/             # Linux 支撑能力，当前实验环境使用 Ubuntu
│   ├── cpp/               # CUDA C++ 所需 C++ 能力
│   ├── rust/              # cuda-oxide 与 AI Infra 所需 Rust 能力
│   └── python/            # benchmark、baseline、correctness 工具层
├── benchmarks/
│   ├── configs/           # benchmark 配置
│   ├── scripts/           # benchmark 脚本
│   └── results/           # 性能结果与分析
├── models/                # DeepSeek 模型结构、部署约束和实验记录，不存权重
├── reports/               # 阶段复盘与最终报告
├── scripts/               # 仓库级通用脚本
├── configs/               # runtime、model、benchmark 通用配置
└── docs/                  # 路线图、profiling、术语与参考资料
```

---

## 🧱 支撑能力层 (Supporting Layers)

### Linux

当前实验环境使用 Ubuntu。该目录只记录 GPU 开发相关的 Linux 能力，包括驱动、CUDA Toolkit、Nsight、动态库路径、权限、性能观察工具与常见环境排查。

### C++

C++ 目录服务于 CUDA C++ 开发，重点关注 CMake、编译链接、内存模型、RAII、模板、host/device 代码组织，以及与 Rust 的 FFI 边界。

### Rust

Rust 是本仓库的重点支撑语言之一，围绕 `cuda-oxide`、`unsafe`、FFI、所有权边界、异步运行时与推理系统调度展开。

### Python

Python 定位为工具层，用于 PyTorch baseline、输入数据生成、correctness check、benchmark 结果处理和可视化。

---

## 🧩 推理运行时 (Inference Runtime)

成熟推理运行时用于对照自研 kernel 与系统设计，不替代 CUDA/Rust 主线。为避免学习范围发散，本仓库当前只保留 SGLang。

### SGLang

SGLang 作为重点学习对象，用于理解高性能 LLM/VLM serving、structured generation、runtime 调度、KV cache 与 RadixAttention 等机制。

---

## 🧠 框架层 (Frameworks)

框架层连接底层 kernel 与上层推理服务，用于理解模型执行、张量抽象、backend 设计和 custom op 集成。

学习优先级：

```text
PyTorch baseline -> Candle
```

### PyTorch

PyTorch 用作 baseline、correctness oracle 和 CUDA extension 学习入口。重点关注 CUDA tensor、memory layout、custom op、`torch.compile` / Inductor / Triton 的基本机制，以及与自研 kernel 的性能对照。

### Candle

Candle 是 Rust-native 推理框架主线，优先用于理解 Rust 中的 tensor、model loading、CUDA backend、custom op 和轻量 LLM inference pipeline。

---

## 🤖 模型层 (Models)

模型目录记录 DeepSeek 系列模型的结构、部署要求、显存占用、runtime 支持情况和 benchmark 观察。本仓库不存放模型权重。

---

## 📊 复盘与度量 (Reports & Benchmarks)

每个实验尽量保留 correctness、benchmark 与 profiling 结论。每 15 天输出一次阶段复盘，最终形成 90 天总结报告。
