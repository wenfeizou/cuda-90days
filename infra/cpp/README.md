# C++ 支撑能力

本目录服务于 CUDA C++ kernel 开发，记录必要的 C++ 工程能力与底层机制。

## 关注范围

- CMake 与 CUDA 工程组织
- 编译、链接、ABI 与运行时库
- 指针、引用、对象生命周期与 RAII
- 模板与 CUDA host/device 代码组织
- CPU/GPU 内存模型差异
- 与 Rust 的 FFI 边界

## 非目标

本目录不作为完整 C++ 语言学习路线，只记录会直接影响 CUDA 开发、kernel 调试、性能优化和跨语言集成的内容。
