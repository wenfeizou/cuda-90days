# PyTorch

PyTorch 在本仓库中承担 baseline、correctness oracle 和 CUDA extension 学习入口的角色。

## 学习目标

- 理解 CUDA tensor、memory layout、stride 与 dtype
- 使用 PyTorch 构造 correctness baseline
- 使用 PyTorch benchmark 对照自研 CUDA/Rust kernel
- 学习 custom CUDA extension 与 custom op
- 建立对 `torch.compile`、Inductor、Triton 的基本认知

## 建议文档

- `tensor_basics.md`: tensor、layout、stride、dtype
- `cuda_extension.md`: custom CUDA extension
- `custom_op.md`: custom op 注册与调用
- `torch_compile.md`: `torch.compile`、Inductor、Triton 基本机制
- `benchmark_baseline.md`: baseline 与 benchmark 方法
