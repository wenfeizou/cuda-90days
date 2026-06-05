# Benchmarks

本目录用于保存 benchmark 脚本、原始结果和分析结论。

- `configs/`: benchmark 配置
- `scripts/`: benchmark 与数据处理脚本
- `results/`: benchmark 输出、表格和阶段性分析

每次 benchmark 建议至少记录：

- GPU 型号
- NVIDIA driver 与 CUDA 版本
- runtime / framework / kernel 版本
- 模型、batch size、input tokens、output tokens
- latency、throughput、tokens/s
- 显存峰值
- 复现实验命令
- profiling 工具与关键观察
