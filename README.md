# from_0_to_1_minivllm

## 教程导航

- [Mini vLLM 从 0 到 1 复现手册](MinivLLM复现手册.md)
- [Day 1：Qwen3 架构与激活函数](day1/Day1.md)
- [Day 2：RMSNorm 与 residual](day2/Day2.md)
- [Day 3：支持张量并行的 Linear](day3/Day3.md)
- [Day 4：Vocab Embedding、LM Head 与 Sampler](day4/Day4.md)
- [Day 5：KV Cache 布局与写入](day5/Day5.md)
- [Day 6：Packed Prefill FlashAttention](day6/Day6.md)
- [Day 7：Decode PagedAttention](day7/Day7.md)
- [Day 8：Q/K RMSNorm 与 RoPE](day8/Day8.md)
- [Day 9：Qwen3 组装与权重加载](day9/Day9.md)
- [Day 10：Sequence 与 BlockManager](day10/Day10.md)
- [Day 11：ModelRunner 与 CUDA Graph](day11/Day11.md)
- [Day 12：Scheduler](day12/Day12.md)
- [Day 13：LLMEngine 端到端推理](day13/Day13.md)

从零开始学习并实现一个轻量级 Mini vLLM。

## 学习记录

- `day1/`：Qwen3 架构与激活函数
- `day2/`：RMSNorm 与 residual
- `day3/`：Replicated、Column、Row、Merged 和 QKV Parallel Linear
- `day4/`：词表并行 Embedding、LM Head、权重绑定与采样
- `day5/`：Paged KV Cache 布局、物理 slot 映射与 Triton 写入
- `day6/`：Packed 变长 Prefill、GQA 与 Triton online softmax
- `day7/`：随机 block table、GQA 与 Decode PagedAttention
- `day8/`：Q/K per-head RMSNorm、packed/batched RoPE 与 HF 对齐
- `day9/`：Qwen3 Decoder/Model/CausalLM、checkpoint 映射与真实 0.6B 对齐
- `day10/`：Sequence、物理块生命周期、前缀缓存与引用计数
- `day11/`：Prefill/Decode 张量契约、KV 容量、共享内存与 CUDA Graph
- `day12/`：Prefill/Decode 调度、共享预算、抢占与停止条件
- `day13/`：Paged Qwen3、LLMEngine、TP 验证与真实 0.6B 性能报告

Day1～Day13 已按照复现手册的依赖顺序实现。运行各 Day 的单元测试可逐层复核，Day13 提供完整端到端验收。

Linux/WSL 全量验收：

```bash
python verify_all.py

# 本地已缓存 Qwen/Qwen3-0.6B 时，再执行真实模型验收
python verify_all.py --real-model
```
