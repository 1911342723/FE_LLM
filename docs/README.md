# FE-LLM 文档

本目录包含 FE-LLM（基于最小自由能原理的认知演化语言模型）的设计文档与论文。

## 内容

- **[FE-LLM论文.md](FE-LLM论文.md)** — 完整的中文论文，涵盖：
  - 核心思想：以最小自由能原理（FEP）替代"被动预测下一个词"的范式
  - 总体架构：马尔可夫毯 / 能量地貌世界模型 / 分层预测编码 / 主动推理 / 能量递减解码
  - 惊奇度的三层数学定义（语义 / 因果 / 噪音）
  - 两个验证实验（算术认知、中英双向翻译）的设置与结果
  - **客观评价**：诚实区分"已验证的主张"与"尚未成立的设想"
  - 局限性与未来工作

## 图表（figures/，均为 SVG 矢量图）

| 图 | 文件 | 说明 |
|----|------|------|
| 图 1 | [architecture.svg](figures/architecture.svg) | FE-LLM 总体架构与认知闭环 |
| 图 2 | [surprise_layers.svg](figures/surprise_layers.svg) | 三层惊奇度与行动策略映射 |
| 图 3 | [energy_decoding.svg](figures/energy_decoding.svg) | 能量递减解码（生成即滚落到吸引子） |
| 图 4 | [distillation_pipeline.svg](figures/distillation_pipeline.svg) | 教师—学生蒸馏与训练管道 |
| 图 5 | [arithmetic_energy.svg](figures/arithmetic_energy.svg) | 算术实验的惊奇能量曲线（核心证据） |

## 一句话总结

FE-LLM 是一项强调**可解释性**的思想验证原型：算术实验有力证明了"生成=能量下降、惊奇=高能量"的核心主张；但在开放序列任务上，当前实现尚未超越标准 Transformer 范式。其价值在于提供了一个"以最小化惊奇组织智能"的可运行视角，而非性能突破。
