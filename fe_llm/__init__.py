# -*- coding: utf-8 -*-
"""
FE-LLM (Free Energy Large Language Model)
==========================================
以「状态趋稳」为第一性原理的非自回归语言生成架构探索。

当前主线（见 docs/FE-LLM论文.md）：
    energy_lm/    —— 能量坍缩对话模型：单个能量函数 E_θ + PER 交互 + 退火去掩码坍缩。
                     不预测下一个词，而让整句从全 [MASK] 并行弛豫到能量最低的稳定配置。
    embedding/    —— 语义嵌入层（kernel 概念空间使用）：DashScope 真实向量 + 哈希降级。
    config        —— 统一配置中心（读 .env）。

概念推理内核见 kernel/（M0–M5 里程碑：可见思考 / 概念学习 / 零重训成长 /
主动推理 / 消融 / 规模化）。
"""

__version__ = "0.3.0"
