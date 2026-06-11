# -*- coding: utf-8 -*-
"""
FE-LLM (Free Energy Large Language Model)
==========================================
以「状态趋稳」为第一性原理的非自回归语言生成架构探索。

包结构（2026-06-11 重构，按"机制层不变、生成底座演进"组织）：
    active_inference/ —— 主动推理控制层（核心贡献，持续主线）：
                         observation → belief update → EFE 行动选择 → 语言实现，
                         12 个核心模块在顶层；training/ 与 experiments/ 为脚本分组。
    energy_lm/        —— 从零训练字符级生成层（参考实现，翻译判定两连阴性后冻结）：
                         models/ data/ training/ evaluation/ generation/ diagnostics/ demos/。
    backbone_lm/      —— 预训练底座 + FE-LLM 机制层（当前生成层主线，见
                         docs/FE-LLM预训练底座路线草案.md）：冻结底座外挂
                         IntentAdapter / EnergyHead，hybrid 能量打分解码。
    embedding/        —— 语义嵌入层：DashScope 真实向量 + 哈希降级。
    config            —— 统一配置中心（读 .env）。
"""

__version__ = "0.4.0"
