# -*- coding: utf-8 -*-
"""
FE-LLM (Free Energy Large Language Model)
==========================================
以「状态趋稳 / 自由能最小化」为第一性原理的非自回归语言模型。核心是因果 PER
（预测-误差-弛豫）语言模型 SeqEnergyNet（含可学突触基底 #2，非 Transformer 底座）：
从 0 训练的字符级 Python 代码模型，在同参数 / 同 token 预算下 held-out bpc 优于标准
Transformer，且可溯源、能后天成长（详见 docs/README.md）。

包结构：
    energy_lm/  —— 因果 PER 语言模型（SeqEnergyNet）：models/ data/ training/
                   evaluation/ generation/ diagnostics/ demos/。
                   代码模型训练入口 training/code_train.py。
    config      —— 统一配置中心（设备探测 + 教师模型配置，读 .env）。
"""

__version__ = "0.5.0"
