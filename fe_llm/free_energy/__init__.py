# -*- coding: utf-8 -*-
"""
free_energy 包 —— ② 自由能数学引擎（量化"惊奇"）
=================================================
对应文档第二步：「先有神经元法则(数学公式)」+ 文档强调的"可被训练固化成权重"的两层之一。

包含两套实现：
    - RuleFreeEnergyEngine : 基于余弦距离/逻辑规则的解析版，无需训练即可运行，
                             同时充当 SurpriseNet 的"教师"(蒸馏标签来源)。
    - SurpriseNet          : 真正用 PyTorch 训练并固化成权重的小型神经网络。
                             输入「输入信号向量 + 世界模型期望向量」，输出多维误差
                             (语义/因果/噪音)。训练它=让它学会精准地给误差打分。

FreeEnergyEngine 是门面：有权重就用 SurpriseNet，否则回退规则版，对上层透明。
"""

from .engine import FreeEnergyEngine
from .report import SurpriseReport
from .rule_engine import RuleFreeEnergyEngine

__all__ = ["FreeEnergyEngine", "SurpriseReport", "RuleFreeEnergyEngine"]
