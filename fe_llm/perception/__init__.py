# -*- coding: utf-8 -*-
"""
perception 包 —— ③ 马尔可夫毯 + 分层预测编码（系统的"神经系统"）
==============================================================
对应文档第三步：「搭建马尔可夫毯与神经通讯总线」。

    - markov_blanket   : 系统边界。感知层接收 Prompt 并转为抽象惊奇信号；
                         行动层把内部决策转为文字输出。外部数据不直接接触核心。
    - predictive_coding: 分层预测编码。高层下发预期、底层上传误差，反复对流
                         直到能量坍缩，得到稳定的潜在意图向量。

这两层是纯计算逻辑（确定性），不需要训练固化权重。
"""

from .markov_blanket import ActiveLayer, SensoryLayer, SensorySignal
from .predictive_coding import CodingResult, PredictiveCodingHierarchy

__all__ = [
    "SensoryLayer",
    "ActiveLayer",
    "SensorySignal",
    "PredictiveCodingHierarchy",
    "CodingResult",
]
