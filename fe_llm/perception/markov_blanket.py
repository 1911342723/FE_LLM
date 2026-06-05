# -*- coding: utf-8 -*-
"""
perception/markov_blanket.py —— 马尔可夫毯
===========================================
对应文档：「架构的物理边界：马尔可夫毯」。

任何"系统"都必须有边界隔离内部状态与外部世界。Transformer 没有边界，
Prompt 直接贯穿所有层。FE-LLM 设立严格边界：
    - 感知层 SensoryLayer : 接收 Prompt → 向量化 + 自由能计算 → 抽象惊奇信号。
    - 行动层 ActiveLayer  : 把内部意图 → 文字输出，反向改变外部环境。
    - 内部隔离区          : 核心引擎只看抽象信号，永远不直接接触原始语料。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..embedding.base import Embedder
from ..free_energy import FreeEnergyEngine, SurpriseReport


@dataclass
class SensorySignal:
    """外部 Prompt 穿过感知层后形成的抽象信号（内部引擎唯一能看到的东西）。"""

    raw_text: str            # 原文（仅供复述/调试，核心引擎不直接消费）
    state_vector: np.ndarray # 高维认知状态向量
    report: SurpriseReport   # 自由能惊奇报告


class SensoryLayer:
    """感知层（马尔可夫毯输入侧）。"""

    def __init__(self, engine: FreeEnergyEngine, embedder: Embedder):
        self.engine = engine
        self.embedder = embedder

    def perceive(self, prompt: str) -> SensorySignal:
        """Prompt → 抽象惊奇信号。"""
        state = self.embedder.embed_one(prompt)
        report = self.engine.compute(prompt)
        return SensorySignal(raw_text=prompt, state_vector=state, report=report)


class ActiveLayer:
    """行动层（马尔可夫毯输出侧）：把内部意图交给解码器生成文字。"""

    def __init__(self, decoder):
        self.decoder = decoder

    def act(self, intent_vector: np.ndarray, context: dict) -> str:
        return self.decoder.decode(intent_vector, context)
