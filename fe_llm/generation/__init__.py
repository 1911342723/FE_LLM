# -*- coding: utf-8 -*-
"""
generation 包 —— ④ 主动推理 + 概念分词器 + 能量递减解码器（系统的"嘴巴"）
=====================================================================
对应文档第四步：「研发概念分词器与能量解码器」。

    - tokenizer        : 概念分词器，词表里映射的是"能量梯度"而非统计概率(替代 BPE)。
    - active_inference : 主动推理。意图定型 + 计算预期自由能 EFE + 选择行动策略。
    - decoder          : 能量递减解码器。沿能量梯度逐元滚落，逼近意图后输出 <EOS>(替代 Softmax)。
    - decoder_net      : 可训练的输出网络(核心权重之二)。给定意图向量，学习最优词汇路径。

DecoderNet 是文档点名"需要被真正训练固化成权重"的第二部分。
"""

from .active_inference import ActionPlan, ActiveInferenceEngine, STRATEGY_TAGS
from .decoder import EnergyDescentDecoder
from .tokenizer import EOS, ConceptTokenizer, SemanticUnit, build_default_tokenizer

__all__ = [
    "ActionPlan",
    "ActiveInferenceEngine",
    "STRATEGY_TAGS",
    "EnergyDescentDecoder",
    "ConceptTokenizer",
    "SemanticUnit",
    "build_default_tokenizer",
    "EOS",
]
