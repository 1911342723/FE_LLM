# -*- coding: utf-8 -*-
"""
FE-LLM (Free Energy Large Language Model)
==========================================
基于 Karl Friston「最小自由能原理(FEP)」与「主动推理(Active Inference)」的
下一代认知架构原型。

设计哲学（与传统 Transformer 的根本区别）：
    - Transformer：被动计算「下一个词的概率分布」，本质是文字接龙。
    - FE-LLM    ：主动计算「我该如何输出，才能让系统内部的惊奇度降到最低」，
                  本质是为了恢复内部平静而采取的目标导向行动。

模块组织（严格对应文档的四步走路线图 + 分层隔离）：
    config            —— 统一配置中心(.env)
    第一步  world_model/       —— ① 世界模型: pgvector 持久化 + 内存降级(出厂世界观)
    第二步  free_energy/       —— ② 自由能引擎: 规则版 + 可训练 SurpriseNet
    第三步  perception/        —— ③ 马尔可夫毯 + 分层预测编码
    第四步  generation/        —— ④ 概念分词器 + 主动推理 + 能量解码器(+ 可训练 DecoderNet)
    embedding/        —— 语义嵌入层: DashScope 真实向量 + 哈希降级

只有两部分需要被深度学习框架真正训练并固化成权重(见 training/ 训练层)：
    - SurpriseNet (free_energy/surprise_net.py) : 给输入与期望的多维误差打分
    - DecoderNet  (generation/decoder_net.py)   : 规划输出词汇的能量下降路径

顶层入口：FreeEnergyLLM(engine.py) / load_model(factory.py)
"""

from .engine import FreeEnergyLLM  # noqa: E402  (定义见下方说明)
from .factory import load_model  # noqa: E402

__all__ = ["FreeEnergyLLM", "load_model"]
__version__ = "0.2.0"
