# -*- coding: utf-8 -*-
"""
training/dataset.py —— 训练数据生成（教师蒸馏）
================================================
为两个可训练网络自动生成训练样本，无需人工标注。

设计：
    - 语料来源：种子知识文本 + 一批合成的正/负/噪音样本。
    - SurpriseNet 样本：(输入向量, 期望向量) → 规则引擎打出的[语义,因果,噪音]标签。
    - DecoderNet  样本：(意图向量, 当前累计向量, 候选向量) → 几何残余能量标签。
"""

from __future__ import annotations

import random

import numpy as np

from fe_llm.embedding.base import Embedder, cosine_distance, unit
from fe_llm.free_energy.rule_engine import RuleFreeEnergyEngine
from fe_llm.generation.tokenizer import ConceptTokenizer
from fe_llm.world_model import WorldModel


# ============================================================
# 合成语料：覆盖低惊奇 / 因果冲突 / 噪音三类，让网络见过各种误差分布
# ============================================================
LOW_SURPRISE_SAMPLES = [
    "你好请问今天天气怎么样", "帮我写一个排序算法的函数",
    "数据库的索引是怎么工作的", "解释一下什么是神经网络",
    "水在一百度会沸腾对吗", "地球绕着太阳公转",
    "谢谢你的帮助再见", "加法满足交换律吗",
]
CONFLICT_SAMPLES = [
    "A大于B并且B大于C所以C大于A", "地球是一个平面边缘有悬崖",
    "造一台永动机不用能量永远转", "太阳从西边升起东边落下",
]
NOISE_SAMPLES = [
    "@@@###$$$%%%^^^", "asdkjqwoeiruzxcvmnb", "!!!???***>>><<<",
    "。。。、、、；；；", "xxxxyyyyzzzz1234567890qqqq",
]


def build_surprise_dataset(
    world: WorldModel, embedder: Embedder, rule: RuleFreeEnergyEngine
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 SurpriseNet 训练集。
    返回 (signals, expectations, labels)：
        signals      : (N, dim) 输入向量
        expectations : (N, dim) 最近吸引子向量
        labels       : (N, 3)   [semantic, causal, noise] 规则引擎标签
    """
    texts = LOW_SURPRISE_SAMPLES + CONFLICT_SAMPLES + NOISE_SAMPLES
    # 加入种子知识文本本身（应得到极低惊奇）
    texts += [c.text for c in world.all_concepts()]

    signals, expectations, labels = [], [], []
    for text in texts:
        state = embedder.embed_one(text)
        nearest = world.nearest(state)
        expectation = nearest[0].vector if nearest else state
        report = rule.compute(text, state)

        signals.append(state)
        expectations.append(expectation)
        labels.append([report.semantic, report.causal, report.noise])

    return (np.vstack(signals).astype(np.float32),
            np.vstack(expectations).astype(np.float32),
            np.array(labels, dtype=np.float32))


def build_decoder_dataset(
    tokenizer: ConceptTokenizer, n_samples: int = 2000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 DecoderNet 训练集。
    采样思路：随机挑一个语义元向量作为"意图"，随机一个"当前累计向量"，
    再对每个候选语义元，标签 = 加入它之后到意图的几何残余距离。
    网络学会预测这个残余，从而在解码时挑选最快降能的候选。

    返回 (intents, currents, candidates, residuals)。
    """
    rng = random.Random(seed)
    units = tokenizer.candidates()
    vecs = [u.vector for u in units]
    if not vecs:
        raise RuntimeError("分词器为空，无法生成解码训练集。")

    intents, currents, candidates, residuals = [], [], [], []
    for _ in range(n_samples):
        # 意图 = 1~3 个随机语义元的组合方向（模拟一个真实意图态）
        k = rng.randint(1, min(3, len(vecs)))
        intent = unit(sum(rng.sample(vecs, k)))
        # 当前累计 = 0~2 个语义元（模拟已输出一部分）
        m = rng.randint(0, min(2, len(vecs)))
        current = (unit(sum(rng.sample(vecs, m))) if m > 0
                   else np.zeros_like(intent))
        # 随机候选
        cand = rng.choice(vecs)
        trial = unit(current + cand)
        residual = cosine_distance(trial, intent)

        intents.append(intent)
        currents.append(current)
        candidates.append(cand)
        residuals.append([residual])

    return (np.vstack(intents).astype(np.float32),
            np.vstack(currents).astype(np.float32),
            np.vstack(candidates).astype(np.float32),
            np.array(residuals, dtype=np.float32))
