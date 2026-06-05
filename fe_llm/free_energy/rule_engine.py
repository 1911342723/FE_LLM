# -*- coding: utf-8 -*-
"""
free_energy/rule_engine.py —— 解析版自由能引擎（规则 + 几何）
=============================================================
基于余弦距离与逻辑规则计算三层惊奇，无需训练即可运行。
两大作用：
    1) 系统在没有训练好的 SurpriseNet 时的默认引擎。
    2) 训练 SurpriseNet 时充当"教师"，自动生成蒸馏标签（见 training/）。

三层惊奇（自下而上累加 + 阈值阻断）：
    浅层 E_semantic : 输入向量到最近吸引子的几何距离 → -ln 映射成惊奇
    中层 E_causal   : 输入是否踩中某公理的"互斥反命题"（布尔逻辑冲突）
    顶层 E_noise    : 可识别词元占比过低(乱码) → 阶跃惩罚 → 超阈值即阻断
"""

from __future__ import annotations

import math

import numpy as np

from ..embedding.base import tokenize_words
from ..world_model import WorldModel
from .report import SurpriseReport


class RuleFreeEnergyEngine:
    """解析版自由能引擎。"""

    def __init__(
        self,
        world: WorldModel,
        semantic_threshold: float = 0.85,
        noise_threshold: float = 2.2,
        precision: float = 1.0,
    ):
        self.world = world
        self.semantic_threshold = semantic_threshold
        self.noise_threshold = noise_threshold
        # 动态置信度：越高对误差越敏感(低容错，数理严苛)，越低越宽容(闲聊/角色扮演)
        self.precision = precision

    # ---------------------- 浅层惊奇 ----------------------
    def _semantic(self, state: np.ndarray) -> tuple[float, str]:
        nearest = self.world.nearest(state)
        if nearest is None:
            return 2.0, "<空世界>"
        concept, dist = nearest
        # -ln(1-距离)：命中(距离→0)惊奇→0；正交(距离→1)惊奇显著上升
        surprise = -math.log(max(1e-6, 1.0 - min(dist, 0.999)))
        return surprise, concept.name

    # ---------------------- 中层惊奇 ----------------------
    def _causal(self, text: str, state: np.ndarray) -> tuple[float, str | None]:
        """
        因果冲突检测。覆盖两种情形：

        情形 A（输入直接落在"谬误"概念上）：
            最近吸引子本身是一个低深度的谬误概念(如 地平说谬误)，且它与某条
            高深度公理(如 地球形状)互斥。输入贴得越近，说明用户越是在主张这个
            谬误，冲突惩罚与"被违背公理的深度 × 贴近度"成正比。

        情形 B（输入在表达对最近公理的反命题）：
            最近吸引子是正向公理，但输入词面同时大量命中其某个互斥反命题，
            说明输入在违背该公理。

        两种情形取较大惩罚。
        """
        nearest = self.world.nearest(state)
        if nearest is None:
            return 0.0, None
        concept, dist = nearest

        penalty = 0.0
        conflict: str | None = None

        # —— 情形 A：最近概念自身是与公理互斥的谬误 ——
        for rival_name in concept.relations.get("互斥", []):
            rival = self.world.get(rival_name)
            if rival is None:
                continue
            # 只有当"被违背方"是更稳固的公理(深度更高)时才算冲突，
            # 避免把公理本身误判为谬误。
            if rival.depth > concept.depth:
                # 贴近度 = 1-距离，越贴近谬误，主张越强烈
                closeness = max(0.0, 1.0 - dist)
                a_penalty = rival.depth * closeness * 2.0
                if a_penalty > penalty:
                    penalty = a_penalty
                    conflict = rival_name

        # —— 情形 B：输入词面命中最近(正向)公理的互斥反命题 ——
        tokens = set(tokenize_words(text))
        for rival_name in concept.relations.get("互斥", []):
            rival = self.world.get(rival_name)
            if rival is None or rival.depth >= concept.depth:
                continue
            overlap = len(tokens & set(tokenize_words(rival.text)))
            if overlap >= 2:
                b_penalty = concept.depth * overlap * 0.5
                if b_penalty > penalty:
                    penalty = b_penalty
                    conflict = rival_name

        return penalty, conflict

    # ---------------------- 顶层噪音 ----------------------
    def _noise(self, text: str, semantic: float) -> float:
        stripped = text.strip()
        if not stripped:
            return 1.5
        recognizable = tokenize_words(text)
        non_space = [c for c in stripped if not c.isspace()]
        ratio = len(recognizable) / max(1, len(non_space))
        noise = 0.0
        if ratio < 0.35:
            noise += (0.35 - ratio) * 4.0
        if semantic > 3.0:
            noise += (semantic - 3.0) * 0.5
        return noise

    # ---------------------- 总自由能 ----------------------
    def compute(self, text: str, state: np.ndarray) -> SurpriseReport:
        """给定文本及其向量，计算三层惊奇与总自由能。"""
        semantic, nearest_name = self._semantic(state)
        causal, conflict_name = self._causal(text, state)
        noise = self._noise(text, semantic)

        semantic *= self.precision
        causal *= self.precision
        total = semantic + causal + noise

        # 顶层阻断：仅由噪音触发（逻辑矛盾交给"反驳"，不在此阻断）
        blocked = noise > 0.5 and total > self.noise_threshold

        reason = self._diagnose(semantic, causal, noise, blocked,
                                nearest_name, conflict_name)
        return SurpriseReport(
            semantic=semantic, causal=causal, noise=noise, total=total,
            blocked=blocked, nearest_concept=nearest_name,
            conflict_concept=conflict_name, reason=reason,
        )

    @staticmethod
    def _diagnose(semantic, causal, noise, blocked, nearest_name, conflict_name):
        if blocked and noise > 0.5:
            return "致命级噪音：输入无法解析，触发顶层阻断。"
        if causal > 1.0 and conflict_name:
            return f"因果断裂：输入与公理「{conflict_name}」发生逻辑冲突。"
        if semantic > 1.5:
            return f"语义偏离：输入远离已知概念，最接近的也只是「{nearest_name}」。"
        return f"低惊奇：输入与「{nearest_name}」高度吻合，系统状态平稳。"
