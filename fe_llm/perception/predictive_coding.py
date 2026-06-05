# -*- coding: utf-8 -*-
"""
perception/predictive_coding.py —— 分层预测编码
================================================
对应文档：「核心计算机制：分层预测编码」，取代 Transformer 的前向传播 + 注意力。

三层信念（高层抽象意图 / 中层事实逻辑 / 底层实际观测）：
    - 自上而下：高层向下发送预期。
    - 自下而上：下层把不符预期的误差上传。
    - 计算本质：上下层反复妥协，直到系统自由能趋近 0（能量坍缩）、内部自洽。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..embedding.base import cosine_distance, unit
from ..world_model import WorldModel


@dataclass
class CodingResult:
    settled_state: np.ndarray  # 坍缩后的稳定内部状态（潜在意图向量雏形）
    residual_energy: float     # 残余自由能（越接近 0 越自洽）
    iterations: int            # 收敛迭代轮数
    trajectory: list[float]    # 每轮残余能量轨迹（观察"能量坍缩"过程）


class PredictiveCodingHierarchy:
    """简化三层预测编码层级。"""

    def __init__(self, world: WorldModel, max_iter: int = 12,
                 tol: float = 1e-3, learning_rate: float = 0.5):
        self.world = world
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate

    def encode(self, sensory_state: np.ndarray) -> CodingResult:
        """把感知状态坍缩为能量最低的稳定内部状态。"""
        bottom = sensory_state.copy()                 # 底层=实际观测（不可动）

        nearest = self.world.nearest(sensory_state)   # 中层=最匹配概念
        mid = nearest[0].vector.copy() if nearest else sensory_state.copy()

        high = unit((bottom + mid) / 2.0)             # 高层=抽象意图

        trajectory: list[float] = []
        prev = high.copy()
        iterations = 0

        for i in range(self.max_iter):
            iterations = i + 1
            # 自上而下 + 自下而上：中层向高层预期与底层现实之间妥协
            mid = unit(mid + self.learning_rate * (
                0.5 * (high - mid) + 0.5 * (bottom - mid)))
            # 高层吸收来自现实的误差，向中层靠拢
            high = unit(high + self.learning_rate * (mid - high))

            residual = cosine_distance(high, bottom)
            trajectory.append(residual)
            if np.linalg.norm(high - prev) < self.tol:
                break
            prev = high.copy()

        return CodingResult(
            settled_state=high,
            residual_energy=cosine_distance(high, bottom),
            iterations=iterations,
            trajectory=trajectory,
        )
