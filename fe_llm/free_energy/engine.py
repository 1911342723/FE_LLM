# -*- coding: utf-8 -*-
"""
free_energy/engine.py —— 自由能引擎门面
========================================
对上层暴露统一的 compute(text) 接口，内部决定用哪套实现：
    - 若加载了训练好的 SurpriseNet 权重 → 用神经网络给语义/因果误差打分。
    - 否则 → 用 RuleFreeEnergyEngine 解析计算。
两种实现产出的 SurpriseReport 结构完全一致，上层无感知。

注意：噪音(noise)与阻断判定属于"系统级保护逻辑"，无论用哪套实现都由规则把关，
      因为它关乎安全（防止把乱码喂给网络做无意义推理）。
"""

from __future__ import annotations

import numpy as np

from ..embedding.base import Embedder
from ..world_model import WorldModel
from .report import SurpriseReport
from .rule_engine import RuleFreeEnergyEngine


class FreeEnergyEngine:
    """自由能引擎门面。"""

    def __init__(
        self,
        world: WorldModel,
        embedder: Embedder,
        precision: float = 1.0,
        semantic_threshold: float = 0.85,
        noise_threshold: float = 2.2,
        surprise_net=None,           # 训练好的 SurpriseNet（可选）
        device: str = "cpu",
    ):
        self.world = world
        self.embedder = embedder
        self.rule = RuleFreeEnergyEngine(
            world, semantic_threshold, noise_threshold, precision
        )
        self.net = surprise_net
        self.device = device
        if self.net is not None:
            self.net.eval()

    @property
    def precision(self) -> float:
        return self.rule.precision

    @precision.setter
    def precision(self, value: float) -> None:
        self.rule.precision = value

    def compute(self, text: str) -> SurpriseReport:
        """计算一段文本的总自由能。"""
        state = self.embedder.embed_one(text)

        # 没有神经网络 → 纯规则
        if self.net is None:
            return self.rule.compute(text, state)

        # 有神经网络 → 网络给语义/因果打分，规则负责噪音与阻断保护
        return self._compute_with_net(text, state)

    def _compute_with_net(self, text: str, state: np.ndarray) -> SurpriseReport:
        import torch

        nearest = self.world.nearest(state)
        expectation = nearest[0].vector if nearest else state
        nearest_name = nearest[0].name if nearest else "<空世界>"

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            e = torch.tensor(expectation, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            out = self.net(s, e).squeeze(0).cpu().numpy()

        semantic = float(out[0]) * self.precision
        # 因果冲突本质是"是否存在互斥公理"的离散逻辑事实，不能由回归网络凭空臆造。
        # 因此用规则层先判定是否真的存在冲突公理：
        #   - 无冲突 → causal 强制为 0（避免网络对分布外输入外推出虚高因果惊奇）。
        #   - 有冲突 → 取网络打分与规则打分的较大者，保留网络的平滑性同时不漏报。
        rule_causal, conflict_name = self.rule._causal(text, state)
        if conflict_name is None:
            causal = 0.0
        else:
            net_causal = float(out[1]) * self.precision
            causal = max(net_causal, rule_causal)

        # 噪音仍由规则把关（安全逻辑不交给网络）
        noise = self.rule._noise(text, semantic)

        total = semantic + causal + noise
        blocked = noise > 0.5 and total > self.rule.noise_threshold
        reason = self.rule._diagnose(semantic, causal, noise, blocked,
                                     nearest_name, conflict_name)
        return SurpriseReport(
            semantic=semantic, causal=causal, noise=noise, total=total,
            blocked=blocked, nearest_concept=nearest_name,
            conflict_concept=conflict_name, reason=reason,
        )
