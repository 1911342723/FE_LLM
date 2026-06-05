# -*- coding: utf-8 -*-
"""
generation/decoder.py —— 能量递减解码器（替代 Softmax）
========================================================
对应文档：「能量递减解码器」。把"目标意图"想象成山谷最深处，每输出一个语义元
就往下走一步，挑选最能降低残余能量的那一个，能量趋近 0 时输出 <EOS>。

两种打分方式：
    - 几何版(默认)   ：用余弦距离衡量"加入候选后离意图多近"。无需训练。
    - 网络版(可选)   ：用训练好的 DecoderNet 预测残余能量，路径更流畅。
"""

from __future__ import annotations

import numpy as np

from ..embedding.base import cosine_distance, unit
from .active_inference import STRATEGY_TAGS
from .tokenizer import ConceptTokenizer


class EnergyDescentDecoder:
    def __init__(self, tokenizer: ConceptTokenizer, max_units: int = 4,
                 energy_floor: float = 0.15, decoder_net=None, device: str = "cpu"):
        self.tokenizer = tokenizer
        self.max_units = max_units        # 一句话最多铺几个语义元
        self.energy_floor = energy_floor  # 残余能量低于此即收尾(输出 <EOS>)
        self.net = decoder_net            # 训练好的 DecoderNet（可选）
        self.device = device
        if self.net is not None:
            self.net.eval()

    def decode(self, intent_vector: np.ndarray, context: dict) -> str:
        strategy = context.get("strategy", "确认")
        candidates = self.tokenizer.candidates(STRATEGY_TAGS.get(strategy))

        # 阻断：直接输出唯一阻断语义元（最小自由能动作，不铺长句）
        if strategy == "阻断" and candidates:
            return candidates[0].surface
        if not candidates:
            return "（无可用语义元）"

        current = np.zeros_like(intent_vector)
        chosen: list[str] = []
        used: set[int] = set()

        for _ in range(self.max_units):
            best_idx, best_residual = -1, float("inf")
            for idx, unit_obj in enumerate(candidates):
                if idx in used:
                    continue
                trial = unit(current + unit_obj.vector)
                residual = self._residual(intent_vector, current, trial, unit_obj.vector)
                if residual < best_residual:
                    best_residual, best_idx = residual, idx
            if best_idx < 0:
                break
            chosen.append(candidates[best_idx].surface)
            used.add(best_idx)
            current = unit(current + candidates[best_idx].vector)
            if best_residual <= self.energy_floor:
                break

        return self._render(chosen)

    def _residual(self, intent, current, trial, candidate_vec) -> float:
        """残余能量打分：优先用 DecoderNet，否则用几何余弦距离。"""
        if self.net is None:
            return cosine_distance(trial, intent)
        import torch
        with torch.no_grad():
            i = torch.tensor(intent, dtype=torch.float32, device=self.device).unsqueeze(0)
            c = torch.tensor(current, dtype=torch.float32, device=self.device).unsqueeze(0)
            k = torch.tensor(candidate_vec, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            return float(self.net(i, c, k).item())

    @staticmethod
    def _render(surfaces: list[str]) -> str:
        if not surfaces:
            return "……"
        text = "，".join(surfaces)
        if not text.endswith(("。", "？", "！")):
            text += "。"
        return text
