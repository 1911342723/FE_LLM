# -*- coding: utf-8 -*-
"""
generation/decoder_net.py —— 可训练的输出网络（核心权重之二）
=============================================================
对应文档："需要被真正训练固化成权重的两部分"之二：
    "能量递减解码器(输出网络)——给定一个代表意图的高维目标向量，
     它能在字典里找到最优的词汇组合路径。"

设计（条件式能量评分网络）：
    输入：[ 意图向量 intent | 当前已输出累计状态 current | 候选语义元向量 cand ]
    输出：一个标量"能量分"，表示『选这个候选后，离意图还剩多少残余能量』。
    解码时：对所有候选打分，选分最低(最快降能)的语义元，循环直至能量足够低。

为什么用网络而非纯几何：
    纯几何(余弦距离)只能线性度量；网络可学到"哪些语义元组合更流畅自然、
    更能表达意图"的非线性路径规划能力。训练标签由解码器的几何残余蒸馏而来。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DecoderNet(nn.Module):
    """条件式能量评分网络。"""

    def __init__(self, embed_dim: int, hidden: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        in_dim = embed_dim * 3  # [intent | current | candidate]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Softplus(),   # 残余能量非负
        )

    def forward(self, intent: torch.Tensor, current: torch.Tensor,
                candidate: torch.Tensor) -> torch.Tensor:
        """
        intent / current / candidate : (B, embed_dim)
        返回 : (B, 1) 预测残余能量（越小越该选）
        """
        x = torch.cat([intent, current, candidate], dim=-1)
        return self.net(x)

    def save(self, path: str) -> None:
        torch.save({"embed_dim": self.embed_dim,
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "DecoderNet":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(embed_dim=ckpt["embed_dim"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
