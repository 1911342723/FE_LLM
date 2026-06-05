# -*- coding: utf-8 -*-
"""
free_energy/surprise_net.py —— 可训练的自由能网络（核心权重之一）
================================================================
对应文档："需要被深度学习框架真正训练并固化成权重的两部分"之一：
    "数学引擎层——这个小型网络不背诵知识，它只负责计算『输入信号』与
     『世界模型期望』之间的多维误差。训练它，就是让它学会如何精准给误差打分。"

网络职责（极轻量，不存知识）：
    输入：拼接[ 输入信号向量 s | 世界模型期望向量 e | 二者差向量 (s-e) ]
    输出：三维误差打分 [semantic, causal, noise]（均为非负）

为什么这样设计：
    - 知识完全在 pgvector 世界模型里，本网络只学"如何衡量偏差"，所以参数极少。
    - 训练标签由 RuleFreeEnergyEngine 蒸馏产生（教师→学生），见 training/。
    - 推理时极快，且可学到比手写规则更平滑、更鲁棒的误差曲面。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SurpriseNet(nn.Module):
    """
    多维误差打分网络。

    参数：
        embed_dim  : 单个向量维度（与嵌入后端一致，真实模型 1536）
        hidden     : 隐藏层宽度
    """

    def __init__(self, embed_dim: int, hidden: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        # 输入是 [s | e | s-e]，故为 3 * embed_dim
        in_dim = embed_dim * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),   # 输出三维：语义/因果/噪音
            nn.Softplus(),               # 保证误差非负（惊奇度不可能为负）
        )

    def forward(self, signal: torch.Tensor, expectation: torch.Tensor) -> torch.Tensor:
        """
        signal      : (B, embed_dim) 输入信号向量
        expectation : (B, embed_dim) 世界模型期望向量（最近吸引子）
        返回         : (B, 3) 三维误差 [semantic, causal, noise]
        """
        diff = signal - expectation
        x = torch.cat([signal, expectation, diff], dim=-1)
        return self.net(x)

    # ---------------------- 权重存取 ----------------------
    def save(self, path: str) -> None:
        torch.save({"embed_dim": self.embed_dim,
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "SurpriseNet":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(embed_dim=ckpt["embed_dim"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
