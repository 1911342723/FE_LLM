# -*- coding: utf-8 -*-
"""
arithmetic/model.py —— 两个固化权重的小网络
=============================================
严格对应 FE-LLM 架构的两个可训练网络（此处为算术任务的特化版）：

    EnergyNet（自由能/惊奇网络）
        输入：题目特征 + 某个候选答案的吸引子向量
        输出：标量能量。正确答案→低能量(不惊奇)，错误答案→高能量(惊奇)。
        这就是把"惊奇 = -ln P(观测|世界模型)"落地成一个可学习的能量函数。

    SolverNet（解码器/能量递减解码）
        输入：题目特征
        输出：一个"答案意图向量"(高维)，它会落在能量地貌中正确答案吸引子附近。
        生成时：把意图向量与所有答案吸引子比距离，滚落到最近的那个 → 得到答案。

两者共享同一张"答案码本"(AnswerCodebook)，即能量地貌中所有吸引子的坐标，
这保证了"解码器生成的意图"和"能量网络评估的吸引子"处在同一个空间。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoding import QUESTION_DIM


class AnswerCodebook(nn.Module):
    """
    答案码本：每个候选答案 = 能量地貌中的一个吸引子向量。

    支持两种模式：
        learnable=True  : 吸引子坐标自由学习（纯能量塑形，几何无结构）。
        learnable=False : 结构化固定吸引子（推荐）。把"答案的数值大小"显式编码进
                          坐标的第一组维度（傅里叶式多尺度编码），使数值相近的答案
                          天然在空间中相近、整体有序可分。这样解码器只要回归出
                          正确的数值意图，滚落到最近吸引子就稳定准确。
    """

    def __init__(self, num_answers: int, embed_dim: int = 64,
                 learnable: bool = False, lo: int = 0):
        super().__init__()
        self.num_answers = num_answers
        self.embed_dim = embed_dim
        self.learnable = learnable
        self.lo = lo

        if learnable:
            self.embedding = nn.Embedding(num_answers, embed_dim)
            nn.init.normal_(self.embedding.weight, std=0.1)
        else:
            # 结构化固定坐标：对每个答案的"归一化数值"做多尺度正弦/余弦编码
            coords = self._build_structured(num_answers, embed_dim)
            # 注册为 buffer（不训练，但随模型保存/迁移设备）
            self.register_buffer("fixed", coords)

    @staticmethod
    def _build_structured(num_answers: int, embed_dim: int) -> torch.Tensor:
        """构造有序的多尺度位置编码坐标 (num_answers, embed_dim)。"""
        pos = torch.arange(num_answers, dtype=torch.float32).unsqueeze(1)  # (N,1)
        i = torch.arange(embed_dim // 2, dtype=torch.float32).unsqueeze(0)
        # 频率从低到高，覆盖不同数值尺度；低频维度保证"整体有序"
        freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * (2 * i) / embed_dim)
        angles = pos * freqs                                  # (N, D/2)
        coords = torch.zeros(num_answers, embed_dim)
        coords[:, 0::2] = torch.sin(angles)
        coords[:, 1::2] = torch.cos(angles)
        # 叠加一个线性维度，强化"数值轴"的单调可分性
        lin = (pos / max(1, num_answers - 1)) * 2.0 - 1.0     # 归一化到[-1,1]
        coords[:, 0] = coords[:, 0] + lin.squeeze(1)
        return coords

    def all_vectors(self) -> torch.Tensor:
        """返回全部吸引子坐标 (num_answers, embed_dim)。"""
        return self.embedding.weight if self.learnable else self.fixed

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if self.learnable:
            return self.embedding(idx)
        return self.fixed[idx]


class EnergyNet(nn.Module):
    """
    自由能/惊奇网络：E(题目, 候选答案吸引子) → 标量能量。
    """

    def __init__(self, embed_dim: int = 64, hidden: int = 128):
        super().__init__()
        # 先把题目特征投影到与吸引子同维的空间
        self.q_proj = nn.Sequential(
            nn.Linear(QUESTION_DIM, hidden), nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        # 能量打分头：吃 [题目投影 | 候选吸引子 | 二者差]
        self.energy_head = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1), nn.Softplus(),   # 能量非负
        )

    def forward(self, question: torch.Tensor, answer_vec: torch.Tensor) -> torch.Tensor:
        """
        question   : (B, QUESTION_DIM)
        answer_vec : (B, embed_dim) 候选答案吸引子
        返回        : (B, 1) 能量
        """
        q = self.q_proj(question)
        diff = q - answer_vec
        x = torch.cat([q, answer_vec, diff], dim=-1)
        return self.energy_head(x)

    def energy_over_all(self, question: torch.Tensor,
                        codebook: AnswerCodebook) -> torch.Tensor:
        """
        对一批题目，计算它们对"所有候选答案"的能量。
        question : (B, QUESTION_DIM)
        返回      : (B, num_answers) 每个答案的能量（用于 argmin 生成）
        """
        B = question.shape[0]
        N = codebook.num_answers
        q = self.q_proj(question)                       # (B, D)
        all_vec = codebook.all_vectors()                # (N, D)
        q_exp = q.unsqueeze(1).expand(B, N, -1)         # (B, N, D)
        a_exp = all_vec.unsqueeze(0).expand(B, N, -1)   # (B, N, D)
        diff = q_exp - a_exp
        x = torch.cat([q_exp, a_exp, diff], dim=-1)     # (B, N, 3D)
        return self.energy_head(x).squeeze(-1)          # (B, N)


class SolverNet(nn.Module):
    """
    解码器：题目特征 → 答案意图。

    输出同时给出两种表示，便于"滚落"到答案吸引子：
        - intent_vec : 高维意图向量（落在正确吸引子附近，用于能量地貌可视化/对齐）
        - scalar     : 归一化标量答案预测（÷scale 后的数值）。这是数值认知的核心——
                       系统先形成"答案大约是多少"的连续意图，再滚落(取整)到最近整数吸引子。
    """

    def __init__(self, embed_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(QUESTION_DIM, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.to_vec = nn.Linear(hidden, embed_dim)   # 高维意图向量
        self.to_scalar = nn.Linear(hidden, 1)        # 归一化标量答案

    def forward(self, question: torch.Tensor):
        """返回 (intent_vec:(B,embed_dim), scalar:(B,1))。"""
        h = self.trunk(question)
        return self.to_vec(h), self.to_scalar(h)
