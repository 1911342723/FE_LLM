# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/seq_net.py —— 双轴自由能生成网络（v4）
========================================================
对应 kernel/docs/设计v4-双轴自由能生成.md。

与 v3 (energy_net.DialogueEnergyNet) 的唯一关键区别：**因果约束**。
PER 的电导矩阵 g_ij 加下三角掩码——位置 i 只能从 c 与 r_{<i} 接收预测，
不能看到右边还没生成的字。于是网络输出的"位置 i 的能量"是真正的
条件能量 E(· | c, r_{<i})，与"顺序逐字生成"一致，链式分解精确、无独立假设。

思考轴（多轮弛豫 + 可溯源 + certainty 停止）在 seq_collapse.py 里实现，
本文件只提供"给定前缀算下一字条件能量"的网络。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalPERBlock(nn.Module):
    """带因果掩码的预测-误差弛豫块（思考轴的一步弛豫单元）。

    与 v3 PERBlock 机制相同（预测 μ → 电导 g → 共识 → 误差 ε → 弛豫 z←z-ηε），
    在电导上加下三角因果掩码：位置 j 只汇聚来自 i≤j 的预测。

    辨识点 #2（可学突触基底 self.synapse）：当 use_synapse=True 时，电导
        g_ij = softplus(突触基底 a_ij) · 因果 softmax(内容相容度)
    其中 a_ij 是 (seq_len×seq_len) 的**可训练参数**，把"经验里高频的位置依赖"刻成
    先天低阻通路（方向三：经验刻在突触）。这正是 PER 区别于注意力、且此前因果版本被
    阉割掉的部分。use_synapse=False 时退化为"因果注意力"（旧行为，作消融对照）。
    """

    def __init__(self, dim: int, seq_len: int | None = None, n_heads: int = 8,
                 dropout: float = 0.1, use_synapse: bool = True):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_synapse = bool(use_synapse and seq_len is not None)
        self.seq_len = seq_len
        self.norm = nn.LayerNorm(dim)
        self.W_pred = nn.Linear(dim, dim)        # 每位置发出的预测 μ
        self.to_q = nn.Linear(dim, dim)          # 内容相容度（即时调制电导）
        self.to_k = nn.Linear(dim, dim)
        self.eta = nn.Parameter(torch.tensor(0.5))     # 可学习弛豫步长
        # 辨识点 #2：可训练突触基底电导（经验刻成先天低阻通路；因果约束下用下三角）
        if self.use_synapse:
            self.synapse = nn.Parameter(torch.zeros(seq_len, seq_len))
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, L, D = z.shape
        h = self.norm(z)
        mu = self.W_pred(h)                                       # (B,L,D)
        q = self.to_q(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        compat = (q @ k.transpose(-2, -1)).mean(1) / (self.head_dim ** 0.5)  # (B,L,L)
        # —— 因果掩码：位置 j 只能从 i<=j 接收预测（不看右边未生成的字）——
        causal = torch.tril(torch.ones(L, L, device=z.device, dtype=torch.bool))
        compat = compat.masked_fill(~causal.unsqueeze(0), float("-inf"))
        attn = torch.softmax(compat, dim=-1)                      # (B,L,L) 因果行和=1
        if self.use_synapse:
            # g = softplus(突触基底) · 因果注意力，再行归一（未来位置为 0，因果保持）
            base = torch.nn.functional.softplus(self.synapse[:L, :L])
            g = base.unsqueeze(0) * attn
            g = g / (g.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            g = attn
        expect = g @ mu                                           # 共识预期 (B,L,D)
        err = h - expect                                          # 预测误差 ε
        if getattr(self, "_capture", False):
            # 可溯源捕获（默认关闭、零开销）：路由电导 g 与逐位置预测误差范数 ||ε||
            self.cap_g = g.detach()                               # (B,L,L)
            self.cap_err_norm = err.detach().norm(dim=-1)        # (B,L)
        z = z - self.eta * self.dropout(err)                      # 弛豫：消除误差
        z = z + self.ffn(self.norm2(z))                           # 整合
        return z


class SeqEnergyNet(nn.Module):
    """因果双轴能量网络：给定 [上文|SEP|BOS|前缀...]，输出每位置的下一字条件能量。

    输出 (B, L, V)：位置 i 的那一行 = E(下一字 | 上文, r_{<=i})。
    生成时取最后一个已知位置的能量行来落下一字（见 seq_collapse.py）。
    """

    # 本网络 forward 返回的是能量 = -logits（取负才是 logits）。下游统一用该标记换算。
    returns_energy = True

    def __init__(self, vocab_size: int, max_len: int, dim: int = 256,
                 depth: int = 6, n_heads: int = 8, dropout: float = 0.1,
                 use_synapse: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dim = dim
        self.depth = depth
        self.n_heads = n_heads
        self.use_synapse = bool(use_synapse)
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            CausalPERBlock(dim, seq_len=max_len, n_heads=n_heads, dropout=dropout,
                           use_synapse=use_synapse)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids:(B,L) → energy:(B,L,V)。能量=-logits。位置 i 行=E(下一字|c,r_{<=i})。
        depth 层 CausalPERBlock = 思考轴上的 depth 轮弛豫。"""
        x = self.embed(ids) + self.pos[:, : ids.size(1)]
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.norm(x))
        return -logits

    # ----------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"vocab_size": self.vocab_size, "max_len": self.max_len,
                    "dim": self.dim, "depth": self.depth, "n_heads": self.n_heads,
                    "use_synapse": self.use_synapse,
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "SeqEnergyNet":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        # 向后兼容：旧 checkpoint 无 use_synapse 键 → 按无突触（阉割版）加载
        net = cls(vocab_size=ck["vocab_size"], max_len=ck["max_len"],
                  dim=ck["dim"], depth=ck["depth"], n_heads=ck.get("n_heads", 8),
                  use_synapse=ck.get("use_synapse", False))
        net.load_state_dict(ck["state_dict"])
        net.eval()
        return net
