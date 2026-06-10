# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/energy_net.py —— 对话能量网络 E_θ（PER 交互，自有机制）
======================================================================
职责：看「上文 + 当前(半填充)回应序列」，为**每个位置、每个候选字**输出一个能量值。
    能量低 = 在该位置填这个字，让整句更稳定（更自洽/更像训练经验里的低能深沟）。

位置间交互用我们**自己的机制 PER（预测-误差弛豫）**，不是抄注意力，也不是凑 Mixer：
    每个位置向其它位置发出预测 μ，按"突触通路电导 g_ij"汇聚成共识预期，
    用预测误差 ε = z - 预期 驱动各位置弛豫，反复直到误差最小（能量最低/稳定）。
    - 方向一：误差驱动（像大脑神经元相互预测）。
    - 方向三：电导 g_ij 含可训练突触基底（经验刻高高频搭配 = 经验就是省电）。
    - 注意力是 PER 的单轮退化特例 → 我们是借鉴不背叛、更本源的版本。
详见 kernel/docs/设计v3-PER-预测误差弛豫交互.md。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PERBlock(nn.Module):
    """
    预测-误差弛豫块（自有交互机制）。

    一轮弛豫：
        预测   μ_i = W_pred(z_i)                      每位置发出的预测
        电导   g_ij = softplus(突触基底 a_ij) · 内容相容度(z_i,z_j)
        共识   expect_j = Σ_i g_ij·μ_i / Σ_i g_ij
        误差   ε = z - expect
        更新   z ← z - η·ε      （η 可学习，残差式弛豫）
    堆叠多个 PERBlock = 多轮弛豫。
    """

    def __init__(self, seq_len: int, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.norm = nn.LayerNorm(dim)
        # 预测函数：每位置用自身状态预测"邻居该是什么"
        self.W_pred = nn.Linear(dim, dim)
        # 内容相容度用的查询/键（决定突触即时开合，按内容调制）
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        # 可训练突触基底电导 a_ij（方向三：经验刻在这；位置对的先天低阻通路）
        self.synapse = nn.Parameter(torch.zeros(seq_len, seq_len))
        # 可学习弛豫步长 η
        self.eta = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(dropout)
        # 弛豫后的通道整合（消化误差修正）
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, L, D = z.shape
        h = self.norm(z)
        mu = self.W_pred(h)                                  # (B,L,D) 每位置的预测
        # —— 内容相容度（多头点积，仅作突触的即时调制，不是注意力主体）——
        q = self.to_q(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        compat = (q @ k.transpose(-2, -1)).mean(1) / (self.head_dim ** 0.5)  # (B,L,L)
        # —— 电导 g_ij = softplus(突触基底) · softmax(相容度) ——
        base = torch.nn.functional.softplus(self.synapse[:L, :L])   # (L,L) 经验刻的低阻通路
        g = base.unsqueeze(0) * torch.softmax(compat, dim=-1)        # (B,L,L)
        g = g / (g.sum(dim=-1, keepdim=True) + 1e-6)
        # —— 共识预期 expect_j = Σ_i g_ji · μ_i ——
        expect = g @ mu                                      # (B,L,D)
        err = h - expect                                     # 预测误差 ε（局部自由能）
        z = z - self.eta * self.dropout(err)                 # 弛豫：消除误差
        z = z + self.ffn(self.norm2(z))                      # 整合
        return z


class DialogueEnergyNet(nn.Module):
    """
    对话去掩码能量网络（PER 交互）。
    输入：拼好的 token 序列 [上文 | SEP | 回应(含MASK)]，定长 seq_len。
    输出：(B, L, vocab) 的能量矩阵 —— 每个位置每个候选字的能量。
    """

    def __init__(self, vocab_size: int, seq_len: int, dim: int = 256,
                 depth: int = 6, n_heads: int = 4, dropout: float = 0.1,
                 hidden: int | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dim = dim
        self.depth = depth
        self.n_heads = n_heads
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            PERBlock(seq_len, dim, n_heads=n_heads, dropout=dropout)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, L) → energy: (B, L, vocab)。能量 = -logits。"""
        x = self.embed(ids) + self.pos[:, : ids.size(1)]
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.norm(x))         # (B, L, vocab)
        return -logits                            # 能量 = -logits（低能=高 logit）

    # ----------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"vocab_size": self.vocab_size, "seq_len": self.seq_len,
                    "dim": self.dim, "depth": self.depth, "n_heads": self.n_heads,
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "DialogueEnergyNet":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        net = cls(vocab_size=ck["vocab_size"], seq_len=ck["seq_len"],
                  dim=ck["dim"], depth=ck["depth"], n_heads=ck.get("n_heads", 4))
        net.load_state_dict(ck["state_dict"])
        net.eval()
        return net
