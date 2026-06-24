# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/models/transformer_lm.py —— 标准因果 Transformer 字符 LM（对照标尺）
=======================================================================================
作为 SeqEnergyNet（因果 PER + 可学突触）的**公平对照**：同字符级、同数据、同 ctx、同词表，
在相同参数量预算下比"标准 Transformer vs 本项目自有 PER 架构"的语言建模/生成能力。

与 lm_scaling_eval.TinyTransformerLM 同源（nn.TransformerEncoder + 因果掩码），
这里抽成可保存/加载、带 ffn_mult 的独立模型，供 code_train.py 用 --arch transformer 调用。

注意：本网络 forward 直接返回 logits（returns_energy=False）；SeqEnergyNet 返回能量=-logits。
下游用 net.returns_energy 统一换算，保证训练/评估/采样是同一套代码路径，对照才公平。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CharTransformerLM(nn.Module):
    """标准因果 Transformer 字符语言模型。forward(ids) -> logits (B, L, V)。"""

    returns_energy = False

    def __init__(self, vocab_size: int, max_len: int, dim: int = 512, depth: int = 8,
                 n_heads: int = 8, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dim = dim
        self.depth = depth
        self.n_heads = n_heads
        self.ffn_mult = ffn_mult
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=ffn_mult * dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids:(B,L) → logits:(B,L,V)。因果掩码保证位置 i 只看 <=i。"""
        L = ids.size(1)
        x = self.embed(ids) + self.pos[:, :L]
        mask = torch.triu(torch.ones(L, L, device=ids.device, dtype=torch.bool), diagonal=1)
        x = self.enc(x, mask=mask)
        return self.head(self.norm(x))

    def save(self, path: str) -> None:
        torch.save({"arch": "transformer", "vocab_size": self.vocab_size, "max_len": self.max_len,
                    "dim": self.dim, "depth": self.depth, "n_heads": self.n_heads,
                    "ffn_mult": self.ffn_mult, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "CharTransformerLM":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        net = cls(vocab_size=ck["vocab_size"], max_len=ck["max_len"], dim=ck["dim"],
                  depth=ck["depth"], n_heads=ck.get("n_heads", 8), ffn_mult=ck.get("ffn_mult", 4))
        net.load_state_dict(ck["state_dict"])
        net.eval()
        return net
