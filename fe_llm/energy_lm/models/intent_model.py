# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/intent_model.py —— 意图驱动自由能生成（v5 核心网络）
====================================================================
对应 kernel/docs/设计v5-意图驱动自由能生成.md。

三阶段：感知（惊奇）→ 思考（弛豫出意图）→ 行动（朝意图递减能量生成文字）。

本文件实现阶段2+3的网络：
    IntentEncoder ：prompt → PER 弛豫 → 意图向量 z*（阶段2）
    EnergyDecoder ：意图向量 z* + 前缀 → 逐字生成（阶段3）
    IntentLM      ：联合模型 = IntentEncoder + EnergyDecoder

决策逻辑的关键区别（非概率机器）：
    - GPT：每步选 argmax P(w|prefix)。
    - 我们：每步选 argmin distance(当前隐状态+w, 意图z*)。
    生成是"朝目标走"，不是"猜最可能的字"。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================
# PER Block（双向版 & 因果版）
# ==================================================================

class PERBlock(nn.Module):
    """预测-误差弛豫块（双向，用于阶段2"思考"）。"""

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm = nn.LayerNorm(dim)
        self.W_pred = nn.Linear(dim, dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.eta = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, L, D = z.shape
        h = self.norm(z)
        mu = self.W_pred(h)
        q = self.to_q(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        compat = (q @ k.transpose(-2, -1)).mean(1) / (self.head_dim ** 0.5)
        g = torch.softmax(compat, dim=-1)
        expect = g @ mu
        err = h - expect
        z = z - self.eta * self.dropout(err)
        z = z + self.ffn(self.norm2(z))
        return z


class CausalPERBlock(nn.Module):
    """因果PER块（用于阶段3"说话"——只看已说的）。"""

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm = nn.LayerNorm(dim)
        self.W_pred = nn.Linear(dim, dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.eta = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, L, D = z.shape
        h = self.norm(z)
        mu = self.W_pred(h)
        q = self.to_q(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        compat = (q @ k.transpose(-2, -1)).mean(1) / (self.head_dim ** 0.5)
        causal = torch.tril(torch.ones(L, L, device=z.device, dtype=torch.bool))
        compat = compat.masked_fill(~causal.unsqueeze(0), float("-inf"))
        g = torch.softmax(compat, dim=-1)
        expect = g @ mu
        err = h - expect
        z = z - self.eta * self.dropout(err)
        z = z + self.ffn(self.norm2(z))
        return z


# ==================================================================
# 阶段2：意图编码器（PER 弛豫 → 意图向量）
# ==================================================================

class IntentEncoder(nn.Module):
    """prompt → PER 弛豫 → 意图向量。
    双向（看全部 prompt），因为这是"思考"不是"说话"。
    取 [CLS]（第一位）的弛豫终态作为意图。"""

    def __init__(self, vocab_size: int, max_len: int, dim: int = 256,
                 depth: int = 4, n_heads: int = 8, intent_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.blocks = nn.ModuleList([PERBlock(dim, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, intent_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, L) → intent: (B, intent_dim)"""
        B, L = ids.shape
        x = self.embed(ids) + self.pos[:, :L]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                    # (B, 1+L, dim)
        for blk in self.blocks:
            x = blk(x)
        z = self.norm(x[:, 0])                             # 取 CLS 位的弛豫终态
        return self.proj(z)                                # (B, intent_dim)


# ==================================================================
# 阶段3：能量递减解码器（意图驱动逐字生成）
# ==================================================================

class EnergyDecoder(nn.Module):
    """给定意图向量 z* + 前缀 tokens → 逐字 logits + 隐状态。
    意图每层注入（加到 residual stream）——让每一步都"看着目标走"。"""

    def __init__(self, vocab_size: int, max_len: int, dim: int = 256,
                 depth: int = 4, n_heads: int = 8, intent_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.intent_proj = nn.Linear(intent_dim, dim)      # 把意图投射到 dim
        self.blocks = nn.ModuleList([CausalPERBlock(dim, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.h_proj = nn.Linear(dim, intent_dim)           # 隐状态投射到意图空间（算距离）

    def forward(self, ids: torch.Tensor, intent: torch.Tensor):
        """ids:(B,L), intent:(B,intent_dim) → logits:(B,L,V), h_intent:(B,L,intent_dim)"""
        B, L = ids.shape
        x = self.embed(ids) + self.pos[:, :L]
        z_inject = self.intent_proj(intent).unsqueeze(1)   # (B,1,dim) 广播注入
        for blk in self.blocks:
            x = x + z_inject                                # 每层都注入意图（看着目标走）
            x = blk(x)
        h = self.norm(x)
        logits = self.head(h)                              # (B,L,V) 用于 CE loss
        h_intent = self.h_proj(h)                          # (B,L,intent_dim) 用于距离计算
        return logits, h_intent


# ==================================================================
# 联合模型
# ==================================================================

class IntentLM(nn.Module):
    """完整的意图驱动语言模型 = IntentEncoder + EnergyDecoder。"""

    def __init__(self, vocab_size: int, enc_max: int = 32, dec_max: int = 32,
                 dim: int = 256, enc_depth: int = 4, dec_depth: int = 4,
                 n_heads: int = 8, intent_dim: int = 256):
        super().__init__()
        self.encoder = IntentEncoder(vocab_size, enc_max, dim, enc_depth, n_heads, intent_dim)
        self.decoder = EnergyDecoder(vocab_size, dec_max, dim, dec_depth, n_heads, intent_dim)
        self.intent_dim = intent_dim

    def forward(self, prompt_ids, resp_ids):
        """训练时的前向：
        1. 编码器从 prompt 弛豫出意图（z_pred）
        2. 编码器从 response 弛豫出目标意图（z_target，梯度截断）
        3. 解码器用 z_target 做 teacher-forcing 生成 response
        返回 logits, z_pred, z_target"""
        z_pred = self.encoder(prompt_ids)
        with torch.no_grad():
            z_target = self.encoder(resp_ids)               # 目标意图（stop-gradient）
        logits, h_intent = self.decoder(resp_ids, z_target)
        return logits, h_intent, z_pred, z_target

    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(),
                    "config": {"vocab_size": self.encoder.embed.num_embeddings,
                               "intent_dim": self.intent_dim,
                               "enc_max": self.encoder.pos.shape[1],
                               "dec_max": self.decoder.pos.shape[1],
                               "dim": self.encoder.embed.embedding_dim,
                               "enc_depth": len(self.encoder.blocks),
                               "dec_depth": len(self.decoder.blocks),
                               "n_heads": self.encoder.blocks[0].n_heads}}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu"):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ck["config"]
        net = cls(vocab_size=cfg["vocab_size"], intent_dim=cfg["intent_dim"],
                  enc_max=cfg.get("enc_max", 32), dec_max=cfg.get("dec_max", 32),
                  dim=cfg.get("dim", 256), enc_depth=cfg.get("enc_depth", 4),
                  dec_depth=cfg.get("dec_depth", 4), n_heads=cfg.get("n_heads", 8))
        net.load_state_dict(ck["state_dict"])
        return net
