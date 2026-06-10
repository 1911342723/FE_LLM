# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/slot_intent_model.py —— 意图序列化网络（M1）
=============================================================
对应 docs/FE-LLM意图序列化架构草案.md（道易"卦/爻/精度"的工程化转译）：

    IntentState = {
        global_intent: R^d        # 全局压缩态（卦）——接口与旧版 z* 兼容
        intent_slots:  R^{K×d}    # K 个局部意图槽（爻）——承载细粒度语义
        slot_salience: R^K        # 各槽位显著性（精度权重）
    }

设计约束（草案 1.5.4 节）：
    - cross-attention 只承担"读取/路由"职责（目），不作为解释本身；
    - 可溯源来自显式能量量：逐字残余能量 + 槽位覆盖度 + salience。

训练铁律（翻译实验教训，经验.md）：
    - 解码器只条件于 encoder(prompt) 的输出，训练/推理同分布。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fe_llm.energy_lm.intent_model import CausalPERBlock, PERBlock


class SlotIntentEncoder(nn.Module):
    """prompt → PER 弛豫 → 结构化意图（global + K 槽位 + salience）。

    实现：K 个 learned query 与 [CLS]、token 序列拼接后一起过 PERBlock 弛豫。
    用 PER 而不是标准 attention pooling，保持"预测-误差弛豫"的机制叙事一致。
    """

    def __init__(self, vocab_size: int, max_len: int, dim: int = 256,
                 depth: int = 4, n_heads: int = 8, intent_dim: int = 128,
                 n_slots: int = 8):
        super().__init__()
        self.n_slots = n_slots
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        # K 个槽位查询：弛豫过程中各自去"读"输入序列的不同要素。
        self.slot_queries = nn.Parameter(torch.randn(1, n_slots, dim) * 0.02)
        self.blocks = nn.ModuleList([PERBlock(dim, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.proj_global = nn.Linear(dim, intent_dim)
        self.proj_slot = nn.Linear(dim, intent_dim)
        self.salience_head = nn.Linear(dim, 1)

    def forward(self, ids: torch.Tensor):
        """ids: (B, L) → global:(B,d), slots:(B,K,d), salience:(B,K)"""
        B, L = ids.shape
        x = self.embed(ids) + self.pos[:, :L]
        cls = self.cls_token.expand(B, -1, -1)
        slots = self.slot_queries.expand(B, -1, -1)
        x = torch.cat([cls, slots, x], dim=1)              # (B, 1+K+L, dim)
        for blk in self.blocks:
            x = blk(x)
        h = self.norm(x)
        global_intent = self.proj_global(h[:, 0])          # CLS 位
        slot_states = h[:, 1 : 1 + self.n_slots]           # (B, K, dim)
        intent_slots = self.proj_slot(slot_states)         # (B, K, d)
        # salience 经 softmax 归一（和为 1），是显式精度权重。
        slot_salience = torch.softmax(self.salience_head(slot_states).squeeze(-1), dim=-1)
        return global_intent, intent_slots, slot_salience


class SlotEnergyDecoder(nn.Module):
    """global_intent 逐层注入（保留旧机制）+ cross-attention 读取槽位（新通路）。"""

    def __init__(self, vocab_size: int, max_len: int, dim: int = 256,
                 depth: int = 4, n_heads: int = 8, intent_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.intent_proj = nn.Linear(intent_dim, dim)      # global 注入（同旧版）
        self.slot_kv_proj = nn.Linear(intent_dim, dim)     # 槽位投到模型维供读取
        self.blocks = nn.ModuleList([CausalPERBlock(dim, n_heads) for _ in range(depth)])
        # 槽位读取通路：标准 MHA，但只作路由（"目"），溯源不依赖其权重。
        self.slot_reader = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.read_norm = nn.LayerNorm(dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.h_proj = nn.Linear(dim, intent_dim)           # 隐状态投到意图空间（算能量）

    def forward(self, ids: torch.Tensor, global_intent: torch.Tensor,
                intent_slots: torch.Tensor):
        """ids:(B,L) → logits:(B,L,V), h_intent:(B,L,d)

        h_intent 同时用于：全局能量（距 global_intent）与槽位覆盖能量（距各槽位）。
        """
        B, L = ids.shape
        x = self.embed(ids) + self.pos[:, :L]
        z_inject = self.intent_proj(global_intent).unsqueeze(1)
        for blk in self.blocks:
            x = x + z_inject                                # 看着全局目标走（旧机制保留）
            x = blk(x)
        # 槽位读取：因果性安全——KV 是编码器槽位（不含未来 token 信息）。
        kv = self.slot_kv_proj(intent_slots)                # (B, K, dim)
        read, _ = self.slot_reader(self.read_norm(x), kv, kv, need_weights=False)
        x = x + read
        h = self.norm(x)
        logits = self.head(h)
        h_intent = self.h_proj(h)
        return logits, h_intent


class SlotIntentLM(nn.Module):
    """结构化意图驱动语言模型 = SlotIntentEncoder + SlotEnergyDecoder。"""

    def __init__(self, vocab_size: int, enc_max: int = 48, dec_max: int = 48,
                 dim: int = 256, enc_depth: int = 4, dec_depth: int = 6,
                 n_heads: int = 8, intent_dim: int = 128, n_slots: int = 8):
        super().__init__()
        self.encoder = SlotIntentEncoder(vocab_size, enc_max, dim, enc_depth,
                                         n_heads, intent_dim, n_slots)
        self.decoder = SlotEnergyDecoder(vocab_size, dec_max, dim, dec_depth,
                                         n_heads, intent_dim)
        self.intent_dim = intent_dim
        self.n_slots = n_slots

    def forward(self, prompt_ids: torch.Tensor, resp_ids: torch.Tensor):
        """训练前向（铁律：解码器只条件于 prompt 侧意图）。"""
        global_intent, intent_slots, slot_salience = self.encoder(prompt_ids)
        logits, h_intent = self.decoder(resp_ids, global_intent, intent_slots)
        return logits, h_intent, global_intent, intent_slots, slot_salience

    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "vocab_size": self.encoder.embed.num_embeddings,
                "intent_dim": self.intent_dim,
                "n_slots": self.n_slots,
                "enc_max": self.encoder.pos.shape[1],
                "dec_max": self.decoder.pos.shape[1],
                "dim": self.encoder.embed.embedding_dim,
                "enc_depth": len(self.encoder.blocks),
                "dec_depth": len(self.decoder.blocks),
                "n_heads": self.encoder.blocks[0].n_heads,
            },
        }, path)

    @classmethod
    def load(cls, path: str, map_location="cpu"):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ck["config"]
        net = cls(vocab_size=cfg["vocab_size"], intent_dim=cfg["intent_dim"],
                  n_slots=cfg.get("n_slots", 8),
                  enc_max=cfg.get("enc_max", 48), dec_max=cfg.get("dec_max", 48),
                  dim=cfg.get("dim", 256), enc_depth=cfg.get("enc_depth", 4),
                  dec_depth=cfg.get("dec_depth", 6), n_heads=cfg.get("n_heads", 8))
        net.load_state_dict(ck["state_dict"])
        return net
