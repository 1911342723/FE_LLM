# -*- coding: utf-8 -*-
"""
fe_llm/world_model/hierarchical_lm.py —— 分层意图语言模型（v2-M1 端到端）
=========================================================================
HierarchicalIntentLM = HierarchicalPredictiveEncoder + SlotEnergyDecoder。

- 编码器产出 z_global（顶层抽象意图，对应 v1 单向量的位置）+ z_local（底层局部 latent）；
- 解码器复用已验证的 SlotEnergyDecoder：global_intent 注入 + 对 z_local 做 cross-attention
  读取（z_local 充当"局部要素/爻"），逐字能量递减生成。

铁律（沿用翻译实验教训，写入 经验.md）：
    解码器只条件于 encoder(prompt) 的输出，训练/推理同分布。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fe_llm.energy_lm.models.slot_intent_model import SlotEnergyDecoder

from .hierarchical_encoder import HierarchicalPredictiveEncoder, HierarchicalState


class HierarchicalIntentLM(nn.Module):
    """分层预测编码 + 能量解码 的端到端语言模型（完全从 0，不依赖底座）。"""

    def __init__(
        self,
        vocab_size: int,
        enc_max: int = 48,
        dec_max: int = 48,
        dim: int = 256,
        enc_depth: int = 4,
        dec_depth: int = 6,
        n_heads: int = 8,
        intent_dim: int = 128,
        relax_steps: int = 5,
        alpha: float = 0.3,
        precision: float = 1.0,
    ):
        super().__init__()
        self.encoder = HierarchicalPredictiveEncoder(
            vocab_size=vocab_size,
            max_len=enc_max,
            dim=dim,
            n_heads=n_heads,
            intent_dim=intent_dim,
            depth=enc_depth,
            relax_steps=relax_steps,
            alpha=alpha,
            precision=precision,
        )
        self.decoder = SlotEnergyDecoder(
            vocab_size=vocab_size,
            max_len=dec_max,
            dim=dim,
            depth=dec_depth,
            n_heads=n_heads,
            intent_dim=intent_dim,
        )
        self.intent_dim = intent_dim

    def forward(
        self,
        prompt_ids: torch.Tensor,
        resp_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, HierarchicalState]:
        """训练前向：解码器条件于 prompt 侧的 z_global + z_local（同分布铁律）。"""

        state = self.encoder(prompt_ids, attention_mask=attention_mask)
        logits, h_intent = self.decoder(resp_ids, state.z_global, state.z_local)
        return logits, h_intent, state

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "vocab_size": self.encoder.embed.num_embeddings,
                    "intent_dim": self.intent_dim,
                    "enc_max": self.encoder.pos.shape[1],
                    "dec_max": self.decoder.pos.shape[1],
                    "dim": self.encoder.embed.embedding_dim,
                    "enc_depth": len(self.encoder.blocks),
                    "dec_depth": len(self.decoder.blocks),
                    "n_heads": self.encoder.blocks[0].n_heads,
                    "relax_steps": self.encoder.relax_steps,
                    "alpha": self.encoder.alpha,
                    "precision": self.encoder.precision,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "HierarchicalIntentLM":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ckpt["config"]
        net = cls(
            vocab_size=cfg["vocab_size"],
            enc_max=cfg.get("enc_max", 48),
            dec_max=cfg.get("dec_max", 48),
            dim=cfg.get("dim", 256),
            enc_depth=cfg.get("enc_depth", 4),
            dec_depth=cfg.get("dec_depth", 6),
            n_heads=cfg.get("n_heads", 8),
            intent_dim=cfg["intent_dim"],
            relax_steps=cfg.get("relax_steps", 5),
            alpha=cfg.get("alpha", 0.3),
            precision=cfg.get("precision", 1.0),
        )
        net.load_state_dict(ckpt["state_dict"])
        return net
