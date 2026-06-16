"""Shared data containers for the pretrained-backbone FE-LLM route."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class IntentState:
    """Structured intent read from a frozen backbone.

    global_intent 继续对接主动推理控制层；intent_slots 与 slot_salience
    服务生成层的显式能量计算，不能把 attention weight 直接当解释。
    """

    global_intent: torch.Tensor
    intent_slots: torch.Tensor
    slot_salience: torch.Tensor

    def validate(self) -> None:
        if self.global_intent.ndim != 2:
            raise ValueError("global_intent must have shape (batch, intent_dim)")
        if self.intent_slots.ndim != 3:
            raise ValueError("intent_slots must have shape (batch, n_slots, intent_dim)")
        if self.slot_salience.ndim != 2:
            raise ValueError("slot_salience must have shape (batch, n_slots)")
        batch, _, intent_dim = self.intent_slots.shape
        if self.global_intent.shape != (batch, intent_dim):
            raise ValueError("global_intent and intent_slots dimensions do not match")
        if self.slot_salience.shape != self.intent_slots.shape[:2]:
            raise ValueError("slot_salience and intent_slots slot dimensions do not match")

    def to_trace_summary(self) -> dict[str, float | int]:
        self.validate()
        return {
            "batch": int(self.global_intent.shape[0]),
            "intent_dim": int(self.global_intent.shape[1]),
            "n_slots": int(self.intent_slots.shape[1]),
            "global_norm": float(self.global_intent.norm(dim=-1).mean().detach().cpu()),
            "salience_entropy": float(
                (-(self.slot_salience.clamp_min(1e-8).log() * self.slot_salience).sum(dim=-1))
                .mean()
                .detach()
                .cpu()
            ),
        }


@dataclass
class HybridDecodeStep:
    """Trace record for one P1 hybrid decoding decision."""

    token_id: int
    prob_token_id: int
    energy_token_id: int
    score: float
    log_prob: float
    residual_energy: float
    alpha: float
