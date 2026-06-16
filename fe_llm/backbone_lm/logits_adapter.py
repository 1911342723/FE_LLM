"""Intent-conditioned logit bias adapter for the P1.5 route."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import IntentState


class IntentLogitsAdapter(nn.Module):
    """Map candidate hidden states and structured intent to logit biases.

    P1 的 energy rerank 是"事后扣分"；P1.5 让 intent 直接产生候选 logit bias，
    用同一 A/B/C 口径检验真实 intent 是否稳定优于随机 intent。
    """

    def __init__(self, hidden_size: int, intent_dim: int = 128, adapter_dim: int = 128) -> None:
        super().__init__()
        self.hidden_proj = nn.Linear(hidden_size, intent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(intent_dim * 4, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, 1),
        )

    def forward(self, candidate_hidden: torch.Tensor, intent_state: IntentState) -> torch.Tensor:
        """Return bias with shape (batch, n_candidates).

        candidate_hidden can be (B, K, H) for K candidates or (B, H) for one candidate.
        """

        intent_state.validate()
        squeeze = False
        if candidate_hidden.ndim == 2:
            candidate_hidden = candidate_hidden.unsqueeze(1)
            squeeze = True
        if candidate_hidden.ndim != 3:
            raise ValueError("candidate_hidden must have shape (batch, n_candidates, hidden_size)")

        h = self.hidden_proj(candidate_hidden)
        global_intent = intent_state.global_intent.unsqueeze(1).expand(-1, h.shape[1], -1)

        h_norm = F.normalize(h, dim=-1)
        slot_norm = F.normalize(intent_state.intent_slots, dim=-1)
        slot_scores = torch.einsum("bkd,bsd->bks", h_norm, slot_norm)
        slot_weights = torch.softmax(slot_scores, dim=-1) * intent_state.slot_salience.unsqueeze(1)
        slot_weights = slot_weights / slot_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        slot_context = torch.einsum("bks,bsd->bkd", slot_weights, intent_state.intent_slots)

        features = torch.cat([h, global_intent, slot_context, h - global_intent], dim=-1)
        bias = self.mlp(features).squeeze(-1)
        return bias.squeeze(1) if squeeze else bias
