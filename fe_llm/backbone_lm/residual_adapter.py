"""Intent-conditioned hidden-state residual adapter for the P2 route."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import IntentState


class IntentResidualAdapter(nn.Module):
    """Generate intent-conditioned delta hidden states.

    P2 比 P1.5 更深：不是只给候选 logit 加标量 bias，而是先改变候选 hidden
    state，再经底座 lm_head 读出 logits。
    """

    def __init__(
        self,
        hidden_size: int,
        intent_dim: int = 128,
        adapter_dim: int = 128,
        max_delta_norm: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_proj = nn.Linear(hidden_size, intent_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(intent_dim * 4, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, hidden_size),
        )
        self.max_delta_norm = max_delta_norm

    def forward(
        self,
        hidden: torch.Tensor,
        intent_state: IntentState,
        gamma: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (adapted_hidden, delta_hidden).

        hidden can be (B, H) or (B, K, H).
        """

        intent_state.validate()
        squeeze = False
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(1)
            squeeze = True
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape (batch, n_items, hidden_size)")

        h_intent = self.hidden_proj(hidden)
        global_intent = intent_state.global_intent.unsqueeze(1).expand(-1, hidden.shape[1], -1)
        h_norm = F.normalize(h_intent, dim=-1)
        slot_norm = F.normalize(intent_state.intent_slots, dim=-1)
        slot_scores = torch.einsum("bkd,bsd->bks", h_norm, slot_norm)
        slot_weights = torch.softmax(slot_scores, dim=-1) * intent_state.slot_salience.unsqueeze(1)
        slot_weights = slot_weights / slot_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        slot_context = torch.einsum("bks,bsd->bkd", slot_weights, intent_state.intent_slots)

        features = torch.cat([h_intent, global_intent, slot_context, h_intent - global_intent], dim=-1)
        delta = self.delta_head(features)
        if self.max_delta_norm > 0:
            norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            scale = torch.clamp(self.max_delta_norm / norm, max=1.0)
            delta = delta * scale
        adapted = hidden + gamma * delta
        if squeeze:
            return adapted.squeeze(1), delta.squeeze(1)
        return adapted, delta
