"""Intent adapter and explicit energy head for the P1 backbone route."""

from __future__ import annotations

import torch
import torch.nn as nn

from .types import IntentState


def _masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states.mean(dim=1)
    mask = attention_mask.to(dtype=hidden_states.dtype, device=hidden_states.device).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_states * mask).sum(dim=1) / denom


class IntentAdapter(nn.Module):
    """Read structured intent from frozen backbone hidden states.

    P1 不改底座参数：adapter 只用 learned queries 读取底座隐状态，并输出
    global_intent、intent_slots、slot_salience 三个机制层变量。
    """

    def __init__(
        self,
        hidden_size: int,
        intent_dim: int = 128,
        n_slots: int = 8,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")
        self.hidden_size = hidden_size
        self.intent_dim = intent_dim
        self.n_slots = n_slots
        self.global_proj = nn.Linear(hidden_size, intent_dim)
        self.slot_queries = nn.Parameter(torch.randn(1, n_slots, hidden_size) * 0.02)
        self.slot_reader = nn.MultiheadAttention(
            hidden_size,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.slot_norm = nn.LayerNorm(hidden_size)
        self.slot_proj = nn.Linear(hidden_size, intent_dim)
        self.salience_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> IntentState:
        """hidden_states: (B, L, H), attention_mask: (B, L) with 1 for valid tokens."""

        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (batch, seq_len, hidden_size)")
        batch = hidden_states.shape[0]
        pooled = _masked_mean(hidden_states, attention_mask)
        global_intent = self.global_proj(pooled)

        queries = self.slot_queries.expand(batch, -1, -1)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask.to(device=hidden_states.device) == 0
        slot_states, _ = self.slot_reader(
            queries,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        slot_states = self.slot_norm(slot_states)
        intent_slots = self.slot_proj(slot_states)
        slot_salience = torch.softmax(self.salience_head(slot_states).squeeze(-1), dim=-1)
        state = IntentState(
            global_intent=global_intent,
            intent_slots=intent_slots,
            slot_salience=slot_salience,
        )
        state.validate()
        return state


class EnergyHead(nn.Module):
    """Compute residual and coverage energy from decoder hidden states."""

    def __init__(
        self,
        hidden_size: int,
        intent_dim: int = 128,
        coverage_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_proj = nn.Linear(hidden_size, intent_dim)
        self.coverage_weight = coverage_weight

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (batch, seq_len, hidden_size)")
        return self.hidden_proj(hidden_states)

    def residual_energy(self, hidden_states: torch.Tensor, intent_state: IntentState) -> torch.Tensor:
        intent_state.validate()
        h_intent = self.project(hidden_states)
        target = intent_state.global_intent.unsqueeze(1)
        return torch.norm(h_intent - target, dim=-1)

    def coverage_energy(self, hidden_states: torch.Tensor, intent_state: IntentState) -> torch.Tensor:
        """Prefix-min coverage energy from the slot draft.

        For each generated position i and slot k, use min_{j<=i} ||h_j - slot_k||.
        This makes coverage an explicit trace quantity: high-salience slots should be
        approached over time before EOS becomes cheap.
        """

        intent_state.validate()
        h_intent = self.project(hidden_states)
        distances = torch.cdist(h_intent, intent_state.intent_slots)
        prefix_min = distances.cummin(dim=1).values
        weighted = prefix_min * intent_state.slot_salience.unsqueeze(1)
        return weighted.sum(dim=-1)

    def forward(self, hidden_states: torch.Tensor, intent_state: IntentState) -> dict[str, torch.Tensor]:
        residual = self.residual_energy(hidden_states, intent_state)
        coverage = self.coverage_energy(hidden_states, intent_state)
        total = residual + self.coverage_weight * coverage
        return {
            "residual_energy": residual,
            "coverage_energy": coverage,
            "total_energy": total,
        }
