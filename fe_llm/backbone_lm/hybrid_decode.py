"""Hybrid score utilities for FE-LLM P1 decoding."""

from __future__ import annotations

import torch

from .types import HybridDecodeStep


def normalize_candidate_energy(residual_energy: torch.Tensor) -> torch.Tensor:
    """Normalize candidate energies inside one decoding step to [0, 1]."""

    if residual_energy.ndim != 1:
        raise ValueError("residual_energy must be a 1D tensor of candidate energies")
    span = residual_energy.max() - residual_energy.min()
    if float(span.detach().cpu()) <= 1e-8:
        return torch.zeros_like(residual_energy)
    return (residual_energy - residual_energy.min()) / span


def hybrid_scores(
    candidate_log_probs: torch.Tensor,
    candidate_residual_energy: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """score = logP_backbone(w) - alpha * normalized_residual_energy(w)."""

    if candidate_log_probs.shape != candidate_residual_energy.shape:
        raise ValueError("candidate_log_probs and candidate_residual_energy must have same shape")
    if candidate_log_probs.ndim != 1:
        raise ValueError("candidate_log_probs must be a 1D tensor")
    energy_norm = normalize_candidate_energy(candidate_residual_energy)
    return candidate_log_probs - alpha * energy_norm


def select_hybrid_candidate(
    candidate_token_ids: torch.Tensor,
    candidate_log_probs: torch.Tensor,
    candidate_residual_energy: torch.Tensor,
    alpha: float = 1.0,
) -> HybridDecodeStep:
    """Select one candidate and return a trace-friendly decision record."""

    if candidate_token_ids.ndim != 1:
        raise ValueError("candidate_token_ids must be a 1D tensor")
    if candidate_token_ids.shape != candidate_log_probs.shape:
        raise ValueError("candidate_token_ids and candidate_log_probs must have same shape")
    scores = hybrid_scores(candidate_log_probs, candidate_residual_energy, alpha=alpha)
    selected_idx = int(scores.argmax())
    prob_idx = int(candidate_log_probs.argmax())
    energy_idx = int(candidate_residual_energy.argmin())
    return HybridDecodeStep(
        token_id=int(candidate_token_ids[selected_idx]),
        prob_token_id=int(candidate_token_ids[prob_idx]),
        energy_token_id=int(candidate_token_ids[energy_idx]),
        score=float(scores[selected_idx].detach().cpu()),
        log_prob=float(candidate_log_probs[selected_idx].detach().cpu()),
        residual_energy=float(candidate_residual_energy[selected_idx].detach().cpu()),
        alpha=alpha,
    )
