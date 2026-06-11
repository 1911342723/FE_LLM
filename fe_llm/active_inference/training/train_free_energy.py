"""Calibrate expected-free-energy weights from policy teacher data."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn.functional as F

from fe_llm.active_inference.free_energy import FreeEnergyScorer
from fe_llm.active_inference.observation import Observation
from fe_llm.active_inference.policy import ActionType, PolicyGenerator
from fe_llm.active_inference.state import BeliefState
from fe_llm.active_inference.surprise import PredictionError, SurpriseEstimator
from fe_llm.active_inference.training.train_policy import DEFAULT_DATA, load_samples, split_samples

DEFAULT_OUTPUT = os.path.join("checkpoints", "active_inference", "free_energy_weights.json")


def build_component_dataset(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    labels = {action.value: idx for idx, action in enumerate(ActionType)}
    scorer = FreeEnergyScorer()
    surprise_estimator = SurpriseEstimator()
    generator = PolicyGenerator()
    belief = BeliefState.empty(128)
    xs, ys = [], []
    for sample in samples:
        observation = Observation.from_text(sample["prompt"])
        comps = sample.get("surprise_components", {})
        pred_error = PredictionError(
            semantic_error=float(comps.get("semantic_error", 0.0)),
            intent_error=float(comps.get("intent_error", 0.0)),
            consistency_error=float(comps.get("consistency_error", 0.0)),
            uncertainty_error=float(comps.get("uncertainty_error", 0.0)),
            safety_error=float(comps.get("safety_error", 0.0)),
        )
        surprise = surprise_estimator.score(pred_error)
        candidates = generator.generate(belief, surprise)
        scores = scorer.score(candidates, belief, surprise, observation)
        rows = []
        for action in ActionType:
            score = scores[action]
            rows.append([score.risk, score.ambiguity, score.action_cost, -score.epistemic_value, 1.0])
        xs.append(rows)
        ys.append(labels[sample["action_type"]])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def train(args: argparse.Namespace) -> dict:
    samples = load_samples(args.data)
    train_samples, val_samples = split_samples(samples, val_ratio=args.val_ratio, seed=args.seed)
    x_train, y_train = build_component_dataset(train_samples)
    x_val, y_val = build_component_dataset(val_samples)

    torch.manual_seed(args.seed)
    init = math.log(math.exp(1.0) - 1.0)
    raw_weights = torch.nn.Parameter(torch.full((len(ActionType), 4), init))
    action_bias = torch.nn.Parameter(torch.zeros(len(ActionType)))
    opt = torch.optim.AdamW([raw_weights, action_bias], lr=args.lr, weight_decay=args.weight_decay)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    class_weight = class_weights(y_train)

    best = {"balanced_accuracy": -1.0}
    best_state = None
    for epoch in range(1, args.epochs + 1):
        weights = F.softplus(raw_weights) + 1e-4
        totals = (x_train_t[:, :, :4] * weights.unsqueeze(0)).sum(-1) + action_bias
        logits = -totals
        loss = F.cross_entropy(logits, y_train_t, weight=class_weight)
        regularizer = args.prior_strength * ((weights - 1.0) ** 2).mean() + args.bias_strength * (action_bias**2).mean()
        objective = loss + regularizer
        opt.zero_grad(set_to_none=True)
        objective.backward()
        opt.step()

        metrics = evaluate(x_val, y_val, raw_weights, action_bias)
        if metrics["balanced_accuracy"] > best["balanced_accuracy"]:
            best = metrics
            best_state = {
                "raw_weights": raw_weights.detach().clone(),
                "action_bias": action_bias.detach().clone(),
            }
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"[efe] epoch={epoch:04d} loss={float(loss.detach()):.4f} "
                f"val_acc={metrics['accuracy']:.3f} val_bal_acc={metrics['balanced_accuracy']:.3f}",
                flush=True,
            )

    if best_state is not None:
        raw_weights.data.copy_(best_state["raw_weights"])
        action_bias.data.copy_(best_state["action_bias"])
        best = evaluate(x_val, y_val, raw_weights, action_bias)

    weights = (F.softplus(raw_weights) + 1e-4).detach().cpu().numpy()
    bias = action_bias.detach().cpu().numpy()
    action_weights = {
        action.value: {
            "risk_weight": float(weights[idx, 0]),
            "ambiguity_weight": float(weights[idx, 1]),
            "action_cost_weight": float(weights[idx, 2]),
            "epistemic_value_weight": float(weights[idx, 3]),
        }
        for idx, action in enumerate(ActionType)
    }
    out = {
        "risk_weight": 1.0,
        "ambiguity_weight": 1.0,
        "action_cost_weight": 1.0,
        "epistemic_value_weight": 1.0,
        "action_weights": action_weights,
        "action_bias": {action.value: float(bias[idx]) for idx, action in enumerate(ActionType)},
        "val_metrics": best,
        "objective": "per-action calibration of risk + ambiguity + action_cost - epistemic_value + action_bias",
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(ActionType)).astype(np.float32)
    present = counts > 0
    weights = np.ones(len(ActionType), dtype=np.float32)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(
    x: np.ndarray,
    y: np.ndarray,
    raw_weights: torch.nn.Parameter,
    action_bias: torch.nn.Parameter,
) -> dict:
    with torch.no_grad():
        weights = F.softplus(raw_weights) + 1e-4
        totals = (torch.tensor(x, dtype=torch.float32)[:, :, :4] * weights.unsqueeze(0)).sum(-1) + action_bias
        pred = (-totals).argmax(-1).cpu().numpy()
    accuracy = float((pred == y).mean())
    per_class_recall = {}
    recalls = []
    for idx, action in enumerate(ActionType):
        mask = y == idx
        if mask.any():
            recall = float((pred[mask] == y[mask]).mean())
            per_class_recall[action.value] = recall
            recalls.append(recall)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "per_class_recall": per_class_recall,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--prior-strength", type=float, default=0.02)
    ap.add_argument("--bias-strength", type=float, default=0.002)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()
    result = train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
