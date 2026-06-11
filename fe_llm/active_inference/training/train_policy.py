"""Train the tiny active-inference policy selector."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fe_llm.active_inference.observation import Observation
from fe_llm.active_inference.perception import PerceptionEncoder
from fe_llm.active_inference.policy import ActionType, build_policy_feature_vector
from fe_llm.active_inference.prediction import Predictor
from fe_llm.active_inference.state import BeliefState
from fe_llm.active_inference.surprise import PredictionError, PredictionErrorEstimator
from fe_llm.config import get_device

DEFAULT_DATA = os.path.join("data", "active_inference", "policy_teacher.jsonl")
DEFAULT_CKPT = os.path.join("checkpoints", "active_inference", "policy_selector.pt")


class TinyPolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = len(ActionType)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_samples(path: str) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("action_type") in {a.value for a in ActionType}:
                samples.append(item)
    return samples


def build_dataset(samples: list[dict], use_intent_model: bool = True) -> tuple[np.ndarray, np.ndarray]:
    encoder = PerceptionEncoder(use_intent_model=use_intent_model)
    predictor = Predictor()
    error_estimator = PredictionErrorEstimator()
    labels = {action.value: i for i, action in enumerate(ActionType)}
    xs, ys = [], []
    prior = BeliefState.empty(encoder.vector_dim)
    prediction = predictor.predict(prior)
    for item in samples:
        obs = Observation.from_text(item["prompt"])
        obs_state = encoder.encode(obs)
        teacher_error = item.get("surprise_components")
        if isinstance(teacher_error, dict):
            pred_error = PredictionError(
                semantic_error=float(teacher_error.get("semantic_error", 0.0)),
                intent_error=float(teacher_error.get("intent_error", 0.0)),
                consistency_error=float(teacher_error.get("consistency_error", 0.0)),
                uncertainty_error=float(teacher_error.get("uncertainty_error", 0.0)),
                safety_error=float(teacher_error.get("safety_error", 0.0)),
            )
        else:
            pred_error = error_estimator.compare(obs_state, prediction)
        xs.append(build_policy_feature_vector(obs_state, pred_error))
        ys.append(labels[item["action_type"]])
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def train(args: argparse.Namespace) -> float:
    samples = load_samples(args.data)
    if len(samples) < 10:
        raise RuntimeError(f"Need at least 10 samples, got {len(samples)} from {args.data}")
    train_samples, val_samples = split_samples(samples, val_ratio=args.val_ratio, seed=args.seed)
    x_train, y_train = build_dataset(train_samples, use_intent_model=not args.no_intent_model)
    x_val, y_val = build_dataset(val_samples, use_intent_model=not args.no_intent_model)

    device = get_device()
    model = TinyPolicyNet(input_dim=x_train.shape[1], hidden_dim=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and args.amp))
    class_weight = None if args.no_class_weights else class_weights(y_train).to(device)

    batch = args.batch
    best_metrics = {"accuracy": 0.0, "balanced_accuracy": -1.0}
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        order = np.random.default_rng(args.seed + epoch).permutation(len(x_train))
        model.train()
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            xb = torch.tensor(x_train[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda" and args.amp)):
                loss = F.cross_entropy(model(xb), yb, weight=class_weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        metrics = evaluate(model, x_val, y_val, device)
        if metrics["balanced_accuracy"] > best_metrics["balanced_accuracy"]:
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"[policy] epoch={epoch:03d} "
            f"val_acc={metrics['accuracy']:.3f} "
            f"val_bal_acc={metrics['balanced_accuracy']:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mlp_state_dict": model.net.state_dict(),
            "input_dim": int(x_train.shape[1]),
            "hidden_dim": int(args.hidden),
            "actions": [action.value for action in ActionType],
            "val_metrics": evaluate(model, x_val, y_val, device),
            "selection_metric": "balanced_accuracy",
        },
        args.output,
    )
    return evaluate(model, x_val, y_val, device)


def split_samples(samples: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified split so small seed files keep every action represented."""

    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for sample in samples:
        buckets.setdefault(sample["action_type"], []).append(sample)
    train_samples: list[dict] = []
    val_samples: list[dict] = []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n_val = max(1, int(round(len(bucket) * val_ratio))) if len(bucket) > 1 else 0
        val_samples.extend(bucket[:n_val])
        train_samples.extend(bucket[n_val:])
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    if not val_samples:
        split = max(1, int(len(samples) * (1.0 - val_ratio)))
        train_samples = samples[:split]
        val_samples = samples[split:] or samples[: max(1, len(samples) // 5)]
    return train_samples, val_samples


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, device: str) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device))
        pred = logits.argmax(-1).cpu().numpy()
    accuracy = float((pred == y).mean())
    recalls = []
    per_class_recall = {}
    for label in range(len(ActionType)):
        mask = y == label
        if mask.any():
            recall = float((pred[mask] == y[mask]).mean())
            recalls.append(recall)
            per_class_recall[list(ActionType)[label].value] = recall
    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "per_class_recall": per_class_recall,
    }


def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(ActionType)).astype(np.float32)
    present = counts > 0
    weights = np.zeros(len(ActionType), dtype=np.float32)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--output", default=DEFAULT_CKPT)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no-intent-model", action="store_true")
    ap.add_argument("--no-class-weights", action="store_true")
    args = ap.parse_args()
    metrics = train(args)
    print(
        f"[policy] saved {args.output} "
        f"val_accuracy={metrics['accuracy']:.3f} "
        f"val_balanced_accuracy={metrics['balanced_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
