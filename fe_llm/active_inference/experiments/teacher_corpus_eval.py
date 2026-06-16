# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/teacher_corpus_eval.py
==========================================================
在教师生成的**任务型多轮语料**上验证控制层 headroom：上下文感知（句子 + belief 槽位）
vs 上下文盲（只看句子）的动作分类。任务型数据上下文强耦合，预期 headroom 远强于开放闲聊。

判定：上下文感知在「歧义子集」(同句出现多种动作) 上的 balanced acc 显著高于盲。
默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.active_inference.experiments.teacher_corpus_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fe_llm.active_inference.policy import ActionType
from fe_llm.config import get_device

CORPUS = os.path.join("data", "dialogue", "teacher_task_oriented.jsonl")
REPORT_JSON = os.path.join("docs", "reports", "teacher_corpus_eval.json")
REPORT_MD = os.path.join("docs", "reports", "teacher_corpus_eval.md")


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def balanced_accuracy(pred: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    recalls = []
    for c in range(n_classes):
        mask = y == c
        if mask.any():
            recalls.append(float((pred[mask] == y[mask]).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def stratified_split(y: np.ndarray, n_classes: int, seed: int, val_frac=0.25):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * (1 - val_frac)))
        tr.extend(idx[:cut].tolist())
        va.extend(idx[cut:].tolist())
    return np.array(tr, dtype=np.int64), np.array(va, dtype=np.int64)


def train_eval(X, y, tr, va, amb_val, n_classes, device, seed, epochs=250):
    torch.manual_seed(seed)
    Xtr = torch.tensor(X[tr], dtype=torch.float32, device=device)
    ytr = torch.tensor(y[tr], dtype=torch.long, device=device)
    Xva = torch.tensor(X[va], dtype=torch.float32, device=device)
    clf = MLP(X.shape[1], n_classes).to(device)
    counts = np.bincount(y[tr], minlength=n_classes).astype(np.float32)
    w = torch.tensor(counts.sum() / (np.maximum(counts, 1) * n_classes), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(clf.parameters(), lr=5e-3, weight_decay=1e-4)
    for _ in range(epochs):
        clf.train()
        opt.zero_grad()
        F.cross_entropy(clf(Xtr), ytr, weight=w).backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        pred = clf(Xva).argmax(-1).cpu().numpy()
    yva = y[va]
    overall = balanced_accuracy(pred, yva, n_classes)
    amb = balanced_accuracy(pred[amb_val], yva[amb_val], n_classes) if amb_val.any() else 0.0
    return overall, amb


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Context-aware vs context-blind action classification on teacher task corpus.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--corpus", default=CORPUS)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[teacher-eval] dry-run：未训练。")
    print(f"[teacher-eval] 任务型语料 {args.corpus} 上：上下文感知(句子+belief) vs 盲(只看句子)。")
    print("[teacher-eval] 真正运行请显式追加 --run。")


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    rows = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    action_idx = {a.value: i for i, a in enumerate(ActionType)}
    n_classes = len(ActionType)
    y = np.array([action_idx[r["action"]] for r in rows], dtype=np.int64)

    # 歧义标记：同一 utterance 文本对应了多于一种动作（只看句子无法区分）
    text_actions = defaultdict(set)
    for r in rows:
        text_actions[r["utterance"]].add(r["action"])
    ambiguous = np.array([len(text_actions[r["utterance"]]) > 1 for r in rows], dtype=bool)

    # 字符袋（utterance）
    vocab = sorted({c for r in rows for c in r["utterance"]})
    cidx = {c: i for i, c in enumerate(vocab)}

    def bow(t: str) -> np.ndarray:
        v = np.zeros(len(vocab), dtype=np.float32)
        for c in t:
            v[cidx[c]] += 1.0
        return v / max(len(t), 1)

    # belief 特征：已知槽位键 multi-hot + 已知槽位数（归一化）
    slot_keys = sorted({k for r in rows for k in r["known_slots"].keys()})
    skidx = {k: i for i, k in enumerate(slot_keys)}

    def belief_vec(known: dict) -> np.ndarray:
        v = np.zeros(len(slot_keys) + 1, dtype=np.float32)
        for k in known:
            if k in skidx:
                v[skidx[k]] = 1.0
        v[-1] = min(len(known), 4) / 4.0
        return v

    U = np.stack([bow(r["utterance"]) for r in rows])
    B = np.stack([belief_vec(r["known_slots"]) for r in rows])
    X_blind = U
    X_ctx = np.concatenate([U, B], axis=1)

    tr, va = stratified_split(y, n_classes, args.seed)
    amb_val = ambiguous[va]
    dist = {a.value: int((y == i).sum()) for a, i in zip(ActionType, range(n_classes))}

    blind_overall, blind_amb = train_eval(X_blind, y, tr, va, amb_val, n_classes, device, args.seed, args.epochs)
    ctx_overall, ctx_amb = train_eval(X_ctx, y, tr, va, amb_val, n_classes, device, args.seed, args.epochs)

    amb_delta = ctx_amb - blind_amb
    verdict = "PASS: 任务型数据上 belief 带来强 headroom" if amb_delta > 0.2 else "WEAK: headroom 不明显"
    result = {
        "n_turns": len(rows),
        "ambiguous_frac": round(float(ambiguous.mean()), 4),
        "class_dist": dist,
        "overall": {"context_blind": round(blind_overall, 4), "context_aware": round(ctx_overall, 4)},
        "ambiguous_subset": {"context_blind": round(blind_amb, 4), "context_aware": round(ctx_amb, 4), "delta": round(amb_delta, 4)},
        "verdict": verdict,
        "note": "任务型多轮语料；歧义子集=同句多动作（只看句子无法区分），belief 槽位特征是唯一变量。对照真实开放闲聊(0.655 弱)，任务型应显著更强。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# FE-LLM 任务型语料控制层 headroom（上下文感知 vs 盲）",
        "",
        f"- 判定：**{verdict}**",
        f"- turns：{result['n_turns']}，歧义占比：{result['ambiguous_frac']}",
        "",
        "## 总体 balanced accuracy",
        f"- 盲（只看句子）：{result['overall']['context_blind']}",
        f"- 上下文感知（句子+belief）：{result['overall']['context_aware']}",
        "",
        "## 歧义子集（同句多动作，headroom 关键）",
        f"- 盲：{result['ambiguous_subset']['context_blind']}",
        f"- 上下文感知：{result['ambiguous_subset']['context_aware']}",
        f"- delta：{result['ambiguous_subset']['delta']}",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print_dry_run(args)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
