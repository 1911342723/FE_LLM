# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/train_task_nlu.py
=====================================================
把"理解层"也学习化到任务数据：用教师任务型语料训练一个**任务领域识别 NLU**
（utterance → 9 个任务领域 flight/train/hotel/restaurant/appointment/delivery/food/topup/repair），
而不是合成模板。按 session 切分留出评测，证明它能从任务语料泛化识别领域。

判定：留出 session 的领域 balanced accuracy 高（学习式理解层在任务数据上有效）。
默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.active_inference.experiments.train_task_nlu --run
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

CORPUS = os.path.join("data", "dialogue", "teacher_task_oriented.jsonl")
REPORT_JSON = os.path.join("docs", "reports", "task_nlu_eval.json")
REPORT_MD = os.path.join("docs", "reports", "task_nlu_eval.md")


def _balanced_acc(pred: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    recalls = []
    for c in range(n_classes):
        m = y == c
        if m.any():
            recalls.append(float((pred[m] == y[m]).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train task-domain NLU on teacher corpus (session split).")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--corpus", default=CORPUS)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.25)
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[task-nlu] dry-run：未训练。")
    print(f"[task-nlu] 用任务语料 {args.corpus} 训练领域识别 NLU（utterance→领域），按 session 留出评测。")
    print("[task-nlu] 真正运行请显式追加 --run。")


def run(args: argparse.Namespace) -> dict:
    rows = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # 只用任务轮（请求/补槽位 = ask/answer 动作）做领域识别；全局轮(refuse/retrieve/memory)领域无意义。
    task_rows = [r for r in rows if r["action"] in ("ask_clarification", "answer")]
    domains = sorted({r["domain"] for r in task_rows})
    did = {d: i for i, d in enumerate(domains)}
    n_classes = len(domains)

    # 按 session 切分防泄漏
    sessions = defaultdict(list)
    for r in task_rows:
        sessions[r["session_id"]].append(r)
    sids = sorted(sessions)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(sids)
    cut = int(len(sids) * (1 - args.val_frac))
    train_rows = [r for s in sids[:cut] for r in sessions[s]]
    val_rows = [r for s in sids[cut:] for r in sessions[s]]

    vocab = sorted({c for r in task_rows for c in r["utterance"]})
    cidx = {c: i for i, c in enumerate(vocab)}

    def bow(t: str) -> np.ndarray:
        v = np.zeros(len(vocab), dtype=np.float32)
        for c in t:
            v[cidx[c]] += 1.0
        return v / max(len(t), 1)

    Xtr = np.stack([bow(r["utterance"]) for r in train_rows])
    ytr = np.array([did[r["domain"]] for r in train_rows], dtype=np.int64)
    Xva = np.stack([bow(r["utterance"]) for r in val_rows])
    yva = np.array([did[r["domain"]] for r in val_rows], dtype=np.int64)

    torch.manual_seed(args.seed)
    clf = nn.Sequential(nn.Linear(len(vocab), 128), nn.ReLU(), nn.Linear(128, n_classes))
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    w = torch.tensor(counts.sum() / (np.maximum(counts, 1) * n_classes), dtype=torch.float32)
    opt = torch.optim.AdamW(clf.parameters(), lr=5e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    for _ in range(args.epochs):
        clf.train()
        opt.zero_grad()
        F.cross_entropy(clf(Xt), yt, weight=w).backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        pred = clf(torch.tensor(Xva)).argmax(-1).numpy()
    bal = _balanced_acc(pred, yva, n_classes)

    verdict = "PASS: 任务领域 NLU 在留出 session 上有效" if bal > 0.85 else "WEAK"
    result = {
        "domains": domains,
        "train_turns": len(train_rows),
        "val_turns": len(val_rows),
        "val_domain_balanced_acc": round(bal, 4),
        "verdict": verdict,
        "note": "学习式领域识别 NLU 用任务语料训练（非合成模板），按 session 切分留出评测。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# FE-LLM 任务领域识别 NLU（任务语料训练）",
        "",
        f"- 判定：**{verdict}**",
        f"- 领域数：{len(domains)}（{', '.join(domains)}）",
        f"- 训练轮 {result['train_turns']} / 留出轮 {result['val_turns']}",
        f"- 留出 session 领域 balanced acc：{result['val_domain_balanced_acc']}",
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
