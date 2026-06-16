# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/train_context_policy.py
===========================================================
用教师任务型语料训练 ContextAwarePolicy，并在**留出 session**（按会话切分，防泄漏）上
端到端逐轮评测：每轮用该轮 belief 预测动作，量化任务型多轮的动作选择质量。

判定：留出 session 的动作 balanced accuracy 高，且歧义子集（同句多动作）显著高于
"只看句子"的盲基线（belief 真有用）。
默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.active_inference.experiments.train_context_policy --run
输出：checkpoints/active_inference/context_policy.pt + 报告
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

from fe_llm.active_inference.context_policy import ACTION_ID, ContextAwarePolicy
from fe_llm.active_inference.policy import ActionType

CORPUS = os.path.join("data", "dialogue", "teacher_task_oriented.jsonl")
CKPT = os.path.join("checkpoints", "active_inference", "context_policy.pt")
REPORT_JSON = os.path.join("docs", "reports", "context_policy_train.json")
REPORT_MD = os.path.join("docs", "reports", "context_policy_train.md")


def _balanced_acc(pred: list[str], gold: list[str]) -> float:
    n = len(ActionType)
    p = np.array([ACTION_ID[x] for x in pred])
    g = np.array([ACTION_ID[x] for x in gold])
    recalls = []
    for c in range(n):
        m = g == c
        if m.any():
            recalls.append(float((p[m] == g[m]).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train + eval context-aware policy on teacher task corpus (session split).")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--corpus", default=CORPUS)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.25)
    parser.add_argument("--ckpt", default=CKPT)
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[ctx-policy] dry-run：未训练。")
    print(f"[ctx-policy] 训练 ContextAwarePolicy（{args.corpus}），按 session 切分留出端到端评测。")
    print("[ctx-policy] 真正运行请显式追加 --run。")


def run(args: argparse.Namespace) -> dict:
    rows = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # 按 session 切分，防止同会话泄漏
    sessions = defaultdict(list)
    for r in rows:
        sessions[r["session_id"]].append(r)
    sids = sorted(sessions)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(sids)
    cut = int(len(sids) * (1 - args.val_frac))
    train_sids, val_sids = set(sids[:cut]), set(sids[cut:])
    train_rows = [r for s in train_sids for r in sessions[s]]
    val_rows = [r for s in val_sids for r in sessions[s]]

    policy = ContextAwarePolicy().fit(
        [r["utterance"] for r in train_rows],
        [r["known_slots"] for r in train_rows],
        [r["action"] for r in train_rows],
        epochs=args.epochs, seed=args.seed,
    )
    policy.save(args.ckpt)

    # 歧义集合：训练全集里同句对应多动作的句子
    text_actions = defaultdict(set)
    for r in rows:
        text_actions[r["utterance"]].add(r["action"])

    # 端到端逐轮评测（留出 session）
    gold = [r["action"] for r in val_rows]
    pred_ctx = [policy.predict(r["utterance"], r["known_slots"]) for r in val_rows]
    pred_blind = [policy.predict(r["utterance"], {}) for r in val_rows]  # 同模型但 belief 清空=盲
    amb_idx = [i for i, r in enumerate(val_rows) if len(text_actions[r["utterance"]]) > 1]

    overall_ctx = _balanced_acc(pred_ctx, gold)
    amb_ctx = _balanced_acc([pred_ctx[i] for i in amb_idx], [gold[i] for i in amb_idx]) if amb_idx else 0.0
    amb_blind = _balanced_acc([pred_blind[i] for i in amb_idx], [gold[i] for i in amb_idx]) if amb_idx else 0.0

    delta = amb_ctx - amb_blind
    verdict = "PASS: 学习式 context-aware policy 在留出 session 上 belief 强有效" if delta > 0.2 and overall_ctx > 0.85 else "WEAK"
    result = {
        "train_turns": len(train_rows),
        "val_turns": len(val_rows),
        "val_sessions": len(val_sids),
        "overall_balanced_acc": round(overall_ctx, 4),
        "ambiguous_subset": {
            "context_aware": round(amb_ctx, 4),
            "belief_cleared_blind": round(amb_blind, 4),
            "delta": round(delta, 4),
        },
        "verdict": verdict,
        "ckpt": args.ckpt,
        "note": "按 session 切分防泄漏；端到端逐轮用该轮 belief 预测动作。belief_cleared_blind=同模型清空 belief 的盲对照。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# FE-LLM 上下文感知 policy 训练 + 留出 session 端到端评测",
        "",
        f"- 判定：**{verdict}**",
        f"- 训练轮 {result['train_turns']} / 留出轮 {result['val_turns']}（{result['val_sessions']} 个留出 session）",
        f"- 留出总体 balanced acc：{result['overall_balanced_acc']}",
        "",
        "## 歧义子集（同句多动作）",
        f"- 上下文感知：{result['ambiguous_subset']['context_aware']}",
        f"- 清空 belief 盲对照：{result['ambiguous_subset']['belief_cleared_blind']}",
        f"- delta：{result['ambiguous_subset']['delta']}",
        "",
        f"- checkpoint：`{result['ckpt']}`",
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
