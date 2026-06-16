# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/memory_distill.py
=====================================================
成长闭环第三层（离线结构成长）：把"经审计、已确认"的记忆回流为训练数据。

对应道易草案第 8 节："经过审计的数据再进入训练或蒸馏"——只有重复且稳定（confirmed）
的记忆才进入离线训练集，一次性候选（candidate）不进。整个过程可审计、可重复。

流程：
  1. 用真实 controller 在隔离 demo 记忆上积累记忆（重复偏好→confirmed；一次性→candidate）；
  2. audit_summary 给出晋升状态；
  3. distill_confirmed 仅取 confirmed → 导出 policy_teacher 兼容的训练样本（含 provenance）；
  4. 写出蒸馏数据集 + 可审计报告。

默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.active_inference.experiments.memory_distill --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.active_inference.controller import ActiveInferenceController

DEMO_MEMORY = os.path.join("docs", "reports", "_memory_distill_demo.jsonl")
DISTILL_OUT = os.path.join("docs", "reports", "memory_distill_dataset.jsonl")
REPORT_JSON = os.path.join("docs", "reports", "memory_distill_report.json")
REPORT_MD = os.path.join("docs", "reports", "memory_distill_report.md")

REPEATED_PREFERENCE = "记住我喜欢简短回答"
ONE_OFF_FACTS = ["记住我住在北京", "记住我的生日是5月"]


def distill_confirmed(audit_rows: list[dict]) -> list[dict]:
    """仅把 confirmed 记忆转成 policy_teacher 兼容训练样本（prompt + action_type + provenance）。

    纯函数，便于单测：candidate（一次性）不进训练集，确保"穷则变"才回流。
    """
    dataset = []
    for row in audit_rows:
        if row.get("status") != "confirmed":
            continue
        dataset.append({
            "prompt": row["text"],
            "action_type": "update_memory",
            "weight": row["count"],
            "confidence": row["confidence"],
            "source": "confirmed_memory",
            "distinct_sessions": row.get("distinct_sessions", 0),
        })
    return dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline growth: distill confirmed memories into retraining data.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--confirm-threshold", type=int, default=2)
    parser.add_argument("--distill-out", default=DISTILL_OUT)
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[mem-distill] dry-run：未运行真实 controller。")
    print(f"[mem-distill] 稳定偏好重复 {args.repeats} 会话→confirmed→进训练集；一次性事实→candidate→不进。")
    print("[mem-distill] 真正运行请显式追加 --run。")


def run(args: argparse.Namespace) -> dict:
    os.makedirs(os.path.dirname(DEMO_MEMORY), exist_ok=True)
    if os.path.exists(DEMO_MEMORY):
        os.remove(DEMO_MEMORY)

    controller = ActiveInferenceController(memory_candidate_path=DEMO_MEMORY)
    for i in range(args.repeats):
        controller.respond(REPEATED_PREFERENCE, session_id=f"distill-pref-{i}")
    for j, fact in enumerate(ONE_OFF_FACTS):
        controller.respond(fact, session_id=f"distill-fact-{j}")

    audit = controller.memory_manager.audit_summary(confirm_threshold=args.confirm_threshold)
    dataset = distill_confirmed(audit)

    os.makedirs(os.path.dirname(args.distill_out), exist_ok=True)
    with open(args.distill_out, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    distilled_texts = {row["prompt"] for row in dataset}
    pref_in = REPEATED_PREFERENCE in distilled_texts
    facts_excluded = all(fact not in distilled_texts for fact in ONE_OFF_FACTS)
    verdict = (
        "PASS: 仅 confirmed 记忆回流训练集，candidate 被正确排除"
        if pref_in and facts_excluded
        else "FAIL: 蒸馏筛选未达预期"
    )
    result = {
        "audit_total": len(audit),
        "distilled_count": len(dataset),
        "distilled_texts": sorted(distilled_texts),
        "verdict": verdict,
        "distill_out": args.distill_out,
        "note": "离线结构成长：只有审计确认（confirmed）的稳定记忆进入再训练集，一次性候选不进（穷则变）。",
    }
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# FE-LLM 成长闭环第三层：confirmed 记忆离线蒸馏回训",
        "",
        f"- 判定：**{verdict}**",
        f"- 审计记忆条目：{result['audit_total']}，蒸馏进训练集：{result['distilled_count']}",
        f"- 进入训练集：{result['distilled_texts']}",
        f"- 训练数据：`{result['distill_out']}`（policy_teacher 兼容，含 provenance）",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if os.path.exists(DEMO_MEMORY):
        os.remove(DEMO_MEMORY)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.run:
        print_dry_run(args)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
