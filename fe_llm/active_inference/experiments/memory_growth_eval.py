# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/memory_growth_eval.py
=========================================================
成长闭环的可审计离线评估：短期 belief → 长期 memory 候选 → 晋升 confirmed。

对应道易草案"穷则变 + 三层成长"：
  - 单次出现 = candidate（短期/待验证）；
  - 重复且稳定出现 = confirmed（可进入长期记忆 / 离线再训练）。

本脚本用真实 ActiveInferenceController（隔离的 demo 记忆文件，可重复运行）：
  1. 跨多会话重复表达同一稳定偏好 → 应晋升 confirmed（高 confidence）；
  2. 一次性事实 → 维持 candidate（低 confidence）；
  3. 输出可审计报告（每条记忆的次数/置信/状态/来源）。

默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.active_inference.experiments.memory_growth_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.active_inference.controller import ActiveInferenceController

DEMO_MEMORY = os.path.join("docs", "reports", "_memory_growth_demo.jsonl")
REPORT_JSON = os.path.join("docs", "reports", "memory_growth_audit.json")
REPORT_MD = os.path.join("docs", "reports", "memory_growth_audit.md")

# 跨多会话重复的稳定偏好（应晋升 confirmed）。
REPEATED_PREFERENCE = "记住我喜欢简短回答"
# 一次性事实（应维持 candidate）。
ONE_OFF_FACTS = ["记住我住在北京", "记住我的生日是5月"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditable memory-growth evaluation: candidate -> confirmed promotion.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repeats", type=int, default=3, help="重复表达稳定偏好的会话数")
    parser.add_argument("--confirm-threshold", type=int, default=2)
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[mem-growth] dry-run：未运行真实 controller。")
    print(f"[mem-growth] 稳定偏好重复 {args.repeats} 会话应→confirmed；一次性事实→candidate。")
    print("[mem-growth] 真正运行请显式追加 --run。")


def run(args: argparse.Namespace) -> dict:
    os.makedirs(os.path.dirname(DEMO_MEMORY), exist_ok=True)
    if os.path.exists(DEMO_MEMORY):
        os.remove(DEMO_MEMORY)  # 隔离 demo 文件，保证可重复

    controller = ActiveInferenceController(memory_candidate_path=DEMO_MEMORY)
    # 1) 跨多会话重复表达稳定偏好。
    for i in range(args.repeats):
        controller.respond(REPEATED_PREFERENCE, session_id=f"growth-pref-{i}")
    # 2) 一次性事实，各表达一次。
    for j, fact in enumerate(ONE_OFF_FACTS):
        controller.respond(fact, session_id=f"growth-fact-{j}")

    summary = controller.memory_manager.audit_summary(confirm_threshold=args.confirm_threshold)
    pref_row = next((row for row in summary if row["text"] == REPEATED_PREFERENCE), None)
    confirmed = [row for row in summary if row["status"] == "confirmed"]
    candidates = [row for row in summary if row["status"] == "candidate"]

    pref_confirmed = bool(pref_row and pref_row["status"] == "confirmed")
    facts_are_candidates = all(
        any(row["text"] == fact and row["status"] == "candidate" for row in summary) for fact in ONE_OFF_FACTS
    )
    verdict = (
        "PASS: 重复偏好晋升 confirmed，一次性事实维持 candidate"
        if pref_confirmed and facts_are_candidates
        else "FAIL: 成长晋升未按预期"
    )
    result = {
        "repeats": args.repeats,
        "confirm_threshold": args.confirm_threshold,
        "total_memories": len(summary),
        "confirmed_count": len(confirmed),
        "candidate_count": len(candidates),
        "repeated_preference": pref_row,
        "verdict": verdict,
        "audit": summary,
        "note": "短期 belief→长期 memory：单次=candidate，重复稳定=confirmed（可进入离线再训练）。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# FE-LLM 成长闭环可审计评估（candidate → confirmed）",
        "",
        f"- 判定：**{verdict}**",
        f"- 记忆条目：{result['total_memories']}（confirmed {result['confirmed_count']} / candidate {result['candidate_count']}）",
        "",
        "## 审计明细",
        "",
        "| 文本 | 次数 | 会话数 | 置信 | 状态 |",
        "|---|---|---|---|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['text']} | {row['count']} | {row['distinct_sessions']} | {row['confidence']} | {row['status']} |"
        )
    lines.append("")
    lines.append(f"- 说明：{result['note']}")
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if os.path.exists(DEMO_MEMORY):
        os.remove(DEMO_MEMORY)  # 清理 demo 文件，不污染真实记忆
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
