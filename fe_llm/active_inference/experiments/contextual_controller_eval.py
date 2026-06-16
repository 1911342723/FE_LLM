# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/contextual_controller_eval.py
=================================================================
把"机制价值首证"落到真实 FE-LLM 闭环：用真正的 ActiveInferenceController，
证明它的 belief（known_slots + pending_clarification）在 headroom 轮次上决定性胜出，
而"无记忆"版本翻车。对齐 controller 既有动作本体（订票域，不碰 weather→retrieve）。

每个 session 三轮：
  A. "帮我订票"          → 期望 ASK_CLARIFICATION（route 未知）
  B. "{出发}到{目的}"     → 期望 ANSWER（提供 route 并满足澄清）
  C. "帮我订票"（重复）   → 期望 ANSWER（route 已记住）  ← headroom 轮

对照：
  stateful  ：同一 session_id 贯穿三轮（belief 跨轮保留）
  memoryless：每轮换新 session_id（无跨轮记忆）

判定：headroom 轮 C 上 stateful ANSWER 准确率显著高于 memoryless。

默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.active_inference.experiments.contextual_controller_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.active_inference.controller import ActiveInferenceController
from fe_llm.active_inference.observation import CITY_SLOT_VALUES
from fe_llm.active_inference.policy import ActionType

REPORT_JSON = os.path.join("docs", "reports", "contextual_controller_headroom.json")
REPORT_MD = os.path.join("docs", "reports", "contextual_controller_headroom.md")


ROUTE_PHRASINGS = ("{a}到{b}", "从{a}去{b}", "{a}飞{b}", "{a}至{b}")
DISTRACTORS = ("你好", "谢谢", "今天有点累")


def make_sessions(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sessions = []
    for _ in range(n):
        a, b = rng.sample(CITY_SLOT_VALUES, 2)
        phrasing = rng.choice(ROUTE_PHRASINGS)
        sessions.append({
            "route_text": phrasing.format(a=a, b=b),   # 改写鲁棒性：多种表达
            "route": f"{a}到{b}",                       # 归一化后的 route 值
            "distractor": rng.choice(DISTRACTORS),       # 干扰轮：检验记忆跨轮持久
        })
    return sessions


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real controller belief headroom on contextual booking task.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--n-sessions", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[ctrl-headroom] dry-run：未运行真实 controller。")
    print(f"[ctrl-headroom] n_sessions={args.n_sessions}；每 session 三轮（订票/提供route/再订票）。")
    print("[ctrl-headroom] 对照 stateful vs memoryless，比 headroom 轮 ANSWER 准确率。")
    print("[ctrl-headroom] 真正运行请显式追加 --run。")


def run(args: argparse.Namespace) -> dict:
    controller = ActiveInferenceController()
    sessions = make_sessions(args.n_sessions, args.seed)

    a_ask = 0                       # 轮A：route 未知应 ASK（sanity）
    stateful_c_answer = 0           # 轮C：stateful 应 ANSWER
    memoryless_c_answer = 0         # 轮C：memoryless 多半 ASK（无记忆）
    for i, s in enumerate(sessions):
        sid = f"stateful-{i}"
        a = controller.respond("帮我订票", session_id=sid)
        controller.respond(s["route_text"], session_id=sid)        # 改写表达提供 route
        controller.respond(s["distractor"], session_id=sid)         # 干扰轮（不应清掉记忆）
        c = controller.respond("帮我订票", session_id=sid)
        if a.selected_action_type == ActionType.ASK_CLARIFICATION:
            a_ask += 1
        if c.selected_action_type == ActionType.ANSWER:
            stateful_c_answer += 1

        # memoryless：每轮独立 session，route 提供后不被后续记住。
        controller.respond("帮我订票", session_id=f"mem-{i}-a")
        controller.respond(s["route_text"], session_id=f"mem-{i}-b")
        cm = controller.respond("帮我订票", session_id=f"mem-{i}-c")
        if cm.selected_action_type == ActionType.ANSWER:
            memoryless_c_answer += 1

    n = len(sessions)
    stateful_acc = stateful_c_answer / max(n, 1)
    memoryless_acc = memoryless_c_answer / max(n, 1)
    delta = stateful_acc - memoryless_acc
    verdict = "PASS: 真实 controller 的 belief 在 headroom 轮决定性胜出" if delta > 0.5 else "FAIL: belief 未在真实 controller 体现 headroom"
    result = {
        "n_sessions": n,
        "round_a_ask_rate": round(a_ask / max(n, 1), 4),
        "stateful_headroom_answer_acc": round(stateful_acc, 4),
        "memoryless_headroom_answer_acc": round(memoryless_acc, 4),
        "delta": round(delta, 4),
        "verdict": verdict,
        "note": "真实 ActiveInferenceController；headroom 轮='提供过 route 后再说帮我订票'。stateful 用 known_slots 记住 route→ANSWER，memoryless 无记忆→ASK。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# FE-LLM 真实 controller 的 belief headroom（订票上下文）",
        "",
        f"- 判定：**{verdict}**",
        f"- session 数：{result['n_sessions']}",
        f"- 轮A（route 未知应追问）ASK 率：{result['round_a_ask_rate']}",
        "",
        "## headroom 轮 C（提供过 route 后再说『帮我订票』）ANSWER 准确率",
        f"- stateful（belief 记住 route）：{result['stateful_headroom_answer_acc']}",
        f"- memoryless（无跨轮记忆）：{result['memoryless_headroom_answer_acc']}",
        f"- delta：{result['delta']}",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
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
