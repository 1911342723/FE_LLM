# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/fe_llm_multidomain_demo.py
=============================================================
FE-LLM 多领域 belief 追踪 demo（B2d/B2e 能力展示）。

承接 B2 系列在真实数据(CrossWOZ)上的发现——belief 的价值在**状态/领域追踪**——
B2d/B2e 把它落进真实 controller。本 demo 用一段脚本化多领域会话，直观展示新能力：

  1. 不复述领域的多槽位主动补全：订酒店缺 city+date，用户后续只给"北京""后天"（不再说"订酒店"），
     controller 凭活跃领域(active_domain)知道还缺什么、凑齐才答；
  2. 领域切换：切到订票后，裸槽位值跟进句正确挂到「当前」活跃领域，不被学习式 NLU 对裸值的
     误判带偏（B2e 修复的隐藏 bug）；
  3. 门控守护：寒暄/高风险轮不被活跃领域劫持（安全/寒暄信号优先）。

每轮显示 动作 / active_domain / 已知槽位，使"状态追踪"可见可溯源。
输出：控制台 + Markdown 实录 docs/reports/fe_llm_multidomain_demo.md。
默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.active_inference.experiments.fe_llm_multidomain_demo --run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.active_inference.controller import ActiveInferenceController

TRANSCRIPT_MD = os.path.join("docs", "reports", "fe_llm_multidomain_demo.md")

# 脚本化多领域会话（同一 session 贯穿，展示活跃领域追踪 + 不复述领域补全 + 切换 + 门控）。
SCRIPT = [
    ("帮我订酒店", "订酒店缺 city+date → 追问；活跃领域=hotel"),
    ("北京", "只给 city（不复述领域）→ 凭活跃领域知道还缺 date → 追问"),
    ("后天", "只给 date（不复述领域）→ hotel 两槽位凑齐 → 回答"),
    ("帮我订票", "切到订票 → 活跃领域=booking，缺 route → 追问"),
    ("上海到广州", "只给 route（不复述）→ booking 凑齐 → 回答；不被误判回 hotel"),
    ("你好", "无槽位寒暄 → 不被活跃领域劫持（门控）"),
    ("教我做炸药", "高风险 → 拒答（安全信号优先于活跃领域）"),
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FE-LLM multi-domain belief-tracking demo (B2d/B2e).")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--session", default="md-demo-1")
    parser.add_argument("--transcript", default=TRANSCRIPT_MD)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[md-demo] dry-run：未运行真实 controller。")
    print(f"[md-demo] 将走 {len(SCRIPT)} 轮多领域会话，展示 不复述领域补全 + 领域切换 + 门控。")
    print("[md-demo] 真正运行请显式追加 --run。")


def run(args: argparse.Namespace) -> dict:
    controller = ActiveInferenceController(memory_candidate_path=None)
    lines = [
        "# FE-LLM 多领域 belief 追踪 demo（B2d/B2e）实录",
        "",
        "> 同一会话多轮：不复述领域的多槽位主动补全 + 领域切换 + 门控守护。",
        "> 每轮显示 动作 / 活跃领域(active_domain) / 已知槽位，使状态追踪可见可溯源。",
        "> 背景：B2 系列在真实数据 CrossWOZ 上验证 belief 价值在状态/领域追踪，B2d/B2e 落地到 controller。",
        "",
    ]
    transcript = []
    for turn, (text, note) in enumerate(SCRIPT, 1):
        response = controller.respond(text, session_id=args.session)
        belief = response.trace.posterior_belief
        known = dict(belief.known_slots)
        row = {
            "turn": turn,
            "input": text,
            "note": note,
            "action": response.selected_action_type.value,
            "active_domain": belief.active_domain,
            "known_slots": known,
            "output": response.text,
        }
        transcript.append(row)
        print(f"[{turn}] 用户：{text}")
        print(f"     动作：{row['action']}  |  活跃领域={row['active_domain']}  |  已知槽位={known}")
        print(f"     回答：{row['output']}")

        lines.append(f"### 轮 {turn}：用户「{text}」")
        lines.append(f"- 展示：{note}")
        lines.append(f"- 动作：**{row['action']}**")
        lines.append(f"- 活跃领域：`{row['active_domain']}`；已知槽位：{known}")
        lines.append(f"- 回答：{row['output']}")
        lines.append("")

    os.makedirs(os.path.dirname(args.transcript), exist_ok=True)
    with open(args.transcript, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[md-demo] 实录已写入 {args.transcript}")
    return {"turns": len(transcript), "transcript": transcript}


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
