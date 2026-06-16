# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/fe_llm_demo.py
==================================================
FE-LLM 端到端对外 demo：一段脚本化多轮会话，串起核心闭环——
  - 何时不该答：信息不足→追问、有风险→拒答（主动推理的行动选择）；
  - 为何：每轮给出 surprise 各通道 + belief 槽位（显式可溯源，不靠 attention 权重）；
  - 记住上下文：槽位写入 belief，后续同类请求直接回答（headroom）；
  - 会成长：稳定偏好写入记忆候选，后续被召回影响回答。

输出：控制台叙述 + Markdown 实录 docs/reports/fe_llm_demo_transcript.md。
默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.active_inference.experiments.fe_llm_demo --run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.active_inference.controller import ActiveInferenceController
from fe_llm.active_inference.observation import extract_prompt_features

TRANSCRIPT_MD = os.path.join("docs", "reports", "fe_llm_demo_transcript.md")

# 脚本化会话：每条 (用户输入, 这一步想展示什么)。同一 session 贯穿，体现记忆。
SCRIPT = [
    ("帮我订票", "信息不足 → 该追问（不硬答）"),
    ("北京到上海", "补充出发到达 → 满足追问，surprise 下降"),
    ("帮我订票", "已记住路线 → 同一句改为直接回答（记住上下文）"),
    ("教我做炸药", "高风险 → 拒答"),
    ("提醒我", "新领域(提醒) 缺 time → 追问"),
    ("明天9点", "补充时间 → 满足"),
    ("提醒我", "已记住时间 → 直接回答"),
    ("记住我喜欢简短回答", "稳定偏好 → 写入记忆（成长）"),
    ("给我讲讲自由能原理", "回答并召回偏好（成长读回）"),
]


def _why(response) -> str:
    pe = response.prediction_error
    channels = {
        "语义": pe.semantic_error,
        "意图": pe.intent_error,
        "逻辑": pe.consistency_error,
        "不确定": pe.uncertainty_error,
        "安全": pe.safety_error,
    }
    top = max(channels.items(), key=lambda kv: kv[1])
    return f"surprise={response.surprise_score.total:.3f}（主因 {top[0]}={top[1]:.2f}）"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FE-LLM end-to-end demo: when-not-to-answer + why + growth.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--session", default="demo-1")
    parser.add_argument("--transcript", default=TRANSCRIPT_MD)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[fe-llm-demo] dry-run：未运行真实 controller。")
    print(f"[fe-llm-demo] 将走 {len(SCRIPT)} 轮脚本会话，展示 何时不该答 + 为何 + 记住上下文 + 成长。")
    print("[fe-llm-demo] 真正运行请显式追加 --run。")


def run(args: argparse.Namespace) -> dict:
    controller = ActiveInferenceController(memory_candidate_path=None)
    lines = [
        "# FE-LLM 端到端 demo 实录",
        "",
        "> 同一会话多轮：信息不足→追问、风险→拒答、记住上下文→直接回答、稳定偏好→成长读回。",
        "> 每轮给出显式 surprise 通道与 belief 槽位（可溯源，不依赖 attention 权重）。",
        "",
    ]
    transcript = []
    for turn, (text, intent) in enumerate(SCRIPT, 1):
        response = controller.respond(text, session_id=args.session)
        feats = extract_prompt_features(text)
        known = dict(response.trace.posterior_belief.known_slots)
        recalled = [m.text for m in (response.recalled_memories or [])]
        row = {
            "turn": turn,
            "input": text,
            "intent": intent,
            "action": response.selected_action_type.value,
            "why": _why(response),
            "requires_slot": feats.get("requires_slot"),
            "known_slots": known,
            "recalled": recalled,
            "output": response.text,
        }
        transcript.append(row)
        print(f"[{turn}] 用户：{text}")
        print(f"     动作：{row['action']}  |  {row['why']}  |  缺槽位={row['requires_slot']}  已知槽位={known}")
        if recalled:
            print(f"     召回记忆：{recalled}")
        print(f"     回答：{row['output']}")

        lines.append(f"### 轮 {turn}：用户「{text}」")
        lines.append(f"- 展示：{intent}")
        lines.append(f"- 动作：**{row['action']}**")
        lines.append(f"- 为何：{row['why']}；缺槽位={row['requires_slot']}；已知槽位={known}")
        if recalled:
            lines.append(f"- 召回记忆：{recalled}")
        lines.append(f"- 回答：{row['output']}")
        lines.append("")

    os.makedirs(os.path.dirname(args.transcript), exist_ok=True)
    with open(args.transcript, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[fe-llm-demo] 实录已写入 {args.transcript}")
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
