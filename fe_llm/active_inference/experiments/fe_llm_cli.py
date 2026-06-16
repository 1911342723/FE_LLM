# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/fe_llm_cli.py
=================================================
FE-LLM 交互式 CLI：逐行输入一句话，实时显示
  - 动作（answer / ask_clarification / retrieve / refuse / update_memory）
  - 为何（surprise 各通道主因 + belief 已知槽位 + 该句仍缺的槽位）
  - 回答文本
同一会话贯穿，体现"记住上下文 + 成长"。

用法：
  交互：python -m fe_llm.active_inference.experiments.fe_llm_cli
        （输入 :reset 换新会话，:quit/exit 退出）
  脚本：python -m fe_llm.active_inference.experiments.fe_llm_cli --inputs "帮我订票||北京到上海||帮我订票"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.active_inference.controller import ActiveInferenceController
from fe_llm.active_inference.observation import extract_prompt_features

_CHANNELS = ("semantic_error", "intent_error", "consistency_error", "uncertainty_error", "safety_error")
_CHANNEL_CN = {
    "semantic_error": "语义",
    "intent_error": "意图",
    "consistency_error": "逻辑",
    "uncertainty_error": "不确定",
    "safety_error": "安全",
}


def format_turn(controller: ActiveInferenceController, text: str, session_id: str) -> str:
    """对一句输入跑闭环并返回可读的多行结果（核心，可测）。"""
    response = controller.respond(text, session_id=session_id)
    pe = response.prediction_error
    channels = {name: getattr(pe, name) for name in _CHANNELS}
    top = max(channels.items(), key=lambda kv: kv[1])
    known = dict(response.trace.posterior_belief.known_slots)
    requires = extract_prompt_features(text).get("requires_slot")
    recalled = [m.text for m in (response.recalled_memories or [])]
    lines = [
        f"  动作 : {response.selected_action_type.value}",
        f"  为何 : surprise={response.surprise_score.total:.3f}（主因 {_CHANNEL_CN[top[0]]}={top[1]:.2f}）"
        f" | 已知槽位={known or '无'} | 该句缺槽位={requires or '无'}",
    ]
    if recalled:
        lines.append(f"  召回 : {recalled}")
    lines.append(f"  回答 : {response.text}")
    return "\n".join(lines)


def run_inputs(inputs: list[str], session_id: str) -> list[str]:
    """脚本化运行一串输入，返回每句的格式化结果（用于测试/批处理）。"""
    controller = ActiveInferenceController(memory_candidate_path=None)
    out = []
    for text in inputs:
        text = text.strip()
        if not text:
            continue
        out.append(f"> {text}\n" + format_turn(controller, text, session_id))
    return out


def _interactive(session_id: str) -> None:
    controller = ActiveInferenceController(memory_candidate_path=None)
    print("FE-LLM 交互 CLI。输入一句话回车；:reset 换会话，exit/:quit 退出。")
    current = session_id
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text in {"exit", ":quit", ":q"}:
            break
        if text == ":reset":
            current = current + "+"
            print("（已换新会话，记忆清空）")
            continue
        if not text:
            continue
        print(format_turn(controller, text, current))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FE-LLM interactive CLI: action + why + answer per utterance.")
    parser.add_argument("--inputs", default="", help="用 || 分隔的脚本化输入；为空则进入交互模式")
    parser.add_argument("--session", default="cli-1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.inputs:
        for block in run_inputs(args.inputs.split("||"), args.session):
            print(block)
        return 0
    _interactive(args.session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
