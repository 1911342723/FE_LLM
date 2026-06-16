# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/fe_llm_demo_web.py
======================================================
把 FE-LLM 端到端 demo 渲染成自包含 HTML 可视化（无需后端，浏览器直接打开）。

每轮一张卡片：用户输入 / 动作（色标徽章）/ surprise 各通道条形 / belief 槽位 chips /
召回记忆 / 回答。直观展示"何时不该答 + 为何 + 记住上下文 + 成长"。

默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.active_inference.experiments.fe_llm_demo_web --run
输出：docs/reports/fe_llm_demo.html
"""

from __future__ import annotations

import argparse
import html
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.active_inference.controller import ActiveInferenceController
from fe_llm.active_inference.experiments.fe_llm_demo import SCRIPT
from fe_llm.active_inference.observation import extract_prompt_features

OUT_HTML = os.path.join("docs", "reports", "fe_llm_demo.html")

ACTION_COLORS = {
    "answer": "#2ea043",
    "ask_clarification": "#d29922",
    "retrieve": "#1f6feb",
    "refuse": "#da3633",
    "update_memory": "#8957e5",
}
CHANNELS = [("semantic_error", "语义"), ("intent_error", "意图"), ("consistency_error", "逻辑"),
            ("uncertainty_error", "不确定"), ("safety_error", "安全")]


def collect_turns(controller: ActiveInferenceController, session_id: str) -> list[dict]:
    turns = []
    for idx, (text, intent) in enumerate(SCRIPT, 1):
        response = controller.respond(text, session_id=session_id)
        pe = response.prediction_error
        turns.append({
            "turn": idx,
            "input": text,
            "intent": intent,
            "action": response.selected_action_type.value,
            "surprise": round(response.surprise_score.total, 3),
            "channels": {cn: round(float(getattr(pe, key)), 2) for key, cn in CHANNELS},
            "known_slots": dict(response.trace.posterior_belief.known_slots),
            "requires_slot": extract_prompt_features(text).get("requires_slot"),
            "recalled": [m.text for m in (response.recalled_memories or [])],
            "output": response.text,
        })
    return turns


def _bar(label: str, value: float) -> str:
    pct = int(min(max(value, 0.0), 1.0) * 100)
    color = "#da3633" if value >= 0.7 else "#d29922" if value >= 0.4 else "#3fb950"
    return (
        f'<div class="bar-row"><span class="bar-label">{label}</span>'
        f'<span class="bar-track"><span class="bar-fill" style="width:{pct}%;background:{color}"></span></span>'
        f'<span class="bar-val">{value:.2f}</span></div>'
    )


def _chips(slots: dict) -> str:
    if not slots:
        return '<span class="chip chip-empty">无</span>'
    return "".join(f'<span class="chip">{html.escape(k)}={html.escape(str(v))}</span>' for k, v in slots.items())


def build_html(turns: list[dict]) -> str:
    cards = []
    for t in turns:
        color = ACTION_COLORS.get(t["action"], "#888")
        bars = "".join(_bar(cn, val) for cn, val in t["channels"].items())
        recalled = ""
        if t["recalled"]:
            items = "、".join(html.escape(r) for r in t["recalled"])
            recalled = f'<div class="recalled">召回记忆：{items}</div>'
        req = t["requires_slot"] or "无"
        cards.append(f"""
    <div class="card">
      <div class="card-head">
        <span class="turn-no">{t['turn']}</span>
        <span class="user-input">{html.escape(t['input'])}</span>
        <span class="action-badge" style="background:{color}">{t['action']}</span>
      </div>
      <div class="intent-note">展示：{html.escape(t['intent'])}</div>
      <div class="grid">
        <div class="panel">
          <div class="panel-title">为何（surprise={t['surprise']}）</div>
          {bars}
        </div>
        <div class="panel">
          <div class="panel-title">belief 槽位</div>
          <div class="chips">{_chips(t['known_slots'])}</div>
          <div class="req">该句缺槽位：{html.escape(str(req))}</div>
          {recalled}
        </div>
      </div>
      <div class="answer"><span class="answer-label">回答</span>{html.escape(t['output'])}</div>
    </div>""")
    body = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FE-LLM 端到端 Demo</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:#0d1117; color:#e6edf3; font-family:-apple-system,"Segoe UI",Roboto,"Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:920px; margin:0 auto; padding:32px 20px 64px; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  .sub {{ color:#8b949e; margin:0 0 24px; font-size:14px; line-height:1.6; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:12px; padding:16px 18px; margin-bottom:16px; }}
  .card-head {{ display:flex; align-items:center; gap:12px; }}
  .turn-no {{ width:26px; height:26px; border-radius:50%; background:#21262d; color:#8b949e; display:flex; align-items:center; justify-content:center; font-size:13px; flex:0 0 auto; }}
  .user-input {{ font-size:17px; font-weight:600; flex:1; }}
  .action-badge {{ color:#fff; font-size:12px; padding:3px 10px; border-radius:20px; font-weight:600; }}
  .intent-note {{ color:#8b949e; font-size:12px; margin:6px 0 12px 38px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  .panel {{ background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:10px 12px; }}
  .panel-title {{ font-size:12px; color:#8b949e; margin-bottom:8px; }}
  .bar-row {{ display:flex; align-items:center; gap:8px; margin:4px 0; font-size:12px; }}
  .bar-label {{ width:42px; color:#8b949e; flex:0 0 auto; }}
  .bar-track {{ flex:1; height:8px; background:#21262d; border-radius:4px; overflow:hidden; }}
  .bar-fill {{ display:block; height:100%; border-radius:4px; }}
  .bar-val {{ width:34px; text-align:right; color:#8b949e; flex:0 0 auto; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
  .chip {{ background:#1f6feb22; border:1px solid #1f6feb55; color:#79c0ff; font-size:12px; padding:2px 8px; border-radius:6px; }}
  .chip-empty {{ background:#21262d; border-color:#30363d; color:#8b949e; }}
  .req {{ color:#8b949e; font-size:12px; margin-top:8px; }}
  .recalled {{ color:#d2a8ff; font-size:12px; margin-top:8px; }}
  .answer {{ margin-top:12px; padding:10px 12px; background:#0d1117; border-left:3px solid #2ea043; border-radius:4px; font-size:15px; }}
  .answer-label {{ color:#8b949e; font-size:11px; margin-right:8px; }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>FE-LLM 端到端 Demo</h1>
    <p class="sub">同一会话多轮：信息不足→追问、风险→拒答、记住上下文→直接回答、稳定偏好→成长读回。<br>
    每轮给出显式 surprise 通道与 belief 槽位（可溯源，不依赖 attention 权重）。</p>
{body}
  </div>
</body>
</html>
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render FE-LLM end-to-end demo to a self-contained HTML.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--session", default="web-demo")
    parser.add_argument("--out", default=OUT_HTML)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[fe-llm-web] dry-run：未运行真实 controller。")
    print(f"[fe-llm-web] 将渲染 {len(SCRIPT)} 轮会话为自包含 HTML（动作色标/surprise 条/槽位 chips）。")
    print("[fe-llm-web] 真正运行请显式追加 --run。")


def run(args: argparse.Namespace) -> dict:
    controller = ActiveInferenceController(memory_candidate_path=None)
    turns = collect_turns(controller, args.session)
    page = build_html(turns)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[fe-llm-web] 已生成 {args.out}（{len(turns)} 轮，浏览器直接打开）")
    return {"turns": len(turns), "out": args.out}


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print_dry_run(args)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
