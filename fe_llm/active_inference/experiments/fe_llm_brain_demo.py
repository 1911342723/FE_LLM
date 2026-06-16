# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/fe_llm_brain_demo.py
=======================================================
**FE-LLM 活大脑 · 统一端到端 demo**：一段对话里串起这颗"脑子"的全部本事，渲染成自包含 HTML
（Linear 风：浅色、纯色、极简）。无需后端，浏览器直接打开。

一轮里展示：
- **知道何时不该答**：不知道→反问，而不是瞎编；
- **多跳推理 + 可溯源思维链**：把多条事实链式组合（"A的经理的工位"），并亮出中间步（CoT trace）；
- **指代消解**：他/她/它 自动回指上文实体（自然录入链）；
- **grounded 可溯源回答**：答案扎根于取回的内容；
- **主动推理 surprise 平复**：问未知→高 surprise→追问→补充→再问→surprise 下降；
- **工作记忆成长**：对话中现场记住的事实逐轮累积。

全程跑**真实 controller**（`controller.respond()`），不是写死的脚本输出。

默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.active_inference.experiments.fe_llm_brain_demo --run
输出：docs/reports/fe_llm_brain_demo.html（+ 同名 .md 实录）
"""

from __future__ import annotations

import argparse
import html
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.active_inference.capcw_chain_memory import CAPCWChainMemory
from fe_llm.active_inference.controller import ActiveInferenceController

OUT_HTML = os.path.join("docs", "reports", "fe_llm_brain_demo.html")
OUT_MD = os.path.join("docs", "reports", "fe_llm_brain_demo.md")

# 脚本会话（text, 这一轮展示的能力）。全程真实 controller 跑。
SCRIPT = [
    ("你好", "寒暄：正常回应，不被记忆/多跳劫持"),
    ("记住张三的经理是李四", "录入事实（in-context 绑定）"),
    ("他的工位是B302", "指代消解：他=李四（自然录入链）"),
    ("张三的经理的工位是多少", "多跳推理 + 可溯源思维链（链式取回链尾）"),
    ("张三的经理的电话是多少", "知道何时不该答：链断（缺李四的电话）→追问（高 surprise）"),
    ("记住李四的电话是139800", "用户补充缺失的那一边（对外行动改变环境）"),
    ("张三的经理的电话是多少", "主动推理：再问→多跳取回，surprise 平复"),
]

ACTION_LABEL = {"answer": "回答", "ask_clarification": "追问", "refuse": "拒答",
                "retrieve": "检索", "update_memory": "记忆"}
# Linear 风浅色：纯色、低饱和、克制。
ACTION_COLOR = {"answer": "#3fa66a", "ask_clarification": "#d99a2b", "refuse": "#d6564a",
                "retrieve": "#5e6ad2", "update_memory": "#8a63d2"}


def _facts(mem: CAPCWChainMemory, sid: str) -> list[tuple[str, str]]:
    """当前会话工作记忆里的事实（key→value 串），供逐轮展示"成长"。"""
    sess = mem._sess(sid)
    rev = sess["sym_rev"]
    return [(rev.get(k, "?"), rev.get(v, "?")) for k, v in sess["pairs"].items()]


def collect_turns(controller: ActiveInferenceController, sid: str) -> list[dict]:
    controller.reset_chain_working_memory(session_id=sid)
    mem = controller.capcw_chain_memory
    turns = []
    for idx, (text, shows) in enumerate(SCRIPT, 1):
        r = controller.respond(text, session_id=sid)
        s = r.incontext_surprise
        turns.append({
            "turn": idx, "input": text, "shows": shows,
            "action": r.selected_action_type.value,
            "reply": r.text,
            "chain": list(r.incontext_chain or []),
            "incontext_surprise": (round(float(s), 3) if isinstance(s, (int, float)) else None),
            "facts": _facts(mem, sid),
        })
    return turns


def _surprise_bar(s: float | None) -> str:
    if s is None:
        return '<span class="muted">—</span>'
    pct = int(min(max(s, 0.0), 1.0) * 100)
    color = "#d6564a" if s >= 0.7 else "#d99a2b" if s >= 0.4 else "#3fa66a"
    return (f'<span class="track"><span class="fill" style="width:{pct}%;background:{color}"></span></span>'
            f'<span class="sval">{s:.3f}</span>')


def _chain(chain: list[str]) -> str:
    if not chain:
        return ""
    steps = '<span class="arrow">→</span>'.join(f'<span class="step">{html.escape(c)}</span>' for c in chain)
    return f'<div class="cot"><span class="cot-label">思维链</span>{steps}</div>'


def _facts_chips(facts: list[tuple[str, str]]) -> str:
    if not facts:
        return '<span class="chip muted">（空）</span>'
    return "".join(f'<span class="chip">{html.escape(k)} <span class="eq">=</span> {html.escape(v)}</span>'
                   for k, v in facts)


def build_html(turns: list[dict]) -> str:
    rows = []
    for t in turns:
        color = ACTION_COLOR.get(t["action"], "#8a8f98")
        label = ACTION_LABEL.get(t["action"], t["action"])
        cot = _chain(t["chain"])
        rows.append(f"""
      <div class="turn">
        <div class="msg user"><div class="bubble">{html.escape(t['input'])}</div></div>
        <div class="shows">{html.escape(t['shows'])}</div>
        <div class="brain">
          <div class="brow">
            <span class="badge" style="background:{color}">{label}</span>
            <span class="why"><span class="why-label">surprise</span>{_surprise_bar(t['incontext_surprise'])}</span>
          </div>
          {cot}
          <div class="wm"><span class="wm-label">工作记忆</span><span class="chips">{_facts_chips(t['facts'])}</span></div>
        </div>
        <div class="msg bot"><div class="bubble bot-bubble">{html.escape(t['reply'])}</div></div>
      </div>""")
    body = "\n".join(rows)
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FE-LLM · 活大脑 Demo</title>
<style>
  :root {{
    --bg:#fbfbfc; --card:#ffffff; --ink:#1c1c28; --sub:#6b6f76; --line:#ececf1;
    --accent:#5e6ad2; --muted:#9a9ea6; --chip:#f4f4f7;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Inter","Segoe UI","Microsoft YaHei",sans-serif;
    -webkit-font-smoothing:antialiased; line-height:1.55; }}
  .wrap {{ max-width:760px; margin:0 auto; padding:56px 24px 96px; }}
  .title {{ font-size:26px; font-weight:680; letter-spacing:-0.02em; margin:0 0 6px; }}
  .lede {{ color:var(--sub); font-size:14.5px; margin:0 0 8px; }}
  .meta {{ color:var(--muted); font-size:12.5px; margin:0 0 40px; }}
  .meta b {{ color:var(--accent); font-weight:600; }}
  .turn {{ margin:0 0 30px; }}
  .msg {{ display:flex; margin:2px 0; }}
  .msg.user {{ justify-content:flex-end; }}
  .bubble {{ max-width:80%; padding:9px 14px; border-radius:13px; font-size:14.5px; }}
  .user .bubble {{ background:var(--accent); color:#fff; border-bottom-right-radius:4px; }}
  .bot-bubble {{ background:var(--card); border:1px solid var(--line); color:var(--ink); border-bottom-left-radius:4px; }}
  .shows {{ text-align:right; color:var(--muted); font-size:11.5px; margin:3px 2px 8px; }}
  .brain {{ background:var(--card); border:1px solid var(--line); border-radius:11px; padding:11px 14px; margin:8px 0; }}
  .brow {{ display:flex; align-items:center; gap:14px; }}
  .badge {{ color:#fff; font-size:11.5px; font-weight:600; padding:2px 10px; border-radius:20px; flex:0 0 auto; }}
  .why {{ display:flex; align-items:center; gap:8px; flex:1; font-size:12px; color:var(--sub); }}
  .why-label {{ color:var(--muted); }}
  .track {{ flex:1; max-width:200px; height:6px; background:#eef0f4; border-radius:3px; overflow:hidden; }}
  .fill {{ display:block; height:100%; border-radius:3px; }}
  .sval {{ color:var(--sub); font-variant-numeric:tabular-nums; }}
  .cot {{ margin-top:10px; display:flex; align-items:center; gap:7px; flex-wrap:wrap; }}
  .cot-label {{ color:var(--muted); font-size:11.5px; }}
  .step {{ background:#eef0fb; color:var(--accent); border:1px solid #dfe2f7; font-size:12px; padding:2px 9px; border-radius:7px; }}
  .arrow {{ color:var(--muted); font-size:12px; }}
  .wm {{ margin-top:10px; display:flex; gap:8px; align-items:baseline; }}
  .wm-label {{ color:var(--muted); font-size:11.5px; flex:0 0 auto; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
  .chip {{ background:var(--chip); border:1px solid var(--line); color:#3c4049; font-size:12px; padding:2px 9px; border-radius:7px; }}
  .chip .eq {{ color:var(--muted); }}
  .muted {{ color:var(--muted); }}
  .foot {{ color:var(--muted); font-size:12px; margin-top:44px; border-top:1px solid var(--line); padding-top:18px; }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="title">FE-LLM · 活大脑</div>
    <p class="lede">一段对话，串起这颗从 0 自建的"脑子"的全部本事——知道何时不该答、多跳推理、指代消解、可溯源回答、主动推理与成长。</p>
    <p class="meta">全程真实 <b>controller.respond()</b>，非写死输出 · 内容寻址工作记忆(CAPCW) · 思维链可溯源 · <b>200</b> 回归测试守护</p>
{body}
    <div class="foot">每轮：<b>用户</b>→（脑内：动作 / surprise / 思维链 / 工作记忆）→<b>回答</b>。surprise 越低越笃定；追问=知道何时不该答；思维链=多跳的中间步可溯源；工作记忆逐轮累积=成长。</div>
  </div>
</body>
</html>
"""


def build_md(turns: list[dict]) -> str:
    lines = ["# FE-LLM · 活大脑 Demo（实录）", "",
             "全程真实 `controller.respond()`。每轮：用户 → 动作/surprise/思维链/工作记忆 → 回答。", "",
             "| # | 用户 | 展示 | 动作 | 思维链 | surprise | 回答 |",
             "|---|---|---|---|---|---:|---|"]
    for t in turns:
        cot = "→".join(t["chain"]) if t["chain"] else "—"
        s = f"{t['incontext_surprise']:.3f}" if t["incontext_surprise"] is not None else "—"
        lines.append(f"| {t['turn']} | {t['input']} | {t['shows']} | {ACTION_LABEL.get(t['action'], t['action'])} "
                     f"| {cot} | {s} | {t['reply']} |")
    return "\n".join(lines) + "\n"


def _train_chain_memory(path: str, seed: int = 0) -> None:
    # demo 用略宽的 ask_threshold=0.65（surprise=1-match，阈值高=更宽容）：让 d=32 下中等置信度的"正确取回"
    # 也回答；未见 key 仍 surprise=1.0→追问，"知道何时不该答"不受影响。这是 demo 呈现校准，非判定阈值。
    mem = CAPCWChainMemory(n_sym=20, d=32, n_slots=8, ask_threshold=0.65, cot=True)
    mem.train_on_chain(max_hops=2, n_pairs=4, n_train=6000, epochs=45, seed=seed)
    mem.save(path)


def run(args: argparse.Namespace) -> dict:
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "brain_chain_wm.pt")
        print("[brain-demo] 训练链式工作记忆…", flush=True)
        _train_chain_memory(ckpt, seed=args.seed)
        controller = ActiveInferenceController(capcw_chain_memory_path=ckpt)
        turns = collect_turns(controller, args.session)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(build_html(turns))
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(build_md(turns))
    for t in turns:
        cot = "→".join(t["chain"]) if t["chain"] else "—"
        print(f"[brain-demo] {t['turn']} 「{t['input']}」 {ACTION_LABEL.get(t['action'], t['action'])} "
              f"思维链={cot} 回答「{t['reply']}」", flush=True)
    print(f"[brain-demo] 已生成 {args.out}（浏览器直接打开）+ {args.out_md}", flush=True)
    return {"turns": len(turns), "out": args.out}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render FE-LLM unified 'live brain' demo to a Linear-style HTML.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--session", default="brain-demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=OUT_HTML)
    parser.add_argument("--out-md", default=OUT_MD)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print("[brain-demo] dry-run：未运行真实 controller。")
        print(f"[brain-demo] 将跑 {len(SCRIPT)} 轮统一活大脑会话并渲染 Linear 风浅色 HTML。")
        print("[brain-demo] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
