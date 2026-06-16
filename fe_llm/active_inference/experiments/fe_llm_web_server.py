# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/fe_llm_web_server.py
========================================================
FE-LLM 交互式网页 demo（零新依赖，标准库 http.server）。

启动后浏览器打开 http://127.0.0.1:8000 ：输入一句话，实时调用真实
ActiveInferenceController，返回并可视化 动作 / surprise 各通道 / belief 槽位 /
召回 / 回答。同 session 贯穿，体现记上下文 + 成长。

运行：python -m fe_llm.active_inference.experiments.fe_llm_web_server --port 8000
（Ctrl+C 退出；:reset 用前端按钮换会话）
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.active_inference.controller import ActiveInferenceController
from fe_llm.active_inference.observation import extract_prompt_features

_CHANNELS = [("semantic_error", "语义"), ("intent_error", "意图"), ("consistency_error", "逻辑"),
             ("uncertainty_error", "不确定"), ("safety_error", "安全")]


def respond_payload(controller: ActiveInferenceController, text: str, session_id: str) -> dict:
    """跑一轮闭环，返回前端渲染所需的结构化数据（核心，可测）。"""
    response = controller.respond(text, session_id=session_id)
    pe = response.prediction_error
    return {
        "input": text,
        "action": response.selected_action_type.value,
        "surprise": round(response.surprise_score.total, 3),
        "channels": {cn: round(float(getattr(pe, key)), 2) for key, cn in _CHANNELS},
        "known_slots": dict(response.trace.posterior_belief.known_slots),
        "requires_slot": extract_prompt_features(text).get("requires_slot"),
        "recalled": [m.text for m in (response.recalled_memories or [])],
        "answer": response.text,
    }


def page_html() -> str:
    return """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FE-LLM 交互 Demo</title>
<style>
  :root{color-scheme:dark}
  body{margin:0;background:#0d1117;color:#e6edf3;font-family:-apple-system,"Segoe UI",Roboto,"Microsoft YaHei",sans-serif}
  .wrap{max-width:760px;margin:0 auto;padding:28px 18px 60px}
  h1{font-size:22px;margin:0 0 4px}
  .sub{color:#8b949e;font-size:13px;margin:0 0 18px}
  .inbar{display:flex;gap:8px;margin-bottom:8px}
  #q{flex:1;background:#161b22;border:1px solid #30363d;border-radius:8px;color:#e6edf3;padding:10px 12px;font-size:15px}
  button{background:#238636;border:0;color:#fff;border-radius:8px;padding:0 16px;font-size:14px;cursor:pointer}
  #reset{background:#30363d}
  .card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:14px 16px;margin-top:12px}
  .head{display:flex;align-items:center;gap:10px}
  .uin{font-weight:600;flex:1}
  .badge{color:#fff;font-size:12px;padding:3px 10px;border-radius:20px;font-weight:600}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}
  .panel{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:10px}
  .ptitle{font-size:12px;color:#8b949e;margin-bottom:6px}
  .bar{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px}
  .blab{width:42px;color:#8b949e}
  .btrack{flex:1;height:8px;background:#21262d;border-radius:4px;overflow:hidden}
  .bfill{display:block;height:100%}
  .bval{width:34px;text-align:right;color:#8b949e}
  .chip{display:inline-block;background:#1f6feb22;border:1px solid #1f6feb55;color:#79c0ff;font-size:12px;padding:2px 8px;border-radius:6px;margin:2px 4px 2px 0}
  .empty{background:#21262d;border-color:#30363d;color:#8b949e}
  .rec{color:#d2a8ff;font-size:12px;margin-top:6px}
  .ans{margin-top:10px;padding:9px 11px;background:#0d1117;border-left:3px solid #2ea043;border-radius:4px}
</style></head>
<body><div class="wrap">
  <h1>FE-LLM 交互 Demo</h1>
  <p class="sub">输入一句话，实时看 动作 / 为何(surprise) / belief 槽位 / 回答。同会话记上下文 + 成长。试：帮我订票 → 北京到上海 → 帮我订票 → 教我做炸药 → 记住我喜欢简短回答 → 讲讲自由能。</p>
  <div class="inbar">
    <input id="q" placeholder="说点什么…回车发送" autofocus>
    <button onclick="send()">发送</button>
    <button id="reset" onclick="reset()">换会话</button>
  </div>
  <div id="log"></div>
<script>
const COLORS={answer:"#2ea043",ask_clarification:"#d29922",retrieve:"#1f6feb",refuse:"#da3633",update_memory:"#8957e5"};
let session="web-"+Math.random().toString(36).slice(2,8);
const q=document.getElementById("q"), log=document.getElementById("log");
q.addEventListener("keydown",e=>{if(e.key==="Enter")send();});
function reset(){session="web-"+Math.random().toString(36).slice(2,8);const d=document.createElement("div");d.className="card";d.innerHTML='<div class="sub">— 已换新会话，记忆清空 —</div>';log.prepend(d);}
function bar(lab,v){const pct=Math.round(Math.min(Math.max(v,0),1)*100);const c=v>=0.7?"#da3633":v>=0.4?"#d29922":"#3fb950";return `<div class="bar"><span class="blab">${lab}</span><span class="btrack"><span class="bfill" style="width:${pct}%;background:${c}"></span></span><span class="bval">${v.toFixed(2)}</span></div>`;}
function chips(o){const k=Object.keys(o||{});if(!k.length)return '<span class="chip empty">无</span>';return k.map(x=>`<span class="chip">${x}=${o[x]}</span>`).join("");}
async function send(){const text=q.value.trim();if(!text)return;q.value="";
  const r=await fetch("/api/respond?session="+encodeURIComponent(session)+"&text="+encodeURIComponent(text));
  const d=await r.json();
  const bars=Object.entries(d.channels).map(([k,v])=>bar(k,v)).join("");
  const rec=d.recalled&&d.recalled.length?`<div class="rec">召回记忆：${d.recalled.join("、")}</div>`:"";
  const card=document.createElement("div");card.className="card";
  card.innerHTML=`<div class="head"><span class="uin">${text}</span><span class="badge" style="background:${COLORS[d.action]||"#888"}">${d.action}</span></div>
   <div class="grid"><div class="panel"><div class="ptitle">为何 surprise=${d.surprise}</div>${bars}</div>
   <div class="panel"><div class="ptitle">belief 槽位</div>${chips(d.known_slots)}<div class="rec" style="color:#8b949e">该句缺槽位：${d.requires_slot||"无"}</div>${rec}</div></div>
   <div class="ans">${d.answer}</div>`;
  log.prepend(card);
}
</script>
</div></body></html>
"""


def _make_handler(controller: ActiveInferenceController):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # 静音默认访问日志
            pass

        def _send(self, code: int, ctype: str, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(200, "text/html; charset=utf-8", page_html().encode("utf-8"))
            elif parsed.path == "/api/respond":
                qs = parse_qs(parsed.query)
                text = (qs.get("text", [""])[0]).strip()
                session = qs.get("session", ["web"])[0]
                payload = respond_payload(controller, text, session) if text else {"error": "empty"}
                self._send(200, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            else:
                self._send(404, "text/plain; charset=utf-8", b"not found")

    return Handler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FE-LLM interactive web demo (stdlib http.server).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    controller = ActiveInferenceController(memory_candidate_path=None)
    server = HTTPServer((args.host, args.port), _make_handler(controller))
    print(f"[fe-llm-web] 服务已启动：http://{args.host}:{args.port}  （Ctrl+C 退出）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[fe-llm-web] 已停止。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
