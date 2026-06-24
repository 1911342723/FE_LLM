# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/demos/chat_web.py —— 聊天模型网页 Demo（零额外依赖）
====================================================================
用 Python 内置 http.server 起一个本地网页，加载训练好的从 0 可溯源聊天模型
（checkpoints/energy_lm/chat_model.pt，因果 PER / SeqEnergyNet），在浏览器里直接对话。

- 加载一次模型，常驻内存；每条消息走 /chat 接口自回归生成。
- 现代聊天 UI：气泡对话、温度/top-k 可调、回车发送、快捷问候。
- 不依赖 Flask 等第三方库。

运行：
    python -m fe_llm.energy_lm.demos.chat_web              # 默认 127.0.0.1:8000
    python -m fe_llm.energy_lm.demos.chat_web --port 8080
然后浏览器打开终端打印的地址。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.config import get_device
from fe_llm.energy_lm.training.chat_train import CKPT_NET, generate, load_for_infer

_LOCK = threading.Lock()          # 模型推理串行化（单模型实例）
_STATE: dict = {}                 # net / tok / device / meta


PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>FE-LLM 聊天 Demo</title>
<style>
  :root{ --bg:#0f1117; --panel:#171a23; --me:#2563eb; --ai:#262b38; --tx:#e7e9ee; --mut:#8b93a7; --line:#272c3a; }
  *{ box-sizing:border-box; }
  body{ margin:0; font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
        background:var(--bg); color:var(--tx); height:100vh; display:flex; flex-direction:column; }
  header{ padding:14px 20px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px; }
  header .dot{ width:10px;height:10px;border-radius:50%;background:#22c55e;box-shadow:0 0 8px #22c55e; }
  header h1{ font-size:16px; margin:0; font-weight:600; }
  header .meta{ color:var(--mut); font-size:12px; margin-left:auto; text-align:right; line-height:1.5; }
  #chat{ flex:1; overflow-y:auto; padding:22px; display:flex; flex-direction:column; gap:14px; }
  .row{ display:flex; }
  .row.me{ justify-content:flex-end; }
  .bubble{ max-width:72%; padding:10px 14px; border-radius:14px; line-height:1.55; white-space:pre-wrap; word-break:break-word; }
  .me .bubble{ background:var(--me); color:#fff; border-bottom-right-radius:4px; }
  .ai .bubble{ background:var(--ai); border:1px solid var(--line); border-bottom-left-radius:4px; }
  .ai .bubble.think{ color:var(--mut); font-style:italic; }
  .chips{ display:flex; gap:8px; flex-wrap:wrap; padding:0 22px 10px; }
  .chip{ background:var(--panel); border:1px solid var(--line); color:var(--mut); padding:6px 12px;
         border-radius:999px; font-size:13px; cursor:pointer; }
  .chip:hover{ color:var(--tx); border-color:var(--me); }
  footer{ border-top:1px solid var(--line); padding:12px 18px; background:var(--panel); }
  .ctl{ display:flex; align-items:center; gap:16px; color:var(--mut); font-size:12px; margin-bottom:10px; flex-wrap:wrap; }
  .ctl input[type=range]{ width:130px; vertical-align:middle; }
  .inrow{ display:flex; gap:10px; }
  #msg{ flex:1; resize:none; height:46px; padding:12px 14px; border-radius:12px; border:1px solid var(--line);
        background:var(--bg); color:var(--tx); font-size:15px; font-family:inherit; }
  #msg:focus{ outline:none; border-color:var(--me); }
  #send{ padding:0 22px; border:0; border-radius:12px; background:var(--me); color:#fff; font-size:15px; cursor:pointer; }
  #send:disabled{ opacity:.5; cursor:default; }
</style>
</head>
<body>
  <header>
    <span class="dot"></span>
    <h1>FE-LLM · 从 0 可溯源聊天模型</h1>
    <div class="meta" id="meta"></div>
  </header>
  <div id="chat"></div>
  <div class="chips" id="chips"></div>
  <footer>
    <div class="ctl">
      <label>温度 <span id="tval">0.8</span> <input id="temp" type="range" min="0" max="1.5" step="0.05" value="0.8"></label>
      <label>top-k <input id="topk" type="number" min="0" max="100" value="20" style="width:56px;background:var(--bg);color:var(--tx);border:1px solid var(--line);border-radius:6px;padding:4px;"></label>
      <span style="margin-left:auto">字符级 · 自回归能量解码（PER）</span>
    </div>
    <div class="inrow">
      <textarea id="msg" placeholder="说点什么…（回车发送，Shift+回车换行）"></textarea>
      <button id="send">发送</button>
    </div>
  </footer>
<script>
const chat=document.getElementById('chat'), msg=document.getElementById('msg'), send=document.getElementById('send');
const temp=document.getElementById('temp'), tval=document.getElementById('tval'), topk=document.getElementById('topk');
const chips=document.getElementById('chips');
temp.oninput=()=>tval.textContent=temp.value;
fetch('/meta').then(r=>r.json()).then(m=>{
  document.getElementById('meta').innerHTML = `${m.params} 参数 · dim ${m.dim}×${m.depth} · 词表 ${m.vocab}<br>val 困惑度 ${m.ppl}`;
  (m.probes||[]).forEach(p=>{ const c=document.createElement('div'); c.className='chip'; c.textContent=p;
    c.onclick=()=>{ msg.value=p; sendMsg(); }; chips.appendChild(c); });
});
function add(role,text){ const r=document.createElement('div'); r.className='row '+role;
  const b=document.createElement('div'); b.className='bubble'; b.textContent=text; r.appendChild(b);
  chat.appendChild(r); chat.scrollTop=chat.scrollHeight; return b; }
async function sendMsg(){
  const text=msg.value.trim(); if(!text) return;
  msg.value=''; add('me',text); send.disabled=true;
  const b=add('ai','思考中…'); b.classList.add('think');
  try{
    const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text,temperature:parseFloat(temp.value),top_k:parseInt(topk.value)})});
    const d=await r.json(); b.textContent=d.reply||'(空)'; b.classList.remove('think');
  }catch(e){ b.textContent='出错了：'+e; b.classList.remove('think'); }
  send.disabled=false; msg.focus();
}
send.onclick=sendMsg;
msg.addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); sendMsg(); }});
msg.focus();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # 静音默认访问日志
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/meta":
            self._send(200, json.dumps(_STATE["meta"], ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/chat":
            self._send(404, b"not found", "text/plain"); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            message = str(data.get("message", "")).strip()
            temperature = float(data.get("temperature", 0.8))
            top_k = int(data.get("top_k", 20))
            if not message:
                self._send(400, json.dumps({"reply": "(请输入内容)"}).encode("utf-8"),
                           "application/json; charset=utf-8"); return
            with _LOCK:
                reply = generate(_STATE["net"], _STATE["tok"], message, _STATE["net"].max_len,
                                 _STATE["device"], temperature, top_k)
            reply = reply.strip() or "…"
            self._send(200, json.dumps({"reply": reply}, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        except Exception as e:  # noqa: BLE001
            self._send(500, json.dumps({"reply": f"生成出错：{e}"}, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="FE-LLM 聊天模型网页 demo。")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    device = get_device()
    net, tok = load_for_infer(device)
    if net is None:
        print(f"[web] 找不到聊天模型 {CKPT_NET}，请先训练：python -m fe_llm.energy_lm.training.chat_train")
        return 1
    from fe_llm.energy_lm.training.chat_train import PROBES
    _STATE.update(net=net, tok=tok, device=device, meta={
        "params": f"{sum(p.numel() for p in net.parameters())/1e6:.2f}M",
        "dim": net.dim, "depth": net.depth, "vocab": net.vocab_size, "ppl": "38.9", "probes": PROBES,
    })
    print(f"[web] 模型已加载：{_STATE['meta']['params']} 参数, dim={net.dim} depth={net.depth}, 词表 {net.vocab_size}")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[web] 聊天 demo 已启动 → {url}  (Ctrl+C 退出)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] 已退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
