# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/demos/code_web.py —— 代码补全网页 Demo（PER vs Transformer 上下对比）
=======================================================================================
同时加载两个代码模型（本项目 PER `SeqEnergyNet` 与标准 `CharTransformerLM`），在浏览器里
输入一段 Python 前缀，**上面板出 PER 的补全、下面板出 Transformer 的补全**，同解码参数，
肉眼直接对比两种架构。

- 一次加载两个模型常驻内存；每次 /complete 用同一 prompt + 同参数分别跑两边。
- 代码风格 UI：左输入 / 右上下两栏对比；前缀灰、补全绿；温度/top-k/长度可调。
- 热重载（默认开）：两个 checkpoint 任一变了，下次补全前自动加载最新（容错）。
- 不依赖 Flask 等第三方库。

运行：
    python -m fe_llm.energy_lm.demos.code_web                 # 默认 127.0.0.1:8001
    python -m fe_llm.energy_lm.demos.code_web --device cpu    # 不抢 GPU
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from fe_llm.energy_lm.models.tokenizer import CharTokenizer
from fe_llm.energy_lm.training.code_train import ckpt_paths, generate, load_any

_LOCK = threading.Lock()
_STATE: dict = {}

ARCH_LABEL = {"per": "PER（自有 · 因果 PER）", "transformer": "Transformer（标准）"}


def _read_meta(meta_path: str, net) -> dict:
    meta = {"params_M": round(sum(p.numel() for p in net.parameters()) / 1e6, 2),
            "dim": net.dim, "depth": net.depth, "ctx": net.max_len, "vocab": net.vocab_size,
            "val_bpc": "-", "val_ppl": "-", "step": "-"}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            meta.update(val_bpc=m.get("val_bpc", "-"), val_ppl=m.get("val_ppl", "-"), step=m.get("step", "-"))
        except Exception:
            pass
    return meta


def _load_one(arch: str, device: str):
    """加载某架构的最优/最近 checkpoint，返回 model dict 或 None。"""
    net_path, last_path, tok_path, meta_path = ckpt_paths(arch)
    path = net_path if os.path.exists(net_path) else last_path
    if not (os.path.exists(path) and os.path.exists(tok_path)):
        return None
    net = load_any(path, device).to(device)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0
    return {"net": net, "path": path, "meta_path": meta_path, "mtime": mtime,
            "meta": _read_meta(meta_path, net), "label": ARCH_LABEL[arch]}


def _maybe_reload() -> None:
    if not _STATE.get("reload"):
        return
    for arch in ("per", "transformer"):
        m = _STATE.get(arch)
        if not m:
            continue
        try:
            mtime = os.path.getmtime(m["path"])
        except OSError:
            continue
        if mtime <= m["mtime"]:
            continue
        try:
            m["net"] = load_any(m["path"], _STATE["device"]).to(_STATE["device"])
            m["mtime"] = mtime
            m["meta"] = _read_meta(m["meta_path"], m["net"])
            print(f"[web] 热重载 {arch} step={m['meta'].get('step')} val_bpc={m['meta'].get('val_bpc')}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[web] 热重载 {arch} 跳过（{e}）", flush=True)


PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>FE-LLM 代码补全 · PER vs Transformer</title>
<style>
  :root{ --bg:#0d1117; --panel:#161b22; --line:#30363d; --tx:#c9d1d9; --mut:#8b949e;
         --acc:#58a6ff; --gen:#7ee787; --per:#7ee787; --tf:#d2a8ff; }
  *{ box-sizing:border-box; }
  body{ margin:0; background:var(--bg); color:var(--tx); height:100vh; display:flex; flex-direction:column;
        font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif; }
  header{ padding:11px 18px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:10px; }
  header .dot{ width:9px;height:9px;border-radius:50%;background:#3fb950;box-shadow:0 0 8px #3fb950; }
  header h1{ font-size:15px; margin:0; font-weight:600; }
  header .hint{ color:var(--mut); font-size:12px; margin-left:auto; }
  .wrap{ flex:1; display:flex; min-height:0; }
  .col{ display:flex; flex-direction:column; min-width:0; min-height:0; }
  .left{ flex:1; border-right:1px solid var(--line); }
  .right{ flex:1.15; }
  .lbl{ padding:7px 14px; color:var(--mut); font-size:12px; border-bottom:1px solid var(--line);
        display:flex; align-items:center; gap:10px; flex-wrap:wrap; background:var(--panel); }
  .lbl b{ color:var(--tx); font-weight:600; }
  .lbl .m{ color:var(--mut); font-size:11px; }
  .mono{ font-family:"JetBrains Mono","Consolas","SF Mono",Menlo,monospace; font-size:13px; line-height:1.55; }
  #code{ flex:1; width:100%; resize:none; border:0; outline:none; padding:14px;
         background:var(--bg); color:var(--tx); tab-size:4; }
  .pane{ flex:1; display:flex; flex-direction:column; min-height:0; }
  .pane.top{ border-bottom:2px solid var(--line); }
  .out{ flex:1; overflow:auto; padding:14px; white-space:pre-wrap; word-break:break-word; }
  .out .pfx{ color:var(--mut); }
  .out .per{ color:var(--per); }
  .out .tf{ color:var(--tf); }
  .tag{ display:inline-block; width:8px;height:8px;border-radius:2px;margin-right:6px; }
  .tag.per{ background:var(--per); } .tag.tf{ background:var(--tf); }
  .chips{ display:flex; gap:7px; flex-wrap:wrap; padding:9px 14px; border-top:1px solid var(--line); }
  .chip{ background:var(--panel); border:1px solid var(--line); color:var(--mut); padding:5px 9px;
         border-radius:6px; font-size:12px; cursor:pointer; font-family:inherit; }
  .chip:hover{ color:var(--tx); border-color:var(--acc); }
  footer{ border-top:1px solid var(--line); padding:9px 14px; background:var(--panel);
          display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
  footer label{ color:var(--mut); font-size:12px; }
  input[type=range]{ vertical-align:middle; width:110px; }
  input[type=number]{ width:58px; background:var(--bg); color:var(--tx); border:1px solid var(--line);
                      border-radius:5px; padding:4px; }
  #run{ margin-left:auto; padding:8px 20px; border:0; border-radius:7px; background:var(--acc); color:#03101f;
        font-size:14px; font-weight:600; cursor:pointer; }
  #run:disabled{ opacity:.5; cursor:default; }
</style>
</head>
<body>
  <header>
    <span class="dot"></span>
    <h1>FE-LLM 代码补全 · 上下对比</h1>
    <span class="hint">同一前缀，<span style="color:var(--per)">上=PER</span> / <span style="color:var(--tf)">下=Transformer</span>，同解码参数</span>
  </header>
  <div class="wrap">
    <div class="col left">
      <div class="lbl"><b>输入代码前缀</b>（Python）</div>
      <textarea id="code" class="mono" spellcheck="false">def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = </textarea>
      <div class="chips" id="chips"></div>
    </div>
    <div class="col right">
      <div class="pane top">
        <div class="lbl"><span class="tag per"></span><b id="perlbl">PER（自有）</b> <span class="m" id="permeta"></span></div>
        <div id="perout" class="mono out"><span class="pfx">点右下「补全」对比两架构…</span></div>
      </div>
      <div class="pane">
        <div class="lbl"><span class="tag tf"></span><b id="tflbl">Transformer</b> <span class="m" id="tfmeta"></span></div>
        <div id="tfout" class="mono out"></div>
      </div>
    </div>
  </div>
  <footer>
    <label>温度 <span id="tval">0.4</span> <input id="temp" type="range" min="0" max="1.2" step="0.05" value="0.4"></label>
    <label>top-k <input id="topk" type="number" min="0" max="100" value="20"></label>
    <label>长度 <input id="maxnew" type="number" min="20" max="600" value="240"></label>
    <button id="run">补全 ▶</button>
  </footer>
<script>
const codeEl=document.getElementById('code'), run=document.getElementById('run');
const perout=document.getElementById('perout'), tfout=document.getElementById('tfout');
const temp=document.getElementById('temp'), tval=document.getElementById('tval');
const topk=document.getElementById('topk'), maxnew=document.getElementById('maxnew'), chips=document.getElementById('chips');
temp.oninput=()=>tval.textContent=temp.value;
const PRESETS=["def quicksort(arr):\\n","import numpy as np\\n\\n","class Stack:\\n    def __init__(self):\\n        ",
  "def fibonacci(n):\\n    ","def binary_search(nums, target):\\n    ","for i in range(10):\\n    ","try:\\n    "];
fetch('/meta').then(r=>r.json()).then(m=>{
  if(m.per){ document.getElementById('perlbl').textContent=m.per.label;
    document.getElementById('permeta').textContent=`${m.per.params_M}M · d${m.per.dim}×${m.per.depth} · val bpc ${m.per.val_bpc}`; }
  if(m.tf){ document.getElementById('tflbl').textContent=m.tf.label;
    document.getElementById('tfmeta').textContent=`${m.tf.params_M}M · d${m.tf.dim}×${m.tf.depth} · val bpc ${m.tf.val_bpc}`; }
});
PRESETS.forEach(p=>{ const c=document.createElement('div'); c.className='chip'; c.textContent=p.replace(/\\\\n/g,'⏎').slice(0,22);
  c.onclick=()=>{ codeEl.value=p.replace(/\\\\n/g,'\\n'); }; chips.appendChild(c); });
function esc(s){ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function render(el,cls,prompt,comp){ el.innerHTML='<span class="pfx">'+esc(prompt)+'</span><span class="'+cls+'">'+esc(comp)+'</span>'; }
async function complete(){
  const code=codeEl.value; if(!code) return;
  run.disabled=true;
  perout.innerHTML='<span class="pfx">'+esc(code)+'</span><span class="per">▍…</span>';
  tfout.innerHTML='<span class="pfx">'+esc(code)+'</span><span class="tf">▍…</span>';
  try{
    const r=await fetch('/complete',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({code:code,temperature:parseFloat(temp.value),top_k:parseInt(topk.value),max_new:parseInt(maxnew.value)})});
    const d=await r.json();
    render(perout,'per',d.prompt, d.per?d.per.completion:'(无 PER 模型)');
    render(tfout,'tf',d.prompt, d.tf?d.tf.completion:'(无 Transformer 模型)');
  }catch(e){ perout.innerHTML='<span class="pfx">出错：</span>'+esc(''+e); }
  run.disabled=false;
}
run.onclick=complete;
codeEl.addEventListener('keydown',e=>{ if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){ e.preventDefault(); complete(); }});
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
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
            meta = {}
            for k in ("per", "transformer"):
                m = _STATE.get(k)
                if m:
                    key = "per" if k == "per" else "tf"
                    meta[key] = {**m["meta"], "label": m["label"]}
            self._send(200, json.dumps(meta, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/complete":
            self._send(404, b"not found", "text/plain"); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            code = str(data.get("code", ""))
            temperature = float(data.get("temperature", 0.4))
            top_k = int(data.get("top_k", 20))
            max_new = max(1, min(600, int(data.get("max_new", 240))))
            if not code.strip():
                self._send(400, json.dumps({"prompt": code}).encode("utf-8"),
                           "application/json; charset=utf-8"); return
            out = {"prompt": code}
            with _LOCK:
                _maybe_reload()
                tok = _STATE["tok"]; device = _STATE["device"]; amp = _STATE["amp_dtype"]
                for k in ("per", "transformer"):
                    m = _STATE.get(k)
                    if not m:
                        continue
                    comp = generate(m["net"], tok, code, m["net"].max_len, device,
                                    max_new=max_new, temperature=temperature, top_k=top_k,
                                    top_p=0.0, repetition_penalty=1.15, amp_dtype=amp)
                    out["per" if k == "per" else "tf"] = {"completion": comp}
            self._send(200, json.dumps(out, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        except Exception as e:  # noqa: BLE001
            self._send(500, json.dumps({"prompt": "", "per": {"completion": f"生成出错：{e}"}},
                                       ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="FE-LLM 代码补全网页 demo（PER vs Transformer 上下对比）。")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--device", default="", help="留空=自动（CUDA 优先）；cpu=不抢训练显卡")
    ap.add_argument("--no-reload", dest="reload", action="store_false", default=True,
                    help="关闭热重载")
    args = ap.parse_args(argv)

    _, _, tok_path, _ = ckpt_paths("per")
    if not os.path.exists(tok_path):
        print(f"[web] 找不到分词器 {tok_path}，请先训练。")
        return 1
    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.startswith("cuda") else None

    tok = CharTokenizer.load(tok_path)
    per = _load_one("per", device)
    tf = _load_one("transformer", device)
    if per is None and tf is None:
        print("[web] 找不到任何代码模型，请先训练。")
        return 1
    _STATE.update(tok=tok, device=device, amp_dtype=amp_dtype, reload=args.reload, per=per, transformer=tf)

    for tagk, m in (("PER", per), ("Transformer", tf)):
        if m:
            print(f"[web] 已加载 {tagk}: {m['meta']['params_M']}M d{m['net'].dim}×{m['net'].depth} "
                  f"step={m['meta'].get('step')} val_bpc={m['meta'].get('val_bpc')} ({os.path.basename(m['path'])})", flush=True)
        else:
            print(f"[web] {tagk} 未找到（该栏将显示占位）", flush=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[web] 对比 demo 已启动 → {url}  设备={device} 热重载={'开' if args.reload else '关'}  (Ctrl+C 退出)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] 已退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
