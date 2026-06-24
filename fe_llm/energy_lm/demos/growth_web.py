# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/demos/growth_web.py —— "教学→后天成长"网页 Demo（synapse-only）
=================================================================================
在浏览器里**教** PER 代码模型一条复制规则（synapse-only、冻结 backbone），每点一次"再教"
就在线学几轮，实时看到：

  - 教学 loss 曲线随交互下降（学到了）
  - held-out（没教过的新名词）复制准确率上升（**真泛化**，不是死记）
  - 对照（无关代码）loss（遗忘指示——in-place 会升）
  - 可学突触 S 的变化量 Δ 热图（经验刻进结构记忆）
  - 一条 held-out 实时补全样例

按"重置"回到底座。基于 fe_llm.energy_lm.growth.GrowthLearner（一等能力）。

运行：
    python -m fe_llm.energy_lm.demos.growth_web                 # 127.0.0.1:8002
    python -m fe_llm.energy_lm.demos.growth_web --device cpu
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from fe_llm.energy_lm.growth import GrowthLearner
from fe_llm.energy_lm.models.tokenizer import CharTokenizer
from fe_llm.energy_lm.training.code_train import ckpt_paths, load_any

_LOCK = threading.Lock()
_STATE: dict = {}

NOUNS = ["button", "slider", "panel", "dialog", "menu", "toolbar", "canvas", "label",
         "cursor", "frame", "badge", "spinner", "switch", "drawer", "tooltip", "modal"]
TEACH, HELD = NOUNS[:8], NOUNS[8:]
SKILL = lambda x: (f'def make_{x}():\n    return Widget("', f'{x}")\n')
CONTROL = ("import os\nimport ", "sys\n")   # 无关代码，测遗忘


def _examples(nouns):
    return [SKILL(x) for x in nouns]


def _held_acc(gl):
    ok = 0
    for x in HELD:
        prefix, comp = SKILL(x)
        out = gl.generate(prefix, max_new=len(comp) + 4, temperature=0.0)
        ok += int(out.startswith(comp))
    return ok / len(HELD)


def _synapse_png(gl) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = gl.synapse_delta(-1)
    crop = min(56, d.shape[0])
    dv = d[:crop, :crop]
    vmax = float(np.abs(dv).max()) or 1.0
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    im = ax.imshow(dv, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("突触 S 变化量 Δ", fontsize=9)
    ax.set_xlabel("源 i", fontsize=8); ax.set_ylabel("目标 j", fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=110); plt.close()
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>FE-LLM 教学→成长 Demo</title>
<style>
  :root{ --bg:#0d1117; --panel:#161b22; --line:#30363d; --tx:#c9d1d9; --mut:#8b949e; --acc:#58a6ff; --ok:#7ee787; --bad:#ff7b72; }
  *{box-sizing:border-box;} body{margin:0;background:var(--bg);color:var(--tx);min-height:100vh;
    font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;}
  header{padding:12px 20px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:10px;}
  header .dot{width:9px;height:9px;border-radius:50%;background:#3fb950;box-shadow:0 0 8px #3fb950;}
  header h1{font-size:15px;margin:0;font-weight:600;} header .m{color:var(--mut);font-size:12px;margin-left:auto;}
  .wrap{padding:18px;display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:1100px;margin:0 auto;}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;}
  .card h2{font-size:13px;margin:0 0 10px;color:var(--mut);font-weight:600;}
  .big{font-size:26px;font-weight:700;} .row{display:flex;gap:18px;align-items:baseline;flex-wrap:wrap;}
  .kv{display:flex;flex-direction:column;gap:2px;} .kv .l{color:var(--mut);font-size:11px;}
  .ok{color:var(--ok);} .bad{color:var(--bad);}
  button{padding:9px 18px;border:0;border-radius:8px;background:var(--acc);color:#03101f;font-size:14px;font-weight:600;cursor:pointer;margin-right:8px;}
  button.sec{background:#30363d;color:var(--tx);} button:disabled{opacity:.5;cursor:default;}
  .mono{font-family:"JetBrains Mono","Consolas",monospace;font-size:12.5px;white-space:pre-wrap;
    background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px;min-height:54px;}
  svg{width:100%;height:160px;background:var(--bg);border:1px solid var(--line);border-radius:8px;}
  img{width:100%;border:1px solid var(--line);border-radius:8px;}
  .note{color:var(--mut);font-size:11.5px;line-height:1.5;margin-top:8px;}
</style></head><body>
<header><span class="dot"></span><h1>FE-LLM · 教学 → 后天成长（synapse-only，冻结 backbone）</h1>
<span class="m" id="meta"></span></header>
<div style="padding:12px 20px 0;max-width:1100px;margin:0 auto;">
  <button id="teach">再教 8 轮 ▶</button>
  <button id="reset" class="sec">重置到底座</button>
  <span class="note" id="status" style="margin-left:8px;"></span>
</div>
<div class="wrap">
  <div class="card"><h2>成长曲线：教学/held-out/对照 loss（bits，↓）</h2><svg id="chart" viewBox="0 0 400 160" preserveAspectRatio="none"></svg>
    <div class="note"><span style="color:#7ee787">绿=教过</span> · <span style="color:#58a6ff">蓝=held-out 没教过(泛化)</span> · <span style="color:#9e9e9e">灰=对照无关代码(遗忘则升)</span></div></div>
  <div class="card"><h2>关键指标（学规则 make_&lt;X&gt; → Widget("&lt;X&gt;")）</h2>
    <div class="row">
      <div class="kv"><span class="l">已教轮数</span><span class="big" id="rounds">0</span></div>
      <div class="kv"><span class="l">held-out 复制准确率（泛化）</span><span class="big ok" id="acc">—</span></div>
      <div class="kv"><span class="l">突触参数</span><span class="big" id="params">—</span></div>
    </div>
    <div class="note">held-out 复制率上升=把规则**泛化**到没教过的名词；对照 loss 升=动了共享突触有遗忘代价（诚实）。</div></div>
  <div class="card"><h2>held-out 实时补全（没教过的名词）</h2><div class="mono" id="sample">点「再教」开始…</div></div>
  <div class="card"><h2>可学突触 S 变化量 Δ（经验刻进结构记忆）</h2><img id="syn" alt="synapse delta"/></div>
</div>
<script>
let H={teach:[],held:[],control:[]}, rounds=0;
fetch('/meta').then(r=>r.json()).then(m=>{document.getElementById('meta').textContent=`${m.params_M}M · d${m.dim}×${m.depth} · val bpc ${m.val_bpc}`;});
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function draw(){
  const svg=document.getElementById('chart'); const W=400,Hh=160,pad=18;
  const all=[...H.teach,...H.held,...H.control]; if(!all.length){svg.innerHTML='';return;}
  const mx=Math.max(...all,0.1), n=H.teach.length;
  const X=i=>pad+(n<=1?0:i*(W-2*pad)/(n-1)), Y=v=>Hh-pad-(v/mx)*(Hh-2*pad);
  const line=(arr,c)=>{if(arr.length<1)return'';let d=arr.map((v,i)=>(i?'L':'M')+X(i).toFixed(1)+' '+Y(v).toFixed(1)).join(' ');
    return `<path d="${d}" fill="none" stroke="${c}" stroke-width="2"/>`+arr.map((v,i)=>`<circle cx="${X(i).toFixed(1)}" cy="${Y(v).toFixed(1)}" r="2.2" fill="${c}"/>`).join('');};
  svg.innerHTML=line(H.teach,'#7ee787')+line(H.held,'#58a6ff')+line(H.control,'#9e9e9e');
}
async function teach(){
  const b=document.getElementById('teach'),rb=document.getElementById('reset'); b.disabled=rb.disabled=true;
  document.getElementById('status').textContent='在线学习中…';
  try{
    const r=await fetch('/teach',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rounds:8})});
    const d=await r.json();
    H.teach=H.teach.concat(d.teach_hist); H.held=H.held.concat(d.held_hist); H.control=H.control.concat(d.control_hist);
    rounds=d.rounds_total; draw();
    document.getElementById('rounds').textContent=rounds;
    document.getElementById('acc').textContent=Math.round(d.held_acc*100)+'%';
    document.getElementById('params').textContent=(d.params_k).toFixed(0)+'k';
    document.getElementById('sample').textContent=d.sample;
    document.getElementById('syn').src=d.synapse_png;
    document.getElementById('status').textContent='已学 '+rounds+' 轮';
  }catch(e){document.getElementById('status').textContent='出错：'+e;}
  b.disabled=rb.disabled=false;
}
async function reset(){
  await fetch('/reset',{method:'POST'}); H={teach:[],held:[],control:[]}; rounds=0; draw();
  document.getElementById('rounds').textContent='0'; document.getElementById('acc').textContent='—';
  document.getElementById('sample').textContent='已重置到底座。'; document.getElementById('syn').src='';
  document.getElementById('status').textContent='已重置';
}
document.getElementById('teach').onclick=teach; document.getElementById('reset').onclick=reset;
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/meta":
            self._send(200, json.dumps(_STATE["meta"], ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path == "/reset":
            with _LOCK:
                _STATE["gl"].reset_to_base(); _STATE["rounds"] = 0
            self._send(200, b"{}", "application/json"); return
        if self.path != "/teach":
            self._send(404, b"not found", "text/plain"); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            rounds = max(1, min(40, int(data.get("rounds", 8))))
            with _LOCK:
                gl = _STATE["gl"]
                teach_ex, held_ex, ctrl_ex = _STATE["teach_ex"], _STATE["held_ex"], [CONTROL]
                seed = _STATE["rounds"]   # 不同批次不同种子，回放更稳
                teach_hist, held_hist, control_hist = [], [], []

                def on_round(r, _):
                    teach_hist.append(round(gl.eval_loss(teach_ex), 4))
                    held_hist.append(round(gl.eval_loss(held_ex), 4))
                    control_hist.append(round(gl.eval_loss(ctrl_ex), 4))

                gl.teach(teach_ex, rounds=rounds, steps=4, replay=4, seed=seed, on_round=on_round)
                _STATE["rounds"] += rounds
                held_acc = _held_acc(gl)
                sample_prefix, _ = SKILL(held_ex_noun := HELD[0])
                sample = sample_prefix + gl.generate(sample_prefix, max_new=18, temperature=0.0)
                syn_png = _synapse_png(gl)
                params_k = gl.syn_per_block / 1e3
                total = _STATE["rounds"]
            self._send(200, json.dumps({
                "teach_hist": teach_hist, "held_hist": held_hist, "control_hist": control_hist,
                "held_acc": held_acc, "sample": sample, "synapse_png": syn_png,
                "params_k": params_k, "rounds_total": total,
            }, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
        except Exception as e:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="FE-LLM 教学→成长 demo。")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--device", default="")
    ap.add_argument("--lr", type=float, default=0.02)
    args = ap.parse_args(argv)

    net_path, last_path, tok_path, meta_path = ckpt_paths("per")
    path = net_path if Path(net_path).exists() else last_path
    if not (Path(path).exists() and Path(tok_path).exists()):
        print(f"[grow-web] 找不到 PER 模型/分词器，请先训练。"); return 1
    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if device.startswith("cuda") else None
    tok = CharTokenizer.load(tok_path)
    net = load_any(path, device).to(device)
    if not getattr(net, "use_synapse", False):
        print("[grow-web] 该 PER 模型无可学突触，无法成长。"); return 1
    gl = GrowthLearner(net, tok, device=device, lr=args.lr, amp_dtype=amp)
    meta = {"params_M": round(sum(p.numel() for p in net.parameters()) / 1e6, 2),
            "dim": net.dim, "depth": net.depth, "val_bpc": "-"}
    if Path(meta_path).exists():
        try:
            meta["val_bpc"] = json.loads(Path(meta_path).read_text(encoding="utf-8")).get("val_bpc", "-")
        except Exception:
            pass
    _STATE.update(gl=gl, meta=meta, rounds=0, teach_ex=_examples(TEACH), held_ex=_examples(HELD))
    print(f"[grow-web] 已加载 PER {meta['params_M']}M 设备={device}", flush=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[grow-web] 教学→成长 demo 已启动 → {url}  (Ctrl+C 退出)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[grow-web] 已退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
