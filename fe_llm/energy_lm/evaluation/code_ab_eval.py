# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/evaluation/code_ab_eval.py —— PER vs Transformer 代码模型同口径对照
====================================================================================
两个模型（本项目 PER `SeqEnergyNet` 与标准 `CharTransformerLM`）在**同参数量预算、同数据、
同 token 量（max_steps 对齐）、同超参**下各训一版后，本脚本做**同口径**对照并出报告：

    - 同一验证集（同 corpus + 同 seed + 同 ctx → 逐块相同）上重算 held-out **bpc / ppl**；
      bpc 单位无关，跨架构直接可比。
    - 同一组 prompt、同解码参数下生成代码，用 `ast.parse` 统计**语法合法率**（结构质量）。
    - 各贴几段样例，写 docs/reports/code_per_vs_transformer.md(.json)。

运行（两版都训完后）：
    python -m fe_llm.energy_lm.evaluation.code_ab_eval
    python -m fe_llm.energy_lm.evaluation.code_ab_eval --temperature 0.4 --top-p 0.9
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from fe_llm.config import get_device
from fe_llm.energy_lm.training.code_train import (
    CORPUS, build_tokenizer, ckpt_paths, encode_corpus, eval_bpc, generate,
    load_any, make_chunks, syntactic_valid,
)

REPORT_MD = os.path.join("docs", "reports", "code_per_vs_transformer.md")
REPORT_JSON = os.path.join("docs", "reports", "code_per_vs_transformer.json")

# 评语法合法率用的一组 Python 起手式（比训练探针更全，降低噪声）
PROMPTS = [
    "def quicksort(arr):\n",
    "def fibonacci(n):\n    ",
    "import numpy as np\n\n",
    "class Stack:\n",
    "class Node:\n    def __init__(self, value):\n        ",
    "for i in range(10):\n    ",
    "while True:\n    ",
    "try:\n    ",
    "with open(path) as f:\n    ",
    "if __name__ == \"__main__\":\n",
    "def main():\n    ",
    "@property\n    def ",
    "return [x for x in ",
    "def __init__(self):\n        self.",
]


def build_val(args):
    """复现训练时的验证集（同 corpus/min_freq/ctx/seed/val_frac → 逐块一致）。"""
    with open(args.corpus, "r", encoding="utf-8") as f:
        text = f.read()
    if args.max_train_mb > 0:
        text = text[: int(args.max_train_mb * 1_000_000)]
    tok = build_tokenizer(text, args.min_char_freq)
    ids = encode_corpus(text, tok)
    chunks = make_chunks(ids, args.ctx)
    rng = np.random.default_rng(args.seed)
    chunks = chunks[rng.permutation(len(chunks))]
    n_val = max(1, int(len(chunks) * args.val_frac))
    return tok, chunks[:n_val]


def eval_arch(arch, tok, val_chunks, device, amp_dtype, args):
    net_path, last_path, _, meta_path = ckpt_paths(arch)
    path = net_path if os.path.exists(net_path) else last_path
    if not os.path.exists(path):
        print(f"[ab] 跳过 {arch}：找不到 {net_path}")
        return None
    net = load_any(path, device).to(device)
    t0 = time.time()
    bpc, ppl = eval_bpc(net, val_chunks, device, args.batch, amp_dtype, max_batches=args.eval_batches)
    n_ok = 0
    samples = []
    for p in PROMPTS:
        out = generate(net, tok, p, net.max_len, device, max_new=args.max_new,
                       temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                       repetition_penalty=args.rep_penalty, amp_dtype=amp_dtype)
        full = p + out
        ok = syntactic_valid(full)
        n_ok += int(ok)
        samples.append({"prompt": p, "text": full, "ast_ok": ok})
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass
    res = {"arch": arch, "ckpt": os.path.basename(path),
           "params_M": round(sum(x.numel() for x in net.parameters()) / 1e6, 2),
           "dim": net.dim, "depth": net.depth, "ctx": net.max_len,
           "step": meta.get("step"), "train_val_bpc": meta.get("val_bpc"),
           "val_bpc": round(bpc, 4), "val_ppl": round(ppl, 2),
           "ast_valid_rate": round(n_ok / max(1, len(PROMPTS)), 3),
           "samples": samples}
    print(f"[ab] {arch:11s} | {res['params_M']}M d{res['dim']}x{res['depth']} step{res['step']} | "
          f"val_bpc {bpc:.4f} ppl {ppl:.2f} | ast合法率 {res['ast_valid_rate']:.0%} | {time.time()-t0:.0f}s", flush=True)
    return res


def write_report(results, args):
    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": results}, f, ensure_ascii=False, indent=2)

    rows = [r for r in results if r]
    lines = ["# 代码模型对照：自有 PER (SeqEnergyNet) vs 标准 Transformer", "",
             "同参数量预算 / 同数据 / 同 token 量（max_steps 对齐）/ 同超参 / 同验证集下的公平 A/B。",
             "指标：held-out **bits/char (bpc，越低越好，单位无关可比)**、困惑度 ppl、生成代码 `ast.parse` **语法合法率**。",
             "", "| 架构 | 参数 | dim×depth | step | val bpc ↓ | val ppl ↓ | ast 合法率 ↑ |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        name = "PER（自有）" if r["arch"] == "per" else "Transformer"
        lines.append(f"| {name} | {r['params_M']}M | {r['dim']}×{r['depth']} | {r['step']} | "
                     f"**{r['val_bpc']}** | {r['val_ppl']} | {r['ast_valid_rate']:.0%} |")
    lines.append("")

    per = next((r for r in rows if r["arch"] == "per"), None)
    tf = next((r for r in rows if r["arch"] == "transformer"), None)
    if per and tf:
        dbpc = round(tf["val_bpc"] - per["val_bpc"], 4)
        dast = round(per["ast_valid_rate"] - tf["ast_valid_rate"], 3)
        win = "PER 更优" if dbpc > 0.005 else ("Transformer 更优" if dbpc < -0.005 else "基本持平")
        lines += ["## 结论（诚实）", "",
                  f"- **bpc**：Transformer − PER = {dbpc:+.4f}（>0=PER 压缩更好）→ **{win}**。",
                  f"- **语法合法率**：PER − Transformer = {dast:+.1%}。",
                  f"- 解码：temperature={args.temperature} top_k={args.top_k} top_p={args.top_p} rep={args.rep_penalty}，"
                  f"max_new={args.max_new}，prompt 数={len(PROMPTS)}。",
                  "- 单位无关的 bpc + 同验证集使该对比直接可比；样例附后，语义正确性两者都有限（小模型边界）。", ""]
    for r in rows:
        name = "PER（自有 SeqEnergyNet）" if r["arch"] == "per" else "标准 Transformer"
        lines += [f"## {name} 生成样例", ""]
        for s in r["samples"][:6]:
            tag = "✓语法合法" if s["ast_ok"] else "✗语法错"
            lines += [f"`[{tag}]` prompt={s['prompt'].strip()[:30]!r}", "", "```python", s["text"][:400], "```", ""]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[ab] 报告已写：{REPORT_MD}", flush=True)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="PER vs Transformer 代码模型同口径对照评测。")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--device", default="")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batches", type=int, default=400, help="验证集评测最多批数")
    ap.add_argument("--min-char-freq", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--max-train-mb", type=float, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep-penalty", type=float, default=1.15)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    args = ap.parse_args(argv)

    device = args.device.strip() or get_device()
    amp_dtype = torch.bfloat16 if (args.amp and device.startswith("cuda")) else None
    print(f"[ab] 设备={device} amp={'bf16' if amp_dtype else 'off'}", flush=True)
    tok, val_chunks = build_val(args)
    print(f"[ab] 验证集块数={len(val_chunks)} 词表={tok.vocab_size}", flush=True)
    results = [eval_arch(a, tok, val_chunks, device, amp_dtype, args) for a in ("per", "transformer")]
    write_report(results, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
