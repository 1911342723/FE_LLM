# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/evaluation/lm_scaling_eval.py
==============================================
**从 0 可溯源架构的语言建模 scaling 曲线**：模型规模 → held-out 困惑度，会不会随规模变好？

这是回答"能不能从 0 训出一个会随规模变好的可溯源 LM"的关键验证。被测主体=`SeqEnergyNet`
（因果 PER / 预测-误差弛豫，作者自有机制，**不是 Transformer 底座**）。配一条**同规模标准
Transformer 字符 LM 当"标尺"**（只用来读斜率与差距，不是要用它）。

口径（诚实、可复跑）：
- 真文本：opus-100 的英文句拼成连续字符流（小词表~40，隔离架构 scaling，不让 embedding 主导参数）。
- 任务：标准自回归 next-char LM，held-out 交叉熵 → bits/char(bpc) 与困惑度 ppl。
- 扫 5 档模型规模(dim×depth)，每档 PER 与 Transformer 各训一个，同数据同 epoch。
- 判据：PER 的 val-bpc 是否随参数量**单调下降**（有 scaling 斜率）；与 Transformer 标尺的差距随规模是缩/平/扩。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.energy_lm.evaluation.lm_scaling_eval --run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fe_llm.config import get_device
from fe_llm.energy_lm.models.seq_net import SeqEnergyNet

REPORT_JSON = os.path.join("docs", "reports", "lm_scaling_eval.json")
REPORT_MD = os.path.join("docs", "reports", "lm_scaling_eval.md")
DATA_PATH = os.path.join("data", "translation", "opus100_train.jsonl")

# 模型规模档位：(dim, depth, n_heads)
SIZE_GRID = [(64, 2, 4), (96, 3, 4), (160, 4, 8), (256, 5, 8), (384, 6, 8)]


class TinyTransformerLM(nn.Module):
    """最小因果 Transformer 字符 LM —— 仅作 scaling 标尺（非项目要采用的架构）。"""

    def __init__(self, vocab_size: int, max_len: int, dim: int, depth: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=2 * dim,
                                           dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        L = ids.size(1)
        x = self.embed(ids) + self.pos[:, :L]
        mask = torch.triu(torch.ones(L, L, device=ids.device, dtype=torch.bool), diagonal=1)
        x = self.enc(x, mask=mask)
        return self.head(self.norm(x))   # logits (B,L,V)


def load_char_stream(path: str, max_sentences: int) -> str:
    parts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_sentences:
                break
            try:
                en = json.loads(line)["en"].strip()
            except (ValueError, KeyError):
                continue
            if en:
                parts.append(en)
    return "\n".join(parts)


def make_chunks(ids: np.ndarray, L: int) -> np.ndarray:
    n = (len(ids) // L) * L
    return ids[:n].reshape(-1, L)


def _logits_per(net, seq):
    return -net(seq)        # SeqEnergyNet 输出能量=-logits


def _logits_tf(net, seq):
    return net(seq)


def train_eval(kind, vocab_size, L, dim, depth, n_heads, train_chunks, val_chunks,
               epochs, lr, batch, device, seed):
    torch.manual_seed(seed)
    if kind == "per":                       # 完整 PER 原型（含可学突触基底 #2）
        net = SeqEnergyNet(vocab_size, L, dim=dim, depth=depth, n_heads=n_heads, use_synapse=True).to(device)
        fwd = _logits_per
    elif kind == "per_nosyn":               # 阉割对照（去 #2，≈因果注意力）—— 升级前的旧被测主体
        net = SeqEnergyNet(vocab_size, L, dim=dim, depth=depth, n_heads=n_heads, use_synapse=False).to(device)
        fwd = _logits_per
    else:
        net = TinyTransformerLM(vocab_size, L, dim=dim, depth=depth, n_heads=n_heads).to(device)
        fwd = _logits_tf
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr = torch.tensor(train_chunks, device=device, dtype=torch.long)
    va = torch.tensor(val_chunks, device=device, dtype=torch.long)
    ntr = len(tr)
    g = torch.Generator(device=device); g.manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(ntr, device=device, generator=g)
        for s in range(0, ntr, batch):
            seq = tr[perm[s:s + batch]]
            logits = fwd(net, seq)
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size), seq[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
    # held-out 评估
    net.eval()
    tot, ntok = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(va), batch):
            seq = va[s:s + batch]
            logits = fwd(net, seq)
            l = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab_size), seq[:, 1:].reshape(-1), reduction="sum")
            tot += float(l); ntok += seq[:, 1:].numel()
    nats = tot / max(1, ntok)
    return {"params": n_params, "val_loss_nats": round(nats, 4),
            "val_bpc": round(nats / math.log(2), 4), "val_ppl": round(math.exp(nats), 2)}


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    text = load_char_stream(args.data, args.max_sentences)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    chunks = make_chunks(ids, args.ctx)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(chunks)
    n_val = max(1, int(len(chunks) * 0.1))
    val_chunks, train_chunks = chunks[:n_val], chunks[n_val:]
    print(f"[lmscale] device={device} chars={len(text)} vocab={vocab_size} ctx={args.ctx} "
          f"train_chunks={len(train_chunks)} val_chunks={len(val_chunks)} epochs={args.epochs}", flush=True)

    grid = SIZE_GRID if args.sizes is None else [tuple(int(x) for x in s.split(":")) for s in args.sizes.split(",")]
    rows = []
    for (dim, depth, nh) in grid:
        per = train_eval("per", vocab_size, args.ctx, dim, depth, nh, train_chunks, val_chunks,
                         args.epochs, args.lr, args.batch, device, args.seed)
        abl = (train_eval("per_nosyn", vocab_size, args.ctx, dim, depth, nh, train_chunks, val_chunks,
                          args.epochs, args.lr, args.batch, device, args.seed) if args.with_ablation else None)
        tf = train_eval("tf", vocab_size, args.ctx, dim, depth, nh, train_chunks, val_chunks,
                        args.epochs, args.lr, args.batch, device, args.seed)
        rows.append({"dim": dim, "depth": depth, "n_heads": nh, "per": per, "per_nosyn": abl, "tf": tf})
        abl_s = f" || PER−syn bpc={abl['val_bpc']:.3f}" if abl else ""
        print(f"[lmscale] dim={dim:>4} depth={depth} | PER+syn {per['params']/1e6:5.2f}M bpc={per['val_bpc']:.3f} "
              f"ppl={per['val_ppl']:.2f}{abl_s} || TF {tf['params']/1e6:5.2f}M bpc={tf['val_bpc']:.3f} "
              f"ppl={tf['val_ppl']:.2f}", flush=True)

    per_bpc = [r["per"]["val_bpc"] for r in rows]
    tf_bpc = [r["tf"]["val_bpc"] for r in rows]
    per_drop = round(per_bpc[0] - per_bpc[-1], 4)        # >0 = 随规模变好
    tf_drop = round(tf_bpc[0] - tf_bpc[-1], 4)
    per_monotone = all(per_bpc[i + 1] <= per_bpc[i] + 0.02 for i in range(len(per_bpc) - 1))
    gap_small = round(per_bpc[0] - tf_bpc[0], 4)
    gap_big = round(per_bpc[-1] - tf_bpc[-1], 4)

    # 升级证据：完整原型(PER+syn) vs 阉割(PER−syn) 的 bpc 差（>0 = 完整更好）
    abl_msg = ""
    abl_bpc = None
    if args.with_ablation and rows[0]["per_nosyn"] is not None:
        abl_bpc = [r["per_nosyn"]["val_bpc"] for r in rows]
        deltas = [round(abl_bpc[i] - per_bpc[i], 4) for i in range(len(rows))]
        wins = sum(1 for d in deltas if d > 0)
        abl_msg = (f"升级证据(完整#2 vs 阉割)：各档 Δbpc(阉割−完整)={deltas}，完整更优 {wins}/{len(rows)} 档"
                   f"（均值 {round(float(np.mean(deltas)),4):+.4f}）→ "
                   f"{'完整原型在 scaling 全程不劣于、整体优于阉割版' if wins >= len(rows) - 1 else '完整与阉割互有胜负，差异小'}。")

    if per_drop >= 0.1 and per_monotone:
        scale_msg = (f"**完整 PER 原型(含可学突触 #2) 有 scaling 斜率**：val-bpc 随规模 {'单调' if per_monotone else ''}下降 "
                     f"{per_bpc[0]:.3f}→{per_bpc[-1]:.3f}（降 {per_drop:+.3f}）→ 从 0 自建的完整原型会随规模变好。")
    elif per_drop >= 0.1:
        scale_msg = (f"**完整 PER 原型总体随规模变好但非严格单调**：val-bpc {per_bpc[0]:.3f}→{per_bpc[-1]:.3f}"
                     f"（降 {per_drop:+.3f}）。有 scaling 趋势，个别档需调超参。")
    else:
        scale_msg = (f"**完整 PER 原型随规模几乎不降（警示）**：val-bpc {per_bpc[0]:.3f}→{per_bpc[-1]:.3f}"
                     f"（降 {per_drop:+.3f}）→ 当前配置下没有 scaling 斜率，需架构/训练工程。")
    gap_msg = (f"与 Transformer 标尺：小规模差距 {gap_small:+.3f} bpc、大规模 {gap_big:+.3f} bpc"
               f"（{'缩小' if gap_big < gap_small - 0.02 else '扩大' if gap_big > gap_small + 0.02 else '基本持平'}）；"
               f"TF 自身降 {tf_drop:+.3f}。")

    result = {
        "task": "from-scratch traceable LM scaling: FULL PER prototype (causal PER + synapse #2) vs ablated vs Transformer ruler, char-LM held-out bpc",
        "config": {"corpus": "opus100-en chars", "vocab": vocab_size, "ctx": args.ctx,
                   "epochs": args.epochs, "train_chunks": len(train_chunks), "val_chunks": len(val_chunks),
                   "subject": "完整 PER 原型(含可学突触 #2)", "with_ablation": bool(args.with_ablation)},
        "rows": rows,
        "per_bpc_curve": per_bpc, "abl_bpc_curve": abl_bpc, "tf_bpc_curve": tf_bpc,
        "per_drop_small_to_big": per_drop, "tf_drop_small_to_big": tf_drop,
        "per_monotone": per_monotone, "gap_small": gap_small, "gap_big": gap_big,
        "verdict": scale_msg + " " + gap_msg + (" " + abl_msg if abl_msg else ""),
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    has_abl = args.with_ablation and rows[0]["per_nosyn"] is not None
    lines = [
        "# 从 0 可溯源架构语言建模 scaling 曲线（完整 PER 原型/含可学突触 #2 vs 阉割 vs Transformer 标尺）",
        "",
        f"- 语料：opus100-en 字符流，vocab={vocab_size}, ctx={args.ctx}, epochs={args.epochs}, "
        f"train_chunks={len(train_chunks)}, val_chunks={len(val_chunks)}",
        "- **被测主体=完整 PER 原型 `SeqEnergyNet(use_synapse=True)`（含辨识点 #2 可学突触基底）**；"
        "PER−syn=去 #2 的阉割对照（升级前的旧被测主体）；Transformer 仅作 scaling 标尺。指标=held-out bits/char（越低越好）。",
        "",
    ]
    if has_abl:
        lines += ["| dim×depth | PER+syn 参数 | PER+syn bpc | PER+syn ppl | PER−syn bpc | TF bpc | TF ppl |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f"| {r['dim']}×{r['depth']} | {r['per']['params']/1e6:.2f}M | **{r['per']['val_bpc']:.3f}** | "
                         f"{r['per']['val_ppl']:.2f} | {r['per_nosyn']['val_bpc']:.3f} | {r['tf']['val_bpc']:.3f} | "
                         f"{r['tf']['val_ppl']:.2f} |")
    else:
        lines += ["| dim×depth | PER+syn 参数 | PER+syn bpc | PER+syn ppl | TF 参数 | TF bpc | TF ppl |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f"| {r['dim']}×{r['depth']} | {r['per']['params']/1e6:.2f}M | {r['per']['val_bpc']:.3f} | "
                         f"{r['per']['val_ppl']:.2f} | {r['tf']['params']/1e6:.2f}M | {r['tf']['val_bpc']:.3f} | "
                         f"{r['tf']['val_ppl']:.2f} |")
    lines += ["", f"- 结论：{scale_msg}", "", f"- 标尺对照：{gap_msg}"]
    if abl_msg:
        lines += ["", f"- {abl_msg}"]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[lmscale] === 结论 ===", flush=True)
    print(scale_msg, flush=True)
    print(gap_msg, flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="From-scratch traceable LM scaling curve (PER vs Transformer ruler).")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--max-sentences", type=int, default=50000)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sizes", default=None, help="覆盖规模档，如 '64:2:4,128:4:8'")
    ap.add_argument("--with-ablation", dest="with_ablation", action="store_true", default=True,
                    help="同时跑阉割 PER−syn 对照（默认开，展示完整#2 vs 阉割的升级证据）")
    ap.add_argument("--no-ablation", dest="with_ablation", action="store_false", help="只跑完整原型 vs Transformer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-json", default=REPORT_JSON)
    ap.add_argument("--report-md", default=REPORT_MD)
    return ap


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print("[lmscale] dry-run：未训练。从 0 可溯源 LM(SeqEnergyNet) 的 规模→困惑度 曲线 + Transformer 标尺。")
        print("[lmscale] 真正运行请显式追加 --run。")
        return 0
    t0 = time.time()
    run(args)
    print(f"[lmscale] 总用时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
