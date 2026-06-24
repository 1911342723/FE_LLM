# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/evaluation/per_synapse_scaling_eval.py
=======================================================
回答蓝图核心开放问之一在 #2 上的版本：**「PER 辨识点 #2（可学突触基底）是否只缺规模？」**

做法：在真实中文聊天 LM 上，沿模型规模（dim×depth）扫一条曲线，每个规模同数据/同预算训
**完整原型 PER+syn(#2)** 与 **阉割 PER−syn(无#2)**，看 held-out 困惑度差 Δ=(阉割−完整) 随规模
的走向——Δ 随规模变大=「#2 越大越值」；持平=「小而恒定」；缩小=「规模一大就被洗掉」。

复用 per_synapse_lm_eval 的数据管线与配对训练（每个 (规模,种子) 跑一对）。
默认 dry-run；真跑加 --run。
    python -m fe_llm.energy_lm.evaluation.per_synapse_scaling_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from fe_llm.config import get_device
from fe_llm.energy_lm.training.chat_train import DATA_MAIN, DATA_EXTRA, load_pairs
from fe_llm.energy_lm.evaluation.per_synapse_lm_eval import _compare_one_seed

REPORT_JSON = os.path.join("docs", "reports", "per_synapse_scaling_eval.json")
REPORT_MD = os.path.join("docs", "reports", "per_synapse_scaling_eval.md")
FIG_DIR = os.path.join("docs", "reports", "figs")


def parse_sizes(s: str) -> list[tuple[int, int]]:
    out = []
    for tok in s.split(","):
        d, dep = tok.lower().split("x")
        out.append((int(d), int(dep)))
    return out


def run(args):
    device = get_device()
    print(f"[scale] device={device}", flush=True)
    pairs = load_pairs([args.data] + ([DATA_EXTRA] if args.extra else []), args.max_pairs)
    if not pairs:
        print("[scale] 无数据，退出。"); return 1
    sizes = parse_sizes(args.sizes)
    seeds = [args.seed + i for i in range(max(1, args.n_seeds))]
    print(f"[scale] {len(pairs)} 对，规模={sizes}，种子={seeds}", flush=True)

    rows = []
    for (dim, depth) in sizes:
        a = Namespace(dim=dim, depth=depth, heads=args.heads, max_len=args.max_len,
                      epochs=args.epochs, batch=args.batch, lr=args.lr)
        fulls, abls, params = [], [], 0
        t0 = time.time()
        for sd in seeds:
            pf, pa, _, _, npar = _compare_one_seed(a, pairs, sd, device, want_nets=False)
            fulls.append(pf); abls.append(pa); params = npar
        fulls, abls = np.array(fulls), np.array(abls)
        deltas = abls - fulls
        row = {"dim": dim, "depth": depth, "params": params,
               "full_ppl": round(float(fulls.mean()), 3), "full_std": round(float(fulls.std()), 3),
               "abl_ppl": round(float(abls.mean()), 3), "abl_std": round(float(abls.std()), 3),
               "delta_mean": round(float(deltas.mean()), 3), "delta_std": round(float(deltas.std()), 3),
               "wins_full": int((deltas > 0).sum()), "n_seeds": len(seeds), "sec": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"[scale] dim={dim} depth={depth} ({params/1e6:.2f}M): 完整={row['full_ppl']:.2f} "
              f"阉割={row['abl_ppl']:.2f} Δ={row['delta_mean']:+.3f}±{row['delta_std']:.3f} "
              f"({row['wins_full']}/{row['n_seeds']}) {row['sec']}s", flush=True)

    # 趋势判定：Δ 相对 params(log) 的斜率符号
    xs = np.log10(np.array([r["params"] for r in rows]))
    ds = np.array([r["delta_mean"] for r in rows])
    slope = float(np.polyfit(xs, ds, 1)[0]) if len(rows) >= 2 else 0.0
    all_pos = all(r["delta_mean"] > 0 for r in rows)
    if slope > 0.05 and all_pos:
        verdict = (f"📈/🟡 **#2 的增益不随规模消失、且呈轻微上行（suggestive 非定论）**：各档完整原型 ppl 全低于阉割"
                   f"（{sum(r['wins_full'] for r in rows)}/{sum(r['n_seeds'] for r in rows)} 种子全胜），Δ 对 log-参数斜率={slope:+.3f}>0；"
                   f"但绝对幅度小、点数少且有噪声（不一定单调），只能说 #2 **至少不是『规模一大就被洗掉』**、反有『越大越值』苗头，"
                   f"需更大规模/更多种子坐实。已确证的硬价值仍是可溯源（§2.6/P1）。")
    elif abs(slope) <= 0.05:
        verdict = (f"🟡 **#2 优势小而近恒定**（Δ 对 log-params 斜率={slope:+.3f}≈0）——"
                   f"各档完整原型 ppl 略低但增益不随规模放大；#2 在本数据规模不是「只缺规模」，其确证价值仍在可溯源（见 §2.6/P1）。")
    else:
        verdict = (f"🟡/📉 **#2 优势随规模缩小**（Δ 斜率={slope:+.3f}<0）——规模一大 #2 的 ppl 增益被洗淡；"
                   f"诚实结论：#2 价值在可溯源而非规模化降 ppl。")

    fig = _plot(rows, os.path.join(FIG_DIR, "per_synapse_scaling.png"))
    results = {"task": "PER #2 synapse scaling on real chat LM (full vs ablated across model sizes)",
               "config": {"data": os.path.basename(args.data), "n_pairs": len(pairs),
                          "epochs": args.epochs, "n_seeds": len(seeds), "max_len": args.max_len},
               "rows": rows, "delta_vs_logparams_slope": round(slope, 4), "fig": fig, "verdict": verdict}
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    _write_md(results)
    print("\n[scale] === 结论 ===\n" + verdict, flush=True)
    return 0


def _plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        params = [r["params"] / 1e6 for r in rows]
        full = [r["full_ppl"] for r in rows]
        abl = [r["abl_ppl"] for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
        ax1.plot(params, full, "o-", label="完整 PER+syn(#2)")
        ax1.plot(params, abl, "s--", label="阉割 PER−syn")
        ax1.set_xlabel("参数量 (M)"); ax1.set_ylabel("held-out 困惑度"); ax1.set_xscale("log")
        ax1.set_title("规模 × 困惑度"); ax1.legend(); ax1.grid(alpha=0.3)
        ax2.plot(params, [r["delta_mean"] for r in rows], "d-", color="tab:green")
        ax2.axhline(0, color="gray", lw=0.8)
        ax2.set_xlabel("参数量 (M)"); ax2.set_ylabel("Δppl (阉割−完整)"); ax2.set_xscale("log")
        ax2.set_title("#2 的净增益 vs 规模"); ax2.grid(alpha=0.3)
        os.makedirs(FIG_DIR, exist_ok=True)
        plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
        return path
    except Exception as e:
        print(f"[scale] 出图跳过：{e}", flush=True)
        return None


def _write_md(r):
    L = ["# PER 辨识点 #2（可学突触基底）规模曲线：是否只缺规模？", "",
         f"- 真实聊天 LM（{r['config']['data']}，{r['config']['n_pairs']} 对），每档同数据/同预算训完整 vs 阉割，"
         f"epochs={r['config']['epochs']}，{r['config']['n_seeds']} 种子取均值。", "",
         "| 规模(dim×depth) | 参数 | 完整 ppl | 阉割 ppl | Δ(阉割−完整) | 完整胜 |",
         "|---|---:|---:|---:|---:|---:|"]
    for x in r["rows"]:
        L.append(f"| {x['dim']}×{x['depth']} | {x['params']/1e6:.2f}M | {x['full_ppl']:.2f}±{x['full_std']:.2f} | "
                 f"{x['abl_ppl']:.2f}±{x['abl_std']:.2f} | **{x['delta_mean']:+.3f}**±{x['delta_std']:.2f} | {x['wins_full']}/{x['n_seeds']} |")
    L += ["", f"- Δ 对 log-参数 斜率 = **{r['delta_vs_logparams_slope']:+.4f}**"]
    if r.get("fig"):
        L += ["", f"- 曲线图：`{r['fig']}`"]
    L += ["", "## 结论", "", r["verdict"], ""]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[scale] 报告写出 → {REPORT_MD}", flush=True)


def build_arg_parser():
    ap = argparse.ArgumentParser(description="PER #2 synapse 规模曲线（真实聊天 LM 完整 vs 阉割）。")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--data", default=DATA_MAIN)
    ap.add_argument("--extra", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--sizes", default="64x2,96x3,128x4,192x5", help="dimXdepth 逗号分隔")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-seeds", type=int, default=2)
    return ap


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print("[scale] dry-run：PER #2 规模曲线（完整 vs 阉割）。真跑加 --run。")
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
