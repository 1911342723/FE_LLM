# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_multihop_v2_eval.py
============================================
多跳链式推理"新机制"：**写回+重弛豫**（方案A）与**深度/逐跳参数**（方案B），对照上一轮失败的
"定 slot 反复读"（capcw_multihop_eval，commit abc2df5）。见 `docs/FE-LLM核心引擎构想.md` 第 18 节。

上一轮诊断：对**固定 slot** 反复读本质还是单跳（中间结果无法干净再注入为下一跳查询）。新机制让
工作空间的**状态跨跳演化**：

- flat            ：单向量池化（地板）。
- capcw_reads     ：relax 一次 → 在固定 slots 上读 H 次（**上次失败的对照**，shared to_next）。
- capcw_writeback ：**方案A** 每跳读出后把中间结果经 writeback 投影**写回工作集**并**重弛豫**（ws(X) 重算），
                    中间结论进入工作记忆、参与下一跳（对应"残差流即工作记忆，每层读/写"）。
- capcw_depth     ：**方案B** relax 一次，但每跳用**独立参数**的 query 变换(to_next[h])链式读
                    （对应 Transformer 靠多层做多跳，每层一跳）。

唯一变量=链式读出机制（4 臂同 d / 同 slot 预算 / 同序列相邻算子；同 seed 初始化）。
判据：H≥2(多跳)时 max(writeback, depth) − max(reads, flat) ≥ +0.10 → 新机制破解多跳；否则诚实负结果。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_multihop_v2_eval --run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn

from fe_llm.config import get_device
from fe_llm.world_model.capcw import PCWorkspace
from fe_llm.world_model.capcw_multihop_eval import (
    CAPCWMultihop,
    FlatMultihop,
    SeqEncoder,
    gen_multihop,
    train_eval,
)

REPORT_JSON = os.path.join("docs", "reports", "capcw_multihop_v2_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_multihop_v2_eval.md")


class CAPCWWriteback(nn.Module):
    """方案A：写回 + 重弛豫。每跳读出中间结果→写回工作集→重弛豫，让状态跨跳演化。"""

    def __init__(self, n_sym, seq_len, d, n_slots, iters, n_hops):
        super().__init__()
        self.enc = SeqEncoder(n_sym, seq_len, d)          # 含序列相邻算子
        self.ws = PCWorkspace(dim=d, n_slots=n_slots, iters=iters)
        self.to_q = nn.Linear(d, d)
        self.to_next = nn.Linear(d, d)
        self.writeback = nn.Linear(d, d)                  # 把读出的中间结果投影成"新观测"
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_sym))
        self.d = d
        self.n_hops = n_hops

    def forward(self, ids, cue):
        X = self.enc.tokens(ids)                          # (B,L,d) 绑定（含相邻算子）
        q = self.to_q(self.enc.emb(cue))
        read = None
        for _ in range(self.n_hops):
            slots = self.ws(X).slots                      # 重弛豫当前工作集
            score = torch.einsum("bmd,bd->bm", slots, q) / math.sqrt(self.d)
            attn = score.softmax(dim=1).unsqueeze(-1)
            read = (slots * attn).sum(dim=1)
            wb = self.writeback(read).unsqueeze(1)        # (B,1,d) 中间结果写回为新观测
            X = torch.cat([X, wb], dim=1)
            q = self.to_next(read)
        return self.head(read)


class CAPCWDepth(nn.Module):
    """方案B：深度/逐跳参数。relax 一次，但每跳用独立参数的 query 变换链式读（每层一跳）。"""

    def __init__(self, n_sym, seq_len, d, n_slots, iters, n_hops):
        super().__init__()
        self.enc = SeqEncoder(n_sym, seq_len, d)
        self.ws = PCWorkspace(dim=d, n_slots=n_slots, iters=iters)
        self.to_q = nn.Linear(d, d)
        self.to_next = nn.ModuleList([nn.Linear(d, d) for _ in range(max(1, n_hops))])  # 每跳独立
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_sym))
        self.d = d
        self.n_hops = n_hops

    def forward(self, ids, cue):
        slots = self.ws(self.enc.tokens(ids)).slots       # relax 一次
        q = self.to_q(self.enc.emb(cue))
        read = None
        for h in range(self.n_hops):
            score = torch.einsum("bmd,bd->bm", slots, q) / math.sqrt(self.d)
            attn = score.softmax(dim=1).unsqueeze(-1)
            read = (slots * attn).sum(dim=1)
            q = self.to_next[h](read)                      # 逐跳独立变换
        return self.head(read)


def _build_arms(args, n_hops, seq_len, seed):
    n_slots = max(args.n_slots, n_hops + args.n_distract + 1)
    arms = {}
    torch.manual_seed(seed)
    arms["flat"] = FlatMultihop(args.n_sym, seq_len, args.d)
    torch.manual_seed(seed)
    arms["capcw_reads"] = CAPCWMultihop(args.n_sym, seq_len, args.d, n_slots, args.iters, n_reads=n_hops)
    torch.manual_seed(seed)
    arms["capcw_writeback"] = CAPCWWriteback(args.n_sym, seq_len, args.d, n_slots, args.iters, n_hops)
    torch.manual_seed(seed)
    arms["capcw_depth"] = CAPCWDepth(args.n_sym, seq_len, args.d, n_slots, args.iters, n_hops)
    return arms


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    hop_list = [int(x) for x in args.hop_list.split(",")]
    names = ["flat", "capcw_reads", "capcw_writeback", "capcw_depth"]
    print(f"[mh-v2] device={device} hop_list={hop_list} n_distract={args.n_distract} d={args.d} seeds={args.seeds}", flush=True)
    results: dict = {}
    for h in hop_list:
        seq_len = max(args.seq_len, 2 * (h + args.n_distract) + 4)
        accs = {n: [] for n in names}
        for si in range(args.seeds):
            seed = args.seed + si
            train = gen_multihop(args.n_sym, h, args.n_distract, seq_len, args.n_train, seed)
            test = gen_multihop(args.n_sym, h, args.n_distract, seq_len, args.n_test, seed + 5000)
            models = _build_arms(args, h, seq_len, seed)
            common = dict(device=device, epochs=args.epochs, lr=args.lr, batch=args.batch, seed=seed)
            for n in names:
                accs[n].append(train_eval(models[n], train, test, **common))
        results[h] = {n: {"mean": round(float(np.mean(accs[n])), 4), "std": round(float(np.std(accs[n])), 4)} for n in names}
        msg = " ".join(f"{n}={results[h][n]['mean']:.3f}" for n in names)
        print(f"[mh-v2] n_hops={h} (seq_len={seq_len}) {msg} (random={1.0/args.n_sym:.3f})", flush=True)

    multi = [h for h in hop_list if h >= 2]

    def m(h, n):
        return results[h][n]["mean"]
    # 新机制相对"失败对照(reads)与地板(flat)的最好者"在多跳上的增益。
    new_best = {h: round(max(m(h, "capcw_writeback"), m(h, "capcw_depth")), 4) for h in hop_list}
    old_best = {h: round(max(m(h, "capcw_reads"), m(h, "flat")), 4) for h in hop_list}
    gain = {h: round(new_best[h] - old_best[h], 4) for h in hop_list}
    multi_gain = round(float(np.mean([gain[h] for h in multi])), 4) if multi else 0.0
    best_arm_overall = None
    if multi:
        # 在多跳上平均表现最好的新机制臂。
        wb = float(np.mean([m(h, "capcw_writeback") for h in multi]))
        dp = float(np.mean([m(h, "capcw_depth") for h in multi]))
        best_arm_overall = "capcw_writeback" if wb >= dp else "capcw_depth"

    if multi_gain >= 0.10:
        verdict = (f"PASS: 新机制({best_arm_overall})在多跳(H≥2)上比'定slot反复读/单向量'最好者高 {multi_gain:+.4f}"
                   "——写回重弛豫/逐跳深度让中间结论进入状态、链式组合成立。")
    elif multi_gain >= 0.03:
        verdict = (f"PARTIAL: 新机制多跳上有正向增益 {multi_gain:+.4f} 但偏弱（未达 +0.10）。")
    else:
        verdict = ("FAIL: 写回重弛豫/逐跳深度仍未破解多跳链式——与上轮一致，d=32 下中间结论难以可靠链式组合，"
                   "记录为诚实边界（多跳很可能需更大容量或显式 key/value 分离）。")

    result = {
        "task": "multi-hop chained binding (v2 mechanisms): c0->...->cH, query c0 predict cH",
        "design": "arms=flat / capcw_reads(failed control) / capcw_writeback(A:write-back+re-relax) / capcw_depth(B:per-hop params); same adjacency+d+slot; same-seed init",
        "config": {"n_sym": args.n_sym, "hop_list": hop_list, "n_distract": args.n_distract, "d": args.d,
                   "n_slots": args.n_slots, "iters": args.iters, "epochs": args.epochs, "seeds": args.seeds,
                   "random_baseline": round(1.0 / args.n_sym, 4)},
        "by_n_hops": results,
        "new_best_by_hops": new_best, "old_best_by_hops": old_best, "gain_by_hops": gain,
        "multihop_gain_mean": multi_gain, "best_new_arm": best_arm_overall,
        "verdict": verdict,
        "note": "对照 capcw_reads=上轮失败的'定slot反复读'。新机制让状态跨跳演化：writeback 把中间结果写回工作集+重弛豫；"
                "depth 用逐跳独立参数。判据=多跳上新机制相对(reads,flat)最好者的增益≥+0.10。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# CAPCW · 多跳链式新机制（写回重弛豫 / 逐跳深度）",
        "",
        f"- 判定：**{verdict}**",
        f"- 任务：链 c0→…→cH 查 c0 答 cH；n_sym={args.n_sym}, d={args.d}, n_distract={args.n_distract}；随机基线 {1.0/args.n_sym:.3f}",
        f"- 唯一变量=链式读出机制；4 臂同 d/同 slot/同相邻算子/同 seed 初始化。",
        "",
        "| n_hops | flat | capcw_reads(失败对照) | capcw_writeback(A) | capcw_depth(B) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for h in hop_list:
        r = results[h]
        lines.append(f"| {h} | {r['flat']['mean']:.3f} | {r['capcw_reads']['mean']:.3f} | "
                     f"{r['capcw_writeback']['mean']:.3f} | {r['capcw_depth']['mean']:.3f} |")
    lines += [
        "",
        f"- 多跳(H≥2) 新机制最好 − (reads,flat)最好 平均增益 = **{multi_gain:+.4f}**；各跳增益：{gain}",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[mh-v2] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    print(f"[mh-v2] 多跳增益={multi_gain:+.4f}  best_new={best_arm_overall}", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW multi-hop v2 mechanisms (write-back / depth) vs failed reads.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-sym", type=int, default=20)
    ap.add_argument("--hop-list", default="1,2,3")
    ap.add_argument("--n-distract", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--n-slots", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=2)
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
        print("[mh-v2] dry-run：未训练。多跳新机制：写回重弛豫(A)/逐跳深度(B) vs 失败对照(reads)/地板(flat)。")
        print("[mh-v2] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
