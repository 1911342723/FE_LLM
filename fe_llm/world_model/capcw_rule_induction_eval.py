# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_rule_induction_eval.py
===============================================
**直面"连连看"质疑**：CAPCW 是不是只会"查已见的关联"（连连看/lookup），还是能**从示例归纳规则、外推到
未见输入**（beyond lookup = 真有点推理）？见 `docs/FE-LLM核心引擎构想.md` 第 31 节。

任务（in-context 规则归纳，规则=循环移位，每例随机 shift k）：
- 给 K 个示例对 (x_i, (x_i+k) mod n)（同一例共享同一 k）；
- 查询一个符号 x_q，预测 (x_q+k) mod n。
- **关键**：x_q 分 **SEEN**（在示例对里出现过）与 **UNSEEN**（没出现过）两种。
  - 纯"连连看"(lookup)：只能答 SEEN（查表）；UNSEEN ≈ 随机。
  - "超连连看"(规则归纳)：从示例**推出 k**、应用到 UNSEEN → UNSEEN 也对。

判据（先写死）
--------------
- **UNSEEN 准确率**（best of flat/capcw）≥ 0.50（≫ 随机 1/n）→ **PASS：engine 会 in-context 规则归纳+外推
  （不止连连看）**；
- UNSEEN ≈ 随机（≤ 2/n）→ **确认"连连看"**：只会查已见关联、不会规则外推（诚实印证质疑）；
- 之间 → PARTIAL。
SEEN 准确率作 sanity（查表应高）。诚实：循环移位含模运算成分，UNSEEN 失败可能混入"算术难"，但 SEEN/UNSEEN
对比仍清楚区分"查表 vs 规则"。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_rule_induction_eval --run
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
import torch.nn.functional as F

from fe_llm.config import get_device
from fe_llm.world_model.capcw import PCWorkspace

REPORT_JSON = os.path.join("docs", "reports", "capcw_rule_induction_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_rule_induction_eval.md")


def gen_shift(n_sym, K, n, seed, p_seen=0.5):
    """每例：随机 shift k；K 个示例对 (x_i,(x_i+k)%n)（x_i 互不同）；查询 x_q(SEEN/UNSEEN)；label=(x_q+k)%n。"""
    rng = np.random.default_rng(seed)
    pk = np.zeros((n, K), dtype=np.int64)
    pv = np.zeros((n, K), dtype=np.int64)
    q = np.zeros((n,), dtype=np.int64)
    y = np.zeros((n,), dtype=np.int64)
    seen = np.zeros((n,), dtype=bool)
    for i in range(n):
        k = int(rng.integers(1, n_sym))                     # shift ∈ [1, n_sym-1]
        xs = rng.choice(n_sym, size=K, replace=False)       # 示例输入互不同
        pk[i] = xs
        pv[i] = (xs + k) % n_sym
        if rng.random() < p_seen:
            xq = int(rng.choice(xs))                          # SEEN：查询在示例里
            seen[i] = True
        else:
            unseen_pool = [s for s in range(n_sym) if s not in set(int(z) for z in xs)]
            xq = int(rng.choice(unseen_pool))                # UNSEEN：查询不在示例里
            seen[i] = False
        q[i] = xq
        y[i] = (xq + k) % n_sym
    return pk, pv, q, y, seen


class RuleModel(nn.Module):
    """in-context 规则模型：示例对→world(flat 单向量 / capcw slot)，readout=head([pooled; query_emb])→n_sym。

    唯一变量=world 结构。规则是全局的（非按键检索），故 readout 用池化世界 + 查询嵌入，给"归纳+应用规则"
    一个公平机会。符号共享一张嵌入（移位是 符号→符号）。
    """

    def __init__(self, n_sym, d, n_slots, iters, world_mode):
        super().__init__()
        self.sym_emb = nn.Embedding(n_sym, d)
        self.pair = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.world_mode = world_mode
        self.d = d
        if world_mode == "capcw":
            self.ws = PCWorkspace(dim=d, n_slots=n_slots, iters=iters)
        self.to_q = nn.Linear(d, d)
        # readout 同时给两条通道：read=内容寻址读出(查表/lookup)、pooled=全局世界(规则/rule)，外加 query 嵌入。
        # SEEN 可走 read 查表；UNSEEN 只能靠 pooled+query 归纳规则——两通道并存=对'规则外推'公平。
        self.head = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, n_sym))

    def forward(self, pk, pv, q):
        pairs = self.pair(torch.cat([self.sym_emb(pk), self.sym_emb(pv)], dim=-1))   # (B,K,d)
        slots = self.ws(pairs).slots if self.world_mode == "capcw" else pairs        # (B,M/K,d)
        qd = self.sym_emb(q)
        score = torch.einsum("bmd,bd->bm", slots, self.to_q(qd)) / math.sqrt(self.d)
        read = (slots * score.softmax(dim=1).unsqueeze(-1)).sum(dim=1)                # 查表通道
        pooled = slots.mean(dim=1)                                                    # 规则通道
        return self.head(torch.cat([read, pooled, qd], dim=-1))


def _train_eval(world_mode, n_sym, K, d, n_slots, iters, n_train, n_test, epochs, lr, batch, device, seed):
    torch.manual_seed(seed)
    model = RuleModel(n_sym, d, n_slots, iters, world_mode).to(device)
    tr = gen_shift(n_sym, K, n_train, seed)
    te = gen_shift(n_sym, K, n_test, seed + 5000)
    pk, pv, q, y, _ = (torch.tensor(a, device=device) if a.dtype != bool else a for a in tr)
    tpk, tpv, tq, ty, tseen = te
    tpk, tpv, tq, ty = (torch.tensor(a, device=device) for a in (tpk, tpv, tq, ty))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(q)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(pk[idx], pv[idx], q[idx]), y[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(tpk, tpv, tq).argmax(-1).cpu().numpy()
    yt = ty.cpu().numpy()
    correct = (pred == yt)
    acc = float(correct.mean())
    acc_seen = float(correct[tseen].mean()) if tseen.any() else 0.0
    acc_unseen = float(correct[~tseen].mean()) if (~tseen).any() else 0.0
    return acc, acc_seen, acc_unseen


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    print(f"[ri] device={device} n_sym={args.n_sym} K={args.k} d={args.d} seeds={args.seeds} (随机={1.0/args.n_sym:.3f})", flush=True)
    arms = {}
    for mode in ("flat", "capcw"):
        accs, seens, unseens = [], [], []
        for si in range(args.seeds):
            seed = args.seed + si
            a, asn, aun = _train_eval(mode, args.n_sym, args.k, args.d, max(args.n_slots, args.k + 1),
                                      args.iters, args.n_train, args.n_test, args.epochs, args.lr, args.batch, device, seed)
            accs.append(a); seens.append(asn); unseens.append(aun)
        arms[mode] = {"acc": round(float(np.mean(accs)), 4), "acc_seen": round(float(np.mean(seens)), 4),
                      "acc_unseen": round(float(np.mean(unseens)), 4)}
        print(f"[ri] {mode}: acc={arms[mode]['acc']:.3f} seen={arms[mode]['acc_seen']:.3f} "
              f"unseen={arms[mode]['acc_unseen']:.3f}", flush=True)

    rnd = 1.0 / args.n_sym
    best_unseen = max(arms["flat"]["acc_unseen"], arms["capcw"]["acc_unseen"])
    best_seen = max(arms["flat"]["acc_seen"], arms["capcw"]["acc_seen"])
    if best_unseen >= 0.50:
        verdict = (f"PASS(超连连看): engine 会 **in-context 规则归纳 + 外推**——UNSEEN 查询准确率 {best_unseen:.3f}"
                   f"（≫ 随机 {rnd:.3f}），即它从示例**推出移位规则、应用到没见过的输入**，不止查已见关联。")
    elif best_unseen <= 2 * rnd:
        verdict = (f"确认\"连连看\"(诚实负): engine **只会查已见关联**——SEEN {best_seen:.3f} 高(查表 OK)，但 UNSEEN"
                   f"仅 {best_unseen:.3f}≈随机 {rnd:.3f}，**不会从示例归纳规则外推到未见输入**。印证'CAPCW 像连连看'的判断："
                   f"它是内容寻址**取回/链接**引擎，单层不做规则归纳（规则归纳要靠深度/多层组合，本项目按容量纪律不堆）。")
    else:
        verdict = (f"PARTIAL: UNSEEN {best_unseen:.3f} 高于随机 {rnd:.3f} 但 <0.50——有**部分**规则外推、不充分"
                   f"（SEEN {best_seen:.3f}）。介于'纯连连看'与'真规则归纳'之间。")

    result = {
        "task": "in-context rule induction (cyclic shift): can engine extrapolate a rule to UNSEEN queries (beyond lookup)?",
        "design": "K demo pairs (x,(x+k)%n) per example (random k); query SEEN vs UNSEEN; var=world(flat vs CAPCW).",
        "config": {"n_sym": args.n_sym, "K": args.k, "d": args.d, "epochs": args.epochs, "seeds": args.seeds,
                   "random_baseline": round(rnd, 4)},
        "arms": arms, "best_unseen": round(best_unseen, 4), "best_seen": round(best_seen, 4),
        "verdict": verdict,
        "note": "SEEN=查询在示例对里(查表可解=连连看)；UNSEEN=查询没出现(必须归纳移位规则并应用=超连连看)。"
                "规则全局，readout=池化 world + query 嵌入。循环移位含模运算，UNSEEN 失败或混入算术难，但 SEEN/UNSEEN "
                "对比清楚区分'查表 vs 规则外推'。直面'CAPCW 像连连看'的质疑。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# CAPCW 直面\"连连看\"：in-context 规则归纳（能否外推到未见查询？）",
        "",
        f"- 判定：**{verdict}**",
        f"- 任务：K={args.k} 个示例对 (x,(x+k)%{args.n_sym})，查询 SEEN/UNSEEN，预测 (x_q+k)%{args.n_sym}；随机 {rnd:.3f}",
        "",
        "| world 结构 | 总 acc | SEEN(查表) | **UNSEEN(规则外推)** |",
        "|---|---:|---:|---:|",
        f"| flat（单向量） | {arms['flat']['acc']:.3f} | {arms['flat']['acc_seen']:.3f} | **{arms['flat']['acc_unseen']:.3f}** |",
        f"| CAPCW（slot） | {arms['capcw']['acc']:.3f} | {arms['capcw']['acc_seen']:.3f} | **{arms['capcw']['acc_unseen']:.3f}** |",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[ri] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    print(f"[ri] best UNSEEN={best_unseen:.3f} best SEEN={best_seen:.3f} (随机={rnd:.3f})", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW in-context rule induction (cyclic shift): lookup vs rule extrapolation.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-sym", type=int, default=20)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--n-slots", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=80)
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
        print("[ri] dry-run：未训练。in-context 规则归纳(环移)：UNSEEN 查询能否靠归纳规则答(超连连看) vs 仅查表。")
        print("[ri] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
