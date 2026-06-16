# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_sequence_relation_eval.py
==================================================
**从序列直接读关系**（更像真语言）：此前 CAPCW 的绑定都喂**显式 (key,value) 对**；这里把关系以**扁平
token 序列**给出（[e1 r1 v1 e2 r2 v2 …]），模型须从序列里读出 (实体·关系→值) 三元组，再按 (查询实体·关系)
取回值。见 `docs/FE-LLM核心引擎构想.md` 第 33 节。唯一变量=world 结构（flat 单向量 vs CAPCW slot），同 d /
同序列前端 / 同预算。

任务：K 个三元组 (e,r,v)（符号互不同）扁平成 3K 序列；查询某三元组的 (e,r)，预测 v。容量受限 d=32 下：
flat 单向量装不下 K 个三元组 → 取回糊；CAPCW 内容寻址把三元组分到 slot → 按 (e,r) 取回 v。

判据（先写死）：多 K(≥6) CAPCW − flat ≥ +0.10 → CAPCW 把"内容寻址取回"扩到**从序列读出的关系**（更贴近
真语言的输入形态），不只对显式 pair 有效。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_sequence_relation_eval --run
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

REPORT_JSON = os.path.join("docs", "reports", "capcw_sequence_relation_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_sequence_relation_eval.md")


def gen_seq_relations(n_sym, K, n, seed):
    """K 个三元组 (e,r,v) 符号互不同，扁平成 3K 序列；查询某三元组 (e,r)→v。返回 seq(N,3K)、qe,qr、y。"""
    rng = np.random.default_rng(seed)
    seq = np.zeros((n, 3 * K), dtype=np.int64)
    qe = np.zeros((n,), dtype=np.int64)
    qr = np.zeros((n,), dtype=np.int64)
    y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        syms = rng.choice(n_sym, size=3 * K, replace=False)      # 三元组符号互不同（无歧义）
        seq[i] = syms
        j = int(rng.integers(K))                                  # 查询第 j 个三元组
        qe[i], qr[i], y[i] = int(syms[3 * j]), int(syms[3 * j + 1]), int(syms[3 * j + 2])
    return seq, qe, qr, y


class SeqRelModel(nn.Module):
    """序列前端读三元组 + (flat/capcw) world + 按 (e,r) 内容寻址取回 v。唯一变量=world_mode。"""

    def __init__(self, n_sym, d, n_slots, iters, world_mode, K):
        super().__init__()
        self.sym_emb = nn.Embedding(n_sym, d)
        self.role_emb = nn.Embedding(3, d)                       # 三元组内角色：0=实体 1=关系 2=值
        self.triple = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))   # 读出一个三元组
        self.qkey = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))     # (e,r)→查询键
        self.world_mode, self.d, self.K = world_mode, d, K
        if world_mode == "capcw":
            self.ws = PCWorkspace(dim=d, n_slots=n_slots, iters=iters)
        self.to_q = nn.Linear(d, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_sym))

    def forward(self, seq, qe, qr):
        B = seq.shape[0]
        roles = torch.arange(3 * self.K, device=seq.device) % 3                       # 0,1,2,0,1,2,...
        pos = self.sym_emb(seq) + self.role_emb(roles)[None]                          # (B,3K,d) 序列位置表示
        triples = self.triple(pos.view(B, self.K, 3 * self.d))                        # (B,K,d) 读出 K 个三元组
        world = self.ws(triples).slots if self.world_mode == "capcw" else triples.mean(dim=1, keepdim=True)
        q = self.to_q(self.qkey(torch.cat([self.sym_emb(qe), self.sym_emb(qr)], dim=-1)))
        score = torch.einsum("bmd,bd->bm", world, q) / math.sqrt(self.d)
        read = (world * score.softmax(dim=1).unsqueeze(-1)).sum(dim=1)
        return self.head(read)


def _train_eval(world_mode, n_sym, K, d, n_slots, iters, n_train, n_test, epochs, lr, batch, device, seed):
    torch.manual_seed(seed)
    model = SeqRelModel(n_sym, d, n_slots, iters, world_mode, K).to(device)
    seq, qe, qr, y = (torch.tensor(a, device=device) for a in gen_seq_relations(n_sym, K, n_train, seed))
    tseq, tqe, tqr, ty = (torch.tensor(a, device=device) for a in gen_seq_relations(n_sym, K, n_test, seed + 5000))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(y)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(seq[idx], qe[idx], qr[idx]), y[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return float((model(tseq, tqe, tqr).argmax(-1) == ty).float().mean())


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    K_list = [int(x) for x in args.k_list.split(",")]
    print(f"[seqrel] device={device} n_sym={args.n_sym} K_list={K_list} d={args.d} seeds={args.seeds}", flush=True)
    results = {}
    for K in K_list:
        n_slots = max(args.n_slots, K + 1)
        flat_a, capcw_a = [], []
        for si in range(args.seeds):
            seed = args.seed + si
            common = dict(n_sym=args.n_sym, K=K, d=args.d, n_slots=n_slots, iters=args.iters,
                          n_train=args.n_train, n_test=args.n_test, epochs=args.epochs, lr=args.lr,
                          batch=args.batch, device=device, seed=seed)
            flat_a.append(_train_eval("flat", **common))
            capcw_a.append(_train_eval("capcw", **common))
        results[K] = {"flat": round(float(np.mean(flat_a)), 4), "capcw": round(float(np.mean(capcw_a)), 4)}
        r = results[K]
        print(f"[seqrel] K={K} flat={r['flat']:.3f} capcw={r['capcw']:.3f} delta={r['capcw']-r['flat']:+.3f} "
              f"(rand={1.0/args.n_sym:.3f})", flush=True)

    hiK = [K for K in K_list if K >= 6] or K_list
    delta = round(float(np.mean([results[K]["capcw"] - results[K]["flat"] for K in hiK])), 4)
    if delta >= 0.10:
        verdict = (f"PASS: CAPCW 把内容寻址取回扩到**从序列读出的关系**——多 K(≥6) CAPCW−flat {delta:+.4f}（≥+0.10）。"
                   f"即关系以扁平 token 序列给出（更贴近真语言）时，slot 工作空间仍按 (实体·关系) 内容寻址取回值，"
                   f"显著胜单向量。内容寻址不限于显式 pair。")
    elif delta >= 0.05:
        verdict = f"PARTIAL: 序列读关系上 CAPCW 有正增益 {delta:+.4f} 但 <0.10。"
    else:
        verdict = f"FAIL: 序列读关系上 CAPCW 未明显胜 flat（{delta:+.4f}）。诚实记录。"

    result = {
        "task": "read (entity,relation,value) triples from a FLAT token sequence, retrieve value by (e,r) (flat vs CAPCW)",
        "design": "var = world structure (flat single-vector vs CAPCW slots); same d/sequence-frontend/budget; sweep K.",
        "config": {"n_sym": args.n_sym, "K_list": K_list, "d": args.d, "epochs": args.epochs, "seeds": args.seeds,
                   "random_baseline": round(1.0 / args.n_sym, 4)},
        "by_K": results, "capcw_minus_flat_highK": delta,
        "verdict": verdict,
        "note": "关系以扁平 3K token 序列给出（[e r v]×K），序列前端(符号+角色嵌入+三元组读出)读出 K 个三元组；"
                "world=flat 均值 / CAPCW slot；按 (查询实体·关系) 内容寻址取回 v。唯一变量=world 结构。更贴近真语言输入。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# CAPCW 从序列直接读关系：(实体·关系→值) 三元组取回（flat vs CAPCW）",
        "",
        f"- 判定：**{verdict}**",
        f"- 任务：K 个三元组扁平成 3K 序列，按 (查询实体·关系) 取回 v；n_sym={args.n_sym}，随机 {1.0/args.n_sym:.3f}",
        "",
        "| K（三元组数） | flat（单向量） | CAPCW（slot） | delta |",
        "|---:|---:|---:|---:|",
    ]
    for K in K_list:
        r = results[K]
        lines.append(f"| {K} | {r['flat']:.3f} | {r['capcw']:.3f} | {r['capcw']-r['flat']:+.3f} |")
    lines += ["", f"- 多 K(≥6) CAPCW−flat = **{delta:+.4f}**", "", f"- 说明：{result['note']}"]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[seqrel] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW read relations from a flat token sequence (flat vs CAPCW).")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-sym", type=int, default=36)    # 须 ≥ 3*max(K)（每例 3K 个互不同三元组符号）
    ap.add_argument("--k-list", default="3,6,9")
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--n-slots", type=int, default=12)
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
        print("[seqrel] dry-run：未训练。从扁平 token 序列读 (实体·关系→值) 三元组并取回，flat vs CAPCW。")
        print("[seqrel] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
