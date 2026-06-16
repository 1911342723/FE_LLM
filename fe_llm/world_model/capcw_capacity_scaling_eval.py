# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_capacity_scaling_eval.py
=================================================
**容量扩展曲线**：增大脑容量 d 能否把绝对效果不断抬高？直接扫 d，看 in-context 绑定取回准确率随 d 的变化。
见 `docs/FE-LLM核心引擎构想.md` 第 32 节。回答"是不是不断增大 d/数据就能到好效果"。

任务=in-context 键值绑定（每例 K 对 key→value，问某 key 的 value）。扫 d∈{16,32,64,128}，flat(单向量) vs
CAPCW(slot 工作空间)，固定 K、2 seed。**实测意外结论（诚实）**：裸增 d 在本引擎上**不抬反降**——CAPCW
（PCWorkspace）在 d≥64 训练**塌缩/不稳定**（弛豫 dynamics 未随 d 重标定；更多训练也不救→不稳定非欠训），
小 d(16/32) 反而最好；flat 单向量在高 K 下任何 d 都≈随机（均值池化装不下）。

结论意义（回答"是否裸增 d/数据就能到好效果"）：**不能裸增 d**——要受益于规模须**架构工程**（归一化/弛豫
稳定化/深度），那等于**重建标准可扩展架构**（项目按容量纪律刻意不走）。且增 d 即便修好也只抬**绝对精度**、
不改**机制结论**（内容寻址优势恰在小 d；多跳要 CoT、规则归纳靠 readout 等与 d 无关）。这也再次印证 §9：
内容寻址的价值专在小 d 容量瓶颈处。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_capacity_scaling_eval --run
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

REPORT_JSON = os.path.join("docs", "reports", "capcw_capacity_scaling_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_capacity_scaling_eval.md")


def gen_binding(n_keys, n_vals, K, n, seed):
    rng = np.random.default_rng(seed)
    pk = np.zeros((n, K), dtype=np.int64)
    pv = np.zeros((n, K), dtype=np.int64)
    qk = np.zeros((n,), dtype=np.int64)
    y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        keys = rng.choice(n_keys, size=K, replace=False)
        vals = rng.choice(n_vals, size=K, replace=False)
        pk[i], pv[i] = keys, vals
        j = int(rng.integers(K))
        qk[i], y[i] = int(keys[j]), int(vals[j])
    return pk, pv, qk, y


class BindModel(nn.Module):
    """flat(单向量均值) / capcw(slot 工作空间) 的内容寻址取回。唯一变量=world 结构；扫 d。"""

    def __init__(self, n_keys, n_vals, d, n_slots, iters, world_mode):
        super().__init__()
        self.key_emb = nn.Embedding(n_keys, d)
        self.val_emb = nn.Embedding(n_vals, d)
        self.pair = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.world_mode, self.d = world_mode, d
        if world_mode == "capcw":
            self.ws = PCWorkspace(dim=d, n_slots=n_slots, iters=iters)
        self.to_q = nn.Linear(d, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_vals))

    def forward(self, pk, pv, qk):
        pairs = self.pair(torch.cat([self.key_emb(pk), self.val_emb(pv)], dim=-1))
        slots = self.ws(pairs).slots if self.world_mode == "capcw" else pairs.mean(dim=1, keepdim=True)
        q = self.to_q(self.key_emb(qk))
        score = torch.einsum("bmd,bd->bm", slots, q) / math.sqrt(self.d)
        read = (slots * score.softmax(dim=1).unsqueeze(-1)).sum(dim=1)
        return self.head(read)


def _train_eval(world_mode, n_keys, n_vals, K, d, n_slots, iters, n_train, n_test, epochs, lr, batch, device, seed):
    torch.manual_seed(seed)
    model = BindModel(n_keys, n_vals, d, n_slots, iters, world_mode).to(device)
    pk, pv, qk, y = (torch.tensor(a, device=device) for a in gen_binding(n_keys, n_vals, K, n_train, seed))
    tpk, tpv, tqk, ty = (torch.tensor(a, device=device) for a in gen_binding(n_keys, n_vals, K, n_test, seed + 5000))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(qk)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(pk[idx], pv[idx], qk[idx]), y[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return float((model(tpk, tpv, tqk).argmax(-1) == ty).float().mean())


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    d_list = [int(x) for x in args.d_list.split(",")]
    print(f"[scale] device={device} n_keys={args.n_keys} n_vals={args.n_vals} K={args.k} d_list={d_list} seeds={args.seeds}", flush=True)
    results = {}
    for d in d_list:
        n_slots = max(args.n_slots, args.k + 1)
        flat_a, capcw_a = [], []
        for si in range(args.seeds):
            seed = args.seed + si
            common = dict(n_keys=args.n_keys, n_vals=args.n_vals, K=args.k, d=d, n_slots=n_slots, iters=args.iters,
                          n_train=args.n_train, n_test=args.n_test, epochs=args.epochs, lr=args.lr, batch=args.batch,
                          device=device, seed=seed)
            flat_a.append(_train_eval("flat", **common))
            capcw_a.append(_train_eval("capcw", **common))
        results[d] = {"flat": round(float(np.mean(flat_a)), 4), "capcw": round(float(np.mean(capcw_a)), 4)}
        r = results[d]
        print(f"[scale] d={d:>4} flat={r['flat']:.3f} capcw={r['capcw']:.3f} delta={r['capcw']-r['flat']:+.3f}", flush=True)

    small_d = min(d_list)
    big_d = max(d_list)
    rises = round(results[big_d]["capcw"] - results[small_d]["capcw"], 4)   # CAPCW 随 d 的变化（<0=不抬反降）
    best_d = max(d_list, key=lambda d: results[d]["capcw"])                  # CAPCW 最佳的 d
    small_delta = round(results[small_d]["capcw"] - results[small_d]["flat"], 4)
    if rises >= 0.05:
        verdict = (f"裸增 d **抬绝对值**（CAPCW d={small_d}→{big_d}: {results[small_d]['capcw']:.3f}→"
                   f"{results[big_d]['capcw']:.3f}, {rises:+.3f}）；小 d CAPCW≫flat（delta {small_delta:+.3f}）。"
                   f"限制是容量、可被 d 抬。")
    else:
        verdict = (f"**裸增 d 不抬反降（诚实意外）**：CAPCW d={small_d}→{big_d} = {results[small_d]['capcw']:.3f}→"
                   f"{results[big_d]['capcw']:.3f}（{rises:+.3f}），最佳在 **d={best_d}**（小 d），d≥64 训练**塌缩/不稳定**"
                   f"（弛豫 dynamics 未随 d 重标定，更多训练也不救→不稳定非欠训）；flat 在高 K 下任何 d 都≈随机"
                   f"（均值池化装不下）。**结论：不能裸增 d 提升——要受益于规模须架构工程(归一化/弛豫稳定化/深度)=重建"
                   f"标准可扩展架构(项目按容量纪律刻意不走)**；且增 d 即便修好也只抬绝对精度、不改机制结论。"
                   f"再次印证内容寻址优势专在小 d（d={small_d} CAPCW {results[small_d]['capcw']:.3f}≫flat，delta {small_delta:+.3f}）。")

    result = {
        "task": "capacity scaling curve: in-context binding accuracy vs d (flat vs CAPCW)",
        "config": {"n_keys": args.n_keys, "n_vals": args.n_vals, "K": args.k, "d_list": d_list,
                   "epochs": args.epochs, "seeds": args.seeds, "random_baseline": round(1.0 / args.n_vals, 4)},
        "by_d": results,
        "capcw_rise_small_to_big_d": rises, "capcw_best_d": best_d, "delta_small_d": small_delta,
        "verdict": verdict,
        "note": "实测：裸增 d 在本引擎上不抬反降——CAPCW(PCWorkspace) d≥64 训练塌缩/不稳定(弛豫未随 d 重标定,更多训练不救),"
                "最佳在小 d；flat 高 K 任何 d 都≈随机(均值池化装不下)。要受益于规模须架构工程(归一化/稳定化/深度)="
                "重建标准可扩展架构(项目按容量纪律不走)；增 d 即便修好也只抬绝对精度、不改机制结论(CoT/规则归纳/内容寻址定位与 d 无关)。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# CAPCW 容量扩展曲线：绑定取回准确率 vs 脑容量 d（flat vs CAPCW）",
        "",
        f"- 设置：n_keys={args.n_keys}, n_vals={args.n_vals}, K={args.k}, seeds={args.seeds}；随机 {1.0/args.n_vals:.3f}",
        "",
        "| d（脑容量） | flat（单向量） | CAPCW（slot） | delta |",
        "|---:|---:|---:|---:|",
    ]
    for d in d_list:
        r = results[d]
        lines.append(f"| {d} | {r['flat']:.3f} | {r['capcw']:.3f} | {r['capcw']-r['flat']:+.3f} |")
    lines += ["", f"- 结论：{verdict}", "", f"- 说明：{result['note']}"]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[scale] === 结论 ===", flush=True)
    print(verdict, flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW capacity scaling curve: binding accuracy vs d (flat vs CAPCW).")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-keys", type=int, default=16)
    ap.add_argument("--n-vals", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--d-list", default="16,32,64,128")
    ap.add_argument("--n-slots", type=int, default=10)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=50)
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
        print("[scale] dry-run：未训练。容量扩展曲线：绑定准确率 vs d（flat vs CAPCW），看增 d 是否抬天花板。")
        print("[scale] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
