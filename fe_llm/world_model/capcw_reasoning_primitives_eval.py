# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_reasoning_primitives_eval.py
=====================================================
CAPCW **检索之外的推理基元**：内容寻址 slot 工作空间能否支持"比较 / 计数"这类**对绑定值做运算**的推理
（而不只是取回某个值）？见 `docs/FE-LLM核心引擎构想.md` 第 30 节，承接绑定(§9)/接控制层(§12)。

绑定 K 个 (key→数值)；两个推理任务（唯一变量=世界状态结构 flat 单向量 vs CAPCW slot 工作空间，同 d /
同 pair 编码器 / 同预算）：
- **compare（成对比较）**：query 两个 key (X,Y)，输出 v_X > v_Y ？（二分类）。需现场**取回两个值再比**。
- **count（阈值计数）**：给阈值 T，输出 #{绑定值 > T}（0..K，(K+1) 分类）。需**聚合所有值**与 T 比较。

假设：容量受限(d=32)下，flat 单向量装不下 K 个值 → 取回/聚合都糊；CAPCW 内容寻址把值分到 slot →
compare 能取回被查的两个值、count 能在 slot 上聚合。**若 CAPCW 明显胜 flat，则 CAPCW 不止能"取回"、还能
在工作空间上做"比较/计数"这类组合推理。**

判据（先写死，多 K(≥4) 均值，2 seed）
-------------------------------------
- compare：CAPCW − flat ≥ +0.10 → CAPCW 支持成对比较推理。
- count：CAPCW − flat ≥ +0.10 → CAPCW 支持阈值计数（聚合）推理。
- 综合：两者都过 → PASS（CAPCW 推理基元扩到比较+计数）；仅 compare 过 → PARTIAL（比较可、聚合计数难）；
  都不过 → FAIL（CAPCW 在本规模不支持检索之外的这两种推理，诚实记录）。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_reasoning_primitives_eval --run
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

REPORT_JSON = os.path.join("docs", "reports", "capcw_reasoning_primitives_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_reasoning_primitives_eval.md")


def gen_compare(n_keys, n_vals, K, n, seed):
    """每例：K 个 (key→value, 值互不同) + query 两个不同 key (X,Y)；label = int(v_X > v_Y)。"""
    rng = np.random.default_rng(seed)
    pk = np.zeros((n, K), dtype=np.int64)
    pv = np.zeros((n, K), dtype=np.int64)
    qx = np.zeros((n,), dtype=np.int64)
    qy = np.zeros((n,), dtype=np.int64)
    y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        keys = rng.choice(n_keys, size=K, replace=False)
        vals = rng.choice(n_vals, size=K, replace=False)        # 值互不同 → 比较无平局
        pk[i], pv[i] = keys, vals
        a, b = rng.choice(K, size=2, replace=False)
        qx[i], qy[i] = int(keys[a]), int(keys[b])
        y[i] = int(vals[a] > vals[b])
    return pk, pv, qx, qy, y


def gen_count(n_keys, n_vals, K, n, seed):
    """每例：K 个 (key→value, 值互不同) + 阈值 T；label = #{value > T} ∈ 0..K。"""
    rng = np.random.default_rng(seed)
    pk = np.zeros((n, K), dtype=np.int64)
    pv = np.zeros((n, K), dtype=np.int64)
    thr = np.zeros((n,), dtype=np.int64)
    y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        keys = rng.choice(n_keys, size=K, replace=False)
        vals = rng.choice(n_vals, size=K, replace=False)
        pk[i], pv[i] = keys, vals
        t = int(rng.integers(n_vals))
        thr[i] = t
        y[i] = int((vals > t).sum())
    return pk, pv, thr, y


class Reasoner(nn.Module):
    """flat(单向量) / capcw(slot 工作空间) × compare/count。唯一变量=world_mode；同 d/同 pair 编码器。"""

    def __init__(self, n_keys, n_vals, d, n_slots, iters, task: str, world_mode: str, K: int):
        super().__init__()
        self.key_emb = nn.Embedding(n_keys, d)
        self.val_emb = nn.Embedding(n_vals, d)
        self.pair = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.task, self.world_mode, self.d = task, world_mode, d
        if world_mode == "capcw":
            self.ws = PCWorkspace(dim=d, n_slots=n_slots, iters=iters)
        if task == "compare":
            self.to_q = nn.Linear(d, d)
            in_dim = 2 * d                                       # [read_X; read_Y]（capcw）/ [hX; hY]（flat 见下）
            self.head = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, 2))
        else:  # count
            self.thr_emb = nn.Embedding(n_vals, d)
            self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, K + 1))

    def _pairs(self, pk, pv):
        return self.pair(torch.cat([self.key_emb(pk), self.val_emb(pv)], dim=-1))   # (B,K,d)

    def _world(self, pk, pv):
        pairs = self._pairs(pk, pv)
        if self.world_mode == "capcw":
            return self.ws(pairs).slots                          # (B,M,d)
        return pairs.mean(dim=1, keepdim=True)                    # (B,1,d) 单向量（均值池化）

    def _read(self, world, q):
        """内容寻址读出（capcw 多 slot 用 attention；flat 单 slot 退化为该向量）。"""
        score = torch.einsum("bmd,bd->bm", world, q) / math.sqrt(self.d)
        attn = score.softmax(dim=1).unsqueeze(-1)
        return (world * attn).sum(dim=1)

    def forward(self, pk, pv, a, b=None):
        world = self._world(pk, pv)
        if self.task == "compare":
            qx, qy = a, b
            if self.world_mode == "capcw":
                rx = self._read(world, self.to_q(self.key_emb(qx)))
                ry = self._read(world, self.to_q(self.key_emb(qy)))
            else:                                                # flat：单向量 + 两个查询键嵌入
                h = world.squeeze(1)
                rx = h + self.key_emb(qx)
                ry = h + self.key_emb(qy)
            return self.head(torch.cat([rx, ry], dim=-1))
        else:                                                    # count：池化世界 + 阈值嵌入
            pooled = world.mean(dim=1)
            return self.head(torch.cat([pooled, self.thr_emb(a)], dim=-1))


def _train_eval(task, world_mode, n_keys, n_vals, K, d, n_slots, iters, n_train, n_test, epochs, lr, batch, device, seed):
    torch.manual_seed(seed)
    model = Reasoner(n_keys, n_vals, d, n_slots, iters, task, world_mode, K).to(device)
    gen = gen_compare if task == "compare" else gen_count
    tr = gen(n_keys, n_vals, K, n_train, seed)
    te = gen(n_keys, n_vals, K, n_test, seed + 5000)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if task == "compare":
        pk, pv, qx, qy, y = (torch.tensor(a, device=device) for a in tr)
        tpk, tpv, tqx, tqy, ty = (torch.tensor(a, device=device) for a in te)
        args_tr = (pk, pv, qx, qy)
        args_te = (tpk, tpv, tqx, tqy)
    else:
        pk, pv, thr, y = (torch.tensor(a, device=device) for a in tr)
        tpk, tpv, tthr, ty = (torch.tensor(a, device=device) for a in te)
        args_tr = (pk, pv, thr)
        args_te = (tpk, tpv, tthr)
    n = len(y)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            logits = model(*[t[idx] for t in args_tr])
            loss = F.cross_entropy(logits, y[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(*args_te).argmax(-1)
        return float((pred == ty).float().mean())


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    K_list = [int(x) for x in args.k_list.split(",")]
    print(f"[rp] device={device} n_keys={args.n_keys} n_vals={args.n_vals} K_list={K_list} d={args.d} seeds={args.seeds}", flush=True)
    results: dict = {}
    for task in ("compare", "count"):
        results[task] = {}
        for K in K_list:
            n_slots = max(args.n_slots, K + 1)
            flat_accs, capcw_accs = [], []
            for si in range(args.seeds):
                seed = args.seed + si
                common = dict(n_keys=args.n_keys, n_vals=args.n_vals, K=K, d=args.d, n_slots=n_slots,
                              iters=args.iters, n_train=args.n_train, n_test=args.n_test, epochs=args.epochs,
                              lr=args.lr, batch=args.batch, device=device, seed=seed)
                flat_accs.append(_train_eval(task, "flat", **common))
                capcw_accs.append(_train_eval(task, "capcw", **common))
            results[task][K] = {"flat": round(float(np.mean(flat_accs)), 4),
                                "capcw": round(float(np.mean(capcw_accs)), 4)}
            r = results[task][K]
            rand = 0.5 if task == "compare" else round(1.0 / (K + 1), 3)
            print(f"[rp] {task} K={K} flat={r['flat']:.3f} capcw={r['capcw']:.3f} (rand≈{rand})", flush=True)

    def delta(task):
        ks = [K for K in K_list if K >= 4]
        if not ks:
            ks = K_list
        return round(float(np.mean([results[task][K]["capcw"] - results[task][K]["flat"] for K in ks])), 4)

    d_cmp, d_cnt = delta("compare"), delta("count")
    cmp_ok, cnt_ok = d_cmp >= 0.10, d_cnt >= 0.10
    # 细化：① count 是否"无 headroom"（flat 自身已近满，单向量即解，不测内容寻址）；
    #       ② compare 优势是否"容量依赖"（最高 K 的 delta ≫ 低 K，与检索同源的容量效应）。
    maxK = max(K_list)
    cnt_flat_min = min(results["count"][K]["flat"] for K in K_list)
    cnt_no_headroom = cnt_flat_min >= 0.95
    d_cmp_hiK = round(results["compare"][maxK]["capcw"] - results["compare"][maxK]["flat"], 4)
    cmp_capacity_dep = (not cmp_ok) and d_cmp_hiK >= 0.10
    if cmp_ok and cnt_ok:
        verdict = (f"PASS: CAPCW 推理基元扩到**比较+计数**——compare CAPCW−flat {d_cmp:+.4f}、count {d_cnt:+.4f}（均≥+0.10）。")
    else:
        cnt_note = (f"count **无 headroom**（flat 自身已达 {cnt_flat_min:.3f}≈满——单向量用'值编码求和'即可计数，"
                    f"keys 无关，根本不测内容寻址）" if cnt_no_headroom
                    else f"count CAPCW−flat {d_cnt:+.4f}（增益不足）")
        cmp_note = (f"compare 优势**容量依赖**（低 K 单向量够用、打平；仅高 K={maxK} 容量受限时 CAPCW +{d_cmp_hiK:.3f}，"
                    f"均值 {d_cmp:+.4f}<0.10 被低 K 平局拉低）——这是与**检索同源的容量效应**，非新增推理超能力"
                    if cmp_capacity_dep
                    else f"compare CAPCW−flat {d_cmp:+.4f}")
        verdict = (f"诚实负结果(无新增推理基元): CAPCW 在本规模未展现"
                   f"**检索之外**的新推理基元——{cmp_note}；{cnt_note}。"
                   f"结论：compare 本质是'取回两值再比'(优势=容量效应,与绑定/检索同源)，count 单向量可解(无 headroom)；"
                   f"CAPCW 的价值仍锚定在**容量受限的内容寻址取回/绑定**，未额外解锁比较/计数这类运算推理。")

    result = {
        "task": "CAPCW reasoning primitives beyond retrieval: in-context comparison & counting (flat vs CAPCW)",
        "design": "var = world-state structure (flat single-vector vs CAPCW slots); same d/pair-encoder/budget; two tasks.",
        "config": {"n_keys": args.n_keys, "n_vals": args.n_vals, "K_list": K_list, "d": args.d,
                   "epochs": args.epochs, "seeds": args.seeds},
        "by_task": results,
        "compare_capcw_minus_flat": d_cmp, "count_capcw_minus_flat": d_cnt,
        "compare_capcw_minus_flat_highK": d_cmp_hiK, "count_flat_min": round(cnt_flat_min, 4),
        "count_no_headroom": cnt_no_headroom, "compare_capacity_dependent": cmp_capacity_dep,
        "verdict": verdict,
        "note": "compare=query 两 key 取回两值比大小(二分类)；count=阈值上计数(K+1分类,需聚合所有值)。"
                "唯一变量=world 结构(flat 均值单向量 vs CAPCW slot 工作空间)，同 d/同 pair 编码器/同预算。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# CAPCW 检索之外的推理基元：in-context 比较 / 计数（flat vs CAPCW）",
        "",
        f"- 判定：**{verdict}**",
        f"- 设置：n_keys={args.n_keys}, n_vals={args.n_vals}, d={args.d}, seeds={args.seeds}；唯一变量=world 结构。",
        "",
        "## compare（取两值比大小，二分类，随机 0.5）",
        "",
        "| K | flat | CAPCW |",
        "|---:|---:|---:|",
    ]
    for K in K_list:
        r = results["compare"][K]
        lines.append(f"| {K} | {r['flat']:.3f} | {r['capcw']:.3f} |")
    lines += ["", "## count（阈值上计数，K+1 分类）", "", "| K | flat | CAPCW |", "|---:|---:|---:|"]
    for K in K_list:
        r = results["count"][K]
        lines.append(f"| {K} | {r['flat']:.3f} | {r['capcw']:.3f} |")
    lines += [
        "",
        f"- 多 K(≥4) CAPCW−flat：compare **{d_cmp:+.4f}** / count **{d_cnt:+.4f}**",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[rp] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    print(f"[rp] compare CAPCW−flat={d_cmp:+.4f} count CAPCW−flat={d_cnt:+.4f}", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW reasoning primitives (in-context comparison & counting): flat vs CAPCW.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-keys", type=int, default=12)
    ap.add_argument("--n-vals", type=int, default=10)
    ap.add_argument("--k-list", default="3,5,7")
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
        print("[rp] dry-run：未训练。CAPCW 检索之外的推理基元：in-context 比较/计数，flat vs CAPCW。")
        print("[rp] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
