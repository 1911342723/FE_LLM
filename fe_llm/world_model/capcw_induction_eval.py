# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_induction_eval.py
==========================================
CAPCW 真 in-context 语言机制验证：induction（归纳头）。见 `docs/FE-LLM核心引擎构想.md` 第 13 节。

Part 1 指出 CAPCW 的价值专属"真 in-context 绑定"（值不可由键预测）。语言里这个机制的原型就是
**induction head**——Transformer in-context learning 的基石：序列里出现过 [A][B]，后面再遇到 [A]
就应预测 [B]。每个序列的 A→B 配对**随机**（不可记忆，必须现场在上下文里绑定）。

任务（序列形态，比键值 pair 更像语言）：
- 序列 = 若干 (a,b) bigram 对随机拼接 + 填充 token，末尾是 cue = 某个出现过的 a；目标=该 a 对应的 b。
- 与 capcw_binding 的区别：这里是**序列**（token 流 + 位置），induction 要求"找到上文 a 的下一个 token"。

对照（唯一变量=序列聚合结构）：
- flat        ：序列 token 嵌入均值池化（单向量）+ cue 读出；
- CAPCW       ：序列 token 经 PCWorkspace 聚成 slot 工作空间 + cue 内容寻址读出。
容量受限 d。判据：CAPCW − flat ≥ +0.10 → CAPCW 在真 in-context 语言机制(induction)上胜单向量。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_induction_eval --run
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


def gen_induction(n_sym, n_pairs, seq_len, n, seed):
    """序列：n_pairs 个随机 (a,b) 相邻 bigram 散布在长度 seq_len 序列里 + cue(=某 a) 在末位；y=该 a 的 b。

    符号表 [0..n_sym)；位置嵌入另加。返回 ids:(N,seq_len)、cue:(N,)、y:(N,)。
    """
    rng = np.random.default_rng(seed)
    ids = np.zeros((n, seq_len), dtype=np.int64)
    cue = np.zeros((n,), dtype=np.int64)
    y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        syms = rng.choice(n_sym, size=2 * n_pairs, replace=False)
        a_list = syms[:n_pairs]
        b_list = syms[n_pairs:2 * n_pairs]
        # 在前 seq_len-1 个槽里放 n_pairs 个相邻 bigram（不重叠）。
        slots = seq_len - 1
        starts = rng.choice(slots - 1, size=n_pairs, replace=False)
        seq = [int(rng.integers(n_sym)) for _ in range(seq_len)]  # filler 噪声
        for j, st in enumerate(starts):
            seq[st] = int(a_list[j])
            seq[st + 1] = int(b_list[j])
        qi = int(rng.integers(n_pairs))
        seq[seq_len - 1] = int(a_list[qi])      # cue 放末位
        ids[i] = seq
        cue[i] = int(a_list[qi])
        y[i] = int(b_list[qi])
    return ids, cue, y


class SeqEncoder(nn.Module):
    def __init__(self, n_sym, seq_len, d):
        super().__init__()
        self.emb = nn.Embedding(n_sym, d)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d))
        nn.init.normal_(self.pos, std=0.02)
        self.cue_emb = nn.Embedding(n_sym, d)

    def tokens(self, ids):
        return self.emb(ids) + self.pos[:, : ids.shape[1]]


class FlatInduction(nn.Module):
    def __init__(self, n_sym, seq_len, d):
        super().__init__()
        self.enc = SeqEncoder(n_sym, seq_len, d)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, n_sym))

    def forward(self, ids, cue):
        tok = self.enc.tokens(ids)
        world = tok.mean(dim=1)
        q = self.enc.cue_emb(cue)
        return self.head(torch.cat([world, q], dim=-1))


class CAPCWInduction(nn.Module):
    def __init__(self, n_sym, seq_len, d, n_slots, iters):
        super().__init__()
        self.enc = SeqEncoder(n_sym, seq_len, d)
        self.ws = PCWorkspace(dim=d, n_slots=n_slots, iters=iters)
        self.to_q = nn.Linear(d, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_sym))
        self.d = d

    def forward(self, ids, cue):
        tok = self.enc.tokens(ids)
        slots = self.ws(tok).slots
        q = self.to_q(self.enc.cue_emb(cue))
        score = torch.einsum("bmd,bd->bm", slots, q) / math.sqrt(self.d)
        attn = score.softmax(dim=1).unsqueeze(-1)
        read = (slots * attn).sum(dim=1)
        return self.head(read)


def train_eval(model, train, test, device, *, epochs, lr, batch, seed):
    torch.manual_seed(seed)
    ids, cue, y = (torch.tensor(t, device=device) for t in train)
    tids, tcue, ty = (torch.tensor(t, device=device) for t in test)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(y)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(ids[idx], cue[idx]), y[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(tids, tcue).argmax(-1)
        acc = float((pred == ty).float().mean())
    return acc


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    print(f"[induction] device={device} n_sym={args.n_sym} n_pairs={args.n_pairs} seq_len={args.seq_len} d={args.d} seeds={args.seeds}", flush=True)
    flat_accs, capcw_accs = [], []
    for si in range(args.seeds):
        seed = args.seed + si
        train = gen_induction(args.n_sym, args.n_pairs, args.seq_len, args.n_train, seed)
        test = gen_induction(args.n_sym, args.n_pairs, args.seq_len, args.n_test, seed + 5000)
        n_slots = max(args.n_slots, args.n_pairs + 1)
        flat = FlatInduction(args.n_sym, args.seq_len, args.d)
        capcw = CAPCWInduction(args.n_sym, args.seq_len, args.d, n_slots=n_slots, iters=args.iters)
        common = dict(device=device, epochs=args.epochs, lr=args.lr, batch=args.batch, seed=seed)
        flat_accs.append(train_eval(flat, train, test, **common))
        capcw_accs.append(train_eval(capcw, train, test, **common))
        print(f"[induction] seed={seed} flat={flat_accs[-1]:.3f} capcw={capcw_accs[-1]:.3f}", flush=True)

    flat_m, capcw_m = float(np.mean(flat_accs)), float(np.mean(capcw_accs))
    delta = round(capcw_m - flat_m, 4)
    verdict = ("PASS: CAPCW 在真 in-context 语言机制(induction)上明显胜单向量" if delta >= 0.10
               else ("PARTIAL: 正向但偏弱" if delta >= 0.03 else "FAIL: induction 上未明显胜"))

    result = {
        "task": "induction (in-context language mechanism): ...A B ... A -> predict B",
        "config": {"n_sym": args.n_sym, "n_pairs": args.n_pairs, "seq_len": args.seq_len, "d": args.d,
                   "epochs": args.epochs, "seeds": args.seeds, "random_baseline": round(1.0 / args.n_sym, 4)},
        "flat_acc": round(flat_m, 4), "capcw_acc": round(capcw_m, 4), "delta_capcw_minus_flat": delta,
        "verdict": verdict,
        "note": "induction head 是 Transformer in-context learning 的基石；每序列 A→B 随机配对(不可记忆)，"
                "必须现场绑定。唯一变量=序列聚合结构(单向量 vs slot 工作空间)。这是 Part1 指出的'真 in-context 语言任务'。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# CAPCW · 真 in-context 语言机制：induction（归纳头）",
        "",
        f"- 判定：**{verdict}**",
        f"- 任务：序列 ...A B ... A → 预测 B（每序列 A→B 随机配对，不可记忆）；n_sym={args.n_sym}, n_pairs={args.n_pairs}, seq_len={args.seq_len}, d={args.d}；随机基线 {1.0/args.n_sym:.3f}",
        "",
        "| 序列聚合结构 | induction accuracy |",
        "|---|---:|",
        f"| flat（单向量均值池化） | {flat_m:.4f} |",
        f"| CAPCW（slot 工作空间） | {capcw_m:.4f} |",
        "",
        f"- delta（CAPCW − flat）= **{delta:+.4f}**",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[induction] === 裁决 ===", flush=True)
    print(f"{verdict}  flat={flat_m:.3f} capcw={capcw_m:.3f} (delta {delta:+.4f})", flush=True)
    return result


REPORT_JSON = os.path.join("docs", "reports", "capcw_induction_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_induction_eval.md")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW induction (in-context language mechanism) eval.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-sym", type=int, default=20)
    ap.add_argument("--n-pairs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--n-slots", type=int, default=6)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-json", default=REPORT_JSON)
    ap.add_argument("--report-md", default=REPORT_MD)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print("[induction] dry-run：未训练。真 in-context 语言机制 induction 上 flat vs CAPCW。")
        print("[induction] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
