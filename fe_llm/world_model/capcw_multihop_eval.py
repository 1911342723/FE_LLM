# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_multihop_eval.py
=========================================
CAPCW 多跳/链式 in-context 推理：slot 工作空间能否在工作空间里**链式组合检索**。
见 `docs/FE-LLM核心引擎构想.md` 第 17 节（序列相邻算子）之后的下一关。

背景
----
序列相邻算子已让 CAPCW 做 1 跳 induction（cue=A → 紧跟的 B）。语言/推理里更关键的是**多跳**：
A→B、B→C，问 A 答 C（值是下一个绑定的键，需现场把多个绑定**链式组合**）。这是 in-context
组合推理的原型，也是 Transformer 靠**多层/多头**做多跳的能力。

任务（链式 in-context 绑定，H 跳）
----------------------------------
- 链：c_0→c_1→…→c_H（H 个相邻 bigram，散布在序列里，符号互不相同）；
- 干扰：n_distract 个随机 bigram（符号与链不相交）；
- filler 不含任何 bigram 符号 → 每个键唯一、链路无歧义；cue=c_0 在末位，目标=c_H。
- 1 跳即退化为 induction；H≥2 必须链式组合多个绑定。

对照（唯一变量=读出/聚合结构；同 d / 同 slot 预算；三臂都带已验证的序列相邻算子）
-------------------------------------------------------------------------------
- flat        ：序列表示均值池化成**单向量** + MLP 读出（地板：连 1 跳都做不了）；
- capcw_1read ：slot 工作空间 + **单次**内容寻址读出（能 1 跳，预期做不了 ≥2 跳）；
- capcw_iter  ：slot 工作空间 + **H 次迭代**内容寻址读出（读出值→to_next→下一跳 query）
               ——假设：迭代式内容寻址读出在 slot 工作空间里实现多跳链式检索（类比 Transformer 深度）。

判据（预先写死）
----------------
- H1 多跳链式成立：多跳(n_hops≥2) acc(capcw_iter) − acc(flat) ≥ +0.10。
- H2 迭代读出是机制：多跳(n_hops≥2) acc(capcw_iter) − acc(capcw_1read) ≥ +0.10
  （单次读出=单跳，链式必须多次读；capcw_1read 在 1 跳应与 capcw_iter 持平、在 ≥2 跳掉队）。
- 综合：H1 且 H2 → **PASS**（CAPCW 支持多跳 in-context 组合推理，迭代内容寻址读出即其机制）；
  仅 H1 → **PARTIAL**；都不成立 → **FAIL**。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_multihop_eval --run
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
from fe_llm.world_model.capcw import PCWorkspace, SequenceAdjacency

REPORT_JSON = os.path.join("docs", "reports", "capcw_multihop_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_multihop_eval.md")


def gen_multihop(n_sym, n_hops, n_distract, seq_len, n, seed):
    """链式序列：c_0→c_1→…→c_H（H 个相邻 bigram）+ n_distract 个干扰 bigram；cue=c_0、y=c_H。

    无歧义口径（与 induction_seq 一致）：bigram 起点偶数不重叠；filler 不含任何 bigram 符号
    → 每个键唯一、链路唯一。返回 ids:(N,seq_len)、cue:(N,)、y:(N,)。
    """
    rng = np.random.default_rng(seed)
    ids = np.zeros((n, seq_len), dtype=np.int64)
    cue = np.zeros((n,), dtype=np.int64)
    y = np.zeros((n,), dtype=np.int64)
    even_slots = np.arange(0, seq_len - 1, 2)
    n_bigrams = n_hops + n_distract
    need_syms = (n_hops + 1) + 2 * n_distract
    if need_syms > n_sym:
        raise ValueError(f"n_sym={n_sym} 太小，链+干扰需要 {need_syms} 个不同符号")
    if n_bigrams > len(even_slots):
        raise ValueError(f"seq_len={seq_len} 放不下 {n_bigrams} 个不重叠 bigram")
    for i in range(n):
        picks = rng.choice(n_sym, size=need_syms, replace=False)
        chain = picks[: n_hops + 1]                       # c_0..c_H
        dsyms = picks[n_hops + 1:]                         # 干扰符号
        d_keys = dsyms[:n_distract]
        d_vals = dsyms[n_distract:]
        bigrams = [(int(chain[h]), int(chain[h + 1])) for h in range(n_hops)]
        bigrams += [(int(d_keys[j]), int(d_vals[j])) for j in range(n_distract)]
        used = set(int(s) for s in picks)
        filler_pool = np.array([s for s in range(n_sym) if s not in used], dtype=np.int64)
        seq = [int(rng.choice(filler_pool)) for _ in range(seq_len)]
        starts = rng.choice(even_slots, size=n_bigrams, replace=False)
        for (a, b), st in zip(bigrams, starts):
            seq[st] = a
            seq[st + 1] = b
        seq[seq_len - 1] = int(chain[0])                  # cue 放末位
        ids[i] = seq
        cue[i] = int(chain[0])
        y[i] = int(chain[n_hops])
    return ids, cue, y


class SeqEncoder(nn.Module):
    """序列编码器：token 嵌入 + 位置 + 已验证的序列相邻算子（三臂共用，保证唯一变量只在读出结构）。"""

    def __init__(self, n_sym, seq_len, d):
        super().__init__()
        self.emb = nn.Embedding(n_sym, d)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d))
        nn.init.normal_(self.pos, std=0.02)
        self.adj = SequenceAdjacency(d)

    def tokens(self, ids):
        rep = self.adj(self.emb(ids))                     # (B,L,d) (prev→cur) bigram 表示
        return rep + self.pos[:, : ids.shape[1]]


class FlatMultihop(nn.Module):
    """地板：单向量池化 + MLP 读出。"""

    def __init__(self, n_sym, seq_len, d):
        super().__init__()
        self.enc = SeqEncoder(n_sym, seq_len, d)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, n_sym))

    def forward(self, ids, cue):
        tok = self.enc.tokens(ids)
        world = tok.mean(dim=1)
        q = self.enc.emb(cue)
        return self.head(torch.cat([world, q], dim=-1))


class CAPCWMultihop(nn.Module):
    """slot 工作空间 + 可迭代的内容寻址读出（n_reads 次）。

    n_reads=1 即单跳读出（capcw_1read）；n_reads=H 即把上一跳读出值经 to_next 映射为下一跳 query，
    在工作空间里链式检索 H 跳（capcw_iter）。唯一变量=读出次数。
    （注：试过显式 key/value 投影读出，反而压低 1 跳精度、未改善多跳，故采用与已验证 induction 一致的直接读出。）
    """

    def __init__(self, n_sym, seq_len, d, n_slots, iters, n_reads):
        super().__init__()
        self.enc = SeqEncoder(n_sym, seq_len, d)
        self.ws = PCWorkspace(dim=d, n_slots=n_slots, iters=iters)
        self.to_q = nn.Linear(d, d)            # cue → 初始 query
        self.to_next = nn.Linear(d, d)         # 读出值 → 下一跳 query（链式）
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_sym))
        self.d = d
        self.n_reads = n_reads

    def forward(self, ids, cue):
        tok = self.enc.tokens(ids)
        slots = self.ws(tok).slots                                   # (B,M,d)
        q = self.to_q(self.enc.emb(cue))                            # (B,d)
        read = None
        for _ in range(self.n_reads):
            score = torch.einsum("bmd,bd->bm", slots, q) / math.sqrt(self.d)
            attn = score.softmax(dim=1).unsqueeze(-1)               # 内容寻址当前跳的 slot
            read = (slots * attn).sum(dim=1)                        # 读出该跳的值
            q = self.to_next(read)                                  # 作为下一跳 query
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


def _build_arms(args, n_hops, seq_len, seed):
    """三臂（同 d / 同 slot 预算；唯一变量=读出结构）。

    每臂构造前都用同一 seed 重置，保证 capcw_1read 与 capcw_iter **初始权重完全一致**（二者架构相同、
    仅 forward 读出次数不同），从而 1 跳时严格相等（sanity），多跳 delta 是"多读几次"的纯效应、无初始化混淆。
    """
    n_slots = max(args.n_slots, n_hops + args.n_distract + 1)
    torch.manual_seed(seed)
    flat = FlatMultihop(args.n_sym, seq_len, args.d)
    torch.manual_seed(seed)
    capcw_1read = CAPCWMultihop(args.n_sym, seq_len, args.d, n_slots, args.iters, n_reads=1)
    torch.manual_seed(seed)
    capcw_iter = CAPCWMultihop(args.n_sym, seq_len, args.d, n_slots, args.iters, n_reads=n_hops)
    return {"flat": flat, "capcw_1read": capcw_1read, "capcw_iter": capcw_iter}


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    hop_list = [int(x) for x in args.hop_list.split(",")]
    print(f"[multihop] device={device} n_sym={args.n_sym} hop_list={hop_list} n_distract={args.n_distract} "
          f"d={args.d} seeds={args.seeds}", flush=True)

    results: dict[int, dict] = {}
    for h in hop_list:
        seq_len = max(args.seq_len, 2 * (h + args.n_distract) + 4)
        arms = {"flat": [], "capcw_1read": [], "capcw_iter": []}
        for si in range(args.seeds):
            seed = args.seed + si
            train = gen_multihop(args.n_sym, h, args.n_distract, seq_len, args.n_train, seed)
            test = gen_multihop(args.n_sym, h, args.n_distract, seq_len, args.n_test, seed + 5000)
            models = _build_arms(args, h, seq_len, seed)   # 每臂同 seed 初始化（1read/iter 起点一致）
            common = dict(device=device, epochs=args.epochs, lr=args.lr, batch=args.batch, seed=seed)
            for name, model in models.items():
                arms[name].append(train_eval(model, train, test, **common))
        summary = {a: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)} for a, v in arms.items()}
        results[h] = summary
        print(f"[multihop] n_hops={h} (seq_len={seq_len}) "
              f"flat={summary['flat']['mean']:.3f} capcw_1read={summary['capcw_1read']['mean']:.3f} "
              f"capcw_iter={summary['capcw_iter']['mean']:.3f} (random={1.0/args.n_sym:.3f})", flush=True)

    # ---- 判据 ----
    multi = [h for h in hop_list if h >= 2]
    d_iter_flat = {h: round(results[h]["capcw_iter"]["mean"] - results[h]["flat"]["mean"], 4) for h in hop_list}
    d_iter_1read = {h: round(results[h]["capcw_iter"]["mean"] - results[h]["capcw_1read"]["mean"], 4) for h in hop_list}
    h1 = round(float(np.mean([d_iter_flat[h] for h in multi])), 4) if multi else 0.0      # 多跳 iter vs flat
    h2 = round(float(np.mean([d_iter_1read[h] for h in multi])), 4) if multi else 0.0     # 多跳 iter vs 单读
    h1_pass = h1 >= 0.10
    h2_pass = h2 >= 0.10
    if h1_pass and h2_pass:
        verdict = ("PASS: CAPCW 支持多跳 in-context 组合推理——多跳上 capcw_iter 同时胜单向量(flat)与单次读出"
                   "(capcw_1read)；迭代式内容寻址读出即链式检索机制（类比 Transformer 靠深度做多跳）。")
    elif h1_pass:
        verdict = ("PARTIAL: capcw_iter 多跳上胜单向量(H1)，但相对单次读出优势未达 +0.10(H2 未过)"
                   "——迭代读出是否为机制证据不足。")
    else:
        verdict = "FAIL: 多跳上 capcw_iter 未明显胜单向量——CAPCW 链式组合检索未成立。"

    result = {
        "task": "multi-hop in-context chained binding: c0->c1->...->cH, query c0 predict cH",
        "design": "vars = readout(flat pool / single content-addr read / iterative reads); same adjacency operator + d + slot budget",
        "config": {"n_sym": args.n_sym, "hop_list": hop_list, "n_distract": args.n_distract,
                   "base_seq_len": args.seq_len, "d": args.d, "n_slots": args.n_slots, "iters": args.iters,
                   "epochs": args.epochs, "seeds": args.seeds, "random_baseline": round(1.0 / args.n_sym, 4)},
        "by_n_hops": results,
        "delta_iter_minus_flat_by_hops": d_iter_flat,
        "delta_iter_minus_1read_by_hops": d_iter_1read,
        "h1_iter_vs_flat_multihop_mean": h1,
        "h2_iter_vs_1read_multihop_mean": h2,
        "h1_pass": h1_pass,
        "h2_pass": h2_pass,
        "verdict": verdict,
        "note": "1 跳即 induction（三臂里 capcw_1read 与 capcw_iter 此时同构、应持平）；≥2 跳必须链式组合多个绑定。"
                "唯一变量=读出结构，三臂共用同一序列相邻算子(SequenceAdjacency)与 slot 预算。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    lines = [
        "# CAPCW · 多跳/链式 in-context 推理（迭代内容寻址读出）",
        "",
        f"- 判定：**{verdict}**",
        f"- 任务：链式绑定 c0→c1→…→cH，查 c0 答 cH（每序列链路随机、不可记忆）；"
        f"n_sym={args.n_sym}, d={args.d}, n_distract={args.n_distract}；随机基线 {1.0/args.n_sym:.3f}",
        f"- 唯一变量=读出结构（flat 单向量 / capcw_1read 单次内容寻址 / capcw_iter H 次迭代）；三臂共用序列相邻算子+同预算。",
        "",
        "| n_hops（跳数） | flat | capcw_1read | capcw_iter |",
        "|---:|---:|---:|---:|",
    ]
    for h in hop_list:
        s = results[h]
        lines.append(f"| {h} | {s['flat']['mean']:.3f}±{s['flat']['std']:.3f} | "
                     f"{s['capcw_1read']['mean']:.3f}±{s['capcw_1read']['std']:.3f} | "
                     f"{s['capcw_iter']['mean']:.3f}±{s['capcw_iter']['std']:.3f} |")
    lines += [
        "",
        f"- **H1（多跳链式成立）**：多跳(n_hops≥2) capcw_iter − flat 平均 = **{h1:+.4f}**"
        f"（阈值 ≥ +0.10 → {'成立' if h1_pass else '不成立'}）；各跳：{d_iter_flat}",
        f"- **H2（迭代读出是机制）**：多跳 capcw_iter − capcw_1read 平均 = **{h2:+.4f}**"
        f"（阈值 ≥ +0.10 → {'成立' if h2_pass else '不成立'}）；各跳：{d_iter_1read}",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n[multihop] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    print(f"[multihop] H1 iter-flat={h1:+.4f}  H2 iter-1read={h2:+.4f}", flush=True)
    print(f"[multihop] 报告：{args.report_json} / {args.report_md}", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW multi-hop chained in-context reasoning eval.")
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
    # Windows 控制台默认 GBK，统一切到 utf-8（报告文件本就 utf-8）。
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print("[multihop] dry-run：未训练。多跳链式 in-context 推理：flat / capcw_1read / capcw_iter，扫 n_hops。")
        print("[multihop] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
