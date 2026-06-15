# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_binding_eval.py
========================================
核心引擎方案一 CAPCW（内容寻址预测编码工作空间）的**绑定任务验证**。
见 `docs/FE-LLM核心引擎构想.md`。

CAPCW 的核心主张：把世界状态从"单向量"换成"一组**内容寻址的 slot 工作空间**"，能拿到
Transformer 级的内容寻址能力。最能体现这一点的、且**单向量可证不足**的任务是「绑定」：

任务（in-context 键值绑定）：
- 每个样本随机给 K 对 (key→value)（每样本绑定都不同 → 不能记忆，必须 in-context 读取）；
- 再问其中某个 key 的 value（n_vals 路分类）。
- 单向量把 K 个绑定挤进一个固定向量会相互干扰，K 越大越糟；内容寻址的 slot 工作空间应能把
  query 路由到对的 slot 读出 value。

对照（唯一变量 = 世界状态结构；同 pair 编码 / 同预算 / 同读出风格）：
- flat        ：pair 表示均值池化成**单向量** + query 读出；
- CAPCW       ：slot-attention 把 pair 聚成 **M 个 slot 工作空间** + query 内容寻址读出；
- hierarchy   ：已封存的 HierarchicalPredictiveEncoder → z_global **单向量**（旁证，shelved 方案也解不了绑定）。

判定（预先写死）：扫 K=2..5，**CAPCW 在高 K 明显优于 flat（如 K≥4 时 +0.10 以上）** → CAPCW 的
内容寻址工作空间确有不可替代价值（验证方向）；否则 → 方向证伪，记录第三个核心引擎负结果。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_binding_eval --run
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

REPORT_JSON = os.path.join("docs", "reports", "capcw_binding_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_binding_eval.md")


def gen_binding(n_keys, n_vals, k_pairs, n_examples, seed):
    """每样本：K 个不同 key 各绑一个不同 value；问一个 key 的 value。"""
    rng = np.random.default_rng(seed)
    pk = np.zeros((n_examples, k_pairs), dtype=np.int64)
    pv = np.zeros((n_examples, k_pairs), dtype=np.int64)
    qk = np.zeros((n_examples,), dtype=np.int64)
    y = np.zeros((n_examples,), dtype=np.int64)
    for i in range(n_examples):
        keys = rng.choice(n_keys, size=k_pairs, replace=False)
        vals = rng.choice(n_vals, size=k_pairs, replace=False)
        pk[i] = keys
        pv[i] = vals
        qi = int(rng.integers(k_pairs))
        qk[i] = keys[qi]
        y[i] = vals[qi]
    return pk, pv, qk, y


class PairEncoder(nn.Module):
    """key/value 各自嵌入 → 每对融合成一个 pair 向量；共享给所有臂（唯一变量只在聚合）。"""

    def __init__(self, n_keys, n_vals, d):
        super().__init__()
        self.key_emb = nn.Embedding(n_keys, d)
        self.val_emb = nn.Embedding(n_vals, d)
        self.pair = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, pk, pv):
        ke = self.key_emb(pk)
        ve = self.val_emb(pv)
        return self.pair(torch.cat([ke, ve], dim=-1))  # (B,K,d)


class FlatModel(nn.Module):
    """单向量世界状态：pair 均值池化 + query 读出。"""

    def __init__(self, n_keys, n_vals, d):
        super().__init__()
        self.enc = PairEncoder(n_keys, n_vals, d)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, n_vals))

    def forward(self, pk, pv, qk):
        pairs = self.enc(pk, pv)
        world = pairs.mean(dim=1)
        q = self.enc.key_emb(qk)
        return self.head(torch.cat([world, q], dim=-1))


class SlotAttention(nn.Module):
    """最小 slot-attention：M slot 迭代竞争性 attend 输入（内容寻址工作空间）。"""

    def __init__(self, d, n_slots, iters=3):
        super().__init__()
        self.n_slots, self.iters, self.d = n_slots, iters, d
        self.slots_mu = nn.Parameter(torch.randn(1, n_slots, d) * 0.02)
        self.to_q = nn.Linear(d, d)
        self.to_k = nn.Linear(d, d)
        self.to_v = nn.Linear(d, d)
        self.gru = nn.GRUCell(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.norm_in = nn.LayerNorm(d)
        self.norm_slots = nn.LayerNorm(d)

    def forward(self, inputs):  # (B,P,d)
        b, p, d = inputs.shape
        slots = self.slots_mu.expand(b, -1, -1)
        inp = self.norm_in(inputs)
        k = self.to_k(inp)
        v = self.to_v(inp)
        for _ in range(self.iters):
            q = self.to_q(self.norm_slots(slots))                  # (B,M,d)
            logits = torch.einsum("bmd,bpd->bmp", q, k) / math.sqrt(d)
            attn = logits.softmax(dim=1)                           # slots 竞争每个输入（dim=1=slots）
            attn = attn + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)           # 对输入归一化
            updates = torch.einsum("bmp,bpd->bmd", attn, v)        # (B,M,d)
            slots = self.gru(updates.reshape(-1, d), slots.reshape(-1, d)).reshape(b, -1, d)
            slots = slots + self.mlp(slots)
        return slots                                              # (B,M,d)


class CAPCWModel(nn.Module):
    """内容寻址 slot 工作空间 + query 内容寻址读出。"""

    def __init__(self, n_keys, n_vals, d, n_slots, iters):
        super().__init__()
        self.enc = PairEncoder(n_keys, n_vals, d)
        self.slot_attn = SlotAttention(d, n_slots, iters)
        self.to_q = nn.Linear(d, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_vals))
        self.d = d

    def forward(self, pk, pv, qk):
        pairs = self.enc(pk, pv)
        slots = self.slot_attn(pairs)                              # (B,M,d)
        q = self.to_q(self.enc.key_emb(qk))                       # (B,d)
        score = torch.einsum("bmd,bd->bm", slots, q) / math.sqrt(self.d)
        attn = score.softmax(dim=1).unsqueeze(-1)                 # (B,M,1) query 内容寻址 slot
        read = (slots * attn).sum(dim=1)                          # (B,d)
        return self.head(read)


class CAPCWPCModel(nn.Module):
    """阶段二：用预测编码/自由能工作空间 PCWorkspace 替代裸 slot-attention（content 路由由
    重建误差导出、弛豫降自由能、可溯源）+ query 内容寻址读出。判据=保持绑定胜势。"""

    def __init__(self, n_keys, n_vals, d, n_slots, iters):
        super().__init__()
        self.enc = PairEncoder(n_keys, n_vals, d)
        self.ws = PCWorkspace(dim=d, n_slots=n_slots, iters=iters)
        self.to_q = nn.Linear(d, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_vals))
        self.d = d

    def forward(self, pk, pv, qk, n_slots=None):
        pairs = self.enc(pk, pv)
        slots = self.ws(pairs, n_slots=n_slots).slots   # n_slots 覆盖供动态生长（默认用 ws 自身 slot 数）
        q = self.to_q(self.enc.key_emb(qk))
        score = torch.einsum("bmd,bd->bm", slots, q) / math.sqrt(self.d)
        attn = score.softmax(dim=1).unsqueeze(-1)
        read = (slots * attn).sum(dim=1)
        return self.head(read)

    @torch.no_grad()
    def query_match(self, pk, pv, qk):
        """query→slot 最大路由权重：匹配到(已绑定)高、匹配不到(未绑定)低=高 surprise。供 surprise→动作用。"""
        pairs = self.enc(pk, pv)
        slots = self.ws(pairs).slots
        q = self.to_q(self.enc.key_emb(qk))
        score = torch.einsum("bmd,bd->bm", slots, q) / math.sqrt(self.d)
        return score.softmax(dim=1).max(dim=1).values


class HierModel(nn.Module):
    """已封存的分层 PC 编码器 → z_global 单向量（旁证 shelved 方案解不了绑定）。"""

    def __init__(self, n_keys, n_vals, d, intent_dim, relax):
        super().__init__()
        from fe_llm.world_model.hierarchical_encoder import HierarchicalPredictiveEncoder

        # 词表：key [0..n_keys) | value [n_keys..) | query 标记
        self.n_keys = n_keys
        self.vocab = n_keys + n_vals + 1
        self.query_tok = n_keys + n_vals
        self.n_vals = n_vals
        self.max_len = 64
        self.encoder = HierarchicalPredictiveEncoder(
            vocab_size=self.vocab, max_len=self.max_len, dim=d, n_heads=4,
            intent_dim=intent_dim, depth=2, relax_steps=relax,
        )
        self.q_emb = nn.Embedding(n_keys, intent_dim)
        self.head = nn.Sequential(nn.Linear(2 * intent_dim, intent_dim), nn.GELU(),
                                  nn.Linear(intent_dim, n_vals))

    def _seq(self, pk, pv, qk):
        b, k = pk.shape
        toks = []
        for i in range(k):
            toks.append(pk[:, i : i + 1])
            toks.append(pv[:, i : i + 1] + self.n_keys)
        toks.append(torch.full((b, 1), self.query_tok, device=pk.device, dtype=pk.dtype))
        toks.append(qk[:, None])
        return torch.cat(toks, dim=1)

    def forward(self, pk, pv, qk):
        ids = self._seq(pk, pv, qk)
        state = self.encoder(ids)
        q = self.q_emb(qk)
        return self.head(torch.cat([state.z_global, q], dim=-1))


def train_eval(model, train, test, device, *, epochs, lr, batch, seed):
    torch.manual_seed(seed)
    pk, pv, qk, y = (torch.tensor(t, device=device) for t in train)
    tpk, tpv, tqk, ty = (torch.tensor(t, device=device) for t in test)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(y)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(pk[idx], pv[idx], qk[idx]), y[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(tpk, tpv, tqk).argmax(-1)
        acc = float((pred == ty).float().mean())
    return acc


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    ks = [int(x) for x in args.k_list.split(",")]
    print(f"[capcw] device={device} K_list={ks} n_keys={args.n_keys} n_vals={args.n_vals} seeds={args.seeds}", flush=True)
    results = {}
    for k in ks:
        arms = {"flat": [], "capcw": [], "capcw_pc": [], "hierarchy": []}
        for si in range(args.seeds):
            seed = args.seed + si
            train = gen_binding(args.n_keys, args.n_vals, k, args.n_train, seed)
            test = gen_binding(args.n_keys, args.n_vals, k, args.n_test, seed + 5000)
            n_slots = max(args.n_slots, k + 1)
            flat = FlatModel(args.n_keys, args.n_vals, args.d)
            capcw = CAPCWModel(args.n_keys, args.n_vals, args.d, n_slots=n_slots, iters=args.iters)
            capcw_pc = CAPCWPCModel(args.n_keys, args.n_vals, args.d, n_slots=n_slots, iters=args.iters)
            hier = HierModel(args.n_keys, args.n_vals, args.d, intent_dim=args.d, relax=args.relax)
            common = dict(device=device, epochs=args.epochs, lr=args.lr, batch=args.batch, seed=seed)
            arms["flat"].append(train_eval(flat, train, test, **common))
            arms["capcw"].append(train_eval(capcw, train, test, **common))
            arms["capcw_pc"].append(train_eval(capcw_pc, train, test, **common))
            arms["hierarchy"].append(train_eval(hier, train, test, **common))
        summary = {a: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)} for a, v in arms.items()}
        results[k] = summary
        print(
            f"[capcw] K={k} flat={summary['flat']['mean']:.3f} capcw={summary['capcw']['mean']:.3f} "
            f"capcw_pc={summary['capcw_pc']['mean']:.3f} hier={summary['hierarchy']['mean']:.3f} "
            f"(random={1.0/args.n_vals:.3f})",
            flush=True,
        )

    # 判定：CAPCW 在最大 K 上相对 flat 的优势。
    kmax = max(ks)
    delta_max = round(results[kmax]["capcw"]["mean"] - results[kmax]["flat"]["mean"], 4)
    high_k = [k for k in ks if k >= args.high_k]
    delta_highk = round(float(np.mean([results[k]["capcw"]["mean"] - results[k]["flat"]["mean"] for k in high_k])), 4) if high_k else 0.0
    delta_pc_highk = round(float(np.mean([results[k]["capcw_pc"]["mean"] - results[k]["flat"]["mean"] for k in high_k])), 4) if high_k else 0.0
    if delta_highk >= 0.10:
        verdict = "PASS: CAPCW 内容寻址工作空间在绑定任务高 K 上明显优于单向量——核心引擎方向得到验证"
    elif delta_highk >= 0.03:
        verdict = "PARTIAL: CAPCW 有正向优势但偏弱"
    else:
        verdict = "FAIL: CAPCW 未明显优于单向量——方向证伪，记录第三个核心引擎负结果"

    result = {
        "task": "in-context key->value binding; predict queried key's value",
        "config": {"n_keys": args.n_keys, "n_vals": args.n_vals, "k_list": ks, "n_slots": args.n_slots,
                   "iters": args.iters, "d": args.d, "epochs": args.epochs, "seeds": args.seeds,
                   "random_baseline": round(1.0 / args.n_vals, 4)},
        "by_k": results,
        "delta_capcw_minus_flat_at_kmax": delta_max,
        "delta_capcw_minus_flat_high_k_mean": delta_highk,
        "delta_capcw_pc_minus_flat_high_k_mean": delta_pc_highk,
        "verdict": verdict,
        "note": "唯一变量=世界状态结构(单向量 vs slot 工作空间)；同 pair 编码/同预算。绑定=内容寻址的"
                "经典场景且单向量可证不足，故为 CAPCW 核心主张的公平验证。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# 核心引擎方案一 CAPCW · 绑定任务验证",
        "",
        f"- 判定：**{verdict}**",
        f"- 任务：in-context 键值绑定（每样本随机 K 对，问某 key 的 value）；随机基线 {1.0/args.n_vals:.3f}",
        f"- 唯一变量=世界状态结构（flat 单向量 / CAPCW slot 工作空间 / hierarchy 已封存 z_global）；同 pair 编码/同预算。",
        "",
        "| K（绑定负载） | flat | CAPCW (slot-attn) | CAPCW_PC (自由能) | hierarchy |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in ks:
        s = results[k]
        lines.append(f"| {k} | {s['flat']['mean']:.3f}±{s['flat']['std']:.3f} | "
                     f"{s['capcw']['mean']:.3f}±{s['capcw']['std']:.3f} | "
                     f"{s['capcw_pc']['mean']:.3f}±{s['capcw_pc']['std']:.3f} | "
                     f"{s['hierarchy']['mean']:.3f}±{s['hierarchy']['std']:.3f} |")
    lines += [
        "",
        f"- 高 K(≥{args.high_k}) 平均 CAPCW(slot) − flat = **{delta_highk:+.4f}**；K={kmax} 时 = {delta_max:+.4f}",
        f"- 阶段二判据 高 K 平均 CAPCW_PC(自由能形态) − flat = **{delta_pc_highk:+.4f}**（应同样明显 > 0，即 PC 形态保持绑定胜势）",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[capcw] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    print(f"[capcw] 报告：{args.report_json} / {args.report_md}", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW binding-task validation (content-addressable workspace vs single vector).")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-keys", type=int, default=8)
    ap.add_argument("--n-vals", type=int, default=10)
    ap.add_argument("--k-list", default="2,3,4,5")
    ap.add_argument("--high-k", type=int, default=4)
    ap.add_argument("--n-slots", type=int, default=6)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--relax", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=30)
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
        print("[capcw] dry-run：未训练。绑定任务上 flat(单向量) vs CAPCW(slot 工作空间) vs hierarchy(已封存)，扫 K。")
        print("[capcw] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
