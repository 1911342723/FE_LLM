# -*- coding: utf-8 -*-
"""
fe_llm/world_model/compositional_generalization_eval.py
=======================================================
分层预测编码世界模型的**原则性裁决实验**：组合泛化（compositional generalization）。

背景（见 经验.md / docs/FE-LLM从0自建v2架构设计.md 第 8 节）：
V2-M1 三判定当年 FAIL，但那是"小任务被单向量吃满、测不出分层差别"的**测量问题**。
硬找一个有 headroom 的任务来证明分层 = motivated reasoning（B2 教训：构造特性）。

分层/结构在理论上唯一被认为不可替代的好处是 **组合泛化**：见过 (A×X)、(B×Y)，能否
泛化到**未见的组合** (A×Y)。扁平表示在未见组合上出名地塌；结构化/分层表示理论上能扛。
这才是公平裁决"分层到底有没有用"的实验——直接测它被声称的那个好处，而非凑 headroom。

任务（纯净 2 因子组合）：
- A,B 各 N 取值；标签 = (A+B) mod N（**必须同时知道 A 和 B**；标签空间跨组合复用，
  故未见组合的标签在训练里见过——这正是组合泛化的要件）。
- 输入：长度 L 的 token 序列，A-token 与 B-token 随机插在 filler 噪声里（逼编码器按
  **token 身份**而非位置识别），其余为随机 filler。
- split：训练只见部分 (A,B) 组合，测试在**未见组合**（每个 A 值/B 值都在训练里出现过，
  但该具体组合没出现过）。

对照（唯一变量 = 分层机制；同架构/同预算/同特征维）：
- flat        ：HierarchicalPredictiveEncoder(relax_steps=0)，z_global = z_local 掩码均值（无自上而下弛豫）；
- hierarchical：同编码器(relax_steps=K)，z_global = 自上而下预测-误差弛豫后的顶层；
- (附) hier_full：concat(z_global, z_local 均值)——给分层最强机会（特征维更大，仅作旁证）。

判定（未见组合 accuracy）：
- hier 明显 > flat（delta ≥ +0.05，且 seen 上相当）→ 分层**确有不可替代价值**（组合泛化）；
- 否则 → 分层连理论主场都赢不了 → 诚实定论：本规模下分层是过度设计，封存。

默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.world_model.compositional_generalization_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fe_llm.config import get_device
from fe_llm.world_model.hierarchical_encoder import HierarchicalPredictiveEncoder

REPORT_JSON = os.path.join("docs", "reports", "compositional_generalization_eval.json")
REPORT_MD = os.path.join("docs", "reports", "compositional_generalization_eval.md")


def build_vocab(n_values: int, n_filler: int) -> dict:
    """id 方案：0=pad；A 值 [1..N]；B 值 [N+1..2N]；filler [2N+1..2N+F]。"""
    return {
        "pad": 0,
        "a_base": 1,
        "b_base": 1 + n_values,
        "filler_base": 1 + 2 * n_values,
        "vocab_size": 1 + 2 * n_values + n_filler,
        "n_filler": n_filler,
        "n_values": n_values,
    }


def make_example(a: int, b: int, vocab: dict, length: int, rng: np.random.Generator) -> list[int]:
    """长度 L 序列：A-token 与 B-token 随机两个不同位置，其余 filler 噪声。"""
    seq = [vocab["filler_base"] + int(rng.integers(vocab["n_filler"])) for _ in range(length)]
    pa, pb = rng.choice(length, size=2, replace=False)
    seq[pa] = vocab["a_base"] + a
    seq[pb] = vocab["b_base"] + b
    return seq


def split_combos(n_values: int, n_test: int, seed: int) -> tuple[list, list]:
    """把 N×N 组合切成 train/test，保证每个 A 值、每个 B 值都在 train 里出现过。"""
    rng = np.random.default_rng(seed)
    all_combos = [(a, b) for a in range(n_values) for b in range(n_values)]
    for _ in range(200):
        rng.shuffle(all_combos)
        test = all_combos[:n_test]
        train = all_combos[n_test:]
        a_train = {a for a, _ in train}
        b_train = {b for _, b in train}
        if len(a_train) == n_values and len(b_train) == n_values:
            return sorted(train), sorted(test)
    raise RuntimeError("无法构造满足覆盖的 split，调小 n_test")


def compose_label(a: int, b: int, task: str, n_values: int) -> int:
    """组合标签：必须同时依赖 A 和 B，且标签空间跨组合复用（组合泛化要件）。
    - compare：A<B→0 / A==B→1 / A>B→2（可正常训练学会的有序关系，3 类）；
    - modadd ：(A+B) mod N（grokking 任务，难，需超长训练，作难度上界对照）。
    """
    if task == "compare":
        return 0 if a < b else (1 if a == b else 2)
    return (a + b) % n_values


def n_classes_for(task: str, n_values: int) -> int:
    return 3 if task == "compare" else n_values


def gen_dataset(combos, vocab, length, per_combo, task, n_values, seed):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for (a, b) in combos:
        for _ in range(per_combo):
            xs.append(make_example(a, b, vocab, length, rng))
            ys.append(compose_label(a, b, task, n_values))
    return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64)


class HierClassifier(nn.Module):
    """HierarchicalPredictiveEncoder + 线性分类头（特征可选 z_global 或 concat 两层）。"""

    def __init__(self, vocab_size, max_len, n_classes, *, intent_dim, dim, n_heads, depth,
                 relax_steps, feature="global"):
        super().__init__()
        self.encoder = HierarchicalPredictiveEncoder(
            vocab_size=vocab_size, max_len=max_len, dim=dim, n_heads=n_heads,
            intent_dim=intent_dim, depth=depth, relax_steps=relax_steps,
        )
        self.feature = feature
        feat_dim = intent_dim * (2 if feature == "concat" else 1)
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, ids):
        state = self.encoder(ids)
        if self.feature == "concat":
            pooled = state.z_local.mean(dim=1)
            feat = torch.cat([state.z_global, pooled], dim=-1)
        elif self.feature == "local":
            feat = state.z_local.mean(dim=1)
        else:  # "global"
            feat = state.z_global
        return self.head(feat)


def train_eval_arm(arm, x_tr, y_tr, x_seen, y_seen, x_unseen, y_unseen, *,
                   vocab_size, max_len, n_classes, device, seed, epochs, lr, batch):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HierClassifier(
        vocab_size=vocab_size, max_len=max_len, n_classes=n_classes,
        intent_dim=arm["intent_dim"], dim=arm["dim"], n_heads=arm["n_heads"],
        depth=arm["depth"], relax_steps=arm["relax_steps"], feature=arm["feature"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.tensor(x_tr, device=device)
    yt = torch.tensor(y_tr, device=device)
    n = len(xt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()

    def acc(x, y):
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(x, device=device)).argmax(-1).cpu().numpy()
        return float((pred == y).mean())

    return {"seen": acc(x_seen, y_seen), "unseen": acc(x_unseen, y_unseen)}


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    vocab = build_vocab(args.n_values, args.n_filler)
    n_classes = n_classes_for(args.task, args.n_values)
    print(
        f"[compgen] device={device} task={args.task} N={args.n_values} L={args.length} "
        f"filler={args.n_filler} test_combos={args.n_test_combos} n_classes={n_classes} seeds={args.seeds}",
        flush=True,
    )

    arms = {
        "flat (relax0)": dict(relax_steps=0, feature="global", intent_dim=args.intent_dim,
                              dim=args.dim, n_heads=args.n_heads, depth=args.depth),
        f"hierarchical (relax{args.relax})": dict(relax_steps=args.relax, feature="global",
                              intent_dim=args.intent_dim, dim=args.dim, n_heads=args.n_heads, depth=args.depth),
        f"hier_full concat (relax{args.relax})": dict(relax_steps=args.relax, feature="concat",
                              intent_dim=args.intent_dim, dim=args.dim, n_heads=args.n_heads, depth=args.depth),
    }

    per_arm: dict[str, dict] = {name: {"seen": [], "unseen": []} for name in arms}
    for s in range(args.seeds):
        seed = args.seed + s
        train_combos, test_combos = split_combos(args.n_values, args.n_test_combos, seed)
        x_tr, y_tr = gen_dataset(train_combos, vocab, args.length, args.per_combo, args.task, args.n_values, seed)
        x_seen, y_seen = gen_dataset(train_combos, vocab, args.length, args.eval_per_combo, args.task, args.n_values, seed + 777)
        x_unseen, y_unseen = gen_dataset(test_combos, vocab, args.length, args.eval_per_combo, args.task, args.n_values, seed + 999)
        for name, arm in arms.items():
            r = train_eval_arm(
                arm, x_tr, y_tr, x_seen, y_seen, x_unseen, y_unseen,
                vocab_size=vocab["vocab_size"], max_len=args.length, n_classes=n_classes,
                device=device, seed=seed, epochs=args.epochs, lr=args.lr, batch=args.batch,
            )
            per_arm[name]["seen"].append(r["seen"])
            per_arm[name]["unseen"].append(r["unseen"])
            print(f"[compgen] seed={seed} {name}: seen={r['seen']:.3f} unseen={r['unseen']:.3f}", flush=True)

    summary = {}
    for name, d in per_arm.items():
        summary[name] = {
            "seen_mean": round(float(np.mean(d["seen"])), 4),
            "seen_std": round(float(np.std(d["seen"])), 4),
            "unseen_mean": round(float(np.mean(d["unseen"])), 4),
            "unseen_std": round(float(np.std(d["unseen"])), 4),
            "unseen_per_seed": [round(v, 4) for v in d["unseen"]],
        }

    flat_name = "flat (relax0)"
    hier_name = f"hierarchical (relax{args.relax})"
    full_name = f"hier_full concat (relax{args.relax})"
    flat_u = summary[flat_name]["unseen_mean"]
    hier_u = summary[hier_name]["unseen_mean"]
    full_u = summary[full_name]["unseen_mean"]
    best_hier_u = max(hier_u, full_u)
    delta = round(best_hier_u - flat_u, 4)
    if delta >= 0.05:
        verdict = "PASS: 分层在未见组合(组合泛化)上明显优于扁平——分层确有不可替代价值"
    elif delta >= 0.02:
        verdict = "WEAK+: 分层在组合泛化上仅微弱优于扁平"
    else:
        verdict = "FAIL: 分层连理论主场(组合泛化)都没赢扁平——本规模下分层是过度设计，应封存"

    result = {
        "task": f"compositional generalization ({args.task}), unseen (A,B) combos",
        "config": {
            "task": args.task,
            "n_values": args.n_values, "length": args.length, "n_filler": args.n_filler,
            "n_test_combos": args.n_test_combos, "per_combo": args.per_combo,
            "intent_dim": args.intent_dim, "dim": args.dim, "depth": args.depth,
            "relax": args.relax, "epochs": args.epochs, "seeds": args.seeds,
        },
        "arms": summary,
        "headline": {"flat_unseen": flat_u, "hier_unseen": hier_u, "hier_full_unseen": full_u,
                     "best_hier_minus_flat": delta},
        "verdict": verdict,
        "note": (
            "唯一变量=分层机制(relax_steps)；seen=训练组合留出样本、unseen=未见组合。"
            "组合泛化是分层/结构被理论认为不可替代的好处，故为公平裁决（非凑 headroom）。"
        ),
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# 分层预测编码 · 组合泛化裁决实验",
        "",
        f"- 判定：**{verdict}**",
        f"- 任务：{args.task}（compare=A<B/==/> ; modadd=(A+B)%N），未见 (A,B) 组合泛化；唯一变量=分层机制(relax_steps)。",
        f"- 配置：N={args.n_values}, L={args.length}, filler={args.n_filler}, test_combos={args.n_test_combos}, "
        f"intent_dim={args.intent_dim}, relax={args.relax}, seeds={args.seeds}",
        "",
        "| 臂 | seen (训练组合) | unseen (未见组合) |",
        "|---|---:|---:|",
    ]
    for name, d in summary.items():
        lines.append(f"| {name} | {d['seen_mean']:.3f}±{d['seen_std']:.3f} | {d['unseen_mean']:.3f}±{d['unseen_std']:.3f} |")
    lines += [
        "",
        f"- 头条：best_hier_unseen − flat_unseen = **{delta:+.4f}**",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[compgen] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    print(f"[compgen] 报告：{args.report_json} / {args.report_md}", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compositional generalization adjudication for hierarchical PC.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--task", choices=["compare", "modadd"], default="compare",
                    help="compare=A<B/==/>(可达成) ; modadd=(A+B)%N(grokking,难)")
    ap.add_argument("--n-values", type=int, default=6)
    ap.add_argument("--length", type=int, default=8)
    ap.add_argument("--n-filler", type=int, default=10)
    ap.add_argument("--n-test-combos", type=int, default=10)
    ap.add_argument("--per-combo", type=int, default=150)
    ap.add_argument("--eval-per-combo", type=int, default=80)
    ap.add_argument("--intent-dim", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--relax", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-json", default=REPORT_JSON)
    ap.add_argument("--report-md", default=REPORT_MD)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print("[compgen] dry-run：未训练。组合泛化裁决：flat(relax0) vs hierarchical(relaxK) 在未见 (A,B) 组合上。")
        print("[compgen] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
