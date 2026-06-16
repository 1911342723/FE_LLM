# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_multihop_selfdistill_eval.py
=====================================================
**自蒸馏中间步**：多跳的主因是中间监督（第 26 节），但中间监督要中间标签。本脚本问——**我们总有最终
答案标签、缺的只是中间步标签，能否让一个只用单跳标签训练的 teacher 自动生成中间步，从而免去中间标注
就恢复多跳？** 见 `docs/FE-LLM核心引擎构想.md` 第 26/27 节。

现实标注设定：有 (pairs, 起点 c0, 最终答案 cH)；**没有**中间 c1..c_{H-1}。三臂（读出统一 decode→re-embed，
唯一变量=中间步从哪来）：
- final_only ：只监督最终跳(GT cH)，无中间监督（≈第 26 节 emergent，预期 FAIL）。
- **self_distill**：中间跳用**单跳 teacher 自生成**的 ĉ1..ĉ_{H-1}（只用单跳标签训练，无 GT 中间标签）+ 最终跳 GT cH。
- gt_intermediate：中间跳 + 最终跳全用 GT 链（天花板=cot）。

teacher = 单跳取回工作空间（随机 query→value，只需单跳标签=绑定本身）；自生成中间步=teacher 迭代单跳
（chain_read cot=True 把自己的解码 re-embed 成下一跳 query）。

判据（先写死）
--------------
- 多跳(H≥2) self_distill − final_only ≥ +0.15（自生成中间监督带来链式）；
- 且 self_distill ≥ 0.7 × gt_intermediate → **PASS**：**中间监督可自举**——单跳标签 + 自生成中间步即可恢复
  多跳，无需 GT 中间标签。否则 PARTIAL/FAIL，诚实记录。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_multihop_selfdistill_eval --run
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
import torch.nn.functional as F

from fe_llm.active_inference.capcw_chain_memory import _ChainWorkspace
from fe_llm.config import get_device

REPORT_JSON = os.path.join("docs", "reports", "capcw_multihop_selfdistill_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_multihop_selfdistill_eval.md")


def gen_pair_chain(n_sym, H, n_pairs, n, seed):
    """链式 pair：c0→…→cH（+干扰边凑满 n_pairs）。返回 pk,pv:(n,n_pairs)、q0:(n,)、chain:(n,H)=c1..cH。"""
    rng = np.random.default_rng(seed)
    n_distract = max(0, n_pairs - H)
    need = (H + 1) + 2 * n_distract
    if need > n_sym:
        raise ValueError(f"n_sym={n_sym} 太小，需 {need}")
    pk = np.zeros((n, n_pairs), dtype=np.int64)
    pv = np.zeros((n, n_pairs), dtype=np.int64)
    q0 = np.zeros((n,), dtype=np.int64)
    chain = np.zeros((n, H), dtype=np.int64)
    for i in range(n):
        picks = rng.choice(n_sym, size=need, replace=False)
        c = picks[: H + 1]
        ds = picks[H + 1:]
        dk, dv = ds[:n_distract], ds[n_distract:]
        keys = [int(c[h]) for h in range(H)] + [int(k) for k in dk]
        vals = [int(c[h + 1]) for h in range(H)] + [int(v) for v in dv]
        pk[i, :len(keys)] = keys
        pv[i, :len(vals)] = vals
        if len(keys) < n_pairs:
            pk[i, len(keys):] = keys[0]
            pv[i, len(vals):] = vals[0]
        q0[i] = int(c[0])
        chain[i] = [int(c[h + 1]) for h in range(H)]
    return pk, pv, q0, chain


def gen_single(n_sym, n_pairs, n, seed):
    """单跳数据：随机 n_pairs 对 (key→value) + 随机 query(=某 key)→其 value。teacher 训练用（只需单跳标签）。"""
    rng = np.random.default_rng(seed)
    pk = np.zeros((n, n_pairs), dtype=np.int64)
    pv = np.zeros((n, n_pairs), dtype=np.int64)
    q = np.zeros((n,), dtype=np.int64)
    y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        keys = rng.choice(n_sym, size=n_pairs, replace=False)
        vals = rng.choice(n_sym, size=n_pairs, replace=False)
        pk[i], pv[i] = keys, vals
        j = int(rng.integers(n_pairs))
        q[i], y[i] = int(keys[j]), int(vals[j])
    return pk, pv, q, y


def _train(net, tensors, *, forward_hops, loss_fn, epochs, lr, batch, device):
    """通用训练循环。tensors=(pk,pv,query,...)；loss_fn(logits_list, idx)->标量。"""
    pk, pv, query = tensors[0], tensors[1], tensors[2]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(query)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            logits, _ = net.chain_read(pk[idx], pv[idx], query[idx], forward_hops, cot=True)
            loss = loss_fn(logits, idx)
            loss.backward()
            opt.step()
    net.eval()
    return net


def train_teacher(n_sym, d, n_slots, iters, data, *, epochs, lr, batch, device, seed):
    torch.manual_seed(seed)
    net = _ChainWorkspace(n_sym, d, n_slots, iters).to(device)
    pk, pv, q, y = (torch.tensor(a, device=device) for a in data)
    return _train(net, (pk, pv, q), forward_hops=1, epochs=epochs, lr=lr, batch=batch, device=device,
                  loss_fn=lambda logits, idx: F.cross_entropy(logits[0], y[idx]))


@torch.no_grad()
def gen_self_labels(teacher, pk, pv, q0, H):
    """teacher 迭代单跳(chain_read cot=True 自解码再嵌入)自生成 ĉ1..ĉH（每跳 argmax）。返回 (n,H)。"""
    teacher.eval()
    logits, _ = teacher.chain_read(pk, pv, q0, H, cot=True)
    return torch.stack([logit.argmax(-1) for logit in logits], dim=1)


def train_student(n_sym, d, n_slots, iters, pk, pv, q0, targets, H, *, mode, epochs, lr, batch, device, seed):
    """学生统一 decode→re-embed 读出；mode: 'final'(仅末跳 GT) / 'all'(各跳监督 targets)。"""
    torch.manual_seed(seed)
    net = _ChainWorkspace(n_sym, d, n_slots, iters).to(device)

    def loss_fn(logits, idx):
        if mode == "final":
            return F.cross_entropy(logits[-1], targets[idx, H - 1])
        return sum(F.cross_entropy(logits[h], targets[idx, h]) for h in range(H)) / H

    return _train(net, (pk, pv, q0), forward_hops=H, epochs=epochs, lr=lr, batch=batch, device=device, loss_fn=loss_fn)


@torch.no_grad()
def eval_final(net, pk, pv, q0, gt_chain, H):
    final = net.chain_read(pk, pv, q0, H, cot=True)[0][-1].argmax(-1)
    return float((final == gt_chain[:, H - 1]).float().mean())


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    hop_list = [int(x) for x in args.hop_list.split(",")]
    n_slots = max(args.n_slots, args.n_pairs + 1)
    print(f"[mh-sd] device={device} n_sym={args.n_sym} n_pairs={args.n_pairs} hop_list={hop_list} seeds={args.seeds}", flush=True)
    results: dict = {}
    for H in hop_list:
        accs = {k: [] for k in ("final_only", "self_distill", "gt_intermediate", "teacher_single", "teacher_iter")}
        for si in range(args.seeds):
            seed = args.seed + si
            pk, pv, q0, chain = (torch.tensor(a, device=device) for a in gen_pair_chain(args.n_sym, H, args.n_pairs, args.n_train, seed))
            tpk, tpv, tq0, tchain = (torch.tensor(a, device=device) for a in gen_pair_chain(args.n_sym, H, args.n_pairs, args.n_test, seed + 5000))
            # teacher：单跳取回（随机 query），只需单跳标签。
            teacher = train_teacher(args.n_sym, args.d, n_slots, args.iters,
                                    gen_single(args.n_sym, args.n_pairs, args.n_train, seed + 100),
                                    epochs=args.epochs, lr=args.lr, batch=args.batch, device=device, seed=seed)
            t_single = eval_final(teacher, tpk, tpv, tq0, tchain, 1)        # teacher 单跳质量（取 c1）
            t_iter = eval_final(teacher, tpk, tpv, tq0, tchain, H)          # teacher 迭代多跳（无学生）
            # 自生成中间标签：teacher 在训练集上迭代单跳；中间跳用自生成、最终跳用 GT（现实里有最终标签）。
            self_lbl = gen_self_labels(teacher, pk, pv, q0, H).clone()
            self_lbl[:, H - 1] = chain[:, H - 1]                            # 最终跳=GT（中间跳才自生成）
            common = dict(epochs=args.epochs, lr=args.lr, batch=args.batch, device=device, seed=seed)
            s_final = train_student(args.n_sym, args.d, n_slots, args.iters, pk, pv, q0, chain, H, mode="final", **common)
            s_self = train_student(args.n_sym, args.d, n_slots, args.iters, pk, pv, q0, self_lbl, H, mode="all", **common)
            s_gt = train_student(args.n_sym, args.d, n_slots, args.iters, pk, pv, q0, chain, H, mode="all", **common)
            accs["final_only"].append(eval_final(s_final, tpk, tpv, tq0, tchain, H))
            accs["self_distill"].append(eval_final(s_self, tpk, tpv, tq0, tchain, H))
            accs["gt_intermediate"].append(eval_final(s_gt, tpk, tpv, tq0, tchain, H))
            accs["teacher_single"].append(t_single)
            accs["teacher_iter"].append(t_iter)
        results[H] = {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)} for k, v in accs.items()}
        r = results[H]
        print(f"[mh-sd] H={H} final_only={r['final_only']['mean']:.3f} self_distill={r['self_distill']['mean']:.3f} "
              f"gt={r['gt_intermediate']['mean']:.3f} | teacher_single={r['teacher_single']['mean']:.3f} "
              f"teacher_iter={r['teacher_iter']['mean']:.3f} (rand={1.0/args.n_sym:.3f})", flush=True)

    multi = [h for h in hop_list if h >= 2]

    def mmean(name):
        return float(np.mean([results[h][name]["mean"] for h in multi])) if multi else 0.0

    fin_m, self_m, gt_m = mmean("final_only"), mmean("self_distill"), mmean("gt_intermediate")
    self_minus_final = round(self_m - fin_m, 4)
    self_over_gt = round(self_m / gt_m, 4) if gt_m > 1e-9 else 0.0
    if self_minus_final >= 0.15 and self_over_gt >= 0.7:
        verdict = (f"PASS: **中间监督可自举**——self_distill {self_m:.3f} 比 final_only {fin_m:.3f} 高 "
                   f"{self_minus_final:+.4f}，且达 GT 中间监督 {gt_m:.3f} 的 {self_over_gt:.0%}。**只用单跳标签 + 单跳"
                   f"teacher 自生成中间步即可恢复多跳，无需 GT 中间标签**——中间监督(第 26 节主因)的标注可被省掉。")
    elif self_minus_final >= 0.10:
        verdict = (f"PARTIAL: 自蒸馏带来增益（self−final {self_minus_final:+.4f}）但未达 GT 的 70%"
                   f"（self/gt {self_over_gt:.0%}）——teacher 自生成中间步有噪声，部分恢复。")
    else:
        verdict = (f"FAIL: 自蒸馏未恢复多跳（self−final {self_minus_final:+.4f}<0.10）——teacher 自生成中间步"
                   f"质量不足以替代 GT 中间监督。诚实记录。")

    result = {
        "task": "self-distilled intermediate supervision: single-hop teacher self-generates intermediates (no GT intermediate labels)",
        "design": "readout=decode->re-embed (fixed); vars=intermediate source {none(final_only)/self/GT}. final hop always GT.",
        "config": {"n_sym": args.n_sym, "n_pairs": args.n_pairs, "hop_list": hop_list, "d": args.d,
                   "epochs": args.epochs, "seeds": args.seeds, "random_baseline": round(1.0 / args.n_sym, 4)},
        "by_n_hops": results,
        "multihop_means": {"final_only": round(fin_m, 4), "self_distill": round(self_m, 4), "gt_intermediate": round(gt_m, 4)},
        "self_minus_final": self_minus_final, "self_over_gt_ratio": self_over_gt,
        "verdict": verdict,
        "note": "三臂读出都用 decode→re-embed，唯一变量=中间步来源(无/self/GT)；最终跳一律 GT(现实有最终标签)。"
                "teacher=单跳取回(随机 query,只需单跳标签)，自生成中间步=teacher 迭代单跳。承接第 26 节(中间监督是主因)。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# CAPCW · 自蒸馏中间步（单跳 teacher 自生成中间监督，免 GT 中间标签）",
        "",
        f"- 判定：**{verdict}**",
        f"- 设置：n_sym={args.n_sym}, n_pairs={args.n_pairs}, d={args.d}；随机基线 {1.0/args.n_sym:.3f}；读出统一 decode→re-embed",
        "",
        "| n_hops | final_only(无中间) | **self_distill(自生成中间)** | gt_intermediate(GT中间·天花板) | teacher 单跳 | teacher 迭代 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for H in hop_list:
        r = results[H]
        lines.append(f"| {H} | {r['final_only']['mean']:.3f} | **{r['self_distill']['mean']:.3f}** | "
                     f"{r['gt_intermediate']['mean']:.3f} | {r['teacher_single']['mean']:.3f} | {r['teacher_iter']['mean']:.3f} |")
    lines += [
        "",
        f"- 多跳(H≥2)：self_distill **{self_m:.3f}** vs final_only **{fin_m:.3f}**（+{self_minus_final:.4f}）vs GT 中间 "
        f"**{gt_m:.3f}**（self 达 GT 的 {self_over_gt:.0%}）。",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[mh-sd] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    print(f"[mh-sd] self−final={self_minus_final:+.4f} self/gt={self_over_gt:.0%}", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW multi-hop self-distilled intermediate supervision.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-sym", type=int, default=20)
    ap.add_argument("--n-pairs", type=int, default=5)
    ap.add_argument("--hop-list", default="2,3")
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--n-slots", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=6000)
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
        print("[mh-sd] dry-run：未训练。自蒸馏中间步：单跳 teacher 自生成中间监督(免 GT 中间标签) vs final-only vs GT。")
        print("[mh-sd] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
