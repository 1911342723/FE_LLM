# -*- coding: utf-8 -*-
"""稳定化是否承担预测、残余自由能是否承担 surprise 的裁决实验。

任务不是为了刷语言指标，而是隔离两个必要机制：

* ID：随机两个符号起步，之后 ``x[t] = x[t-2]``。预测下一符号必须记住上一位置，
  只看当前 token 的无弛豫模型做不到。
* OOD：改成 ``x[t] = x[t-3]``。它要求未训练过的状态周期，用于检验生成性预测误差
  能否识别现有结构解释不了的经验。

两臂参数与数据完全相同：

* ``CE+F``：next-token CE + 残余自由能外循环目标；
* ``CE-only``：只用 CE，检验没有生成性自由能学习时 surprise 是否失去意义。

对每臂同时评估完整弛豫与 ``0-step`` 消融。完整模型显著优于 0-step 才说明上下文能力
来自稳定化动力学，而不是 token embedding/head 偷做了任务。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from fe_llm.config import get_device
from fe_llm.energy_lm.models.free_energy_lm import FreeEnergyLM

REPORT_JSON = os.path.join("docs", "reports", "free_energy_sequence_eval.json")
REPORT_MD = os.path.join("docs", "reports", "free_energy_sequence_eval.md")
FIG_PATH = os.path.join("docs", "reports", "figs", "free_energy_sequence_eval.png")


def make_lag_sequences(batch: int, length: int, lag: int, vocab: int, device: str) -> torch.Tensor:
    seq = torch.empty(batch, length, dtype=torch.long, device=device)
    seq[:, :lag] = torch.randint(0, vocab, (batch, lag), device=device)
    for position in range(lag, length):
        seq[:, position] = seq[:, position - lag]
    return seq


def task_loss(logits: torch.Tensor, seq: torch.Tensor, lag: int) -> torch.Tensor:
    # logits[:,i] 预测 x[i+1]；从 i=lag-1 开始，目标才由规则确定。
    return F.cross_entropy(
        logits[:, lag - 1:-1].reshape(-1, logits.size(-1)),
        seq[:, lag:].reshape(-1),
    )


@torch.no_grad()
def accuracy(net: FreeEnergyLM, seq: torch.Tensor, lag: int, max_relax_steps: int) -> float:
    logits = net(seq, max_relax_steps=max_relax_steps)
    pred = logits[:, lag - 1:-1].argmax(dim=-1)
    return float((pred == seq[:, lag:]).float().mean().cpu())


def pairwise_auc(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> float:
    # OOD 应有更高 surprise。Mann-Whitney 形式避免引入 sklearn。
    gt = (ood_scores[:, None] > id_scores[None, :]).float().mean()
    eq = (ood_scores[:, None] == id_scores[None, :]).float().mean()
    return float((gt + 0.5 * eq).cpu())


@torch.no_grad()
def diagnostics(net: FreeEnergyLM, args: argparse.Namespace) -> dict:
    id_seq = make_lag_sequences(args.eval_batch, args.length, 2, args.vocab, args.device)
    ood_seq = make_lag_sequences(args.eval_batch, args.length, 3, args.vocab, args.device)

    id_logits, id_trace = net(id_seq, return_trace=True)
    _, ood_trace = net(ood_seq, return_trace=True)
    id_pred = id_logits[:, 1:-1].argmax(dim=-1)
    id_acc = float((id_pred == id_seq[:, 2:]).float().mean().cpu())
    no_relax_acc = accuracy(net, id_seq, lag=2, max_relax_steps=0)
    ood_acc = accuracy(net, ood_seq, lag=3, max_relax_steps=net.relaxation_steps)

    # 两类都从第 4 个 token 起比较，避免不同随机起始区间污染判别。
    start = 3
    id_surprise = id_trace["surprise_per_dim"][:, start:].mean(dim=1)
    ood_surprise = ood_trace["surprise_per_dim"][:, start:].mean(dim=1)
    id_residual = id_trace["residual_free_energy_per_dim"][:, start:].mean(dim=1)
    ood_residual = ood_trace["residual_free_energy_per_dim"][:, start:].mean(dim=1)
    energy = id_trace["free_energy"].cpu()

    return {
        "id_accuracy": round(id_acc, 6),
        "id_accuracy_zero_relax": round(no_relax_acc, 6),
        "relaxation_gain": round(id_acc - no_relax_acc, 6),
        "ood_lag3_accuracy": round(ood_acc, 6),
        "id_surprise": round(float(id_surprise.mean().cpu()), 6),
        "ood_surprise": round(float(ood_surprise.mean().cpu()), 6),
        "surprise_gap": round(float((ood_surprise.mean() - id_surprise.mean()).cpu()), 6),
        "surprise_auroc": round(pairwise_auc(id_surprise, ood_surprise), 6),
        "id_residual_free_energy": round(float(id_residual.mean().cpu()), 6),
        "ood_residual_free_energy": round(float(ood_residual.mean().cpu()), 6),
        "residual_auroc": round(pairwise_auc(id_residual, ood_residual), 6),
        "free_energy_trace": [round(float(v), 6) for v in energy],
        "energy_monotonic": bool(torch.all(energy[1:] <= energy[:-1] + 1e-5)),
        "mean_relax_steps": round(float(id_trace["steps_per_position"].float().mean().cpu()), 4),
    }


def train_one(seed: int, use_free_energy: bool, args: argparse.Namespace) -> dict:
    torch.manual_seed(seed)
    net = FreeEnergyLM(
        vocab_size=args.vocab,
        max_len=args.length,
        dim=args.dim,
        relaxation_steps=args.relax_steps,
        tolerance=args.tolerance,
        transition_mult=args.transition_mult,
    ).to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    t0 = time.time()
    net.train()

    for step in range(1, args.steps + 1):
        seq = make_lag_sequences(args.batch, args.length, 2, args.vocab, args.device)
        logits = net(seq)
        ce = task_loss(logits, seq, lag=2)
        assert net.last_position_free_energy is not None
        structural_f = net.last_position_free_energy[:, 2:].mean()
        loss = ce + (args.free_energy_weight * structural_f if use_free_energy else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            acc = accuracy(net, seq[: min(128, len(seq))], lag=2,
                           max_relax_steps=net.relaxation_steps)
            row = {
                "step": step,
                "ce": round(float(ce.detach().cpu()), 6),
                "residual_f": round(float(structural_f.detach().cpu()), 6),
                "train_acc": round(acc, 6),
            }
            history.append(row)
            arm = "CE+F" if use_free_energy else "CE-only"
            print(f"[seq] seed={seed} arm={arm:7s} step={step:4d}/{args.steps} "
                  f"ce={row['ce']:.4f} F={row['residual_f']:.4f} acc={acc:.1%}", flush=True)

    net.eval()
    result = diagnostics(net, args)
    result.update({
        "seed": seed,
        "arm": "CE+F" if use_free_energy else "CE-only",
        "seconds": round(time.time() - t0, 2),
        "params": sum(p.numel() for p in net.parameters()),
        "history": history,
    })
    return result


def summarize(rows: list[dict]) -> dict:
    metrics = [
        "id_accuracy", "id_accuracy_zero_relax", "relaxation_gain",
        "ood_lag3_accuracy", "id_surprise", "ood_surprise", "surprise_gap",
        "surprise_auroc", "id_residual_free_energy", "ood_residual_free_energy",
        "residual_auroc", "mean_relax_steps",
    ]
    out: dict[str, dict] = {}
    for arm in ("CE+F", "CE-only"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        out[arm] = {}
        for metric in metrics:
            values = np.asarray([row[metric] for row in arm_rows], dtype=float)
            out[arm][metric] = {
                "mean": round(float(values.mean()), 6),
                "std": round(float(values.std()), 6),
            }
        out[arm]["all_energy_monotonic"] = all(row["energy_monotonic"] for row in arm_rows)
    return out


def write_reports(result: dict) -> None:
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    summary = result["summary"]
    fe, ce = summary["CE+F"], summary["CE-only"]
    verdict = result["verdict"]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(
            "# 自由能稳定化承担预测 + 残余自由能承担 surprise\n\n"
            "任务：ID 为 lag-2 周期序列，OOD 为未训练 lag-3；多种子、参数/数据完全一致。\n\n"
            "| 臂 | ID acc | 0-step acc | 弛豫增益 | surprise AUROC | residual AUROC |\n"
            "|---|---:|---:|---:|---:|---:|\n"
            f"| CE+F | {fe['id_accuracy']['mean']:.1%} | {fe['id_accuracy_zero_relax']['mean']:.1%} | "
            f"{fe['relaxation_gain']['mean']:+.1%} | {fe['surprise_auroc']['mean']:.3f} | "
            f"{fe['residual_auroc']['mean']:.3f} |\n"
            f"| CE-only | {ce['id_accuracy']['mean']:.1%} | {ce['id_accuracy_zero_relax']['mean']:.1%} | "
            f"{ce['relaxation_gain']['mean']:+.1%} | {ce['surprise_auroc']['mean']:.3f} | "
            f"{ce['residual_auroc']['mean']:.3f} |\n\n"
            f"## 裁决\n\n{verdict}\n\n"
            "## 边界\n\n"
            "这是机制裁决，不是语言规模结论。它只回答：稳定化是否真的承载上下文计算，"
            "以及自由能外循环是否让 surprise 获得结构含义。下一步才是用该信号触发容量生长。\n"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["CE+F", "CE-only"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    x = np.arange(2)
    axes[0].bar(x - 0.17, [summary[a]["id_accuracy"]["mean"] for a in labels], 0.34,
                label="完整弛豫", color="#1565c0")
    axes[0].bar(x + 0.17, [summary[a]["id_accuracy_zero_relax"]["mean"] for a in labels], 0.34,
                label="0-step", color="#b0bec5")
    axes[0].set_title("预测是否来自稳定化")
    axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=8)
    axes[1].bar(x, [summary[a]["surprise_auroc"]["mean"] for a in labels], color=["#2e7d32", "#ef6c00"])
    axes[1].axhline(0.5, color="black", ls="--", lw=1)
    axes[1].set_title("初始 surprise：ID vs OOD")
    axes[1].set_ylim(0, 1.05)
    axes[2].bar(x, [summary[a]["residual_auroc"]["mean"] for a in labels], color=["#2e7d32", "#ef6c00"])
    axes[2].axhline(0.5, color="black", ls="--", lw=1)
    axes[2].set_title("稳定后残余自由能：ID vs OOD")
    axes[2].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("自由能核心机制裁决（非 Transformer、共享递归稳定化）")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    args.device = args.device.strip() or get_device()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for seed in seeds:
        rows.append(train_one(seed, True, args))
        rows.append(train_one(seed, False, args))
    summary = summarize(rows)
    fe, ce = summary["CE+F"], summary["CE-only"]
    mechanism_pass = (
        fe["id_accuracy"]["mean"] >= 0.90
        and fe["relaxation_gain"]["mean"] >= 0.50
        and fe["surprise_auroc"]["mean"] >= 0.80
        and fe["surprise_auroc"]["mean"] - ce["surprise_auroc"]["mean"] >= 0.20
        and fe["all_energy_monotonic"]
        and ce["all_energy_monotonic"]
    )
    verdict = (
        "✅ 稳定化成为真实计算：完整弛豫学会 lag-2，而 0-step 不能；CE+F 的自由能可区分未见结构，"
        "满足用 residual surprise 触发生长的前提。"
        if mechanism_pass else
        "🟡 当前核心尚未同时通过预测承载与结构 surprise 判据；应先修正动力学，不进入容量生长。"
    )
    result = {
        "task": "lag-2 contextual prediction vs lag-3 structural OOD",
        "config": {k: v for k, v in vars(args).items()},
        "rows": rows,
        "summary": summary,
        "mechanism_pass": mechanism_pass,
        "verdict": verdict,
    }
    if args.write_report:
        write_reports(result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="裁决稳定化是否承担预测、残余自由能是否承担 surprise。")
    ap.add_argument("--device", default="")
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--length", type=int, default=14)
    ap.add_argument("--vocab", type=int, default=8)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--relax-steps", type=int, default=5)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--transition-mult", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--free-energy-weight", type=float, default=2.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--write-report", action="store_true", default=True)
    ap.add_argument("--no-write-report", dest="write_report", action="store_false")
    return ap


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = build_arg_parser().parse_args(argv)
    result = run(args)
    print(json.dumps({"summary": result["summary"], "verdict": result["verdict"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
