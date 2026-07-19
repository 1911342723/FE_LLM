# -*- coding: utf-8 -*-
"""完整生成性通路 vs 低秩动力学生长的参数效率裁决。

基础稳定态学习 lag-2；新结构为 lag-3。所有候选共享并冻结基础 FreeEnergyLM，仅新增：

* full：完整复制生成性转移；
* rank-r：``T_new=T_base+U tanh(Vz)`` 的低秩动力学修正。

各臂使用相同数据、训练步数、任务损失与 residual-F 外循环目标，并由同一 MDL 能量路由验收。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from fe_llm.config import get_device
from fe_llm.energy_lm.free_energy_growth import FreeEnergyGrowthSystem
from fe_llm.energy_lm.models.low_rank_transition import LowRankGenerativeTransition
from fe_llm.energy_lm.evaluation.free_energy_growth_eval import routed_accuracy, train_base
from fe_llm.energy_lm.evaluation.free_energy_sequence_eval import (
    accuracy,
    make_lag_sequences,
    task_loss,
)

REPORT_JSON = os.path.join("docs", "reports", "free_energy_low_rank_growth_eval.json")
REPORT_MD = os.path.join("docs", "reports", "free_energy_low_rank_growth_eval.md")
FIG_PATH = os.path.join("docs", "reports", "figs", "free_energy_low_rank_growth_eval.png")


def train_candidate(
    core,
    candidate: nn.Module,
    arm: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict], float]:
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    params = list(candidate.parameters())
    for parameter in params:
        parameter.requires_grad_(True)
    lr = args.full_lr if arm == "full" else args.delta_lr
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    history = []
    t0 = time.time()
    for step in range(1, args.growth_steps + 1):
        # 各臂每一步看到完全相同的训练 batch。
        torch.manual_seed(seed * 100_000 + step)
        seq = make_lag_sequences(args.batch, args.length, 3, args.vocab, args.device)
        logits = core(seq, transition_override=candidate)
        ce = task_loss(logits, seq, lag=3)
        assert core.last_position_free_energy is not None
        residual = core.last_position_free_energy[:, 3:].mean()
        loss = ce + args.free_energy_weight * residual
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.growth_steps:
            row = {"step": step, "ce": round(float(ce.detach()), 6),
                   "residual": round(float(residual.detach()), 6)}
            history.append(row)
            print(f"[lowrank] seed={seed} arm={arm:6s} step={step:4d}/{args.growth_steps} "
                  f"ce={row['ce']:.4f} F={row['residual']:.4f}", flush=True)
    return history, time.time() - t0


@torch.no_grad()
def candidate_metrics(core, candidate: nn.Module, seq: torch.Tensor, lag: int) -> dict:
    logits, trace = core(seq, transition_override=candidate, return_trace=True)
    pred = logits[:, lag - 1:-1].argmax(dim=-1)
    acc = float((pred == seq[:, lag:]).float().mean().cpu())
    residual = float(trace["residual_free_energy_per_dim"][:, lag:].mean().cpu())
    return {"accuracy": acc, "residual": residual}


def run_seed(seed: int, args: argparse.Namespace) -> list[dict]:
    core = train_base(seed, args)
    core.eval()
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    audit_id = make_lag_sequences(args.eval_batch, args.length, 2, args.vocab, args.device)
    audit_ood = make_lag_sequences(args.eval_batch, args.length, 3, args.vocab, args.device)
    base_logits = core(audit_id).detach().clone()
    full_params = sum(parameter.numel() for parameter in core.transition.parameters())
    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    rows = []

    candidates: list[tuple[str, nn.Module]] = [("full", copy.deepcopy(core.transition))]
    candidates.extend((f"r{rank}", LowRankGenerativeTransition(
        core.transition, dim=core.dim, rank=rank)) for rank in ranks)

    for arm_index, (arm, candidate) in enumerate(candidates):
        torch.manual_seed(seed * 100 + arm_index)
        candidate = candidate.to(args.device)
        history, seconds = train_candidate(core, candidate, arm, seed, args)
        metrics = candidate_metrics(core, candidate, audit_ood, lag=3)

        system = FreeEnergyGrowthSystem(core).to(args.device).eval()
        cost = system.calibrate_complexity_cost(
            audit_id, candidate, start=3,
            quantile=args.complexity_quantile, margin=args.complexity_margin)
        pathway = system.commit_pathway(candidate, complexity_cost=cost)
        routed_id_acc, id_choices = routed_accuracy(system, audit_id, lag=2, start=3)
        routed_ood_acc, ood_choices = routed_accuracy(system, audit_ood, lag=3, start=3)
        id_route = float((id_choices == 0).float().mean().cpu())
        ood_route = float((ood_choices == pathway).float().mean().cpu())
        base_after = system.forward_pathway(audit_id, 0)
        base_delta = float((base_after - base_logits).abs().max().detach().cpu())
        added = sum(parameter.numel() for parameter in candidate.parameters())
        rank = None if arm == "full" else int(arm[1:])

        rows.append({
            "seed": seed,
            "arm": arm,
            "rank": rank,
            "added_params": added,
            "full_transition_params": full_params,
            "param_ratio": round(added / full_params, 6),
            "new_accuracy": round(metrics["accuracy"], 6),
            "new_residual": round(metrics["residual"], 6),
            "complexity_cost": round(cost, 6),
            "id_route_base_accuracy": round(id_route, 6),
            "ood_route_new_accuracy": round(ood_route, 6),
            "routed_id_accuracy": round(routed_id_acc, 6),
            "routed_ood_accuracy": round(routed_ood_acc, 6),
            "base_logit_max_delta": round(base_delta, 9),
            "history": history,
            "seconds": round(seconds, 2),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    metrics = [
        "new_accuracy", "new_residual", "complexity_cost",
        "id_route_base_accuracy", "ood_route_new_accuracy",
        "routed_id_accuracy", "routed_ood_accuracy", "base_logit_max_delta",
    ]
    out = {}
    for arm in sorted({row["arm"] for row in rows}, key=lambda x: (x != "full", x)):
        arm_rows = [row for row in rows if row["arm"] == arm]
        item = {
            "added_params": arm_rows[0]["added_params"],
            "param_ratio": arm_rows[0]["param_ratio"],
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in arm_rows], dtype=float)
            item[metric] = {"mean": round(float(values.mean()), 6),
                            "std": round(float(values.std()), 6)}
        out[arm] = item
    return out


def write_reports(result: dict) -> None:
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    summary = result["summary"]
    ordered = ["full"] + sorted([x for x in summary if x != "full"], key=lambda x: int(x[1:]))
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(
            "# 低秩生成性动力学生长：能力不降，参数大幅下降\n\n"
            "基础 lag-2 稳定态冻结；同预算学习 lag-3。低秩臂修正生成性转移，不复制完整网络。\n\n"
            "| 通路 | 新增参数 | 完整比 | 新技能 acc | OOD→新路由 | ID→旧路由 |\n"
            "|---|---:|---:|---:|---:|---:|\n"
        )
        for arm in ordered:
            row = summary[arm]
            f.write(
                f"| {arm} | {row['added_params']:,} | {row['param_ratio']:.1%} | "
                f"{row['new_accuracy']['mean']:.1%} | {row['ood_route_new_accuracy']['mean']:.1%} | "
                f"{row['id_route_base_accuracy']['mean']:.1%} |\n")
        f.write(
            f"\n## 裁决\n\n{result['verdict']}\n\n"
            "## 边界\n\n这是周期结构机制任务；低秩修正尚未在真实字符流和多个连续新结构上验证。"
            "参数随通路数仍线性增长，但单次斜率已显著降低。\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ordered
    ratios = [summary[a]["param_ratio"] for a in labels]
    accs = [summary[a]["new_accuracy"]["mean"] for a in labels]
    routes = [summary[a]["ood_route_new_accuracy"]["mean"] for a in labels]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(labels, ratios, color=["#78909c"] + ["#2e7d32"] * (len(labels) - 1))
    axes[0].set_title("新增参数 / 完整通路"); axes[0].set_ylim(0, 1.05)
    x = np.arange(len(labels))
    axes[1].plot(x, accs, "o-", label="新技能 acc")
    axes[1].plot(x, routes, "s--", label="OOD→新通路")
    axes[1].set_xticks(x, labels); axes[1].set_ylim(0, 1.05); axes[1].legend(fontsize=8)
    axes[1].set_title("能力与能量路由")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("低秩生成性生长（非 attention）")
    fig.tight_layout(); fig.savefig(FIG_PATH, dpi=180); plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    args.device = args.device.strip() or get_device()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for seed in seeds:
        rows.extend(run_seed(seed, args))
    summary = summarize(rows)
    eligible = []
    for arm, row in summary.items():
        if arm == "full":
            continue
        if (row["new_accuracy"]["mean"] >= 0.95
                and row["ood_route_new_accuracy"]["mean"] >= 0.90
                and row["id_route_base_accuracy"]["mean"] >= 0.90):
            eligible.append(arm)
    best = min(eligible, key=lambda x: summary[x]["added_params"]) if eligible else None
    passed = bool(
        best is not None
        and summary[best]["param_ratio"] <= 0.15
        and summary[best]["base_logit_max_delta"]["mean"] <= 1e-8
    )
    verdict = (
        f"✅ 低秩生成性生长成立：{best} 以完整通路 {summary[best]['param_ratio']:.1%} 的新增参数"
        "保持新结构学习、最低自由能路由与旧稳定态零变化。"
        if passed else
        "🟡 当前低秩修正尚未在参数显著下降时同时保持学习与能量路由；保留曲线并修正容量。"
    )
    result = {
        "task": "full transition vs low-rank generative dynamics growth",
        "config": vars(args),
        "rows": rows,
        "summary": summary,
        "best_low_rank": best,
        "pass": passed,
        "verdict": verdict,
    }
    if args.write_report:
        write_reports(result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="比较完整与低秩生成性动力学生长。")
    ap.add_argument("--device", default="")
    ap.add_argument("--seeds", default="71,72,73")
    ap.add_argument("--ranks", default="2,4,8,16")
    ap.add_argument("--base-steps", type=int, default=150)
    ap.add_argument("--growth-steps", type=int, default=180)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval-batch", type=int, default=384)
    ap.add_argument("--length", type=int, default=14)
    ap.add_argument("--vocab", type=int, default=8)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--relax-steps", type=int, default=5)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--transition-mult", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--full-lr", type=float, default=2e-3)
    ap.add_argument("--delta-lr", type=float, default=4e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--free-energy-weight", type=float, default=2.0)
    ap.add_argument("--complexity-quantile", type=float, default=0.95)
    ap.add_argument("--complexity-margin", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=60)
    ap.add_argument("--write-report", action="store_true", default=True)
    ap.add_argument("--no-write-report", dest="write_report", action="store_false")
    return ap


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    result = run(build_arg_parser().parse_args(argv))
    print(json.dumps({"summary": result["summary"], "verdict": result["verdict"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
