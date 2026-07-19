# -*- coding: utf-8 -*-
"""低秩生成性生长在真实字符流上的端到端裁决。

沿用 OPUS 英文→反向英文结构变化与等字符边际噪声协议。完整臂复制转移+读出；低秩臂
分别用低秩动力学修正与低秩读出。所有臂先做纯 residual-F probe，通过可约性后才学习读出、
支付 MDL 结构代价并固化。
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
from fe_llm.energy_lm.models.low_rank_transition import (
    LowRankGenerativeTransition,
    LowRankReadout,
)
from fe_llm.energy_lm.evaluation.free_energy_reducibility_eval import (
    DATA_PATH,
    path_metrics,
    prepare_data,
    sample_chunks,
    shuffle_chunks,
    train_base,
    train_probe,
    train_readout,
)

REPORT_JSON = os.path.join("docs", "reports", "free_energy_low_rank_char_eval.json")
REPORT_MD = os.path.join("docs", "reports", "free_energy_low_rank_char_eval.md")
FIG_PATH = os.path.join("docs", "reports", "figs", "free_energy_low_rank_char_eval.png")


def make_candidate(core, arm: str, rank: int | None):
    if arm == "full":
        return copy.deepcopy(core.transition), copy.deepcopy(core.head)
    assert rank is not None
    return (
        LowRankGenerativeTransition(core.transition, dim=core.dim, rank=rank),
        LowRankReadout(core.head, in_dim=core.dim, out_dim=core.vocab_size, rank=rank),
    )


def run_seed(seed: int, data: dict, args: argparse.Namespace) -> list[dict]:
    core = train_base(seed, data["base_train"], data["vocab_size"], args)
    system_base = FreeEnergyGrowthSystem(core).to(args.device).eval()
    gen = torch.Generator(device=args.device).manual_seed(seed + 1901)
    base_valid = sample_chunks(data["base_valid"], args.eval_batch, args.length, gen)
    structure_valid = sample_chunks(data["structure_valid"], args.eval_batch, args.length, gen)
    noise_valid = shuffle_chunks(structure_valid.clone(), gen)
    base_logits = core(base_valid).detach().clone()
    structure_before = path_metrics(core, core.transition, structure_valid)
    noise_before = path_metrics(core, core.transition, noise_valid)

    full_transition_params = sum(p.numel() for p in core.transition.parameters())
    full_head_params = sum(p.numel() for p in core.head.parameters())
    full_total = full_transition_params + full_head_params
    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    arms: list[tuple[str, int | None]] = [("full", None)] + [(f"r{rank}", rank) for rank in ranks]
    rows = []

    for arm_index, (arm, rank) in enumerate(arms):
        torch.manual_seed(seed * 100 + arm_index)
        transition, head = make_candidate(core, arm, rank)
        transition = transition.to(args.device); head = head.to(args.device)
        system = FreeEnergyGrowthSystem(core).to(args.device).eval()
        t0 = time.time()

        probe_history = train_probe(
            system, transition, data["structure_train"], noise=False, seed=seed, args=args)
        probe_metrics = path_metrics(core, transition, structure_valid)
        reducibility = (
            structure_before["residual"] - probe_metrics["residual"]
        ) / max(1e-8, structure_before["residual"])
        reducible = reducibility >= args.min_reducibility

        readout_history: list[dict] = []
        consolidated = probe_metrics
        route_base = route_structure = None
        complexity_cost = None
        if reducible:
            for parameter in transition.parameters():
                parameter.requires_grad_(False)
            readout_history = train_readout(
                system, transition, head, data["structure_train"], seed=seed, args=args)
            consolidated = path_metrics(core, transition, structure_valid, head=head)
            complexity_cost = system.calibrate_complexity_cost(
                base_valid, transition, start=2,
                quantile=args.complexity_quantile, margin=args.complexity_margin)
            pathway = system.commit_pathway(
                transition, complexity_cost=complexity_cost, head=head)
            base_choices, _ = system.route(base_valid, start=2)
            structure_choices, _ = system.route(structure_valid, start=2)
            route_base = float((base_choices == 0).float().mean().cpu())
            route_structure = float((structure_choices == pathway).float().mean().cpu())

        # 噪声 probe 使用同类型、同秩但独立的临时容量，绝不固化。
        noise_transition, _ = make_candidate(core, arm, rank)
        noise_transition = noise_transition.to(args.device)
        noise_system = FreeEnergyGrowthSystem(core).to(args.device).eval()
        noise_history = train_probe(
            noise_system, noise_transition, data["structure_train"],
            noise=True, seed=seed, args=args)
        noise_after = path_metrics(core, noise_transition, noise_valid)
        noise_reducibility = (
            noise_before["residual"] - noise_after["residual"]
        ) / max(1e-8, noise_before["residual"])

        base_after = system.forward_pathway(base_valid, 0)
        base_delta = float((base_after - base_logits).abs().max().detach().cpu())
        transition_params = sum(p.numel() for p in transition.parameters())
        head_params = sum(p.numel() for p in head.parameters())

        def compact(metrics: dict) -> dict:
            return {k: round(float(v), 6) for k, v in metrics.items()
                    if k != "residual_per_sample"}

        rows.append({
            "seed": seed,
            "arm": arm,
            "rank": rank,
            "transition_params": transition_params,
            "head_params": head_params,
            "added_params": transition_params + head_params,
            "full_added_params": full_total,
            "transition_ratio": round(transition_params / full_transition_params, 6),
            "total_ratio": round((transition_params + head_params) / full_total, 6),
            "structure_reducibility": round(reducibility, 6),
            "noise_reducibility": round(noise_reducibility, 6),
            "reducible": reducible,
            "structure_before": {k: round(float(v), 6) for k, v in structure_before.items()
                                 if k != "residual_per_sample"},
            "structure_probe": compact(probe_metrics),
            "structure_consolidated": compact(consolidated),
            "noise_before": {k: round(float(v), 6) for k, v in noise_before.items()
                             if k != "residual_per_sample"},
            "noise_after": compact(noise_after),
            "complexity_cost": None if complexity_cost is None else round(complexity_cost, 6),
            "base_route_accuracy": None if route_base is None else round(route_base, 6),
            "structure_route_accuracy": None if route_structure is None else round(route_structure, 6),
            "base_logit_max_delta": round(base_delta, 9),
            "probe_history": probe_history,
            "readout_history": readout_history,
            "noise_history": noise_history,
            "seconds": round(time.time() - t0, 2),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for arm in sorted({row["arm"] for row in rows}, key=lambda x: (x != "full", x)):
        arm_rows = [row for row in rows if row["arm"] == arm]
        item = {key: arm_rows[0][key] for key in (
            "transition_params", "head_params", "added_params",
            "transition_ratio", "total_ratio")}
        metrics = [
            "structure_reducibility", "noise_reducibility",
            "base_route_accuracy", "structure_route_accuracy", "base_logit_max_delta",
        ]
        for metric in metrics:
            values = np.asarray([row[metric] for row in arm_rows if row[metric] is not None], dtype=float)
            item[metric] = ({"mean": None, "std": None} if values.size == 0 else {
                "mean": round(float(values.mean()), 6),
                "std": round(float(values.std()), 6),
            })
        for stage in ("structure_probe", "structure_consolidated", "noise_after"):
            for metric in ("bpc", "residual", "accuracy"):
                values = np.asarray([row[stage][metric] for row in arm_rows], dtype=float)
                item[f"{stage}_{metric}"] = {
                    "mean": round(float(values.mean()), 6),
                    "std": round(float(values.std()), 6),
                }
        item["reducible_rate"] = round(float(np.mean([row["reducible"] for row in arm_rows])), 6)
        out[arm] = item
    return out


def write_reports(result: dict) -> None:
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    summary = result["summary"]
    ordered = ["full"] + sorted([a for a in summary if a != "full"], key=lambda x: int(x[1:]))
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(
            "# 真实字符流：低秩自由能生长端到端裁决\n\n"
            "OPUS 英文→反向英文；等字符边际噪声对照。低秩转移与低秩读出均只新增残差参数。\n\n"
            "| 通路 | 转移比 | 总参数比 | 结构可约率 | 噪声可约率 | 固化 BPC | 结构→新路由 |\n"
            "|---|---:|---:|---:|---:|---:|---:|\n")
        for arm in ordered:
            row = summary[arm]
            route = row["structure_route_accuracy"]["mean"]
            f.write(
                f"| {arm} | {row['transition_ratio']:.1%} | {row['total_ratio']:.1%} | "
                f"{row['structure_reducibility']['mean']:.1%} | {row['noise_reducibility']['mean']:.1%} | "
                f"{row['structure_consolidated_bpc']['mean']:.3f} | "
                f"{('n/a' if route is None else f'{route:.1%}')} |\n")
        f.write(
            f"\n## 裁决\n\n{result['verdict']}\n\n"
            "## 边界\n\n结构变化仍是受控反向英文；低秩容量尚未验证真实跨领域迁移与多个连续通路。\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ordered
    ratios = [summary[a]["total_ratio"] for a in labels]
    bpcs = [summary[a]["structure_consolidated_bpc"]["mean"] for a in labels]
    reducibility = [summary[a]["structure_reducibility"]["mean"] for a in labels]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(labels, ratios, color=["#78909c"] + ["#2e7d32"] * (len(labels)-1))
    axes[0].set_title("端到端新增参数 / 完整生长"); axes[0].set_ylim(0, 1.05)
    x = np.arange(len(labels))
    axes[1].plot(x, bpcs, "o-", label="固化后 BPC")
    axes[1].plot(x, reducibility, "s--", label="结构可约率")
    axes[1].set_xticks(x, labels); axes[1].legend(fontsize=8); axes[1].set_title("字符能力与自由能")
    for ax in axes: ax.grid(axis="y", alpha=0.2)
    fig.suptitle("真实字符流低秩自由能生长")
    fig.tight_layout(); fig.savefig(FIG_PATH, dpi=180); plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    args.device = args.device.strip() or get_device()
    data = prepare_data(args)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for seed in seeds:
        rows.extend(run_seed(seed, data, args))
    summary = summarize(rows)
    eligible = []
    full_bpc = summary["full"]["structure_consolidated_bpc"]["mean"]
    for arm, row in summary.items():
        if arm == "full" or row["structure_route_accuracy"]["mean"] is None:
            continue
        if (row["reducible_rate"] == 1.0
                and row["structure_reducibility"]["mean"] >= args.min_reducibility
                and row["noise_reducibility"]["mean"] < args.min_reducibility
                and row["structure_consolidated_bpc"]["mean"]
                    <= full_bpc * (1.0 + args.max_relative_bpc_gap)
                and row["base_route_accuracy"]["mean"] >= 0.90
                and row["structure_route_accuracy"]["mean"] >= 0.90):
            eligible.append(arm)
    best = min(eligible, key=lambda a: summary[a]["added_params"]) if eligible else None
    passed = bool(best is not None and summary[best]["total_ratio"] <= 0.25
                  and summary[best]["base_logit_max_delta"]["mean"] <= 1e-8)
    verdict = (
        f"✅ 真实字符流低秩生长成立：{best} 以完整端到端生长 {summary[best]['total_ratio']:.1%} 参数，"
        "通过结构可约性、噪声拒绝、字符 BPC、MDL 路由与旧态零变化全部判据。"
        if passed else
        "🟡 真实字符流低秩生长尚未在显著省参时通过全部能量与能力判据；保留秩曲线继续修正。"
    )
    result = {
        "task": "real char stream full vs low-rank free-energy growth",
        "config": vars(args),
        "data": {"vocab_size": data["vocab_size"], "sentence_counts": data["sentence_counts"]},
        "rows": rows,
        "summary": summary,
        "best_low_rank": best,
        "pass": passed,
        "verdict": verdict,
    }
    if args.write_report: write_reports(result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="真实字符流 full vs 低秩自由能生长。")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--device", default="")
    ap.add_argument("--seeds", default="81,82,83")
    ap.add_argument("--ranks", default="8,16,20")
    ap.add_argument("--sentences", type=int, default=12000)
    ap.add_argument("--base-steps", type=int, default=350)
    ap.add_argument("--probe-steps", type=int, default=240)
    ap.add_argument("--readout-steps", type=int, default=240)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval-batch", type=int, default=384)
    ap.add_argument("--stream-batch", type=int, default=128)
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--relax-steps", type=int, default=5)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--transition-mult", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--probe-lr", type=float, default=4e-3)
    ap.add_argument("--readout-lr", type=float, default=4e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--free-energy-weight", type=float, default=2.0)
    ap.add_argument("--probe-ce-weight", type=float, default=0.0)
    ap.add_argument("--threshold-quantile", type=float, default=0.99)
    ap.add_argument("--min-growth-fraction", type=float, default=0.25)
    ap.add_argument("--min-reducibility", type=float, default=0.30)
    ap.add_argument("--probe-noise", type=float, default=1e-3)
    ap.add_argument("--complexity-quantile", type=float, default=0.95)
    ap.add_argument("--complexity-margin", type=float, default=1e-4)
    ap.add_argument("--max-relative-bpc-gap", type=float, default=0.10,
                    help="低秩固化 BPC 相对完整通路最多允许高 10%")
    ap.add_argument("--log-every", type=int, default=60)
    ap.add_argument("--write-report", action="store_true", default=True)
    ap.add_argument("--no-write-report", dest="write_report", action="store_false")
    return ap


def main(argv: list[str] | None = None) -> int:
    try: sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception: pass
    result = run(build_arg_parser().parse_args(argv))
    print(json.dumps({"summary": result["summary"], "verdict": result["verdict"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
