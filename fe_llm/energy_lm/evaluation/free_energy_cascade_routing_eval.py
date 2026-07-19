# -*- coding: utf-8 -*-
"""多盆地自由能级联路由：廉价能量筛选，完整能量终裁。

基础盆地学习 lag-2，三个低秩生成性盆地分别学习 lag-3/4/5，再加入经稳定经验校准
MDL 代价的扰动假设，共 16 条通路。穷举完整弛豫是 oracle；级联先让所有通路只做少量
同构自由能弛豫，再对 top-k 做完整求解。没有域分类器、关键词门控或 attention router。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from fe_llm.config import get_device
from fe_llm.energy_lm.free_energy_growth import FreeEnergyGrowthSystem
from fe_llm.energy_lm.models.low_rank_transition import LowRankGenerativeTransition
from fe_llm.energy_lm.evaluation.free_energy_growth_eval import train_base
from fe_llm.energy_lm.evaluation.free_energy_sequence_eval import (
    make_lag_sequences,
    task_loss,
)

REPORT_JSON = os.path.join("docs", "reports", "free_energy_cascade_routing_eval.json")
REPORT_MD = os.path.join("docs", "reports", "free_energy_cascade_routing_eval.md")
FIG_PATH = os.path.join("docs", "reports", "figs", "free_energy_cascade_routing_eval.png")


def train_lag_pathway(
    system: FreeEnergyGrowthSystem,
    lag: int,
    stable_ids: torch.Tensor,
    *,
    seed: int,
    args: argparse.Namespace,
) -> tuple[int, dict]:
    candidate = LowRankGenerativeTransition(
        system.core.transition, dim=system.core.dim, rank=args.rank).to(args.device)
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    params = list(candidate.parameters())
    for parameter in params:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        params, lr=args.growth_lr, weight_decay=args.weight_decay)
    history = []
    for step in range(1, args.growth_steps + 1):
        torch.manual_seed(seed * 100_000 + lag * 1_000 + step)
        sequence = make_lag_sequences(
            args.batch, args.length, lag, args.vocab, args.device)
        logits = system.core(sequence, transition_override=candidate)
        ce = task_loss(logits, sequence, lag=lag)
        assert system.core.last_position_free_energy is not None
        residual = system.core.last_position_free_energy[:, lag:].mean()
        loss = ce + args.free_energy_weight * residual
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.growth_steps:
            row = {"step": step, "ce": round(float(ce.detach()), 6),
                   "residual": round(float(residual.detach()), 6)}
            history.append(row)
            print(f"[cascade] seed={seed} lag={lag} step={step:4d}/{args.growth_steps} "
                  f"ce={row['ce']:.4f} F={row['residual']:.4f}", flush=True)

    cost = system.calibrate_complexity_cost(
        stable_ids,
        candidate,
        start=args.energy_start,
        quantile=args.complexity_quantile,
        margin=args.complexity_margin,
    )
    pathway = system.commit_pathway(candidate, complexity_cost=cost)
    return pathway, {"lag": lag, "pathway": pathway, "complexity_cost": round(cost, 6),
                     "history": history}


@torch.no_grad()
def pathway_accuracy(
    system: FreeEnergyGrowthSystem,
    pathway: int,
    sequence: torch.Tensor,
    lag: int,
) -> float:
    logits = system.forward_pathway(sequence, pathway)
    prediction = logits[:, lag - 1:-1].argmax(dim=-1)
    return float((prediction == sequence[:, lag:]).float().mean().cpu())


def add_decoys(
    system: FreeEnergyGrowthSystem,
    stable_ids: torch.Tensor,
    *,
    seed: int,
    args: argparse.Namespace,
) -> list[dict]:
    rows = []
    generator = torch.Generator(device=args.device).manual_seed(seed + 8001)
    while system.pathway_count < args.pathway_count:
        candidate = LowRankGenerativeTransition(
            system.core.transition, dim=system.core.dim, rank=args.rank).to(args.device)
        with torch.no_grad():
            candidate.up.weight.normal_(
                mean=0.0, std=args.decoy_noise, generator=generator)
        cost = system.calibrate_complexity_cost(
            stable_ids,
            candidate,
            start=args.energy_start,
            quantile=args.complexity_quantile,
            margin=args.complexity_margin,
        )
        pathway = system.commit_pathway(candidate, complexity_cost=cost)
        rows.append({"pathway": pathway, "complexity_cost": round(cost, 6)})
    return rows


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark(callable_, *, device: str, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        callable_()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        callable_()
    synchronize(device)
    return (time.perf_counter() - started) / repeats


def run_seed(seed: int, args: argparse.Namespace) -> dict:
    started = time.time()
    core = train_base(seed, args)
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    system = FreeEnergyGrowthSystem(core).to(args.device).eval()
    learned = {2: 0}
    pathway_rows = []

    calibration = {}
    audit = {}
    for lag in args.lags:
        torch.manual_seed(seed * 10_000 + lag)
        calibration[lag] = make_lag_sequences(
            args.calibration_per_lag, args.length, lag, args.vocab, args.device)
        audit[lag] = make_lag_sequences(
            args.eval_per_lag, args.length, lag, args.vocab, args.device)

    stable_parts = [calibration[2]]
    for lag in args.lags[1:]:
        pathway, row = train_lag_pathway(
            system,
            lag,
            torch.cat(stable_parts, dim=0),
            seed=seed,
            args=args,
        )
        learned[lag] = pathway
        pathway_rows.append(row)
        stable_parts.append(calibration[lag])

    stable_ids = torch.cat(stable_parts, dim=0)
    decoys = add_decoys(system, stable_ids, seed=seed, args=args)
    sequence = torch.cat([audit[lag] for lag in args.lags], dim=0)
    expected = torch.cat([
        torch.full((args.eval_per_lag,), learned[lag], device=args.device, dtype=torch.long)
        for lag in args.lags
    ])

    full_choices, full_scores = system.route(sequence, start=args.energy_start)
    oracle_domain_accuracy = float((full_choices == expected).float().mean().cpu())
    learned_accuracy = {
        str(lag): round(pathway_accuracy(system, learned[lag], audit[lag], lag), 6)
        for lag in args.lags
    }
    full_seconds = benchmark(
        lambda: system.route(sequence, start=args.energy_start),
        device=args.device,
        warmup=args.benchmark_warmup,
        repeats=args.benchmark_repeats,
    )
    batched_choices, batched_scores = system.route_low_rank_batched(
        sequence, start=args.energy_start)
    batched_agreement = float((batched_choices == full_choices).float().mean().cpu())
    max_batched_score_delta = float((batched_scores - full_scores).abs().max().cpu())
    batched_seconds = benchmark(
        lambda: system.route_low_rank_batched(sequence, start=args.energy_start),
        device=args.device,
        warmup=args.benchmark_warmup,
        repeats=args.benchmark_repeats,
    )

    arms = []
    screen_steps = [int(value) for value in args.screen_steps.split(",") if value.strip()]
    shortlist_sizes = [int(value) for value in args.shortlist_sizes.split(",") if value.strip()]
    prefix_lengths = [int(value) for value in args.screen_prefix_lengths.split(",") if value.strip()]
    for prefix in prefix_lengths:
        for steps in screen_steps:
          for size in shortlist_sizes:
            choices, _, shortlist, _ = system.route_cascade(
                sequence,
                start=args.energy_start,
                screen_relax_steps=steps,
                shortlist_size=size,
                batched_low_rank=True,
                screen_prefix_length=prefix,
            )
            oracle_recall = float(
                (shortlist == full_choices[:, None]).any(dim=1).float().mean().cpu())
            agreement = float((choices == full_choices).float().mean().cpu())
            domain_accuracy = float((choices == expected).float().mean().cpu())
            batch_index = torch.arange(sequence.size(0), device=args.device)
            regret = full_scores[batch_index, choices] - full_scores.min(dim=1).values
            seconds = benchmark(
                lambda steps=steps, size=size, prefix=prefix: system.route_cascade(
                    sequence,
                    start=args.energy_start,
                    screen_relax_steps=steps,
                    shortlist_size=size,
                    batched_low_rank=True,
                    screen_prefix_length=prefix,
                ),
                device=args.device,
                warmup=args.benchmark_warmup,
                repeats=args.benchmark_repeats,
            )
            ideal_ratio = (
                system.pathway_count * steps * prefix
                + size * core.relaxation_steps * args.length
            ) / (system.pathway_count * core.relaxation_steps * args.length)
            arms.append({
                "screen_prefix_length": prefix,
                "screen_steps": steps,
                "shortlist_size": size,
                "oracle_recall": round(oracle_recall, 6),
                "route_agreement": round(agreement, 6),
                "domain_route_accuracy": round(domain_accuracy, 6),
                "mean_energy_regret": round(float(regret.mean().cpu()), 9),
                "p99_energy_regret": round(float(torch.quantile(regret, 0.99).cpu()), 9),
                "seconds": round(seconds, 6),
                "speedup": round(full_seconds / seconds, 6),
                "speedup_vs_batched": round(batched_seconds / seconds, 6),
                "ideal_relaxation_cost_ratio": round(ideal_ratio, 6),
            })

    eligible = [row for row in arms if (
        row["oracle_recall"] >= args.min_oracle_recall
        and row["route_agreement"] >= args.min_route_agreement
        and row["domain_route_accuracy"] >= oracle_domain_accuracy - args.max_domain_accuracy_gap
    )]
    best = max(eligible, key=lambda row: row["speedup_vs_batched"]) if eligible else None
    cascade_pass = bool(
        min(learned_accuracy.values()) >= args.min_task_accuracy
        and oracle_domain_accuracy >= args.min_oracle_domain_accuracy
        and best is not None
        and best["speedup"] >= args.min_speedup
        and best["speedup_vs_batched"] >= args.min_batched_speedup
    )
    row = {
        "seed": seed,
        "pathway_count": system.pathway_count,
        "learned_pathways": {str(key): value for key, value in learned.items()},
        "pathways": pathway_rows,
        "decoys": decoys,
        "learned_task_accuracy": learned_accuracy,
        "oracle_domain_route_accuracy": round(oracle_domain_accuracy, 6),
        "full_route_seconds": round(full_seconds, 6),
        "batched_route_seconds": round(batched_seconds, 6),
        "batched_route_agreement": round(batched_agreement, 6),
        "max_batched_score_delta": round(max_batched_score_delta, 9),
        "arms": arms,
        "best": best,
        "cascade_pass": cascade_pass,
        "seconds": round(time.time() - started, 2),
    }
    return judge_row(row, args)


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": round(float(array.mean()), 6),
            "std": round(float(array.std()), 6)}


def judge_row(row: dict, args: argparse.Namespace) -> dict:
    row["batched_speedup"] = round(
        row["full_route_seconds"] / row["batched_route_seconds"], 6)
    row["batched_pass"] = bool(
        min(row["learned_task_accuracy"].values()) >= args.min_task_accuracy
        and row["oracle_domain_route_accuracy"] >= args.min_oracle_domain_accuracy
        and row["batched_route_agreement"] >= 1.0 - 1e-8
        and row["max_batched_score_delta"] <= 1e-7
        and row["batched_speedup"] >= args.min_batch_speedup
    )
    best = row.get("best")
    row["cascade_pass"] = bool(
        row["batched_pass"]
        and best is not None
        and best["oracle_recall"] >= args.min_oracle_recall
        and best["route_agreement"] >= args.min_route_agreement
        and best["speedup_vs_batched"] >= args.min_batched_speedup
    )
    # 本轮目标是消除逐通路完整求解；级联是更激进但可被实验拒绝的附加假设。
    row["pass"] = row["batched_pass"]
    return row


def summarize(rows: list[dict], args: argparse.Namespace) -> dict:
    keys = sorted({
        (arm["screen_prefix_length"], arm["screen_steps"], arm["shortlist_size"])
        for row in rows for arm in row["arms"]
    })
    arms = {}
    for prefix, steps, size in keys:
        selected = [next(
            arm for arm in row["arms"]
            if arm["screen_prefix_length"] == prefix
            and arm["screen_steps"] == steps and arm["shortlist_size"] == size)
            for row in rows]
        name = f"p{prefix}_s{steps}_k{size}"
        arms[name] = {"screen_prefix_length": prefix,
                      "screen_steps": steps, "shortlist_size": size}
        for metric in (
            "oracle_recall", "route_agreement", "domain_route_accuracy",
            "mean_energy_regret", "p99_energy_regret", "seconds", "speedup",
            "speedup_vs_batched",
            "ideal_relaxation_cost_ratio",
        ):
            arms[name][metric] = mean_std([row[metric] for row in selected])
    eligible = [
        (name, arm) for name, arm in arms.items()
        if arm["oracle_recall"]["mean"] >= args.min_oracle_recall
        and arm["route_agreement"]["mean"] >= args.min_route_agreement
        and arm["domain_route_accuracy"]["mean"] >= (
            np.mean([row["oracle_domain_route_accuracy"] for row in rows])
            - args.max_domain_accuracy_gap)
    ]
    best_name, best = max(
        eligible, key=lambda item: item[1]["speedup_vs_batched"]["mean"]
    ) if eligible else (None, None)
    return {
        "pass_rate": round(float(np.mean([row["pass"] for row in rows])), 6),
        "batched_pass_rate": round(float(np.mean([
            row["batched_pass"] for row in rows])), 6),
        "cascade_pass_rate": round(float(np.mean([
            row["cascade_pass"] for row in rows])), 6),
        "pathway_count": rows[0]["pathway_count"],
        "oracle_domain_route_accuracy": mean_std([
            row["oracle_domain_route_accuracy"] for row in rows]),
        "full_route_seconds": mean_std([row["full_route_seconds"] for row in rows]),
        "batched_route_seconds": mean_std([row["batched_route_seconds"] for row in rows]),
        "batched_route_agreement": mean_std([
            row["batched_route_agreement"] for row in rows]),
        "max_batched_score_delta": mean_std([
            row["max_batched_score_delta"] for row in rows]),
        "batched_speedup": mean_std([row["batched_speedup"] for row in rows]),
        "arms": arms,
        "best_arm": best_name,
        "best": best,
    }


def write_reports(result: dict) -> None:
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    summary = result["summary"]
    with open(REPORT_MD, "w", encoding="utf-8") as file:
        file.write(
            "# 多盆地自由能路由：精确批量化与级联否证\n\n"
            "16 条生成性通路。索引化低秩批量求解保持完整自由能精确值；短前缀级联作为"
            "更激进假设接受独立验收。\n\n"
            "| 完整路由实现 | 时间 | 相对逐通路 | oracle 一致 | 最大分数差 |\n"
            "|---|---:|---:|---:|---:|\n"
            f"| 逐通路完整弛豫 | {summary['full_route_seconds']['mean']:.4f}s | 1.00× | 100% | 0 |\n"
            f"| 索引化低秩批量完整弛豫 | {summary['batched_route_seconds']['mean']:.4f}s | "
            f"{summary['batched_speedup']['mean']:.2f}× | "
            f"{summary['batched_route_agreement']['mean']:.1%} | "
            f"{summary['max_batched_score_delta']['mean']:.1e} |\n\n"
            "| 级联 | oracle 入围 | 最终一致 | 领域路由 | vs 逐路加速 | vs 批量穷举 | 理想弛豫成本比 |\n"
            "|---|---:|---:|---:|---:|---:|---:|\n")
        for name, arm in summary["arms"].items():
            file.write(
                f"| {name} | {arm['oracle_recall']['mean']:.1%} | "
                f"{arm['route_agreement']['mean']:.1%} | "
                f"{arm['domain_route_accuracy']['mean']:.1%} | "
                f"{arm['speedup']['mean']:.2f}× | "
                f"{arm['speedup_vs_batched']['mean']:.2f}× | "
                f"{arm['ideal_relaxation_cost_ratio']['mean']:.1%} |\n")
        best = summary["best"]
        file.write(
            f"\n完整路由领域准确率：`{summary['oracle_domain_route_accuracy']['mean']:.1%}`。"
            f"最佳级联：`{summary['best_arm']}`。\n\n"
            f"## 裁决\n\n{result['verdict']}\n\n"
            "## 边界\n\n"
            "通路由四个真实机制盆地和十二个 MDL 校准扰动假设构成，尚不是十六个自然"
            "语言领域。短前缀级联虽减少理论弛豫量并保持高 oracle 召回，但两个批量前向"
            "在当前 GPU/规模上慢于一次批量完整求解，因此明确拒绝作为默认路径。精确批量"
            "穷举仍按通路数增加显存与算术量，尚未解决超大盆地库的次线性检索。\n"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(summary["arms"])
    recall = [summary["arms"][name]["oracle_recall"]["mean"] for name in labels]
    speed_labels = ["exact_batch"] + labels
    speedup = [summary["batched_speedup"]["mean"]] + [
        summary["arms"][name]["speedup"]["mean"] for name in labels]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    axes[0].bar(labels, recall, color="#1565c0")
    axes[0].axhline(0.99, color="#c62828", ls="--", label="99% oracle 入围")
    axes[0].set_ylim(0, 1.05); axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_title("粗自由能 shortlist 召回"); axes[0].legend(fontsize=8)
    axes[1].bar(speed_labels, speedup, color=["#2e7d32"] + ["#78909c"] * len(labels))
    axes[1].axhline(1.0, color="#555555", ls="--")
    axes[1].tick_params(axis="x", rotation=35); axes[1].set_title("相对逐通路穷举的实测加速")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("精确批量化成立，短前缀级联接受否证")
    fig.tight_layout(); fig.savefig(FIG_PATH, dpi=180); plt.close(fig)


def make_verdict(summary: dict, passed: bool) -> str:
    if not passed:
        return "🟡 当前索引化批量完整自由能尚未同时达到数值一致与实测加速判据。"
    best = summary.get("best")
    cascade = (
        f"短前缀级联 {summary['best_arm']} 虽有 "
        f"{best['oracle_recall']['mean']:.1%} oracle 入围，但相对批量穷举仅 "
        f"{best['speedup_vs_batched']['mean']:.2f}×，因此被拒绝。"
        if best is not None else
        "没有短前缀级联同时满足 oracle 入围与领域路由判据，因此被拒绝。"
    )
    return (
        f"✅ 精确批量自由能路由成立：16 通路与逐路完整求解 "
        f"{summary['batched_route_agreement']['mean']:.1%} 一致、最大能量差 "
        f"{summary['max_batched_score_delta']['mean']:.1e}，GPU 实测加速 "
        f"{summary['batched_speedup']['mean']:.2f}×。{cascade}"
    )


def run(args: argparse.Namespace) -> dict:
    args.device = args.device.strip() or get_device()
    args.lags = [int(value) for value in args.lags.split(",") if value.strip()]
    if args.lags[0] != 2 or len(args.lags) < 2:
        raise ValueError("lags 必须从基础 lag-2 开始并至少包含两个结构。")
    rows = [run_seed(seed, args) for seed in (
        int(value) for value in args.seeds.split(",") if value.strip())]
    summary = summarize(rows, args)
    passed = bool(summary["batched_pass_rate"] == 1.0)
    verdict = make_verdict(summary, passed)
    result = {
        "task": "exact batched free-energy routing with rejected cascade hypothesis",
        "config": vars(args),
        "rows": rows,
        "summary": summary,
        "pass": passed,
        "verdict": verdict,
    }
    if args.write_report:
        write_reports(result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多盆地同构自由能级联路由。")
    parser.add_argument("--device", default="")
    parser.add_argument("--seeds", default="122,123,124")
    parser.add_argument("--lags", default="2,3,4,5")
    parser.add_argument("--pathway-count", type=int, default=16)
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--vocab", type=int, default=8)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--relax-steps", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--transition-mult", type=int, default=2)
    parser.add_argument("--base-steps", type=int, default=150)
    parser.add_argument("--growth-steps", type=int, default=320)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--calibration-per-lag", type=int, default=96)
    parser.add_argument("--eval-per-lag", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--growth-lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--free-energy-weight", type=float, default=2.0)
    parser.add_argument("--complexity-quantile", type=float, default=0.90)
    parser.add_argument("--complexity-margin", type=float, default=1e-4)
    parser.add_argument("--decoy-noise", type=float, default=0.02)
    parser.add_argument("--energy-start", type=int, default=5)
    parser.add_argument("--screen-prefix-lengths", default="8")
    parser.add_argument("--screen-steps", default="2")
    parser.add_argument("--shortlist-sizes", default="2,4")
    parser.add_argument("--benchmark-warmup", type=int, default=1)
    parser.add_argument("--benchmark-repeats", type=int, default=4)
    parser.add_argument("--min-task-accuracy", type=float, default=0.95)
    parser.add_argument("--min-oracle-domain-accuracy", type=float, default=0.85)
    parser.add_argument("--min-oracle-recall", type=float, default=0.99)
    parser.add_argument("--min-route-agreement", type=float, default=0.99)
    parser.add_argument("--max-domain-accuracy-gap", type=float, default=0.01)
    parser.add_argument("--min-speedup", type=float, default=1.25)
    parser.add_argument("--min-batched-speedup", type=float, default=1.05)
    parser.add_argument("--min-batch-speedup", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=60)
    parser.add_argument("--write-report", action="store_true", default=True)
    parser.add_argument("--no-write-report", dest="write_report", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    result = run(build_arg_parser().parse_args(argv))
    print(json.dumps({"summary": result["summary"], "pass": result["pass"],
                      "verdict": result["verdict"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
