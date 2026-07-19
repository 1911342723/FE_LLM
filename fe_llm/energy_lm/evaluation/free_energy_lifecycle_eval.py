# -*- coding: utf-8 -*-
"""真实字符流上的多盆地自由能生命周期实验。

基础模型只学习正常 OPUS 英文。随后依次出现两个保持逐句字符边际的结构变化：整句
反向与句内字符排序。系统只能根据所有现有通路的 residual free-energy 决定是否增加
低秩生成性动力学，不使用领域名、技能标签或关键词门控。

实验同时审计：连续生长、重复结构复用、最低能路由、冗余盆地合并、长期不活跃盆地
回收，以及整个过程中基础稳定态 logits 严格不变。合并环节显式注入一个持久化层面的
重复快照，用于验证容量去重机制；它不冒充“两个独立学习盆地自然收敛为同一结构”。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections.abc import Callable

import numpy as np
import torch

from fe_llm.config import get_device
from fe_llm.energy_lm.free_energy_growth import FreeEnergyGrowthSystem
from fe_llm.energy_lm.models.low_rank_transition import (
    LowRankGenerativeTransition,
    LowRankReadout,
)
from fe_llm.energy_lm.evaluation.free_energy_reducibility_eval import (
    DATA_PATH,
    build_stream,
    encode_stream,
    load_sentences,
    make_vocab,
    path_metrics,
    sample_chunks,
    train_base,
    train_probe,
    train_readout,
)

REPORT_JSON = os.path.join("docs", "reports", "free_energy_lifecycle_eval.json")
REPORT_MD = os.path.join("docs", "reports", "free_energy_lifecycle_eval.md")
FIG_PATH = os.path.join("docs", "reports", "figs", "free_energy_lifecycle_eval.png")


def reverse_sentence(text: str) -> str:
    return text[::-1]


def sort_characters(text: str) -> str:
    return "".join(sorted(text))


def transformed_stream(sentences: list[str], transform: Callable[[str], str]) -> str:
    return "\n".join(transform(text) for text in sentences) + "\n"


def prepare_lifecycle_data(args: argparse.Namespace) -> dict:
    sentences = load_sentences(args.data, args.sentences)
    n_base_train = int(len(sentences) * 0.50)
    n_base_valid = int(len(sentences) * 0.125)
    cursor = n_base_train + n_base_valid
    remaining = sentences[cursor:]
    split_structures = len(remaining) // 2
    reverse_rows = remaining[:split_structures]
    sorted_rows = remaining[split_structures:]

    def split(rows: list[str]) -> tuple[list[str], list[str]]:
        point = len(rows) // 2
        return rows[:point], rows[point:]

    reverse_train, reverse_valid = split(reverse_rows)
    sorted_train, sorted_valid = split(sorted_rows)
    texts = {
        "base_train": build_stream(sentences[:n_base_train]),
        "base_valid": build_stream(sentences[n_base_train:cursor]),
        "reverse_train": transformed_stream(reverse_train, reverse_sentence),
        "reverse_valid": transformed_stream(reverse_valid, reverse_sentence),
        "sorted_train": transformed_stream(sorted_train, sort_characters),
        "sorted_valid": transformed_stream(sorted_valid, sort_characters),
    }
    vocab, id_to_char = make_vocab(*texts.values())
    encoded = {name: encode_stream(text, vocab, args.device) for name, text in texts.items()}
    encoded.update({
        "vocab_size": len(vocab),
        "vocab": "".join(id_to_char),
        "sentence_counts": {
            "base_train": n_base_train,
            "base_valid": n_base_valid,
            "reverse_train": len(reverse_train),
            "reverse_valid": len(reverse_valid),
            "sorted_train": len(sorted_train),
            "sorted_valid": len(sorted_valid),
        },
    })
    return encoded


def make_low_rank_candidate(core, rank: int):
    return (
        LowRankGenerativeTransition(core.transition, dim=core.dim, rank=rank),
        LowRankReadout(core.head, in_dim=core.dim, out_dim=core.vocab_size, rank=rank),
    )


@torch.no_grad()
def best_raw_residual(
    system: FreeEnergyGrowthSystem,
    ids: torch.Tensor,
    *,
    start: int,
) -> float:
    scores = torch.stack([
        system.residual_scores(ids, pathway, start=start)
        for pathway in range(system.pathway_count)
    ], dim=1)
    return float(scores.min(dim=1).values.mean().cpu())


@torch.no_grad()
def route_fraction(
    system: FreeEnergyGrowthSystem,
    ids: torch.Tensor,
    pathway: int,
    *,
    start: int,
) -> float:
    choices, _ = system.route(ids, start=start)
    return float((choices == pathway).float().mean().cpu())


def compact_metrics(metrics: dict) -> dict[str, float]:
    return {
        key: round(float(value), 6)
        for key, value in metrics.items()
        if key != "residual_per_sample"
    }


def grow_structure(
    name: str,
    system: FreeEnergyGrowthSystem,
    train_stream: torch.Tensor,
    valid_ids: torch.Tensor,
    stable_ids: torch.Tensor,
    *,
    seed: int,
    args: argparse.Namespace,
) -> tuple[int | None, dict]:
    generator = torch.Generator(device=args.device).manual_seed(seed + 31)
    encounter = sample_chunks(train_stream, args.stream_batch, args.length, generator)
    high_energy, pressure, encounter_energy = system.should_grow(
        encounter, start=args.energy_start, min_fraction=args.min_growth_fraction)
    before = best_raw_residual(system, valid_ids, start=args.energy_start)

    transition, head = make_low_rank_candidate(system.core, args.rank)
    transition = transition.to(args.device)
    head = head.to(args.device)
    probe_history = train_probe(
        system, transition, train_stream, noise=False, seed=seed + 101, args=args)
    probe = path_metrics(
        system.core, transition, valid_ids, start=args.energy_start)
    reducibility = (before - probe["residual"]) / max(1e-8, before)
    committed = bool(high_energy and reducibility >= args.min_reducibility)
    pathway = None
    complexity_cost = None
    readout_history: list[dict] = []
    consolidated = probe
    if committed:
        for parameter in transition.parameters():
            parameter.requires_grad_(False)
        readout_history = train_readout(
            system, transition, head, train_stream, seed=seed + 211, args=args)
        consolidated = path_metrics(
            system.core, transition, valid_ids, start=args.energy_start, head=head)
        complexity_cost = system.calibrate_complexity_cost(
            stable_ids,
            transition,
            start=args.energy_start,
            quantile=args.complexity_quantile,
            margin=args.complexity_margin,
        )
        pathway = system.commit_pathway(
            transition, complexity_cost=complexity_cost, head=head)

    return pathway, {
        "name": name,
        "high_energy": high_energy,
        "growth_pressure": round(pressure, 6),
        "encounter_energy": round(encounter_energy, 6),
        "before_best_residual": round(before, 6),
        "probe": compact_metrics(probe),
        "reducibility": round(reducibility, 6),
        "committed": committed,
        "pathway": pathway,
        "complexity_cost": None if complexity_cost is None else round(complexity_cost, 6),
        "consolidated": compact_metrics(consolidated),
        "probe_history": probe_history,
        "readout_history": readout_history,
    }


def run_seed(seed: int, data: dict, args: argparse.Namespace) -> dict:
    started = time.time()
    core = train_base(seed, data["base_train"], data["vocab_size"], args)
    system = FreeEnergyGrowthSystem(core).to(args.device).eval()
    generator = torch.Generator(device=args.device).manual_seed(seed + 3001)
    samples = {
        name: sample_chunks(data[f"{name}_valid"], args.eval_batch, args.length, generator)
        for name in ("base", "reverse", "sorted")
    }
    base_logits_before = core(samples["base"]).detach().clone()
    base_metrics = path_metrics(core, core.transition, samples["base"], start=args.energy_start)
    threshold = system.calibrate_threshold(
        samples["base"], start=args.energy_start, quantile=args.threshold_quantile)
    path_counts = [system.pathway_count]

    reverse_path, reverse_growth = grow_structure(
        "reverse",
        system,
        data["reverse_train"],
        samples["reverse"],
        samples["base"],
        seed=seed + 1000,
        args=args,
    )
    path_counts.append(system.pathway_count)
    stable_for_sorted = torch.cat((samples["base"], samples["reverse"]), dim=0)
    sorted_path, sorted_growth = grow_structure(
        "sorted",
        system,
        data["sorted_train"],
        samples["sorted"],
        stable_for_sorted,
        seed=seed + 2000,
        args=args,
    )
    path_counts.append(system.pathway_count)

    if reverse_path is None or sorted_path is None:
        return {
            "seed": seed,
            "threshold": round(threshold, 6),
            "base": compact_metrics(base_metrics),
            "growth": [reverse_growth, sorted_growth],
            "path_counts": path_counts,
            "pass": False,
            "failure": "两个真实字符结构未全部通过可约性闸门",
            "seconds": round(time.time() - started, 2),
        }

    route_before_lifecycle = {
        "base": route_fraction(system, samples["base"], 0, start=args.energy_start),
        "reverse": route_fraction(
            system, samples["reverse"], reverse_path, start=args.energy_start),
        "sorted": route_fraction(
            system, samples["sorted"], sorted_path, start=args.energy_start),
    }
    reverse_metrics_raw = path_metrics(
        core, system.transition_for(reverse_path), samples["reverse"],
        start=args.energy_start, head=system.head_for(reverse_path))
    sorted_metrics_raw = path_metrics(
        core, system.transition_for(sorted_path), samples["sorted"],
        start=args.energy_start, head=system.head_for(sorted_path))
    metrics_before_lifecycle = {
        "reverse": compact_metrics(reverse_metrics_raw),
        "sorted": compact_metrics(sorted_metrics_raw),
    }

    repeat_generator = torch.Generator(device=args.device).manual_seed(seed + 4001)
    repeat_ids = sample_chunks(
        data["reverse_train"], args.stream_batch, args.length, repeat_generator)
    repeat_grow, repeat_pressure, repeat_energy = system.should_grow(
        repeat_ids, start=args.energy_start, min_fraction=args.min_growth_fraction)
    repeat_route = route_fraction(
        system, samples["reverse"], reverse_path, start=args.energy_start)
    path_counts.append(system.pathway_count)

    # 故障注入：持久化层把同一盆地快照重复注册。旧盆地因同能量、更低 MDL 代价胜出。
    duplicate_transition = copy.deepcopy(system.transition_for(reverse_path))
    duplicate_head = copy.deepcopy(system.head_for(reverse_path))
    duplicate_cost = float(system.pathway_costs[reverse_path].cpu()) + args.duplicate_cost_margin
    duplicate_path = system.commit_pathway(
        duplicate_transition, complexity_cost=duplicate_cost, head=duplicate_head)
    params_with_duplicate = system.added_parameter_count()
    path_counts.append(system.pathway_count)
    merged, merge_audit = system.merge_pathways_if_redundant(
        reverse_path,
        duplicate_path,
        samples["reverse"],
        start=args.energy_start,
        energy_tolerance=args.merge_energy_tolerance,
        min_covered_fraction=args.merge_min_covered,
    )
    params_after_merge = system.added_parameter_count()
    path_counts.append(system.pathway_count)

    recent_ids = torch.cat((samples["base"], samples["sorted"]), dim=0)
    recent_stats = system.pathway_statistics(recent_ids, start=args.energy_start)
    reverse_recent_fraction = float(
        recent_stats["route_fraction"][reverse_path].cpu())
    retired, retire_audit = system.retire_pathway_if_inactive(
        reverse_path,
        recent_ids,
        start=args.energy_start,
        max_route_fraction=args.retire_max_route_fraction,
    )
    path_counts.append(system.pathway_count)
    sorted_after_index = (
        sorted_path - 1 if retired and sorted_path > reverse_path else sorted_path)

    route_after_retire = {
        "base": route_fraction(system, samples["base"], 0, start=args.energy_start),
        "sorted": route_fraction(
            system, samples["sorted"], sorted_after_index, start=args.energy_start),
    }
    sorted_after = path_metrics(
        core,
        system.transition_for(sorted_after_index),
        samples["sorted"],
        start=args.energy_start,
        head=system.head_for(sorted_after_index),
    )
    reverse_returns, return_pressure, return_energy = system.should_grow(
        samples["reverse"],
        start=args.energy_start,
        min_fraction=args.min_growth_fraction,
    )
    base_delta = float((core(samples["base"]) - base_logits_before).abs().max().cpu())
    sorted_bpc_delta = abs(sorted_after["bpc"] - sorted_metrics_raw["bpc"])

    passed = bool(
        reverse_growth["committed"]
        and sorted_growth["committed"]
        and min(route_before_lifecycle.values()) >= args.min_route_accuracy
        and not repeat_grow
        and repeat_route >= args.min_route_accuracy
        and merged
        and params_after_merge < params_with_duplicate
        and retired
        and reverse_recent_fraction <= args.retire_max_route_fraction
        and min(route_after_retire.values()) >= args.min_route_accuracy
        and reverse_returns
        and sorted_bpc_delta <= 1e-8
        and base_delta <= 1e-8
    )
    return {
        "seed": seed,
        "threshold": round(threshold, 6),
        "base": compact_metrics(base_metrics),
        "growth": [reverse_growth, sorted_growth],
        "route_before_lifecycle": {
            key: round(value, 6) for key, value in route_before_lifecycle.items()
        },
        "metrics_before_lifecycle": metrics_before_lifecycle,
        "repeat_reverse": {
            "growth_triggered": repeat_grow,
            "growth_pressure": round(repeat_pressure, 6),
            "energy": round(repeat_energy, 6),
            "route_existing": round(repeat_route, 6),
        },
        "merge": {
            "fault_injection": "exact persisted snapshot duplicate",
            "merged": merged,
            **{key: round(float(value), 6) if key != "survivor" else int(value)
               for key, value in merge_audit.items()},
            "params_with_duplicate": params_with_duplicate,
            "params_after_merge": params_after_merge,
        },
        "retirement": {
            "retired": retired,
            "recent_route_fraction": round(reverse_recent_fraction, 6),
            **{key: round(float(value), 6) for key, value in retire_audit.items()},
            "route_after": {
                key: round(value, 6) for key, value in route_after_retire.items()
            },
            "sorted_bpc_delta": round(sorted_bpc_delta, 9),
            "reverse_return_triggers_growth": reverse_returns,
            "reverse_return_pressure": round(return_pressure, 6),
            "reverse_return_energy": round(return_energy, 6),
        },
        "path_counts": path_counts,
        "base_logit_max_delta": round(base_delta, 9),
        "pass": passed,
        "seconds": round(time.time() - started, 2),
    }


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(array.mean()), 6),
        "std": round(float(array.std()), 6),
    }


def summarize(rows: list[dict]) -> dict:
    complete = [row for row in rows if "route_before_lifecycle" in row]
    if not complete:
        return {"complete_seed_count": 0, "pass_rate": 0.0}
    summary = {
        "complete_seed_count": len(complete),
        "pass_rate": round(float(np.mean([row["pass"] for row in rows])), 6),
        "base_logit_max_delta": mean_std([row["base_logit_max_delta"] for row in complete]),
        "path_counts": [
            round(float(np.mean([row["path_counts"][index] for row in complete])), 6)
            for index in range(len(complete[0]["path_counts"]))
        ],
    }
    for domain in ("base", "reverse", "sorted"):
        summary[f"route_{domain}"] = mean_std([
            row["route_before_lifecycle"][domain] for row in complete
        ])
    for index, domain in enumerate(("reverse", "sorted")):
        summary[f"{domain}_reducibility"] = mean_std([
            row["growth"][index]["reducibility"] for row in complete
        ])
        summary[f"{domain}_bpc"] = mean_std([
            row["growth"][index]["consolidated"]["bpc"] for row in complete
        ])
    summary["repeat_growth_rate"] = round(float(np.mean([
        row["repeat_reverse"]["growth_triggered"] for row in complete
    ])), 6)
    summary["merge_rate"] = round(float(np.mean([
        row["merge"]["merged"] for row in complete
    ])), 6)
    summary["retire_rate"] = round(float(np.mean([
        row["retirement"]["retired"] for row in complete
    ])), 6)
    summary["return_growth_rate"] = round(float(np.mean([
        row["retirement"]["reverse_return_triggers_growth"] for row in complete
    ])), 6)
    summary["route_after_retire_base"] = mean_std([
        row["retirement"]["route_after"]["base"] for row in complete
    ])
    summary["route_after_retire_sorted"] = mean_std([
        row["retirement"]["route_after"]["sorted"] for row in complete
    ])
    return summary


def write_reports(result: dict) -> None:
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    summary = result["summary"]
    with open(REPORT_MD, "w", encoding="utf-8") as file:
        file.write(
            "# 多盆地自由能生命周期\n\n"
            "真实 OPUS 英文字符流；连续结构为整句反向和句内字符排序，二者都保持逐句"
            "字符边际。新增容量为 rank-16 低秩生成性转移与低秩读出。\n\n"
            "| 裁决 | 三种子结果 |\n|---|---:|\n"
            f"| 反向结构 residual-F 可约率 | {summary['reverse_reducibility']['mean']:.1%} |\n"
            f"| 排序结构 residual-F 可约率 | {summary['sorted_reducibility']['mean']:.1%} |\n"
            f"| 正常流→基础盆地 | {summary['route_base']['mean']:.1%} |\n"
            f"| 反向流→反向盆地 | {summary['route_reverse']['mean']:.1%} |\n"
            f"| 排序流→排序盆地 | {summary['route_sorted']['mean']:.1%} |\n"
            f"| 重复结构再次生长率 | {summary['repeat_growth_rate']:.1%} |\n"
            f"| 冗余快照合并率 | {summary['merge_rate']:.1%} |\n"
            f"| 不活跃盆地回收率 | {summary['retire_rate']:.1%} |\n"
            f"| 被回收结构返回后重新触发生长 | {summary['return_growth_rate']:.1%} |\n"
            f"| 盆地数量轨迹 | {'→'.join(f'{value:g}' for value in summary['path_counts'])} |\n"
            f"| 基础 logits 最大变化 | {summary['base_logit_max_delta']['mean']:.1e} |\n\n"
            f"## 裁决\n\n{result['verdict']}\n\n"
            "## 诚实边界\n\n"
            "本实验的合并对象是显式注入的同一盆地持久化快照，用于审计去重机制；尚未"
            "证明两个独立训练但相近的结构能自动合并。回收依据是近期证据窗口，因此被"
            "回收的旧结构返回时会重新触发生长，不代表无损长期记忆。先导实验中的相邻"
            "字符交换在 seed 91、rank-16 下未越过 30% 可约率闸门，系统拒绝固化；正式"
            "实验改用生成规律更清晰的句内字符排序，因此当前结论仍是受控结构机制证明。\n"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    domains = ["base", "reverse", "sorted"]
    routes = [summary[f"route_{name}"]["mean"] for name in domains]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(domains, routes, color=["#546e7a", "#6a1b9a", "#1565c0"])
    axes[0].set_ylim(0, 1.05); axes[0].set_title("最低自由能路由")
    events = ["初始", "反向", "排序", "重复", "冗余注入", "合并", "回收"]
    axes[1].plot(events, summary["path_counts"], "o-", color="#00838f", lw=2)
    axes[1].set_ylim(0.8, 4.2); axes[1].set_title("盆地容量生命周期")
    axes[1].tick_params(axis="x", rotation=25)
    for x, value in enumerate(summary["path_counts"]):
        axes[1].annotate(f"{value:g}", (x, value), xytext=(0, 6),
                         textcoords="offset points", ha="center")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("真实字符流：多盆地自由能生命周期")
    fig.tight_layout(); fig.savefig(FIG_PATH, dpi=180); plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    args.device = args.device.strip() or get_device()
    data = prepare_lifecycle_data(args)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows = [run_seed(seed, data, args) for seed in seeds]
    summary = summarize(rows)
    passed = bool(
        len(rows) == summary.get("complete_seed_count")
        and summary.get("pass_rate") == 1.0
        and summary["repeat_growth_rate"] == 0.0
        and summary["merge_rate"] == 1.0
        and summary["retire_rate"] == 1.0
        and summary["return_growth_rate"] == 1.0
        and summary["base_logit_max_delta"]["mean"] <= 1e-8
    )
    verdict = (
        "✅ 多盆地生命周期成立：两个未知结构按 residual-F 连续长成独立低秩盆地；"
        "重复结构复用已有盆地，冗余快照按能量等价合并，不活跃容量被回收且不扰动"
        "仍活跃盆地；旧结构返回后重新形成生长压力。"
        if passed else
        "🟡 多盆地生命周期尚未通过全部多种子判据；保留失败证据并修正结构或裁决阈值。"
    )
    result = {
        "task": "real char multi-basin free-energy lifecycle",
        "config": vars(args),
        "data": {
            "vocab_size": data["vocab_size"],
            "vocab": data["vocab"],
            "sentence_counts": data["sentence_counts"],
            "transforms": {
                "reverse": "reverse every sentence",
                "sorted": "sort characters inside every sentence",
            },
        },
        "rejected_pilot": {
            "seed": 91,
            "rank": 16,
            "transform": "swap every adjacent character pair",
            "outcome": "candidate rejected below the fixed 0.30 reducibility threshold",
        },
        "rows": rows,
        "summary": summary,
        "pass": passed,
        "verdict": verdict,
    }
    if args.write_report:
        write_reports(result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="真实字符流多盆地自由能生命周期实验。")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--device", default="")
    parser.add_argument("--seeds", default="91,92,93")
    parser.add_argument("--sentences", type=int, default=16000)
    parser.add_argument("--base-steps", type=int, default=350)
    parser.add_argument("--probe-steps", type=int, default=240)
    parser.add_argument("--readout-steps", type=int, default=240)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=384)
    parser.add_argument("--stream-batch", type=int, default=128)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--relax-steps", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--transition-mult", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--probe-lr", type=float, default=4e-3)
    parser.add_argument("--readout-lr", type=float, default=4e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--free-energy-weight", type=float, default=2.0)
    parser.add_argument("--probe-ce-weight", type=float, default=0.0)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--energy-start", type=int, default=2)
    parser.add_argument("--min-growth-fraction", type=float, default=0.25)
    parser.add_argument("--min-reducibility", type=float, default=0.30)
    parser.add_argument("--complexity-quantile", type=float, default=0.95)
    parser.add_argument("--complexity-margin", type=float, default=1e-4)
    parser.add_argument("--duplicate-cost-margin", type=float, default=1e-3)
    parser.add_argument("--merge-energy-tolerance", type=float, default=1e-8)
    parser.add_argument("--merge-min-covered", type=float, default=1.0)
    parser.add_argument("--retire-max-route-fraction", type=float, default=0.03,
                        help="与稳定流约 5% 的能量路由校准误差对齐；近期占比不超过 3% 才回收")
    parser.add_argument("--min-route-accuracy", type=float, default=0.90)
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
    print(json.dumps({
        "summary": result["summary"],
        "pass": result["pass"],
        "verdict": result["verdict"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
