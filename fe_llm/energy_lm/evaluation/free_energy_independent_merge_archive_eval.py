# -*- coding: utf-8 -*-
"""真实字符流上的独立盆地合并、CPU 冷存储与无训练再激活。

两个 rank-16 低秩盆地从互斥 OPUS 反向字符子流、不同初始化独立学习。共同 held-out
证据只允许在保留盆地能以有限 residual-F 增量覆盖另一盆地时合并。合并后的盆地在长期
不用时从活动 GPU 路由卸到 CPU；结构返回后先比较归档 residual-F 与当前活动解释，只有
归档明显更低才原样恢复。模型不读取结构名进行路由或恢复。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

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
    lm_loss,
    path_metrics,
    prepare_data,
    sample_chunks,
    train_base,
    train_probe,
    train_readout,
)

REPORT_JSON = os.path.join(
    "docs", "reports", "free_energy_independent_merge_archive_eval.json")
REPORT_MD = os.path.join(
    "docs", "reports", "free_energy_independent_merge_archive_eval.md")
FIG_PATH = os.path.join(
    "docs", "reports", "figs", "free_energy_independent_merge_archive_eval.png")


def make_candidate(core, rank: int):
    return (
        LowRankGenerativeTransition(core.transition, dim=core.dim, rank=rank),
        LowRankReadout(core.head, in_dim=core.dim, out_dim=core.vocab_size, rank=rank),
    )


def split_stream(stream: torch.Tensor, minimum: int) -> tuple[torch.Tensor, torch.Tensor]:
    midpoint = stream.numel() // 2
    first = stream[:midpoint]
    second = stream[midpoint:]
    if min(first.numel(), second.numel()) < minimum:
        raise RuntimeError("互斥结构子流过短。")
    return first, second


@torch.no_grad()
def routed_bpc(
    system: FreeEnergyGrowthSystem,
    ids: torch.Tensor,
    *,
    start: int,
) -> tuple[float, torch.Tensor]:
    logits, choices = system.routed_logits(ids, start=start)
    return float(lm_loss(logits, ids).cpu()) / math.log(2), choices


def compact(metrics: dict) -> dict[str, float]:
    return {
        key: round(float(value), 6)
        for key, value in metrics.items()
        if key != "residual_per_sample"
    }


def train_independent_candidate(
    system: FreeEnergyGrowthSystem,
    stream: torch.Tensor,
    valid_ids: torch.Tensor,
    base_residual: float,
    base_valid: torch.Tensor,
    *,
    seed: int,
    index: int,
    args: argparse.Namespace,
) -> tuple[int | None, dict, torch.nn.Module | None, torch.nn.Module | None]:
    torch.manual_seed(seed * 100 + index)
    transition, head = make_candidate(system.core, args.rank)
    transition = transition.to(args.device)
    head = head.to(args.device)
    probe_history = train_probe(
        system, transition, stream, noise=False,
        seed=seed + index * 1000, args=args)
    probe = path_metrics(
        system.core, transition, valid_ids, start=args.energy_start)
    reducibility = (base_residual - probe["residual"]) / max(1e-8, base_residual)
    if reducibility < args.min_reducibility:
        return None, {
            "index": index,
            "reducibility": round(reducibility, 6),
            "probe": compact(probe),
            "committed": False,
            "probe_history": probe_history,
        }, transition, head

    for parameter in transition.parameters():
        parameter.requires_grad_(False)
    readout_history = train_readout(
        system, transition, head, stream,
        seed=seed + index * 1000 + 101, args=args)
    consolidated = path_metrics(
        system.core, transition, valid_ids, start=args.energy_start, head=head)
    cost = system.calibrate_complexity_cost(
        base_valid,
        transition,
        start=args.energy_start,
        quantile=args.complexity_quantile,
        margin=args.complexity_margin,
    )
    pathway = system.commit_pathway(transition, complexity_cost=cost, head=head)
    return pathway, {
        "index": index,
        "pathway": pathway,
        "reducibility": round(reducibility, 6),
        "probe": compact(probe),
        "consolidated": compact(consolidated),
        "complexity_cost": round(cost, 6),
        "committed": True,
        "probe_history": probe_history,
        "readout_history": readout_history,
    }, transition, head


def run_seed(seed: int, data: dict, args: argparse.Namespace) -> dict:
    started = time.time()
    core = train_base(seed, data["base_train"], data["vocab_size"], args)
    system = FreeEnergyGrowthSystem(core).to(args.device).eval()
    generator = torch.Generator(device=args.device).manual_seed(seed + 9001)
    base_valid = sample_chunks(data["base_valid"], args.eval_batch, args.length, generator)
    reverse_valid = sample_chunks(
        data["structure_valid"], args.eval_batch, args.length, generator)
    stream_a, stream_b = split_stream(
        data["structure_train"], args.length * args.batch * 2)
    base_logits_before = core(base_valid).detach().clone()
    base_reverse = path_metrics(
        core, core.transition, reverse_valid, start=args.energy_start)
    threshold = system.calibrate_threshold(
        base_valid, start=args.energy_start, quantile=args.threshold_quantile)

    candidates = []
    modules = []
    pathways = []
    for index, stream in enumerate((stream_a, stream_b), start=1):
        pathway, row, transition, head = train_independent_candidate(
            system,
            stream,
            reverse_valid,
            base_reverse["residual"],
            base_valid,
            seed=seed,
            index=index,
            args=args,
        )
        candidates.append(row)
        modules.append((transition, head))
        if pathway is not None:
            pathways.append(pathway)

    if len(pathways) != 2:
        return {
            "seed": seed,
            "threshold": round(threshold, 6),
            "base_reverse": compact(base_reverse),
            "candidates": candidates,
            "pass": False,
            "failure": "两个独立盆地未全部越过可约率闸门",
            "seconds": round(time.time() - started, 2),
        }

    transition_difference = max(
        float((left - right).abs().max().cpu())
        for left, right in zip(
            modules[0][0].parameters(), modules[1][0].parameters()))
    scores = system.score_all(reverse_valid, start=args.energy_start)
    first, second = pathways
    means = {first: float(scores[:, first].mean().cpu()),
             second: float(scores[:, second].mean().cpu())}
    keep = min(pathways, key=lambda pathway: means[pathway])
    remove = second if keep == first else first
    tolerance = args.max_relative_energy_increase * means[remove]
    active_params_two = system.added_parameter_count()
    routed_bpc_before, choices_before = routed_bpc(
        system, reverse_valid, start=args.energy_start)
    base_route_before = float(
        (system.route(base_valid, start=args.energy_start)[0] == 0).float().mean().cpu())

    merged, merge_audit = system.merge_pathways_if_redundant(
        keep,
        remove,
        reverse_valid,
        start=args.energy_start,
        energy_tolerance=tolerance,
        min_covered_fraction=args.min_merge_coverage,
    )
    survivor = int(merge_audit["survivor"])
    recalibrated_cost = None
    if merged:
        recalibrated_cost = system.recalibrate_pathway_cost(
            base_valid,
            survivor,
            start=args.energy_start,
            quantile=args.complexity_quantile,
            margin=args.complexity_margin,
        )
    active_params_merged = system.added_parameter_count()
    routed_bpc_after, choices_after = routed_bpc(
        system, reverse_valid, start=args.energy_start)
    reverse_route_after = float((choices_after == survivor).float().mean().cpu())
    base_route_after = float(
        (system.route(base_valid, start=args.energy_start)[0] == 0).float().mean().cpu())
    survivor_logits = system.forward_pathway(reverse_valid, survivor).detach().clone()

    archive_id = None
    mapping = None
    archived_params = None
    active_params_cold = None
    archive_better_fraction = None
    archive_mean_advantage = None
    restored = None
    restored_logit_delta = None
    restored_bpc = None
    restored_reverse_route = None
    restored_base_route = None
    if merged:
        archive_id, mapping = system.archive_pathway(survivor)
        archived_params = system.archived_parameter_count()
        active_params_cold = system.added_parameter_count()
        active_scores = system.score_all(reverse_valid, start=args.energy_start).min(dim=1).values
        cold_scores = system.archived_scores(
            reverse_valid, archive_id, start=args.energy_start)
        advantage = active_scores - cold_scores
        archive_better_fraction = float((advantage > 0).float().mean().cpu())
        archive_mean_advantage = float(advantage.mean().cpu())
        if (archive_better_fraction >= args.min_reactivation_fraction
                and float(cold_scores.mean().cpu()) <= threshold):
            restored = system.restore_archived_pathway(archive_id)
            restored_logits = system.forward_pathway(reverse_valid, restored)
            restored_logit_delta = float(
                (restored_logits - survivor_logits).abs().max().cpu())
            restored_bpc, restored_choices = routed_bpc(
                system, reverse_valid, start=args.energy_start)
            restored_reverse_route = float(
                (restored_choices == restored).float().mean().cpu())
            restored_base_route = float(
                (system.route(base_valid, start=args.energy_start)[0] == 0).float().mean().cpu())

    base_delta = float((core(base_valid) - base_logits_before).abs().max().cpu())
    bpc_relative_increase = (
        routed_bpc_after - routed_bpc_before) / max(1e-8, routed_bpc_before)
    parameter_reduction = 1.0 - active_params_merged / active_params_two
    passed = bool(
        transition_difference > 1e-5
        and merged
        and merge_audit["covered_fraction"] >= args.min_merge_coverage
        and bpc_relative_increase <= args.max_relative_bpc_increase
        and parameter_reduction >= args.min_parameter_reduction
        and base_route_after >= args.min_route_accuracy
        and reverse_route_after >= args.min_route_accuracy
        and active_params_cold == 0
        and archived_params == active_params_merged
        and archive_better_fraction is not None
        and archive_better_fraction >= args.min_reactivation_fraction
        and restored is not None
        and restored_logit_delta is not None and restored_logit_delta <= 1e-8
        and restored_bpc is not None
        and abs(restored_bpc - routed_bpc_after) <= 1e-8
        and restored_reverse_route is not None
        and restored_reverse_route >= args.min_route_accuracy
        and restored_base_route is not None
        and restored_base_route >= args.min_route_accuracy
        and base_delta <= 1e-8
    )
    return {
        "seed": seed,
        "threshold": round(threshold, 6),
        "base_reverse": compact(base_reverse),
        "candidates": candidates,
        "transition_max_difference": round(transition_difference, 9),
        "independent_pathways": pathways,
        "pathway_mean_scores": {str(key): round(value, 6) for key, value in means.items()},
        "keep": keep,
        "remove": remove,
        "merge_energy_tolerance": round(tolerance, 6),
        "merge": {
            "merged": merged,
            "covered_fraction": round(float(merge_audit["covered_fraction"]), 6),
            "mean_energy_increase": round(float(merge_audit["mean_energy_increase"]), 6),
            "survivor": survivor,
            "recalibrated_complexity_cost": (
                None if recalibrated_cost is None else round(recalibrated_cost, 6)),
            "routed_bpc_before": round(routed_bpc_before, 6),
            "routed_bpc_after": round(routed_bpc_after, 6),
            "relative_bpc_increase": round(bpc_relative_increase, 6),
            "base_route_before": round(base_route_before, 6),
            "base_route_after": round(base_route_after, 6),
            "reverse_route_after": round(reverse_route_after, 6),
            "active_params_before": active_params_two,
            "active_params_after": active_params_merged,
            "parameter_reduction": round(parameter_reduction, 6),
            "pathway_usage_before": {
                str(pathway): round(float((choices_before == pathway).float().mean().cpu()), 6)
                for pathway in pathways
            },
        },
        "archive": {
            "archive_id": archive_id,
            "active_mapping": mapping,
            "archived_params": archived_params,
            "active_params_cold": active_params_cold,
            "archive_better_fraction": (
                None if archive_better_fraction is None
                else round(archive_better_fraction, 6)),
            "archive_mean_energy_advantage": (
                None if archive_mean_advantage is None
                else round(archive_mean_advantage, 6)),
            "restored_pathway": restored,
            "restored_logit_max_delta": (
                None if restored_logit_delta is None else round(restored_logit_delta, 9)),
            "restored_bpc": None if restored_bpc is None else round(restored_bpc, 6),
            "restored_reverse_route": (
                None if restored_reverse_route is None else round(restored_reverse_route, 6)),
            "restored_base_route": (
                None if restored_base_route is None else round(restored_base_route, 6)),
        },
        "base_logit_max_delta": round(base_delta, 9),
        "pass": passed,
        "seconds": round(time.time() - started, 2),
    }


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": round(float(array.mean()), 6),
            "std": round(float(array.std()), 6)}


def summarize(rows: list[dict]) -> dict:
    complete = [row for row in rows if "merge" in row]
    if not complete:
        return {"complete_seed_count": 0, "pass_rate": 0.0}
    summary = {
        "complete_seed_count": len(complete),
        "pass_rate": round(float(np.mean([row["pass"] for row in rows])), 6),
        "merge_rate": round(float(np.mean([
            row["merge"]["merged"] for row in complete])), 6),
        "restore_rate": round(float(np.mean([
            row["archive"]["restored_pathway"] is not None for row in complete])), 6),
    }
    for key, getter in {
        "merge_coverage": lambda row: row["merge"]["covered_fraction"],
        "merge_bpc_increase": lambda row: row["merge"]["relative_bpc_increase"],
        "parameter_reduction": lambda row: row["merge"]["parameter_reduction"],
        "base_route_after_merge": lambda row: row["merge"]["base_route_after"],
        "reverse_route_after_merge": lambda row: row["merge"]["reverse_route_after"],
        "archive_better_fraction": lambda row: row["archive"]["archive_better_fraction"],
        "restored_logit_max_delta": lambda row: row["archive"]["restored_logit_max_delta"],
        "restored_base_route": lambda row: row["archive"]["restored_base_route"],
        "restored_reverse_route": lambda row: row["archive"]["restored_reverse_route"],
        "base_logit_max_delta": lambda row: row["base_logit_max_delta"],
    }.items():
        values = [getter(row) for row in complete if getter(row) is not None]
        summary[key] = mean_std(values)
    return summary


def write_reports(result: dict) -> None:
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    summary = result["summary"]
    with open(REPORT_MD, "w", encoding="utf-8") as file:
        file.write(
            "# 独立盆地合并、冷存储与无训练再激活\n\n"
            "两个 rank-16 盆地来自互斥 OPUS 反向子流和不同初始化；合并与返回都只读取"
            " held-out residual-F + MDL。\n\n"
            "| 裁决 | 三种子结果 |\n|---|---:|\n"
            f"| 独立盆地合并率 | {summary['merge_rate']:.1%} |\n"
            f"| 合并能量覆盖率 | {summary['merge_coverage']['mean']:.1%} |\n"
            f"| 合并后 BPC 相对变化 | {summary['merge_bpc_increase']['mean']:+.2%} |\n"
            f"| 活动新增参数减少 | {summary['parameter_reduction']['mean']:.1%} |\n"
            f"| 合并后基础 / 结构路由 | {summary['base_route_after_merge']['mean']:.1%} / "
            f"{summary['reverse_route_after_merge']['mean']:.1%} |\n"
            f"| 冷存储期间活动新增参数 | 0 |\n"
            f"| 归档优于当前活动解释 | {summary['archive_better_fraction']['mean']:.1%} |\n"
            f"| 无训练恢复率 | {summary['restore_rate']:.1%} |\n"
            f"| 恢复前后 logits 最大差 | {summary['restored_logit_max_delta']['mean']:.1e} |\n"
            f"| 基础 logits 最大变化 | {summary['base_logit_max_delta']['mean']:.1e} |\n\n"
            f"## 裁决\n\n{result['verdict']}\n\n"
            "## 边界\n\n"
            "两个独立盆地学习的是同一种受控反向字符规律，尚未证明语义相近但不等价的"
            "真实领域可以安全合并。冷存储当前是进程内 CPU 模块，不包含磁盘持久化、版本"
            "迁移或大规模归档索引。\n"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(
        ["合并前", "合并后", "冷存储", "恢复后"],
        [2.0, 1.0, 0.0, 1.0], color=["#6a1b9a", "#1565c0", "#78909c", "#2e7d32"])
    axes[0].set_ylabel("活动新增盆地数"); axes[0].set_title("活动容量生命周期")
    axes[1].bar(
        ["合并覆盖", "归档返回优势", "恢复率"],
        [summary["merge_coverage"]["mean"],
         summary["archive_better_fraction"]["mean"], summary["restore_rate"]],
        color=["#1565c0", "#ef6c00", "#2e7d32"])
    axes[1].set_ylim(0, 1.05); axes[1].set_title("能量裁决")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("独立低能盆地：合并、冷存储、无训练再激活")
    fig.tight_layout(); fig.savefig(FIG_PATH, dpi=180); plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    args.device = args.device.strip() or get_device()
    data = prepare_data(args)
    rows = [run_seed(int(value), data, args) for value in args.seeds.split(",") if value.strip()]
    summary = summarize(rows)
    passed = bool(
        summary.get("complete_seed_count") == len(rows)
        and summary.get("pass_rate") == 1.0
        and summary["merge_rate"] == 1.0
        and summary["restore_rate"] == 1.0
        and summary["base_logit_max_delta"]["mean"] <= 1e-8
        and summary["restored_logit_max_delta"]["mean"] <= 1e-8
    )
    verdict = (
        "✅ 独立盆地生命周期成立：互斥经验上独立学习的两个低能盆地通过 held-out "
        "能量等价性合并；不活跃时活动 GPU 容量降为零；结构返回时归档 residual-F "
        "重新胜出并无训练恢复，基础稳定态和恢复 logits 均严格不变。"
        if passed else
        "🟡 独立盆地尚未同时通过能量合并、能力保持、CPU 冷存储与无训练再激活判据。"
    )
    result = {
        "task": "independently learned basin merge, cold archive and reactivation",
        "config": vars(args),
        "data": {"vocab_size": data["vocab_size"],
                 "sentence_counts": data["sentence_counts"]},
        "rows": rows,
        "summary": summary,
        "pass": passed,
        "verdict": verdict,
    }
    if args.write_report:
        write_reports(result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="真实字符流独立盆地合并与冷存储实验。")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--device", default="")
    parser.add_argument("--seeds", default="132,133,134")
    parser.add_argument("--sentences", type=int, default=12000)
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
    parser.add_argument("--min-reducibility", type=float, default=0.30)
    parser.add_argument("--complexity-quantile", type=float, default=0.90)
    parser.add_argument("--complexity-margin", type=float, default=1e-4)
    parser.add_argument("--max-relative-energy-increase", type=float, default=0.05)
    parser.add_argument("--min-merge-coverage", type=float, default=0.95)
    parser.add_argument("--max-relative-bpc-increase", type=float, default=0.03)
    parser.add_argument("--min-parameter-reduction", type=float, default=0.45)
    parser.add_argument("--min-reactivation-fraction", type=float, default=0.90)
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
    print(json.dumps({"summary": result["summary"], "pass": result["pass"],
                      "verdict": result["verdict"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
