# -*- coding: utf-8 -*-
"""可校验磁盘盆地归档的三种子机制裁决。

lag-2 基础稳定态上独立学习 lag-3 低秩生成性盆地。归档后用新建系统从磁盘加载：必须
通过基础核心 SHA-256、payload 摘要和逐模块张量摘要，随后仍以 residual-F + MDL 复核
并无训练恢复。另行篡改张量和基础核心，要求加载明确拒绝。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

from fe_llm.config import get_device
from fe_llm.energy_lm.free_energy_growth import FreeEnergyGrowthSystem
from fe_llm.energy_lm.evaluation.free_energy_cascade_routing_eval import (
    pathway_accuracy,
    train_lag_pathway,
)
from fe_llm.energy_lm.evaluation.free_energy_growth_eval import train_base
from fe_llm.energy_lm.evaluation.free_energy_sequence_eval import make_lag_sequences

REPORT_JSON = os.path.join("docs", "reports", "free_energy_archive_persistence_eval.json")
REPORT_MD = os.path.join("docs", "reports", "free_energy_archive_persistence_eval.md")
FIG_PATH = os.path.join("docs", "reports", "figs", "free_energy_archive_persistence_eval.png")


def run_seed(seed: int, args: argparse.Namespace) -> dict:
    started = time.time()
    core = train_base(seed, args)
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    system = FreeEnergyGrowthSystem(core).to(args.device).eval()
    torch.manual_seed(seed * 10_000)
    stable = make_lag_sequences(
        args.calibration_batch, args.length, 2, args.vocab, args.device)
    audit_base = make_lag_sequences(
        args.eval_batch, args.length, 2, args.vocab, args.device)
    audit_new = make_lag_sequences(
        args.eval_batch, args.length, 3, args.vocab, args.device)
    base_logits_before = core(audit_base).detach().clone()
    pathway, training = train_lag_pathway(
        system, 3, stable, seed=seed, args=args)
    task_accuracy = pathway_accuracy(system, pathway, audit_new, 3)
    logits_before = system.forward_pathway(audit_new, pathway).detach().clone()
    score_before = (
        system.residual_scores(audit_new, pathway, start=args.energy_start)
        + system.pathway_costs[pathway]).detach().clone()
    core_snapshot = copy.deepcopy(core)
    active_params = system.added_parameter_count()
    archive_id, _ = system.archive_pathway(pathway)
    cold_params = system.added_parameter_count()

    with tempfile.TemporaryDirectory(prefix="fe-archive-") as directory:
        archive_path = os.path.join(directory, f"seed-{seed}.pt")
        manifest = system.save_archives(archive_path)
        dangling_temp_files = [
            name for name in os.listdir(directory) if ".tmp-" in name]

        restored_system = FreeEnergyGrowthSystem(
            copy.deepcopy(core_snapshot)).to(args.device).eval()
        loaded_ids = restored_system.load_archives(archive_path)
        loaded_score = restored_system.archived_scores(
            audit_new, loaded_ids[0], start=args.energy_start)
        energy_delta = float((loaded_score - score_before).abs().max().cpu())
        active_scores = restored_system.score_all(
            audit_new, start=args.energy_start).min(dim=1).values
        archive_advantage = active_scores - loaded_score
        archive_better_fraction = float((archive_advantage > 0).float().mean().cpu())
        restored = restored_system.restore_archived_pathway(loaded_ids[0])
        restored_logits = restored_system.forward_pathway(audit_new, restored)
        logit_delta = float((restored_logits - logits_before).abs().max().cpu())
        choices_new, _ = restored_system.route(audit_new, start=args.energy_start)
        choices_base, _ = restored_system.route(audit_base, start=args.energy_start)
        new_route = float((choices_new == restored).float().mean().cpu())
        base_route = float((choices_base == 0).float().mean().cpu())

        mismatched = FreeEnergyGrowthSystem(
            copy.deepcopy(core_snapshot)).to(args.device).eval()
        with torch.no_grad():
            mismatched.core.root_state.add_(0.1)
        core_mismatch_rejected = False
        try:
            mismatched.load_archives(archive_path)
        except ValueError as error:
            core_mismatch_rejected = "指纹" in str(error)

        corrupted_path = os.path.join(directory, f"seed-{seed}-corrupted.pt")
        payload = torch.load(archive_path, map_location="cpu", weights_only=True)
        payload["entries"][0]["transition"]["state"]["up.weight"][0, 0] += 1.0
        torch.save(payload, corrupted_path)
        tamper_rejected = False
        try:
            FreeEnergyGrowthSystem(
                copy.deepcopy(core_snapshot)).load_archives(corrupted_path)
        except ValueError as error:
            tamper_rejected = "摘要" in str(error)

    base_delta = float((core(audit_base) - base_logits_before).abs().max().cpu())
    passed = bool(
        task_accuracy >= args.min_task_accuracy
        and active_params > 0 and cold_params == 0
        and manifest["archive_count"] == 1
        and len(manifest["file_sha256"]) == 64
        and not dangling_temp_files
        and energy_delta <= 1e-8
        and archive_better_fraction >= args.min_reactivation_fraction
        and logit_delta <= 1e-8
        and new_route >= args.min_route_accuracy
        and base_route >= args.min_route_accuracy
        and core_mismatch_rejected
        and tamper_rejected
        and base_delta <= 1e-8
    )
    return {
        "seed": seed,
        "task_accuracy": round(task_accuracy, 6),
        "training": training,
        "active_params_before_archive": active_params,
        "active_params_cold": cold_params,
        "manifest": manifest,
        "atomic_temp_files_remaining": len(dangling_temp_files),
        "loaded_archive_ids": loaded_ids,
        "energy_max_delta": round(energy_delta, 9),
        "archive_better_fraction": round(archive_better_fraction, 6),
        "restored_logit_max_delta": round(logit_delta, 9),
        "new_route_accuracy": round(new_route, 6),
        "base_route_accuracy": round(base_route, 6),
        "core_mismatch_rejected": core_mismatch_rejected,
        "tamper_rejected": tamper_rejected,
        "base_logit_max_delta": round(base_delta, 9),
        "pass": passed,
        "seconds": round(time.time() - started, 2),
    }


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": round(float(array.mean()), 6),
            "std": round(float(array.std()), 6)}


def summarize(rows: list[dict]) -> dict:
    summary = {
        "pass_rate": round(float(np.mean([row["pass"] for row in rows])), 6),
        "core_mismatch_reject_rate": round(float(np.mean([
            row["core_mismatch_rejected"] for row in rows])), 6),
        "tamper_reject_rate": round(float(np.mean([
            row["tamper_rejected"] for row in rows])), 6),
        "atomic_save_rate": round(float(np.mean([
            row["atomic_temp_files_remaining"] == 0 for row in rows])), 6),
    }
    for key in (
        "task_accuracy", "energy_max_delta", "archive_better_fraction",
        "restored_logit_max_delta", "new_route_accuracy", "base_route_accuracy",
        "base_logit_max_delta",
    ):
        summary[key] = mean_std([row[key] for row in rows])
    summary["file_size"] = mean_std([row["manifest"]["file_size"] for row in rows])
    return summary


def write_reports(result: dict) -> None:
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    summary = result["summary"]
    with open(REPORT_MD, "w", encoding="utf-8") as file:
        file.write(
            "# 可校验磁盘盆地归档\n\n"
            "低秩生成性盆地离开活动 GPU 后原子保存；新系统必须通过核心指纹、payload 与"
            "逐模块摘要，再接受 residual-F 返回复核。\n\n"
            "| 裁决 | 三种子结果 |\n|---|---:|\n"
            f"| 原子保存率 | {summary['atomic_save_rate']:.1%} |\n"
            f"| 核心不匹配拒绝率 | {summary['core_mismatch_reject_rate']:.1%} |\n"
            f"| 张量篡改拒绝率 | {summary['tamper_reject_rate']:.1%} |\n"
            f"| 磁盘往返 residual-F 最大差 | {summary['energy_max_delta']['mean']:.1e} |\n"
            f"| 归档优于当前活动解释 | {summary['archive_better_fraction']['mean']:.1%} |\n"
            f"| 恢复前后 logits 最大差 | {summary['restored_logit_max_delta']['mean']:.1e} |\n"
            f"| 基础 / 恢复结构路由 | {summary['base_route_accuracy']['mean']:.1%} / "
            f"{summary['new_route_accuracy']['mean']:.1%} |\n"
            f"| 单盆地归档大小 | {summary['file_size']['mean']/1024:.1f} KiB |\n\n"
            f"## 裁决\n\n{result['verdict']}\n\n"
            "## 边界\n\n"
            "当前格式只接受低秩生成性转移和低秩/线性读出，避免任意模块反序列化；尚无"
            "跨模型版本迁移、密钥签名、远程对象存储和大规模归档检索。\n"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["核心错配拒绝", "张量篡改拒绝", "能量精确", "输出精确"]
    values = [
        summary["core_mismatch_reject_rate"], summary["tamper_reject_rate"],
        float(summary["energy_max_delta"]["mean"] <= 1e-8),
        float(summary["restored_logit_max_delta"]["mean"] <= 1e-8),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(labels, values, color=["#c62828", "#ef6c00", "#1565c0", "#2e7d32"])
    axes[0].set_ylim(0, 1.05); axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_title("归档完整性与精确恢复")
    axes[1].bar(
        ["归档前活动", "冷存储活动", "恢复后活动"],
        [1, 0, 1], color=["#1565c0", "#78909c", "#2e7d32"])
    axes[1].set_title("活动 GPU 盆地数")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("低能盆地持久化：受限格式、完整性校验、能量复核")
    fig.tight_layout(); fig.savefig(FIG_PATH, dpi=180); plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    args.device = args.device.strip() or get_device()
    rows = [run_seed(int(value), args) for value in args.seeds.split(",") if value.strip()]
    summary = summarize(rows)
    passed = bool(
        summary["pass_rate"] == 1.0
        and summary["core_mismatch_reject_rate"] == 1.0
        and summary["tamper_reject_rate"] == 1.0
        and summary["energy_max_delta"]["mean"] <= 1e-8
        and summary["restored_logit_max_delta"]["mean"] <= 1e-8
    )
    verdict = (
        "✅ 可校验盆地持久化成立：冷盆地以受限张量格式原子落盘；不同基础稳定态和被"
        "篡改权重均拒绝；新系统加载后 residual-F 精确复现，归档重新成为低能解释才"
        "无训练恢复，输出与基础稳定态均严格不变。"
        if passed else
        "🟡 磁盘归档尚未同时通过完整性拒绝、能量复现与无训练输出恢复。"
    )
    result = {
        "task": "verified persistent archive for low-rank free-energy basins",
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
    parser = argparse.ArgumentParser(description="可校验磁盘盆地归档机制裁决。")
    parser.add_argument("--device", default="")
    parser.add_argument("--seeds", default="142,143,144")
    parser.add_argument("--vocab", type=int, default=8)
    parser.add_argument("--length", type=int, default=14)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--relax-steps", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--transition-mult", type=int, default=2)
    parser.add_argument("--base-steps", type=int, default=150)
    parser.add_argument("--growth-steps", type=int, default=180)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--calibration-batch", type=int, default=192)
    parser.add_argument("--eval-batch", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--growth-lr", type=float, default=4e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--free-energy-weight", type=float, default=2.0)
    parser.add_argument("--energy-start", type=int, default=3)
    parser.add_argument("--complexity-quantile", type=float, default=0.90)
    parser.add_argument("--complexity-margin", type=float, default=1e-4)
    parser.add_argument("--min-task-accuracy", type=float, default=0.95)
    parser.add_argument("--min-route-accuracy", type=float, default=0.90)
    parser.add_argument("--min-reactivation-fraction", type=float, default=0.90)
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
