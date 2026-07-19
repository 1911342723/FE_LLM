# -*- coding: utf-8 -*-
"""真实领域压力上的慢时间尺度结构自由能与迟滞裁决。

输入不是人工概率，而是 ``free_energy_domain_drift_eval`` 在 OPUS→Python 三种子实验中
实测的逐窗口高 residual-F 比例。结构不稳定度 ``s`` 对每个窗口的显式自由能 ``G_t``
弛豫：孤立污染 burst 不越过生长势垒，持续漂移累积后触发，概念回返后再越过较低复位
势垒。不同窗口可由环境做功抬高初始能量，但每个窗口内部必须单调耗散。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from fe_llm.energy_lm.free_energy_growth import StructuralFreeEnergyStabilizer

SOURCE_REPORT = os.path.join("docs", "reports", "free_energy_domain_drift_eval.json")
REPORT_JSON = os.path.join("docs", "reports", "free_energy_online_drift_eval.json")
REPORT_MD = os.path.join("docs", "reports", "free_energy_online_drift_eval.md")
FIG_PATH = os.path.join("docs", "reports", "figs", "free_energy_online_drift_eval.png")


def load_source(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        result = json.load(file)
    if not result.get("pass"):
        raise RuntimeError("真实领域漂移源报告尚未通过，不能据此裁决慢动力学。")
    return result


def build_stabilizer(args: argparse.Namespace) -> StructuralFreeEnergyStabilizer:
    return StructuralFreeEnergyStabilizer(
        observation_precision=args.observation_precision,
        persistence_precision=args.persistence_precision,
        complexity_precision=args.complexity_precision,
        relaxation_steps=args.relaxation_steps,
        relaxation_fraction=args.relaxation_fraction,
        tolerance=args.tolerance,
        activation_barrier=args.activation_barrier,
        reset_barrier=args.reset_barrier,
    )


def append_window(
    timeline: list[dict],
    stabilizer: StructuralFreeEnergyStabilizer,
    *,
    phase: str,
    label: str,
    pressure: float,
) -> None:
    state, active, trace = stabilizer.observe(pressure, return_trace=True)
    assert trace is not None
    energy = trace["free_energy"]
    monotonic = bool((energy[1:] <= energy[:-1] + 1e-8).all())
    timeline.append({
        "index": len(timeline),
        "phase": phase,
        "label": label,
        "raw_pressure": round(float(pressure), 6),
        "stable_instability": round(float(state.cpu()), 6),
        "active": active,
        "activated": bool(trace["activated"]),
        "deactivated": bool(trace["deactivated"]),
        "initial_free_energy": round(float(energy[0].cpu()), 9),
        "final_free_energy": round(float(energy[-1].cpu()), 9),
        "dissipation": round(float((energy[0] - energy[-1]).cpu()), 9),
        "inner_energy_monotonic": monotonic,
        "relaxation_steps": int(energy.numel() - 1),
    })


def run_seed(source_row: dict, args: argparse.Namespace) -> dict:
    stabilizer = build_stabilizer(args)
    timeline: list[dict] = []
    stable = source_row["gradual_drift"][0]["high_energy_fraction"]
    local = source_row["local_pollution"]["high_energy_fraction"]
    burst = source_row["burst_pollution"]["high_energy_fraction"]
    drift = source_row["gradual_drift"][1:]

    for _ in range(args.warmup_windows):
        append_window(timeline, stabilizer, phase="stable", label="base", pressure=stable)
    for _ in range(args.local_events):
        append_window(timeline, stabilizer, phase="local_pollution", label="local", pressure=local)
        append_window(timeline, stabilizer, phase="stable", label="base", pressure=stable)
    append_window(timeline, stabilizer, phase="burst", label="burst", pressure=burst)
    for _ in range(args.recovery_windows):
        append_window(timeline, stabilizer, phase="recovery", label="base", pressure=stable)

    pre_drift_active = any(row["active"] for row in timeline)
    naive_burst_trigger = burst >= args.activation_barrier
    for point in drift:
        fraction = point["shifted_fraction"]
        pressure = point["high_energy_fraction"]
        for _ in range(args.dwell_windows):
            append_window(
                timeline,
                stabilizer,
                phase="gradual_drift",
                label=f"code={fraction:g}",
                pressure=pressure,
            )

    activation = next((row for row in timeline if row["activated"]), None)
    for _ in range(args.return_windows):
        append_window(
            timeline, stabilizer, phase="concept_return", label="base", pressure=stable)
    deactivation = next((row for row in timeline if row["deactivated"]), None)
    hysteresis_observed = any(
        row["phase"] == "concept_return"
        and row["active"]
        and args.reset_barrier < row["stable_instability"] < args.activation_barrier
        for row in timeline
    )
    all_monotonic = all(row["inner_energy_monotonic"] for row in timeline)
    total_dissipation = sum(row["dissipation"] for row in timeline)
    passed = bool(
        all_monotonic
        and naive_burst_trigger
        and not pre_drift_active
        and activation is not None
        and activation["phase"] == "gradual_drift"
        and deactivation is not None
        and deactivation["phase"] == "concept_return"
        and hysteresis_observed
        and total_dissipation > 0
    )
    return {
        "seed": source_row["seed"],
        "source_pressures": {
            "stable": stable,
            "local": local,
            "burst": burst,
        },
        "naive_burst_would_trigger": naive_burst_trigger,
        "pre_drift_active": pre_drift_active,
        "activation_index": None if activation is None else activation["index"],
        "activation_label": None if activation is None else activation["label"],
        "deactivation_index": None if deactivation is None else deactivation["index"],
        "return_windows_to_reset": (
            None if deactivation is None else
            deactivation["index"]
            - next(row["index"] for row in timeline if row["phase"] == "concept_return") + 1
        ),
        "hysteresis_observed": hysteresis_observed,
        "all_inner_energy_monotonic": all_monotonic,
        "total_dissipation": round(total_dissipation, 9),
        "timeline": timeline,
        "pass": passed,
    }


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(array.mean()), 6),
        "std": round(float(array.std()), 6),
    }


def summarize(rows: list[dict]) -> dict:
    return {
        "pass_rate": round(float(np.mean([row["pass"] for row in rows])), 6),
        "inner_energy_monotonic_rate": round(float(np.mean([
            row["all_inner_energy_monotonic"] for row in rows])), 6),
        "naive_burst_trigger_rate": round(float(np.mean([
            row["naive_burst_would_trigger"] for row in rows])), 6),
        "stabilized_burst_trigger_rate": round(float(np.mean([
            row["pre_drift_active"] for row in rows])), 6),
        "sustained_drift_activation_rate": round(float(np.mean([
            row["activation_index"] is not None for row in rows])), 6),
        "concept_return_reset_rate": round(float(np.mean([
            row["deactivation_index"] is not None for row in rows])), 6),
        "hysteresis_rate": round(float(np.mean([
            row["hysteresis_observed"] for row in rows])), 6),
        "activation_index": mean_std([row["activation_index"] for row in rows]),
        "return_windows_to_reset": mean_std([
            row["return_windows_to_reset"] for row in rows]),
        "total_dissipation": mean_std([row["total_dissipation"] for row in rows]),
    }


def write_reports(result: dict) -> None:
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    summary = result["summary"]
    with open(REPORT_MD, "w", encoding="utf-8") as file:
        file.write(
            "# 在线漂移：结构自由能慢动力学\n\n"
            "输入为真实 OPUS→Python 实验测得的高 residual-F 窗口比例。慢状态按显式结构"
            "自由能弛豫，并用双势垒产生生长/复位迟滞。\n\n"
            "| 裁决 | 三种子结果 |\n|---|---:|\n"
            f"| 每窗口结构自由能单调率 | {summary['inner_energy_monotonic_rate']:.1%} |\n"
            f"| 单窗口阈值对 50% burst 的触发率 | {summary['naive_burst_trigger_rate']:.1%} |\n"
            f"| 慢动力学对同一 burst 的触发率 | {summary['stabilized_burst_trigger_rate']:.1%} |\n"
            f"| 持续渐变漂移激活率 | {summary['sustained_drift_activation_rate']:.1%} |\n"
            f"| 概念回返复位率 | {summary['concept_return_reset_rate']:.1%} |\n"
            f"| 双势垒迟滞观测率 | {summary['hysteresis_rate']:.1%} |\n"
            f"| 回返后复位窗口数 | {summary['return_windows_to_reset']['mean']:.1f}±"
            f"{summary['return_windows_to_reset']['std']:.1f} |\n\n"
            f"## 裁决\n\n{result['verdict']}\n\n"
            "## 边界\n\n"
            "慢动力学使用真实模型测得的离散窗口压力，但时间线由报告离线重放；尚未把"
            "临时通路训练耗时、异步数据到达和多盆地并发竞争纳入同一在线进程。\n"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for row in result["rows"]:
        timeline = row["timeline"]
        x = [point["index"] for point in timeline]
        axes[0].plot(x, [point["raw_pressure"] for point in timeline], alpha=0.7,
                     label=f"seed {row['seed']}")
        axes[1].plot(x, [point["stable_instability"] for point in timeline], alpha=0.8,
                     label=f"seed {row['seed']}")
    axes[0].axhline(result["config"]["activation_barrier"], color="#c62828", ls="--")
    axes[0].set_title("真实窗口 residual-F 压力"); axes[0].set_ylabel("高能样本比例")
    axes[1].axhline(result["config"]["activation_barrier"], color="#c62828", ls="--",
                    label="生长势垒")
    axes[1].axhline(result["config"]["reset_barrier"], color="#2e7d32", ls=":",
                    label="复位势垒")
    axes[1].set_title("弛豫后的结构不稳定度"); axes[1].set_ylabel("稳定慢状态")
    for axis in axes:
        axis.set_xlabel("在线窗口"); axis.set_ylim(0, 0.75); axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.suptitle("短暂污染耗散，持续漂移越过结构势垒")
    fig.tight_layout(); fig.savefig(FIG_PATH, dpi=180); plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    source = load_source(args.source_report)
    rows = [run_seed(row, args) for row in source["rows"]]
    summary = summarize(rows)
    passed = bool(
        summary["pass_rate"] == 1.0
        and summary["inner_energy_monotonic_rate"] == 1.0
        and summary["naive_burst_trigger_rate"] == 1.0
        and summary["stabilized_burst_trigger_rate"] == 0.0
        and summary["sustained_drift_activation_rate"] == 1.0
        and summary["concept_return_reset_rate"] == 1.0
        and summary["hysteresis_rate"] == 1.0
    )
    verdict = (
        "✅ 结构自由能慢动力学成立：单窗口规则会响应的污染 burst 被结构惯性耗散；"
        "持续真实漂移才越过生长势垒；概念回返后状态跨越较低复位势垒，避免临界抖动。"
        if passed else
        "🟡 慢时间尺度结构自由能尚未同时分离 burst、持续漂移与概念回返。"
    )
    result = {
        "task": "slow structural free-energy dynamics on real domain pressure timeline",
        "source_report": args.source_report,
        "source_commit": "fe9c64e",
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
    parser = argparse.ArgumentParser(description="真实漂移压力上的结构自由能慢动力学。")
    parser.add_argument("--source-report", default=SOURCE_REPORT)
    parser.add_argument("--observation-precision", type=float, default=1.0)
    parser.add_argument("--persistence-precision", type=float, default=4.0)
    parser.add_argument("--complexity-precision", type=float, default=0.01)
    parser.add_argument("--relaxation-steps", type=int, default=8)
    parser.add_argument("--relaxation-fraction", type=float, default=0.8)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--activation-barrier", type=float, default=0.25)
    parser.add_argument("--reset-barrier", type=float, default=0.10)
    parser.add_argument("--warmup-windows", type=int, default=5)
    parser.add_argument("--local-events", type=int, default=3)
    parser.add_argument("--recovery-windows", type=int, default=5)
    parser.add_argument("--dwell-windows", type=int, default=3)
    parser.add_argument("--return-windows", type=int, default=12)
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
        "summary": result["summary"], "pass": result["pass"],
        "verdict": result["verdict"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
