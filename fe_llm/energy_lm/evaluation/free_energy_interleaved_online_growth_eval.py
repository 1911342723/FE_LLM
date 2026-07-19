# -*- coding: utf-8 -*-
"""真实 OPUS→Python 在线流中的慢能量触发、交错 probe 与原子固化。

活动系统持续按现有盆地服务。窗口 residual-F 压力先进入结构自由能慢动力学；临时盆地
从未注册到活动路由，probe/readout 每训练一小段就恢复一次服务。只有独立 held-out 可约
率越过固定门槛后才一次性固化。持续的边际匹配噪声也可越过慢势垒并获得 probe，但最终
不可约，因此拒绝。第一版是单进程交错调度，不声称 CUDA 真并发。
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
from fe_llm.energy_lm.free_energy_growth import (
    FreeEnergyGrowthSystem,
    StructuralFreeEnergyStabilizer,
)
from fe_llm.energy_lm.evaluation.free_energy_domain_drift_eval import (
    CODE_TRAIN,
    CODE_VALID,
    DATA_PATH,
    make_candidate,
    make_mixture,
    prepare_data,
)
from fe_llm.energy_lm.evaluation.free_energy_reducibility_eval import (
    lm_loss,
    path_metrics,
    sample_chunks,
    shuffle_chunks,
    train_base,
)

REPORT_JSON = os.path.join(
    "docs", "reports", "free_energy_interleaved_online_growth_eval.json")
REPORT_MD = os.path.join(
    "docs", "reports", "free_energy_interleaved_online_growth_eval.md")
FIG_PATH = os.path.join(
    "docs", "reports", "figs", "free_energy_interleaved_online_growth_eval.png")


@torch.no_grad()
def service_snapshot(
    system: FreeEnergyGrowthSystem,
    service_ids: torch.Tensor,
    base_audit: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    start: int,
) -> dict:
    logits, choices = system.routed_logits(service_ids, start=start)
    base_delta = float((system.core(base_audit) - base_logits).abs().max().cpu())
    return {
        "pathway_count": system.pathway_count,
        "finite": bool(torch.isfinite(logits).all()),
        "base_logit_max_delta": round(base_delta, 9),
        "route_histogram": [
            int((choices == pathway).sum().cpu())
            for pathway in range(system.pathway_count)
        ],
    }


def observe_window(
    system: FreeEnergyGrowthSystem,
    stabilizer: StructuralFreeEnergyStabilizer,
    ids: torch.Tensor,
    base_audit: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    phase: str,
    label: str,
    args: argparse.Namespace,
) -> dict:
    instant, pressure, mean_energy = system.should_grow(
        ids,
        start=args.energy_start,
        min_fraction=args.min_growth_fraction,
    )
    state, active, trace = stabilizer.observe(pressure, return_trace=True)
    assert trace is not None
    energy = trace["free_energy"]
    return {
        "phase": phase,
        "label": label,
        "instant_growth": instant,
        "high_energy_fraction": round(pressure, 6),
        "mean_best_energy": round(mean_energy, 6),
        "stable_instability": round(float(state.cpu()), 6),
        "slow_active": active,
        "slow_activated": bool(trace["activated"]),
        "structural_energy_monotonic": bool(
            (energy[1:] <= energy[:-1] + 1e-8).all()),
        "service": service_snapshot(
            system, ids, base_audit, base_logits, start=args.energy_start),
    }


def steps_for_window(total: int, windows: int, index: int) -> int:
    return total // windows + int(index < total % windows)


def train_probe_interleaved(
    system: FreeEnergyGrowthSystem,
    transition,
    train_stream: torch.Tensor,
    heldout: torch.Tensor,
    service_ids: torch.Tensor,
    base_audit: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    noise: bool,
    seed: int,
    args: argparse.Namespace,
) -> list[dict]:
    params = system.train_only_provisional(transition)
    optimizer = torch.optim.AdamW(
        params, lr=args.probe_lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device=args.device).manual_seed(seed)
    history = []
    global_step = 0
    for window in range(args.probe_windows):
        block_steps = steps_for_window(args.probe_steps, args.probe_windows, window)
        transition.train()
        for _ in range(block_steps):
            sequence = sample_chunks(
                train_stream, args.batch, args.length, generator)
            if noise:
                sequence = shuffle_chunks(sequence, generator)
            logits = system.core(sequence, transition_override=transition)
            ce = lm_loss(logits, sequence)
            assert system.core.last_position_free_energy is not None
            residual = system.core.last_position_free_energy[:, 1:].mean()
            loss = residual + args.probe_ce_weight * ce
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step()
            global_step += 1
        transition.eval()
        metrics = path_metrics(
            system.core, transition, heldout, start=args.energy_start)
        history.append({
            "window": window + 1,
            "global_step": global_step,
            "heldout_residual": round(metrics["residual"], 6),
            "heldout_bpc": round(metrics["bpc"], 6),
            "service": service_snapshot(
                system, service_ids, base_audit, base_logits,
                start=args.energy_start),
        })
    return history


def train_readout_interleaved(
    system: FreeEnergyGrowthSystem,
    transition,
    head,
    train_stream: torch.Tensor,
    heldout: torch.Tensor,
    service_ids: torch.Tensor,
    base_audit: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    seed: int,
    args: argparse.Namespace,
) -> list[dict]:
    for parameter in transition.parameters():
        parameter.requires_grad_(False)
    params = system.train_only_provisional_head(head)
    optimizer = torch.optim.AdamW(
        params, lr=args.readout_lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device=args.device).manual_seed(seed)
    history = []
    global_step = 0
    for window in range(args.readout_windows):
        block_steps = steps_for_window(
            args.readout_steps, args.readout_windows, window)
        head.train()
        for _ in range(block_steps):
            sequence = sample_chunks(
                train_stream, args.batch, args.length, generator)
            logits = system.core(
                sequence, transition_override=transition, head_override=head)
            ce = lm_loss(logits, sequence)
            optimizer.zero_grad(set_to_none=True); ce.backward(); optimizer.step()
            global_step += 1
        head.eval()
        metrics = path_metrics(
            system.core, transition, heldout,
            start=args.energy_start, head=head)
        history.append({
            "window": window + 1,
            "global_step": global_step,
            "heldout_bpc": round(metrics["bpc"], 6),
            "service": service_snapshot(
                system, service_ids, base_audit, base_logits,
                start=args.energy_start),
        })
    return history


def make_stabilizer(args: argparse.Namespace) -> StructuralFreeEnergyStabilizer:
    return StructuralFreeEnergyStabilizer(
        observation_precision=args.structural_observation_precision,
        persistence_precision=args.structural_persistence_precision,
        complexity_precision=args.structural_complexity_precision,
        relaxation_steps=args.structural_relax_steps,
        relaxation_fraction=args.structural_relaxation_fraction,
        activation_barrier=args.activation_barrier,
        reset_barrier=args.reset_barrier,
    ).to(args.device)


def run_seed(seed: int, data: dict, args: argparse.Namespace) -> dict:
    started = time.time()
    core = train_base(seed, data["base_train"], data["vocab_size"], args)
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator(device=args.device).manual_seed(seed + 10001)
    base_audit = sample_chunks(
        data["base_valid"], args.eval_batch, args.length, generator)
    code_heldout = sample_chunks(
        data["code_valid"], args.eval_batch, args.length, generator)
    noise_heldout = shuffle_chunks(code_heldout.clone(), generator)
    base_window = sample_chunks(
        data["base_valid"], args.stream_batch, args.length, generator)
    code_window = sample_chunks(
        data["code_train"], args.stream_batch, args.length, generator)
    noise_window = shuffle_chunks(code_window.clone(), generator)
    burst_window, _ = make_mixture(
        base_window, noise_window, args.burst_pollution_fraction)
    base_logits = core(base_audit).detach().clone()

    code_system = FreeEnergyGrowthSystem(core).to(args.device).eval()
    threshold = code_system.calibrate_threshold(
        base_audit, start=args.energy_start, quantile=args.threshold_quantile)
    code_stabilizer = make_stabilizer(args)
    timeline = []
    for _ in range(args.warmup_windows):
        timeline.append(observe_window(
            code_system, code_stabilizer, base_window, base_audit, base_logits,
            phase="warmup", label="base", args=args))
    timeline.append(observe_window(
        code_system, code_stabilizer, burst_window, base_audit, base_logits,
        phase="burst", label="50% shuffled", args=args))
    for _ in range(args.recovery_windows):
        timeline.append(observe_window(
            code_system, code_stabilizer, base_window, base_audit, base_logits,
            phase="recovery", label="base", args=args))
    for index in range(args.max_drift_windows):
        event = observe_window(
            code_system, code_stabilizer, code_window, base_audit, base_logits,
            phase="sustained_code", label=f"code-{index+1}", args=args)
        timeline.append(event)
        if event["slow_activated"]:
            break

    burst_event = next(row for row in timeline if row["phase"] == "burst")
    pre_code_slow_trigger = any(
        row["slow_active"] for row in timeline if row["phase"] != "sustained_code")
    code_trigger = next(
        (row for row in timeline if row["phase"] == "sustained_code"
         and row["slow_activated"]), None)
    code_before = path_metrics(
        core, core.transition, code_heldout, start=args.energy_start)
    torch.manual_seed(seed * 100 + 1)
    code_transition, code_head = make_candidate(core, args.rank)
    code_transition = code_transition.to(args.device)
    code_head = code_head.to(args.device)
    code_probe = train_probe_interleaved(
        code_system,
        code_transition,
        data["code_train"],
        code_heldout,
        code_window,
        base_audit,
        base_logits,
        noise=False,
        seed=seed + 11001,
        args=args,
    ) if code_trigger is not None else []
    code_probe_metrics = path_metrics(
        core, code_transition, code_heldout, start=args.energy_start)
    code_reducibility = (
        code_before["residual"] - code_probe_metrics["residual"]
    ) / max(1e-8, code_before["residual"])
    code_reducible = code_reducibility >= args.min_reducibility
    code_readout = []
    committed = None
    consolidated = code_probe_metrics
    precommit_count = code_system.pathway_count
    if code_trigger is not None and code_reducible:
        code_readout = train_readout_interleaved(
            code_system,
            code_transition,
            code_head,
            data["code_train"],
            code_heldout,
            code_window,
            base_audit,
            base_logits,
            seed=seed + 12001,
            args=args,
        )
        consolidated = path_metrics(
            core, code_transition, code_heldout,
            start=args.energy_start, head=code_head)
        complexity_cost = code_system.calibrate_complexity_cost(
            base_audit,
            code_transition,
            start=args.energy_start,
            quantile=args.complexity_quantile,
            margin=args.complexity_margin,
        )
        for module in (code_transition, code_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        committed = code_system.commit_pathway(
            code_transition, complexity_cost=complexity_cost, head=code_head)
    else:
        complexity_cost = None
    postcommit_count = code_system.pathway_count
    code_choices, _ = code_system.route(code_heldout, start=args.energy_start)
    base_choices, _ = code_system.route(base_audit, start=args.energy_start)
    code_route = (
        None if committed is None
        else float((code_choices == committed).float().mean().cpu()))
    base_route = float((base_choices == 0).float().mean().cpu())

    noise_system = FreeEnergyGrowthSystem(core).to(args.device).eval()
    noise_system.calibrate_threshold(
        base_audit, start=args.energy_start, quantile=args.threshold_quantile)
    noise_stabilizer = make_stabilizer(args)
    noise_timeline = []
    for _ in range(args.warmup_windows):
        noise_timeline.append(observe_window(
            noise_system, noise_stabilizer, base_window, base_audit, base_logits,
            phase="warmup", label="base", args=args))
    for index in range(args.max_drift_windows):
        event = observe_window(
            noise_system, noise_stabilizer, noise_window, base_audit, base_logits,
            phase="sustained_noise", label=f"noise-{index+1}", args=args)
        noise_timeline.append(event)
        if event["slow_activated"]:
            break
    noise_trigger = next(
        (row for row in noise_timeline if row["phase"] == "sustained_noise"
         and row["slow_activated"]), None)
    noise_before = path_metrics(
        core, core.transition, noise_heldout, start=args.energy_start)
    torch.manual_seed(seed * 100 + 2)
    noise_transition, _ = make_candidate(core, args.rank)
    noise_transition = noise_transition.to(args.device)
    noise_probe = train_probe_interleaved(
        noise_system,
        noise_transition,
        data["code_train"],
        noise_heldout,
        noise_window,
        base_audit,
        base_logits,
        noise=True,
        seed=seed + 13001,
        args=args,
    ) if noise_trigger is not None else []
    noise_probe_metrics = path_metrics(
        core, noise_transition, noise_heldout, start=args.energy_start)
    noise_reducibility = (
        noise_before["residual"] - noise_probe_metrics["residual"]
    ) / max(1e-8, noise_before["residual"])
    noise_rejected = noise_reducibility < args.min_reducibility

    service_rows = (
        [row["service"] for row in timeline + noise_timeline]
        + [row["service"] for row in code_probe + code_readout + noise_probe]
    )
    service_all_finite = all(row["finite"] for row in service_rows)
    service_base_delta = max(row["base_logit_max_delta"] for row in service_rows)
    provisional_hidden = all(
        row["service"]["pathway_count"] == 1
        for row in code_probe + code_readout + noise_probe)
    structural_monotonic = all(
        row["structural_energy_monotonic"] for row in timeline + noise_timeline)
    base_delta = float((core(base_audit) - base_logits).abs().max().cpu())
    passed = bool(
        burst_event["instant_growth"]
        and not pre_code_slow_trigger
        and code_trigger is not None
        and noise_trigger is not None
        and structural_monotonic
        and provisional_hidden
        and precommit_count == 1 and postcommit_count == 2
        and code_reducible and committed is not None
        and code_route is not None and code_route >= args.min_route_accuracy
        and base_route >= args.min_route_accuracy
        and consolidated["bpc"] < code_before["bpc"]
        and noise_rejected and noise_system.pathway_count == 1
        and service_all_finite and service_base_delta <= 1e-8
        and base_delta <= 1e-8
    )
    return {
        "seed": seed,
        "threshold": round(threshold, 6),
        "timeline": timeline,
        "burst_instant_trigger": burst_event["instant_growth"],
        "burst_slow_trigger": pre_code_slow_trigger,
        "code_activation_window": (
            None if code_trigger is None else code_trigger["label"]),
        "code_before": {
            key: round(float(value), 6) for key, value in code_before.items()
            if key != "residual_per_sample"},
        "code_probe": code_probe,
        "code_probe_metrics": {
            key: round(float(value), 6) for key, value in code_probe_metrics.items()
            if key != "residual_per_sample"},
        "code_reducibility": round(code_reducibility, 6),
        "code_readout": code_readout,
        "code_consolidated": {
            key: round(float(value), 6) for key, value in consolidated.items()
            if key != "residual_per_sample"},
        "complexity_cost": (
            None if complexity_cost is None else round(complexity_cost, 6)),
        "precommit_pathway_count": precommit_count,
        "postcommit_pathway_count": postcommit_count,
        "code_committed": committed is not None,
        "code_route_accuracy": None if code_route is None else round(code_route, 6),
        "base_route_accuracy": round(base_route, 6),
        "noise_timeline": noise_timeline,
        "noise_activation_window": (
            None if noise_trigger is None else noise_trigger["label"]),
        "noise_probe": noise_probe,
        "noise_reducibility": round(noise_reducibility, 6),
        "noise_rejected": noise_rejected,
        "noise_pathway_count": noise_system.pathway_count,
        "structural_energy_monotonic": structural_monotonic,
        "service_interleavings": len(service_rows),
        "service_all_finite": service_all_finite,
        "provisional_hidden_from_service": provisional_hidden,
        "service_base_logit_max_delta": round(service_base_delta, 9),
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
        "burst_instant_trigger_rate": round(float(np.mean([
            row["burst_instant_trigger"] for row in rows])), 6),
        "burst_slow_trigger_rate": round(float(np.mean([
            row["burst_slow_trigger"] for row in rows])), 6),
        "code_commit_rate": round(float(np.mean([
            row["code_committed"] for row in rows])), 6),
        "noise_reject_rate": round(float(np.mean([
            row["noise_rejected"] for row in rows])), 6),
        "structural_energy_monotonic_rate": round(float(np.mean([
            row["structural_energy_monotonic"] for row in rows])), 6),
        "service_finite_rate": round(float(np.mean([
            row["service_all_finite"] for row in rows])), 6),
        "provisional_hidden_rate": round(float(np.mean([
            row["provisional_hidden_from_service"] for row in rows])), 6),
    }
    for key in (
        "code_reducibility", "noise_reducibility", "code_route_accuracy",
        "base_route_accuracy", "service_interleavings",
        "service_base_logit_max_delta", "base_logit_max_delta",
    ):
        summary[key] = mean_std([row[key] for row in rows if row[key] is not None])
    for stage in ("code_before", "code_consolidated"):
        summary[f"{stage}_bpc"] = mean_std([row[stage]["bpc"] for row in rows])
    return summary


def write_reports(result: dict) -> None:
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    summary = result["summary"]
    with open(REPORT_MD, "w", encoding="utf-8") as file:
        file.write(
            "# 交错在线生长：慢能量触发、临时 probe、原子固化\n\n"
            "真实 OPUS→Python 流；活动系统在每个 probe/readout 窗口之间继续服务，临时"
            "盆地通过 held-out 可约性前不可见。\n\n"
            "| 裁决 | 三种子结果 |\n|---|---:|\n"
            f"| 单窗口规则对 burst 触发率 | {summary['burst_instant_trigger_rate']:.1%} |\n"
            f"| 慢结构自由能对 burst 触发率 | {summary['burst_slow_trigger_rate']:.1%} |\n"
            f"| 结构自由能单调率 | {summary['structural_energy_monotonic_rate']:.1%} |\n"
            f"| 真实代码 / 噪声可约率 | {summary['code_reducibility']['mean']:.1%} / "
            f"{summary['noise_reducibility']['mean']:.1%} |\n"
            f"| 代码固化 / 噪声拒绝率 | {summary['code_commit_rate']:.1%} / "
            f"{summary['noise_reject_rate']:.1%} |\n"
            f"| 临时盆地对服务隐藏率 | {summary['provisional_hidden_rate']:.1%} |\n"
            f"| 交错服务有限输出率 | {summary['service_finite_rate']:.1%} |\n"
            f"| 每种子交错服务次数 | {summary['service_interleavings']['mean']:.1f} |\n"
            f"| 基础 / 代码路由 | {summary['base_route_accuracy']['mean']:.1%} / "
            f"{summary['code_route_accuracy']['mean']:.1%} |\n"
            f"| held-out 代码 BPC | {summary['code_before_bpc']['mean']:.3f}→"
            f"{summary['code_consolidated_bpc']['mean']:.3f} |\n"
            f"| 服务期间基础 logits 最大变化 | "
            f"{summary['service_base_logit_max_delta']['mean']:.1e} |\n\n"
            f"## 裁决\n\n{result['verdict']}\n\n"
            "## 边界\n\n"
            "当前是单进程按窗口交错训练与服务，证明了临时容量隔离和原子可见性，但不是"
            "真正并发 CUDA stream 或多进程服务；也尚未处理 probe 尚未结束时第二个新域"
            "同时到达的竞争。\n"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = result["rows"][0]
    timeline = first["timeline"]
    raw = [row["high_energy_fraction"] for row in timeline]
    stable = [row["stable_instability"] for row in timeline]
    x = np.arange(len(timeline))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    axes[0].plot(x, raw, "o-", label="窗口 residual-F 压力")
    axes[0].plot(x, stable, "s-", label="稳定结构慢状态")
    axes[0].axhline(result["config"]["activation_barrier"], color="#c62828", ls="--")
    axes[0].set_title("burst 耗散，持续代码漂移激活"); axes[0].legend(fontsize=8)
    axes[1].bar(
        ["真实代码", "等边际噪声"],
        [summary["code_reducibility"]["mean"], summary["noise_reducibility"]["mean"]],
        color=["#2e7d32", "#78909c"])
    axes[1].axhline(result["config"]["min_reducibility"], color="#c62828", ls="--")
    axes[1].set_title("交错 probe 后的 held-out 可约性")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("在线服务不中断：临时盆地隔离，能量通过后原子固化")
    fig.tight_layout(); fig.savefig(FIG_PATH, dpi=180); plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    args.device = args.device.strip() or get_device()
    data = prepare_data(args)
    rows = [run_seed(int(value), data, args) for value in args.seeds.split(",") if value.strip()]
    summary = summarize(rows)
    passed = bool(
        summary["pass_rate"] == 1.0
        and summary["burst_instant_trigger_rate"] == 1.0
        and summary["burst_slow_trigger_rate"] == 0.0
        and summary["code_commit_rate"] == 1.0
        and summary["noise_reject_rate"] == 1.0
        and summary["provisional_hidden_rate"] == 1.0
        and summary["service_base_logit_max_delta"]["mean"] <= 1e-8
    )
    verdict = (
        "✅ 交错在线自由能生长成立：慢结构动力学过滤单次 burst；持续真实漂移才启动"
        "未注册 probe；训练期间活动系统持续服务且旧 logits 不变；代码 held-out 可约后"
        "原子固化，持续噪声 probe 后仍不可约而拒绝。"
        if passed else
        "🟡 在线流程尚未同时通过慢触发、临时隔离、服务连续、代码固化与噪声拒绝。"
    )
    result = {
        "task": "interleaved online free-energy growth with atomic commit",
        "config": vars(args),
        "data": {"vocab_size": data["vocab_size"], "sizes": data["sizes"]},
        "rows": rows,
        "summary": summary,
        "pass": passed,
        "verdict": verdict,
    }
    if args.write_report:
        write_reports(result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="真实领域交错在线自由能生长实验。")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--code-train", default=CODE_TRAIN)
    parser.add_argument("--code-valid", default=CODE_VALID)
    parser.add_argument("--device", default="")
    parser.add_argument("--seeds", default="152,153,154")
    parser.add_argument("--sentences", type=int, default=12000)
    parser.add_argument("--code-train-chars", type=int, default=4_000_000)
    parser.add_argument("--code-valid-chars", type=int, default=2_000_000)
    parser.add_argument("--base-steps", type=int, default=350)
    parser.add_argument("--probe-steps", type=int, default=240)
    parser.add_argument("--readout-steps", type=int, default=240)
    parser.add_argument("--probe-windows", type=int, default=8)
    parser.add_argument("--readout-windows", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=384)
    parser.add_argument("--stream-batch", type=int, default=256)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=48)
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
    parser.add_argument("--complexity-quantile", type=float, default=0.90)
    parser.add_argument("--complexity-margin", type=float, default=1e-4)
    parser.add_argument("--burst-pollution-fraction", type=float, default=0.50)
    parser.add_argument("--warmup-windows", type=int, default=5)
    parser.add_argument("--recovery-windows", type=int, default=5)
    parser.add_argument("--max-drift-windows", type=int, default=10)
    parser.add_argument("--structural-observation-precision", type=float, default=1.0)
    parser.add_argument("--structural-persistence-precision", type=float, default=4.0)
    parser.add_argument("--structural-complexity-precision", type=float, default=0.01)
    parser.add_argument("--structural-relax-steps", type=int, default=8)
    parser.add_argument("--structural-relaxation-fraction", type=float, default=0.8)
    parser.add_argument("--activation-barrier", type=float, default=0.25)
    parser.add_argument("--reset-barrier", type=float, default=0.10)
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
