# -*- coding: utf-8 -*-
"""真实自然语言→Python 代码漂移下的自由能生长裁决。

稳定域来自 OPUS 英文句子，新域来自 CodeParrot Python 源码；代码训练与 held-out 使用
不同数据分片。系统不读取域名，只观察所有现有生成性通路能否把因果字符流稳定到低
residual free-energy。实验同时检查三种外部过程：

1. 局部污染：少量逐 chunk 打乱的代码字符不应触发全局扩容；
2. 渐变漂移：代码比例上升时，高能样本比例应随之上升并越过固定生长闸门；
3. 真实换域：代码临时低秩动力学在独立 held-out 上可约才固化，而等字符边际噪声拒绝。

新通路仍是 ``T_base(z)+U tanh(Vz)`` 与低秩读出；没有 Q/K attention、Transformer
层、域分类器或关键词 router。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
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
    build_stream,
    encode_stream,
    load_sentences,
    make_vocab,
    path_metrics,
    sample_chunks,
    shuffle_chunks,
    train_base,
    train_probe,
    train_readout,
)

CODE_TRAIN = os.path.join("data", "code", "python_corpus.txt")
CODE_VALID = os.path.join("data", "code", "python_heldout.txt")
REPORT_JSON = os.path.join("docs", "reports", "free_energy_domain_drift_eval.json")
REPORT_MD = os.path.join("docs", "reports", "free_energy_domain_drift_eval.md")
FIG_PATH = os.path.join("docs", "reports", "figs", "free_energy_domain_drift_eval.png")

CODE_CHARS = set(
    string.ascii_lowercase + string.digits
    + " \n.,!?'-_()[]{}:=+*/<>\"\\#@;%&|^~`"
)


def normalize_code(text: str) -> str:
    """保留真实代码符号与行边界，同时去掉稀有 Unicode 对词表的偶然影响。"""
    text = text.lower().replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    text = "".join(character if character in CODE_CHARS else " " for character in text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def read_prefix(path: str, characters: int) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"缺少代码语料 {path}。先运行 python -m fe_llm.energy_lm.data.prepare_code")
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read(characters)


def prepare_data(args: argparse.Namespace) -> dict:
    sentences = load_sentences(args.data, args.sentences)
    split = int(len(sentences) * 0.75)
    base_train_text = build_stream(sentences[:split])
    base_valid_text = build_stream(sentences[split:])
    code_train_text = normalize_code(read_prefix(args.code_train, args.code_train_chars))
    code_valid_text = normalize_code(read_prefix(args.code_valid, args.code_valid_chars))
    if min(len(code_train_text), len(code_valid_text)) < args.length * args.eval_batch:
        raise RuntimeError("规范化后的代码训练或验证流过短。")

    vocab, id_to_char = make_vocab(
        base_train_text, base_valid_text, code_train_text, code_valid_text)
    return {
        "base_train": encode_stream(base_train_text, vocab, args.device),
        "base_valid": encode_stream(base_valid_text, vocab, args.device),
        "code_train": encode_stream(code_train_text, vocab, args.device),
        "code_valid": encode_stream(code_valid_text, vocab, args.device),
        "vocab_size": len(vocab),
        "vocab": "".join(id_to_char),
        "sizes": {
            "base_train_chars": len(base_train_text),
            "base_valid_chars": len(base_valid_text),
            "code_train_chars": len(code_train_text),
            "code_valid_chars": len(code_valid_text),
        },
    }


def make_candidate(core, rank: int):
    return (
        LowRankGenerativeTransition(core.transition, dim=core.dim, rank=rank),
        LowRankReadout(core.head, in_dim=core.dim, out_dim=core.vocab_size, rank=rank),
    )


def make_mixture(
    base_ids: torch.Tensor,
    shifted_ids: torch.Tensor,
    shifted_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造固定大小的混合窗口，并返回逐样本真实来源，仅供事后审计。"""
    if base_ids.shape != shifted_ids.shape:
        raise ValueError("base_ids 与 shifted_ids 必须形状相同。")
    if not 0.0 <= shifted_fraction <= 1.0:
        raise ValueError("shifted_fraction 必须在 [0,1] 内。")
    count = base_ids.size(0)
    shifted_count = int(round(count * shifted_fraction))
    base_count = count - shifted_count
    mixed = torch.cat((base_ids[:base_count], shifted_ids[:shifted_count]), dim=0)
    source = torch.cat((
        torch.zeros(base_count, dtype=torch.long, device=base_ids.device),
        torch.ones(shifted_count, dtype=torch.long, device=base_ids.device),
    ))
    return mixed, source


def compact(metrics: dict) -> dict[str, float]:
    return {
        key: round(float(value), 6)
        for key, value in metrics.items()
        if key != "residual_per_sample"
    }


@torch.no_grad()
def pressure_curve(
    system: FreeEnergyGrowthSystem,
    base_ids: torch.Tensor,
    shifted_ids: torch.Tensor,
    fractions: list[float],
    *,
    start: int,
    min_growth_fraction: float,
) -> list[dict]:
    rows = []
    for fraction in fractions:
        mixed, _ = make_mixture(base_ids, shifted_ids, fraction)
        decision, pressure, energy = system.should_grow(
            mixed, start=start, min_fraction=min_growth_fraction)
        rows.append({
            "shifted_fraction": round(fraction, 6),
            "high_energy_fraction": round(pressure, 6),
            "mean_best_energy": round(energy, 6),
            "growth_triggered": decision,
        })
    return rows


@torch.no_grad()
def mixed_route_accuracy(
    system: FreeEnergyGrowthSystem,
    base_ids: torch.Tensor,
    code_ids: torch.Tensor,
    fractions: list[float],
    code_pathway: int,
    *,
    start: int,
) -> list[dict]:
    rows = []
    for fraction in fractions:
        mixed, source = make_mixture(base_ids, code_ids, fraction)
        choices, _ = system.route(mixed, start=start)
        expected = torch.where(source == 0, 0, code_pathway)
        rows.append({
            "code_fraction": round(fraction, 6),
            "route_accuracy": round(float((choices == expected).float().mean().cpu()), 6),
            "route_to_code": round(float((choices == code_pathway).float().mean().cpu()), 6),
        })
    return rows


def run_seed(seed: int, data: dict, args: argparse.Namespace) -> dict:
    started = time.time()
    core = train_base(seed, data["base_train"], data["vocab_size"], args)
    system = FreeEnergyGrowthSystem(core).to(args.device).eval()
    generator = torch.Generator(device=args.device).manual_seed(seed + 5001)
    base_valid = sample_chunks(data["base_valid"], args.eval_batch, args.length, generator)
    code_valid = sample_chunks(data["code_valid"], args.eval_batch, args.length, generator)
    base_window = sample_chunks(data["base_valid"], args.stream_batch, args.length, generator)
    code_window = sample_chunks(data["code_train"], args.stream_batch, args.length, generator)
    shuffled_window = shuffle_chunks(code_window.clone(), generator)
    noise_valid = shuffle_chunks(code_valid.clone(), generator)
    base_logits_before = core(base_valid).detach().clone()

    base_before = path_metrics(core, core.transition, base_valid, start=args.energy_start)
    code_before = path_metrics(core, core.transition, code_valid, start=args.energy_start)
    noise_before = path_metrics(core, core.transition, noise_valid, start=args.energy_start)
    threshold = system.calibrate_threshold(
        base_valid, start=args.energy_start, quantile=args.threshold_quantile)

    fractions = [float(value) for value in args.drift_fractions.split(",") if value.strip()]
    gradual = pressure_curve(
        system,
        base_window,
        code_window,
        fractions,
        start=args.energy_start,
        min_growth_fraction=args.min_growth_fraction,
    )
    local_mixed, _ = make_mixture(
        base_window, shuffled_window, args.local_pollution_fraction)
    local_trigger, local_pressure, local_energy = system.should_grow(
        local_mixed, start=args.energy_start, min_fraction=args.min_growth_fraction)
    burst_mixed, _ = make_mixture(
        base_window, shuffled_window, args.burst_pollution_fraction)
    burst_trigger, burst_pressure, burst_energy = system.should_grow(
        burst_mixed, start=args.energy_start, min_fraction=args.min_growth_fraction)

    code_transition, code_head = make_candidate(core, args.rank)
    code_transition = code_transition.to(args.device)
    code_head = code_head.to(args.device)
    full_growth_params = (
        sum(parameter.numel() for parameter in core.transition.parameters())
        + sum(parameter.numel() for parameter in core.head.parameters())
    )
    added_params = (
        sum(parameter.numel() for parameter in code_transition.parameters())
        + sum(parameter.numel() for parameter in code_head.parameters())
    )
    code_history = train_probe(
        system, code_transition, data["code_train"], noise=False,
        seed=seed + 5101, args=args)
    code_probe = path_metrics(
        core, code_transition, code_valid, start=args.energy_start)
    code_reducibility = (
        code_before["residual"] - code_probe["residual"]
    ) / max(1e-8, code_before["residual"])

    noise_system = FreeEnergyGrowthSystem(core).to(args.device).eval()
    noise_transition, _ = make_candidate(core, args.rank)
    noise_transition = noise_transition.to(args.device)
    noise_history = train_probe(
        noise_system, noise_transition, data["code_train"], noise=True,
        seed=seed + 5201, args=args)
    noise_probe = path_metrics(
        core, noise_transition, noise_valid, start=args.energy_start)
    noise_reducibility = (
        noise_before["residual"] - noise_probe["residual"]
    ) / max(1e-8, noise_before["residual"])

    full_code_trigger = gradual[-1]["growth_triggered"]
    code_commit = bool(full_code_trigger and code_reducibility >= args.min_reducibility)
    noise_commit = bool(burst_trigger and noise_reducibility >= args.min_reducibility)
    code_pathway = None
    code_consolidated = code_probe
    readout_history: list[dict] = []
    complexity_cost = None
    if code_commit:
        for parameter in code_transition.parameters():
            parameter.requires_grad_(False)
        readout_history = train_readout(
            system, code_transition, code_head, data["code_train"],
            seed=seed + 5301, args=args)
        code_consolidated = path_metrics(
            core, code_transition, code_valid, start=args.energy_start, head=code_head)
        complexity_cost = system.calibrate_complexity_cost(
            base_valid,
            code_transition,
            start=args.energy_start,
            quantile=args.complexity_quantile,
            margin=args.complexity_margin,
        )
        code_pathway = system.commit_pathway(
            code_transition, complexity_cost=complexity_cost, head=code_head)

    base_route = code_route = None
    mixed_routes: list[dict] = []
    if code_pathway is not None:
        base_choices, _ = system.route(base_valid, start=args.energy_start)
        code_choices, _ = system.route(code_valid, start=args.energy_start)
        base_route = float((base_choices == 0).float().mean().cpu())
        code_route = float((code_choices == code_pathway).float().mean().cpu())
        mixed_routes = mixed_route_accuracy(
            system,
            base_valid,
            code_valid,
            fractions,
            code_pathway,
            start=args.energy_start,
        )

    pressure_values = [row["high_energy_fraction"] for row in gradual]
    monotonic = all(
        current + args.monotonic_tolerance >= previous
        for previous, current in zip(pressure_values, pressure_values[1:])
    )
    onset = next(
        (row["shifted_fraction"] for row in gradual if row["growth_triggered"]), None)
    base_delta = float((core(base_valid) - base_logits_before).abs().max().cpu())
    mixed_min_route = (
        min(row["route_accuracy"] for row in mixed_routes) if mixed_routes else None)
    passed = bool(
        monotonic
        and not local_trigger
        and burst_trigger
        and code_commit
        and not noise_commit
        and code_reducibility >= args.min_reducibility
        and noise_reducibility < args.min_reducibility
        and base_route is not None and base_route >= args.min_route_accuracy
        and code_route is not None and code_route >= args.min_route_accuracy
        and mixed_min_route is not None and mixed_min_route >= args.min_mixed_route_accuracy
        and code_consolidated["bpc"] < code_before["bpc"]
        and base_delta <= 1e-8
    )
    return {
        "seed": seed,
        "threshold": round(threshold, 6),
        "base_before": compact(base_before),
        "code_before": compact(code_before),
        "noise_before": compact(noise_before),
        "gradual_drift": gradual,
        "gradual_pressure_monotonic": monotonic,
        "growth_onset_code_fraction": onset,
        "local_pollution": {
            "fraction": args.local_pollution_fraction,
            "growth_triggered": local_trigger,
            "high_energy_fraction": round(local_pressure, 6),
            "mean_best_energy": round(local_energy, 6),
        },
        "burst_pollution": {
            "fraction": args.burst_pollution_fraction,
            "growth_triggered": burst_trigger,
            "high_energy_fraction": round(burst_pressure, 6),
            "mean_best_energy": round(burst_energy, 6),
        },
        "code_probe": compact(code_probe),
        "noise_probe": compact(noise_probe),
        "code_reducibility": round(code_reducibility, 6),
        "noise_reducibility": round(noise_reducibility, 6),
        "code_committed": code_commit,
        "noise_committed": noise_commit,
        "full_growth_params": full_growth_params,
        "added_params": added_params,
        "parameter_ratio": round(added_params / full_growth_params, 6),
        "code_consolidated": compact(code_consolidated),
        "complexity_cost": None if complexity_cost is None else round(complexity_cost, 6),
        "base_route_accuracy": None if base_route is None else round(base_route, 6),
        "code_route_accuracy": None if code_route is None else round(code_route, 6),
        "mixed_routes": mixed_routes,
        "mixed_min_route_accuracy": mixed_min_route,
        "base_logit_max_delta": round(base_delta, 9),
        "code_probe_history": code_history,
        "noise_probe_history": noise_history,
        "readout_history": readout_history,
        "pass": passed,
        "seconds": round(time.time() - started, 2),
    }


def mean_std(rows: list[dict], key: str) -> dict[str, float | None]:
    values = np.asarray([row[key] for row in rows if row[key] is not None], dtype=float)
    if values.size == 0:
        return {"mean": None, "std": None}
    return {
        "mean": round(float(values.mean()), 6),
        "std": round(float(values.std()), 6),
    }


def summarize(rows: list[dict]) -> dict:
    out = {
        "pass_rate": round(float(np.mean([row["pass"] for row in rows])), 6),
        "monotonic_rate": round(float(np.mean([
            row["gradual_pressure_monotonic"] for row in rows])), 6),
        "local_pollution_trigger_rate": round(float(np.mean([
            row["local_pollution"]["growth_triggered"] for row in rows])), 6),
        "burst_pollution_trigger_rate": round(float(np.mean([
            row["burst_pollution"]["growth_triggered"] for row in rows])), 6),
        "code_commit_rate": round(float(np.mean([row["code_committed"] for row in rows])), 6),
        "noise_commit_rate": round(float(np.mean([row["noise_committed"] for row in rows])), 6),
        "full_growth_params": rows[0]["full_growth_params"],
        "added_params": rows[0]["added_params"],
        "parameter_ratio": rows[0]["parameter_ratio"],
    }
    for key in (
        "growth_onset_code_fraction",
        "code_reducibility",
        "noise_reducibility",
        "base_route_accuracy",
        "code_route_accuracy",
        "mixed_min_route_accuracy",
        "base_logit_max_delta",
    ):
        out[key] = mean_std(rows, key)
    for stage in ("code_before", "code_consolidated"):
        for metric in ("bpc", "residual", "accuracy"):
            values = np.asarray([row[stage][metric] for row in rows], dtype=float)
            out[f"{stage}_{metric}"] = {
                "mean": round(float(values.mean()), 6),
                "std": round(float(values.std()), 6),
            }
    fractions = [row["shifted_fraction"] for row in rows[0]["gradual_drift"]]
    out["gradual_curve"] = [{
        "code_fraction": fraction,
        "high_energy_fraction": mean_std(
            [{"value": row["gradual_drift"][index]["high_energy_fraction"]}
             for row in rows], "value"),
    } for index, fraction in enumerate(fractions)]
    return out


def write_reports(result: dict) -> None:
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    summary = result["summary"]
    with open(REPORT_MD, "w", encoding="utf-8") as file:
        file.write(
            "# 真实领域漂移：自然语言→Python 代码\n\n"
            "稳定域为 OPUS 英文，漂移域为 CodeParrot Python；训练与 held-out 代码来自"
            "不同分片。模型不读取域标签。\n\n"
            "| 裁决 | 三种子结果 |\n|---|---:|\n"
            f"| 渐变漂移压力单调率 | {summary['monotonic_rate']:.1%} |\n"
            f"| 首次触发生长的代码比例 | {summary['growth_onset_code_fraction']['mean']:.1%} |\n"
            f"| 10% 局部污染触发率 | {summary['local_pollution_trigger_rate']:.1%} |\n"
            f"| 50% 污染 burst 高能触发率 | {summary['burst_pollution_trigger_rate']:.1%} |\n"
            f"| 真实代码 residual-F 可约率 | {summary['code_reducibility']['mean']:.1%} |\n"
            f"| 等字符边际噪声可约率 | {summary['noise_reducibility']['mean']:.1%} |\n"
            f"| 代码 / 噪声固化率 | {summary['code_commit_rate']:.1%} / {summary['noise_commit_rate']:.1%} |\n"
            f"| rank-{result['config']['rank']} 新增参数 / 完整生长 | "
            f"{summary['added_params']:,} / {summary['full_growth_params']:,} "
            f"({summary['parameter_ratio']:.1%}) |\n"
            f"| 正常流→基础盆地 | {summary['base_route_accuracy']['mean']:.1%} |\n"
            f"| 代码流→代码盆地 | {summary['code_route_accuracy']['mean']:.1%} |\n"
            f"| 所有混合比例最低路由准确率 | {summary['mixed_min_route_accuracy']['mean']:.1%} |\n"
            f"| held-out 代码 BPC | {summary['code_before_bpc']['mean']:.3f}→"
            f"{summary['code_consolidated_bpc']['mean']:.3f} |\n"
            f"| 基础 logits 最大变化 | {summary['base_logit_max_delta']['mean']:.1e} |\n\n"
            f"## 裁决\n\n{result['verdict']}\n\n"
            "## 边界\n\n"
            "这是自然语言到 Python 的真实突变，但仍是字符级小模型；代码包含自然语言"
            " docstring，且新增符号此前未训练。渐变曲线由离线混合窗口构造，尚不是长期"
            "在线流中的迟滞、自适应阈值或概念回返实验。seed 101 容量先导中，rank-16 "
            "代码可约率只有 26.2%，被固定 30% 闸门拒绝；rank-32 首次越线，正式选用"
            " rank-48，但其新增参数已是完整生长的 59.5%，说明真实跨域容量效率仍不足。\n"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = summary["gradual_curve"]
    x = [row["code_fraction"] for row in curve]
    y = [row["high_energy_fraction"]["mean"] for row in curve]
    yerr = [row["high_energy_fraction"]["std"] for row in curve]
    labels = ["真实代码", "边际匹配噪声"]
    reductions = [
        summary["code_reducibility"]["mean"],
        summary["noise_reducibility"]["mean"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].errorbar(x, y, yerr=yerr, marker="o", capsize=3, color="#1565c0")
    axes[0].axhline(
        result["config"]["min_growth_fraction"], color="#c62828", ls="--",
        label="生长压力阈值")
    axes[0].set_xlabel("窗口中的代码比例"); axes[0].set_ylabel("高 residual-F 样本比例")
    axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=8)
    axes[0].set_title("渐变漂移的自由能压力")
    axes[1].bar(labels, reductions, color=["#2e7d32", "#78909c"])
    axes[1].axhline(
        result["config"]["min_reducibility"], color="#c62828", ls="--",
        label="固化可约率阈值")
    axes[1].set_ylim(0, max(0.55, max(reductions) * 1.2)); axes[1].legend(fontsize=8)
    axes[1].set_title("结构可约，不可约噪声拒绝")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("真实领域漂移的自由能裁决")
    fig.tight_layout(); fig.savefig(FIG_PATH, dpi=180); plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    args.device = args.device.strip() or get_device()
    data = prepare_data(args)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows = [run_seed(seed, data, args) for seed in seeds]
    summary = summarize(rows)
    passed = bool(
        summary["pass_rate"] == 1.0
        and summary["monotonic_rate"] == 1.0
        and summary["local_pollution_trigger_rate"] == 0.0
        and summary["code_commit_rate"] == 1.0
        and summary["noise_commit_rate"] == 0.0
        and summary["base_logit_max_delta"]["mean"] <= 1e-8
    )
    verdict = (
        "✅ 真实领域漂移裁决成立：自由能压力随代码比例上升；局部污染不触发全局扩容，"
        "高能噪声虽进入 probe 但因 held-out 不可约而拒绝；真实代码形成可泛化低能盆地，"
        "最低能路由保持旧域且基础状态零变化。"
        if passed else
        "🟡 真实领域漂移尚未通过全部固定判据；保留失败曲线并修正模型容量或裁决协议。"
    )
    result = {
        "task": "real OPUS English to held-out Python domain drift",
        "config": vars(args),
        "data": {
            "vocab_size": data["vocab_size"],
            "vocab": data["vocab"],
            "sizes": data["sizes"],
            "code_train_source": "CodeParrot shard 1 corpus",
            "code_valid_source": "CodeParrot shard 2 held-out",
        },
        "capacity_pilot_seed_101": {
            "r16": {"code_reducibility": 0.26249, "noise_reducibility": 0.236082,
                    "decision": "reject both"},
            "r32": {"code_reducibility": 0.302715, "noise_reducibility": 0.270675,
                    "decision": "code crosses threshold"},
            "r40": {"code_reducibility": 0.30958, "noise_reducibility": 0.275211,
                    "decision": "code crosses with wider margin"},
            "r48": {"code_reducibility": 0.31389, "noise_reducibility": 0.276272,
                    "decision": "selected before held-out seeds 104-106"},
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
    parser = argparse.ArgumentParser(description="真实自然语言→Python 代码自由能漂移实验。")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--code-train", default=CODE_TRAIN)
    parser.add_argument("--code-valid", default=CODE_VALID)
    parser.add_argument("--device", default="")
    parser.add_argument("--seeds", default="104,105,106")
    parser.add_argument("--sentences", type=int, default=12000)
    parser.add_argument("--code-train-chars", type=int, default=4_000_000)
    parser.add_argument("--code-valid-chars", type=int, default=2_000_000)
    parser.add_argument("--base-steps", type=int, default=350)
    parser.add_argument("--probe-steps", type=int, default=240)
    parser.add_argument("--readout-steps", type=int, default=240)
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
    parser.add_argument("--complexity-quantile", type=float, default=0.90,
                        help="MDL 代价覆盖 90% 旧稳定流的虚假能量优势，与旧域路由验收线一致")
    parser.add_argument("--complexity-margin", type=float, default=1e-4)
    parser.add_argument("--local-pollution-fraction", type=float, default=0.10)
    parser.add_argument("--burst-pollution-fraction", type=float, default=0.50)
    parser.add_argument("--drift-fractions", default="0,0.1,0.25,0.5,0.75,1")
    parser.add_argument("--monotonic-tolerance", type=float, default=0.02)
    parser.add_argument("--min-route-accuracy", type=float, default=0.90)
    parser.add_argument("--min-mixed-route-accuracy", type=float, default=0.90)
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
