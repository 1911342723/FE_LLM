# -*- coding: utf-8 -*-
"""用残余自由能完成“穷则变”的端到端机制裁决。

流程完全由能量状态驱动：先学 lag-2 稳定结构并校准 residual-F 的 99% 分位；流入
lag-3 后，若多数样本在所有现有通路下仍高于阈值，则自动复制最低能通路长出新容量；
冻结共享核心和旧通路，只训练新生成性转移；推理时按最低 residual-F 自动选通路。
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
from fe_llm.energy_lm.models.free_energy_lm import FreeEnergyLM
from fe_llm.energy_lm.evaluation.free_energy_sequence_eval import (
    accuracy,
    make_lag_sequences,
    task_loss,
)

REPORT_JSON = os.path.join("docs", "reports", "free_energy_growth_eval.json")
REPORT_MD = os.path.join("docs", "reports", "free_energy_growth_eval.md")
FIG_PATH = os.path.join("docs", "reports", "figs", "free_energy_growth_eval.png")


@torch.no_grad()
def routed_accuracy(
    system: FreeEnergyGrowthSystem,
    seq: torch.Tensor,
    lag: int,
    start: int,
) -> tuple[float, torch.Tensor]:
    logits, choices = system.routed_logits(seq, start=start)
    pred = logits[:, lag - 1:-1].argmax(dim=-1)
    acc = float((pred == seq[:, lag:]).float().mean().cpu())
    return acc, choices


def train_base(seed: int, args: argparse.Namespace) -> FreeEnergyLM:
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
    net.train()
    for step in range(1, args.base_steps + 1):
        seq = make_lag_sequences(args.batch, args.length, 2, args.vocab, args.device)
        logits = net(seq)
        ce = task_loss(logits, seq, lag=2)
        assert net.last_position_free_energy is not None
        residual = net.last_position_free_energy[:, 2:].mean()
        loss = ce + args.free_energy_weight * residual
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.base_steps:
            print(f"[grow] seed={seed} base step={step:4d}/{args.base_steps} "
                  f"ce={float(ce.detach()):.4f} F={float(residual.detach()):.4f}", flush=True)
    return net.eval()


def train_new_pathway(
    system: FreeEnergyGrowthSystem,
    pathway: int,
    seed: int,
    args: argparse.Namespace,
) -> list[dict]:
    params = system.train_only_pathway(pathway)
    opt = torch.optim.AdamW(params, lr=args.growth_lr, weight_decay=args.weight_decay)
    history = []
    for step in range(1, args.growth_steps + 1):
        seq = make_lag_sequences(args.batch, args.length, 3, args.vocab, args.device)
        logits = system.forward_pathway(seq, pathway)
        ce = task_loss(logits, seq, lag=3)
        assert system.core.last_position_free_energy is not None
        residual = system.core.last_position_free_energy[:, 3:].mean()
        loss = ce + args.free_energy_weight * residual
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.growth_steps:
            row = {
                "step": step,
                "ce": round(float(ce.detach().cpu()), 6),
                "residual_f": round(float(residual.detach().cpu()), 6),
            }
            history.append(row)
            print(f"[grow] seed={seed} new  step={step:4d}/{args.growth_steps} "
                  f"ce={row['ce']:.4f} F={row['residual_f']:.4f}", flush=True)
    return history


def run_seed(seed: int, args: argparse.Namespace) -> dict:
    t0 = time.time()
    core = train_base(seed, args)
    system = FreeEnergyGrowthSystem(core).to(args.device).eval()

    audit_id = make_lag_sequences(args.eval_batch, args.length, 2, args.vocab, args.device)
    audit_ood = make_lag_sequences(args.eval_batch, args.length, 3, args.vocab, args.device)
    calibration = make_lag_sequences(args.calibration_batch, args.length, 2, args.vocab, args.device)
    known_stream = make_lag_sequences(args.stream_batch, args.length, 2, args.vocab, args.device)
    novel_stream = make_lag_sequences(args.stream_batch, args.length, 3, args.vocab, args.device)

    base_logits_before = system.forward_pathway(audit_id, 0).detach().clone()
    base_id_before = accuracy(core, audit_id, lag=2, max_relax_steps=core.relaxation_steps)
    base_ood_before = accuracy(core, audit_ood, lag=3, max_relax_steps=core.relaxation_steps)

    threshold = system.calibrate_threshold(
        calibration, pathway=0, start=3, quantile=args.threshold_quantile)
    known_grow, known_fraction, known_energy = system.should_grow(
        known_stream, start=3, min_fraction=args.min_growth_fraction)
    novel_grow, novel_fraction, novel_energy = system.should_grow(
        novel_stream, start=3, min_fraction=args.min_growth_fraction)

    history: list[dict] = []
    new_pathway = None
    if novel_grow:
        source_scores = system.score_all(novel_stream, start=3).mean(dim=0)
        source = int(source_scores.argmin().cpu())
        new_pathway = system.add_pathway(source=source, noise_std=args.growth_noise)
        history = train_new_pathway(system, new_pathway, seed, args)
    system.eval()

    base_logits_after = system.forward_pathway(audit_id, 0)
    base_logit_delta = float(
        (base_logits_after - base_logits_before).abs().max().detach().cpu())
    base_id_after = accuracy(core, audit_id, lag=2, max_relax_steps=core.relaxation_steps)

    if new_pathway is None:
        return {
            "seed": seed,
            "growth_triggered": False,
            "known_false_trigger": known_grow,
            "known_pressure_fraction": known_fraction,
            "novel_pressure_fraction": novel_fraction,
            "threshold": threshold,
            "base_id_accuracy_before": base_id_before,
            "base_id_accuracy_after": base_id_after,
            "base_retention_delta": base_id_after - base_id_before,
            "base_logit_max_delta": base_logit_delta,
            "base_ood_accuracy_before": base_ood_before,
            "seconds": round(time.time() - t0, 2),
        }

    new_logits = system.forward_pathway(audit_ood, new_pathway)
    new_pred = new_logits[:, 2:-1].argmax(dim=-1)
    new_ood_acc = float((new_pred == audit_ood[:, 3:]).float().mean().cpu())

    routed_id_acc, id_choices = routed_accuracy(system, audit_id, lag=2, start=3)
    routed_ood_acc, ood_choices = routed_accuracy(system, audit_ood, lag=3, start=3)
    id_route_base = float((id_choices == 0).float().mean().cpu())
    ood_route_new = float((ood_choices == new_pathway).float().mean().cpu())
    id_scores = system.score_all(audit_id, start=3).mean(dim=0).cpu()
    ood_scores = system.score_all(audit_ood, start=3).mean(dim=0).cpu()

    return {
        "seed": seed,
        "growth_triggered": True,
        "known_false_trigger": known_grow,
        "threshold": round(threshold, 6),
        "known_pressure_fraction": round(known_fraction, 6),
        "novel_pressure_fraction": round(novel_fraction, 6),
        "known_stream_energy": round(known_energy, 6),
        "novel_stream_energy": round(novel_energy, 6),
        "base_id_accuracy_before": round(base_id_before, 6),
        "base_id_accuracy_after": round(base_id_after, 6),
        "base_retention_delta": round(base_id_after - base_id_before, 6),
        "base_logit_max_delta": round(base_logit_delta, 9),
        "base_ood_accuracy_before": round(base_ood_before, 6),
        "new_ood_accuracy": round(new_ood_acc, 6),
        "routed_id_accuracy": round(routed_id_acc, 6),
        "routed_ood_accuracy": round(routed_ood_acc, 6),
        "id_route_base_accuracy": round(id_route_base, 6),
        "ood_route_new_accuracy": round(ood_route_new, 6),
        "id_pathway_energy": [round(float(x), 6) for x in id_scores],
        "ood_pathway_energy": [round(float(x), 6) for x in ood_scores],
        "core_params": sum(p.numel() for p in core.parameters()),
        "added_params": system.added_parameter_count(),
        "history": history,
        "seconds": round(time.time() - t0, 2),
    }


def summarize(rows: list[dict]) -> dict:
    metrics = [
        "known_pressure_fraction", "novel_pressure_fraction",
        "base_id_accuracy_before", "base_id_accuracy_after", "base_retention_delta",
        "base_logit_max_delta", "base_ood_accuracy_before", "new_ood_accuracy",
        "routed_id_accuracy", "routed_ood_accuracy", "id_route_base_accuracy",
        "ood_route_new_accuracy",
    ]
    summary = {
        "growth_trigger_rate": round(float(np.mean([r["growth_triggered"] for r in rows])), 6),
        "known_false_trigger_rate": round(float(np.mean([r["known_false_trigger"] for r in rows])), 6),
    }
    for metric in metrics:
        values = np.asarray([row[metric] for row in rows if metric in row], dtype=float)
        summary[metric] = ({"mean": None, "std": None} if values.size == 0 else {
            "mean": round(float(values.mean()), 6),
            "std": round(float(values.std()), 6),
        })
    return summary


def write_reports(result: dict) -> None:
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    s = result["summary"]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(
            "# Residual Free Energy 驱动的自动容量生长\n\n"
            "稳定 lag-2 → 流入 lag-3；阈值由旧稳定分布自动校准，旧通路冻结，最低自由能路由。\n\n"
            "| 指标 | 三种子均值 |\n|---|---:|\n"
            f"| 新结构生长触发率 | {s['growth_trigger_rate']:.1%} |\n"
            f"| 已知流误触发率 | {s['known_false_trigger_rate']:.1%} |\n"
            f"| 旧技能保持 | {s['base_id_accuracy_after']['mean']:.1%} |\n"
            f"| 旧通路 logits 最大变化 | {s['base_logit_max_delta']['mean']:.2e} |\n"
            f"| 新技能学习后 acc | {s['new_ood_accuracy']['mean']:.1%} |\n"
            f"| ID 路由到旧通路 | {s['id_route_base_accuracy']['mean']:.1%} |\n"
            f"| OOD 路由到新通路 | {s['ood_route_new_accuracy']['mean']:.1%} |\n"
            f"| 路由后 ID / OOD acc | {s['routed_id_accuracy']['mean']:.1%} / "
            f"{s['routed_ood_accuracy']['mean']:.1%} |\n\n"
            f"## 裁决\n\n{result['verdict']}\n\n"
            "## 边界\n\n这是两种可解释周期结构上的机制证明。新增参数仍随结构数线性增长，"
            "最低自由能路由需要逐通路评估；下一步需研究通路合并、先验惩罚与真实字符流。\n"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["旧技能\n生长前", "旧技能\n生长后", "新技能\n新通路", "路由后\n新技能"]
    values = [
        s["base_id_accuracy_before"]["mean"],
        s["base_id_accuracy_after"]["mean"],
        s["new_ood_accuracy"]["mean"],
        s["routed_ood_accuracy"]["mean"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].bar(labels, values, color=["#1565c0", "#1565c0", "#2e7d32", "#2e7d32"])
    axes[0].set_ylim(0, 1.05); axes[0].set_title("生长前后能力")
    route_values = [s["id_route_base_accuracy"]["mean"], s["ood_route_new_accuracy"]["mean"]]
    axes[1].bar(["ID→旧通路", "OOD→新通路"], route_values, color=["#1565c0", "#2e7d32"])
    axes[1].set_ylim(0, 1.05); axes[1].set_title("最低自由能路由")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Residual Free Energy 驱动“穷则变”")
    fig.tight_layout(); fig.savefig(FIG_PATH, dpi=180); plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    args.device = args.device.strip() or get_device()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = [run_seed(seed, args) for seed in seeds]
    summary = summarize(rows)
    all_grew = summary["growth_trigger_rate"] == 1.0
    passed = bool(
        all_grew
        and summary["known_false_trigger_rate"] == 0.0
        and abs(summary["base_retention_delta"]["mean"]) <= 1e-8
        and summary["base_logit_max_delta"]["mean"] <= 1e-8
        and summary["new_ood_accuracy"]["mean"] >= 0.95
        and summary["id_route_base_accuracy"]["mean"] >= 0.90
        and summary["ood_route_new_accuracy"]["mean"] >= 0.90
    )
    verdict = (
        "✅ ‘穷则变’闭环成立：残余自由能自动识别现有结构解释失败，按需长出冻结隔离的新生成性通路；"
        "最低自由能在推理时恢复正确结构，旧稳定态逐 logit 不变。"
        if passed else
        "🟡 自动生长闭环尚未通过全部判据；保留负结果，先修正阈值、可塑性或自由能路由。"
    )
    result = {
        "task": "residual-F triggered lag-2 to lag-3 structural growth",
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
    ap = argparse.ArgumentParser(description="用 residual free energy 验证‘穷则变’自动容量生长。")
    ap.add_argument("--device", default="")
    ap.add_argument("--seeds", default="51,52,53")
    ap.add_argument("--base-steps", type=int, default=150)
    ap.add_argument("--growth-steps", type=int, default=180)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval-batch", type=int, default=384)
    ap.add_argument("--calibration-batch", type=int, default=1024)
    ap.add_argument("--stream-batch", type=int, default=128)
    ap.add_argument("--length", type=int, default=14)
    ap.add_argument("--vocab", type=int, default=8)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--relax-steps", type=int, default=5)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--transition-mult", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--growth-lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--free-energy-weight", type=float, default=2.0)
    ap.add_argument("--threshold-quantile", type=float, default=0.99)
    ap.add_argument("--min-growth-fraction", type=float, default=0.5)
    ap.add_argument("--growth-noise", type=float, default=1e-3)
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
