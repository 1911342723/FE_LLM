# -*- coding: utf-8 -*-
"""用 held-out 自由能可约性区分结构变化与不可约噪声。

基础模型学习真实 OPUS 英文字符流。候选结构是独立 held-out 句子的反向字符流；噪声对照
逐 chunk 随机打乱同一批字符，因此词表、长度和逐样本字符边际完全相同，唯一差别是顺序结构。

高 residual-F 只获得临时通路。临时通路短训后，只有 held-out residual-F 降幅超过阈值才
固化；随机噪声若不能形成可泛化的低能稳定态，就丢弃临时容量。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import string
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fe_llm.config import get_device
from fe_llm.energy_lm.free_energy_growth import FreeEnergyGrowthSystem
from fe_llm.energy_lm.models.free_energy_lm import FreeEnergyLM

DATA_PATH = os.path.join("data", "translation", "opus100_train.jsonl")
REPORT_JSON = os.path.join("docs", "reports", "free_energy_reducibility_eval.json")
REPORT_MD = os.path.join("docs", "reports", "free_energy_reducibility_eval.md")
FIG_PATH = os.path.join("docs", "reports", "figs", "free_energy_reducibility_eval.png")
ALLOWED = set(string.ascii_lowercase + " .,!?'-")


def normalize_english(text: str) -> str:
    text = text.lower()
    text = "".join(ch if ch in ALLOWED else " " for ch in text)
    return re.sub(r" +", " ", text).strip()


def load_sentences(path: str, limit: int) -> list[str]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                text = normalize_english(json.loads(line)["en"])
            except (KeyError, json.JSONDecodeError):
                continue
            if len(text) >= 12:
                rows.append(text)
            if len(rows) >= limit:
                break
    if len(rows) < 100:
        raise RuntimeError(f"可用英文句子过少：{len(rows)}")
    return rows


def build_stream(sentences: list[str], reverse: bool = False) -> str:
    return "\n".join(text[::-1] if reverse else text for text in sentences) + "\n"


def make_vocab(*streams: str) -> tuple[dict[str, int], list[str]]:
    chars = sorted(set().union(*(set(stream) for stream in streams)))
    return {ch: i for i, ch in enumerate(chars)}, chars


def encode_stream(text: str, vocab: dict[str, int], device: str) -> torch.Tensor:
    return torch.tensor([vocab[ch] for ch in text], dtype=torch.long, device=device)


def sample_chunks(
    stream: torch.Tensor,
    batch: int,
    length: int,
    generator: torch.Generator,
) -> torch.Tensor:
    starts = torch.randint(
        0, stream.numel() - length,
        (batch,), device=stream.device, generator=generator)
    offsets = torch.arange(length, device=stream.device)
    return stream[starts[:, None] + offsets[None, :]]


def shuffle_chunks(chunks: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    order = torch.rand(chunks.shape, device=chunks.device, generator=generator).argsort(dim=1)
    return chunks.gather(1, order)


def lm_loss(logits: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), seq[:, 1:].reshape(-1))


@torch.no_grad()
def path_metrics(
    core: FreeEnergyLM,
    transition: nn.Module,
    seq: torch.Tensor,
    start: int = 2,
    head: nn.Module | None = None,
) -> dict:
    logits, trace = core(seq, transition_override=transition, head_override=head,
                         return_trace=True)
    ce = lm_loss(logits, seq)
    pred = logits[:, :-1].argmax(dim=-1)
    acc = (pred == seq[:, 1:]).float().mean()
    per_sample = trace["residual_free_energy_per_dim"][:, start:].mean(dim=1)
    return {
        "ce": float(ce.cpu()),
        "bpc": float(ce.cpu()) / math.log(2),
        "accuracy": float(acc.cpu()),
        "residual": float(per_sample.mean().cpu()),
        "residual_per_sample": per_sample,
    }


def train_base(
    seed: int,
    train_stream: torch.Tensor,
    vocab_size: int,
    args: argparse.Namespace,
) -> FreeEnergyLM:
    torch.manual_seed(seed)
    generator = torch.Generator(device=args.device).manual_seed(seed + 101)
    core = FreeEnergyLM(
        vocab_size=vocab_size,
        max_len=args.length,
        dim=args.dim,
        relaxation_steps=args.relax_steps,
        tolerance=args.tolerance,
        transition_mult=args.transition_mult,
    ).to(args.device)
    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    core.train()
    for step in range(1, args.base_steps + 1):
        seq = sample_chunks(train_stream, args.batch, args.length, generator)
        logits = core(seq)
        ce = lm_loss(logits, seq)
        assert core.last_position_free_energy is not None
        residual = core.last_position_free_energy[:, 1:].mean()
        loss = ce + args.free_energy_weight * residual
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0); opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.base_steps:
            print(f"[reduce] seed={seed} base step={step:4d}/{args.base_steps} "
                  f"ce={float(ce.detach()):.3f} F={float(residual.detach()):.3f}", flush=True)
    return core.eval()


def train_probe(
    system: FreeEnergyGrowthSystem,
    provisional: nn.Module,
    stream: torch.Tensor,
    *,
    noise: bool,
    seed: int,
    args: argparse.Namespace,
) -> list[dict]:
    params = system.train_only_provisional(provisional)
    opt = torch.optim.AdamW(params, lr=args.probe_lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device=args.device).manual_seed(seed + (701 if noise else 601))
    history = []
    for step in range(1, args.probe_steps + 1):
        seq = sample_chunks(stream, args.batch, args.length, generator)
        if noise:
            seq = shuffle_chunks(seq, generator)
        logits = system.core(seq, transition_override=provisional)
        ce = lm_loss(logits, seq)
        assert system.core.last_position_free_energy is not None
        residual = system.core.last_position_free_energy[:, 1:].mean()
        loss = residual + args.probe_ce_weight * ce
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.probe_steps:
            row = {"step": step, "ce": round(float(ce.detach()), 6),
                   "residual": round(float(residual.detach()), 6)}
            history.append(row)
            tag = "noise" if noise else "structure"
            print(f"[reduce] seed={seed} {tag:9s} step={step:4d}/{args.probe_steps} "
                  f"ce={row['ce']:.3f} F={row['residual']:.3f}", flush=True)
    return history


def train_readout(
    system: FreeEnergyGrowthSystem,
    transition: nn.Module,
    head: nn.Module,
    stream: torch.Tensor,
    *,
    seed: int,
    args: argparse.Namespace,
) -> list[dict]:
    """冻结已通过可约性检验的稳定态，只学习新结构的 next-char 读出。"""
    params = system.train_only_provisional_head(head)
    opt = torch.optim.AdamW(params, lr=args.readout_lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device=args.device).manual_seed(seed + 801)
    history = []
    for step in range(1, args.readout_steps + 1):
        seq = sample_chunks(stream, args.batch, args.length, generator)
        logits = system.core(seq, transition_override=transition, head_override=head)
        ce = lm_loss(logits, seq)
        opt.zero_grad(set_to_none=True); ce.backward(); opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.readout_steps:
            row = {"step": step, "ce": round(float(ce.detach()), 6)}
            history.append(row)
            print(f"[reduce] seed={seed} readout   step={step:4d}/{args.readout_steps} "
                  f"ce={row['ce']:.3f}", flush=True)
    return history


def run_seed(seed: int, data: dict, args: argparse.Namespace) -> dict:
    t0 = time.time()
    core = train_base(seed, data["base_train"], data["vocab_size"], args)
    system = FreeEnergyGrowthSystem(core).to(args.device).eval()
    gen = torch.Generator(device=args.device).manual_seed(seed + 901)

    base_valid = sample_chunks(data["base_valid"], args.eval_batch, args.length, gen)
    structure_valid = sample_chunks(data["structure_valid"], args.eval_batch, args.length, gen)
    noise_valid = shuffle_chunks(structure_valid.clone(), gen)
    structure_stream = sample_chunks(data["structure_train"], args.stream_batch, args.length, gen)
    noise_stream = shuffle_chunks(structure_stream.clone(), gen)

    base_logits_before = core(base_valid).detach().clone()
    base_metrics = path_metrics(core, core.transition, base_valid)
    threshold = system.calibrate_threshold(
        base_valid, pathway=0, start=2, quantile=args.threshold_quantile)
    structure_grow, structure_pressure, structure_energy = system.should_grow(
        structure_stream, start=2, min_fraction=args.min_growth_fraction)
    noise_grow, noise_pressure, noise_energy = system.should_grow(
        noise_stream, start=2, min_fraction=args.min_growth_fraction)

    structure_before = path_metrics(core, core.transition, structure_valid)
    noise_before = path_metrics(core, core.transition, noise_valid)

    structure_probe = system.create_provisional_pathway(noise_std=args.probe_noise)
    structure_history = train_probe(
        system, structure_probe, data["structure_train"], noise=False, seed=seed, args=args)
    structure_after = path_metrics(core, structure_probe, structure_valid)
    structure_reducibility = (
        structure_before["residual"] - structure_after["residual"]
    ) / max(1e-8, structure_before["residual"])

    noise_probe = system.create_provisional_pathway(noise_std=args.probe_noise)
    noise_history = train_probe(
        system, noise_probe, data["structure_train"], noise=True, seed=seed, args=args)
    noise_after = path_metrics(core, noise_probe, noise_valid)
    noise_reducibility = (
        noise_before["residual"] - noise_after["residual"]
    ) / max(1e-8, noise_before["residual"])

    structure_commit = bool(structure_grow and structure_reducibility >= args.min_reducibility)
    noise_commit = bool(noise_grow and noise_reducibility >= args.min_reducibility)
    complexity_cost = None
    readout_history: list[dict] = []
    structure_head = None
    structure_consolidated = structure_after
    if structure_commit:
        structure_head = system.create_provisional_head()
        readout_history = train_readout(
            system, structure_probe, structure_head, data["structure_train"],
            seed=seed, args=args)
        structure_consolidated = path_metrics(
            core, structure_probe, structure_valid, head=structure_head)
        complexity_cost = system.calibrate_complexity_cost(
            base_valid,
            structure_probe,
            start=2,
            quantile=args.complexity_quantile,
            margin=args.complexity_margin,
        )
        committed_index = system.commit_pathway(
            structure_probe, complexity_cost=complexity_cost, head=structure_head)
    else:
        committed_index = None
    # noise_probe 未通过时不注册，临时参数随函数退出释放。

    system.eval()
    base_logits_after = system.forward_pathway(base_valid, 0)
    base_logit_delta = float(
        (base_logits_after - base_logits_before).abs().max().detach().cpu())
    route_base = route_structure = None
    if committed_index is not None:
        base_choices, _ = system.route(base_valid, start=2)
        structure_choices, _ = system.route(structure_valid, start=2)
        route_base = float((base_choices == 0).float().mean().cpu())
        route_structure = float((structure_choices == committed_index).float().mean().cpu())

    def compact(metrics: dict) -> dict:
        return {k: round(float(v), 6) for k, v in metrics.items()
                if k != "residual_per_sample"}

    return {
        "seed": seed,
        "threshold": round(threshold, 6),
        "structure_pressure_fraction": round(structure_pressure, 6),
        "noise_pressure_fraction": round(noise_pressure, 6),
        "structure_stream_energy": round(structure_energy, 6),
        "noise_stream_energy": round(noise_energy, 6),
        "structure_high_energy": structure_grow,
        "noise_high_energy": noise_grow,
        "structure_before": compact(structure_before),
        "structure_after": compact(structure_after),
        "structure_consolidated": compact(structure_consolidated),
        "noise_before": compact(noise_before),
        "noise_after": compact(noise_after),
        "structure_reducibility": round(structure_reducibility, 6),
        "noise_reducibility": round(noise_reducibility, 6),
        "structure_committed": structure_commit,
        "noise_committed": noise_commit,
        "structure_complexity_cost": (None if complexity_cost is None
                                      else round(complexity_cost, 6)),
        "pathway_count": system.pathway_count,
        "base_metrics": compact(base_metrics),
        "base_logit_max_delta": round(base_logit_delta, 9),
        "base_route_accuracy": None if route_base is None else round(route_base, 6),
        "structure_route_accuracy": None if route_structure is None else round(route_structure, 6),
        "structure_history": structure_history,
        "readout_history": readout_history,
        "noise_history": noise_history,
        "seconds": round(time.time() - t0, 2),
    }


def summarize(rows: list[dict]) -> dict:
    metrics = [
        "structure_pressure_fraction", "noise_pressure_fraction",
        "structure_reducibility", "noise_reducibility", "base_logit_max_delta",
        "base_route_accuracy", "structure_route_accuracy", "structure_complexity_cost",
    ]
    out = {
        "structure_commit_rate": round(float(np.mean([r["structure_committed"] for r in rows])), 6),
        "noise_commit_rate": round(float(np.mean([r["noise_committed"] for r in rows])), 6),
        "mean_pathway_count": round(float(np.mean([r["pathway_count"] for r in rows])), 6),
    }
    for metric in metrics:
        values = np.asarray([r[metric] for r in rows if r[metric] is not None], dtype=float)
        out[metric] = ({"mean": None, "std": None} if values.size == 0 else {
            "mean": round(float(values.mean()), 6),
            "std": round(float(values.std()), 6),
        })
    for kind in ("structure", "noise"):
        for stage in ("before", "after"):
            for metric in ("bpc", "residual", "accuracy"):
                values = np.asarray([r[f"{kind}_{stage}"][metric] for r in rows], dtype=float)
                out[f"{kind}_{stage}_{metric}"] = {
                    "mean": round(float(values.mean()), 6),
                    "std": round(float(values.std()), 6),
                }
    for metric in ("bpc", "residual", "accuracy"):
        values = np.asarray([r["structure_consolidated"][metric] for r in rows], dtype=float)
        out[f"structure_consolidated_{metric}"] = {
            "mean": round(float(values.mean()), 6),
            "std": round(float(values.std()), 6),
        }
    return out


def prepare_data(args: argparse.Namespace) -> dict:
    sentences = load_sentences(args.data, args.sentences)
    n_base_train = int(len(sentences) * 0.60)
    n_base_valid = int(len(sentences) * 0.15)
    base_train_sent = sentences[:n_base_train]
    base_valid_sent = sentences[n_base_train:n_base_train + n_base_valid]
    candidate = sentences[n_base_train + n_base_valid:]
    split = len(candidate) // 2
    structure_train_sent, structure_valid_sent = candidate[:split], candidate[split:]

    base_train_text = build_stream(base_train_sent)
    base_valid_text = build_stream(base_valid_sent)
    structure_train_text = build_stream(structure_train_sent, reverse=True)
    structure_valid_text = build_stream(structure_valid_sent, reverse=True)
    vocab, id_to_char = make_vocab(
        base_train_text, base_valid_text, structure_train_text, structure_valid_text)
    return {
        "base_train": encode_stream(base_train_text, vocab, args.device),
        "base_valid": encode_stream(base_valid_text, vocab, args.device),
        "structure_train": encode_stream(structure_train_text, vocab, args.device),
        "structure_valid": encode_stream(structure_valid_text, vocab, args.device),
        "vocab_size": len(vocab),
        "vocab": "".join(id_to_char),
        "sentence_counts": {
            "base_train": len(base_train_sent), "base_valid": len(base_valid_sent),
            "structure_train": len(structure_train_sent),
            "structure_valid": len(structure_valid_sent),
        },
    }


def write_reports(result: dict) -> None:
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    s = result["summary"]
    route_text = ("n/a" if s["structure_route_accuracy"]["mean"] is None
                  else f"{s['structure_route_accuracy']['mean']:.1%}")
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(
            "# Held-out 自由能可约性：结构生长，随机噪声拒绝\n\n"
            "基础为真实 OPUS 英文字符流；结构变化为独立句子的反向流；噪声逐 chunk 保持完全相同字符边际。\n\n"
            "| 指标 | 结构变化 | 等边际随机噪声 |\n|---|---:|---:|\n"
            f"| 高能压力样本 | {s['structure_pressure_fraction']['mean']:.1%} | "
            f"{s['noise_pressure_fraction']['mean']:.1%} |\n"
            f"| held-out residual-F 可约率 | {s['structure_reducibility']['mean']:.1%} | "
            f"{s['noise_reducibility']['mean']:.1%} |\n"
            f"| 临时通路固化率 | {s['structure_commit_rate']:.1%} | {s['noise_commit_rate']:.1%} |\n"
            f"| probe 后字符 bpc | {s['structure_after_bpc']['mean']:.3f} | "
            f"{s['noise_after_bpc']['mean']:.3f} |\n\n"
            f"结构固化后专属读出 bpc：`{s['structure_consolidated_bpc']['mean']:.3f}`，"
            f"字符准确率：`{s['structure_consolidated_accuracy']['mean']:.1%}`。\n\n"
            f"旧通路 logits 最大变化：`{s['base_logit_max_delta']['mean']:.2e}`。"
            f"结构流最低能路由到新通路：`{route_text}`。\n\n"
            f"## 裁决\n\n{result['verdict']}\n\n"
            "## 边界\n\n反向英文仍是受控结构变化，不等于真实领域漂移；可约率阈值需要更多类型校准。"
            "本实验只证明‘高能→临时探测→held-out 可约才固化’能阻止最明显的随机噪声扩容。\n"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    labels = ["结构变化", "等边际噪声"]
    axes[0].bar(labels, [s["structure_reducibility"]["mean"], s["noise_reducibility"]["mean"]],
                color=["#2e7d32", "#ef6c00"])
    axes[0].axhline(result["config"]["min_reducibility"], color="black", ls="--", label="固化阈值")
    axes[0].set_title("held-out residual-F 可约率"); axes[0].legend(fontsize=8)
    axes[1].bar(labels, [s["structure_commit_rate"], s["noise_commit_rate"]],
                color=["#2e7d32", "#ef6c00"])
    axes[1].set_title("临时通路是否固化")
    for ax in axes:
        ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.2)
    fig.suptitle("高自由能不等于盲目生长：先验证可约性")
    fig.tight_layout(); fig.savefig(FIG_PATH, dpi=180); plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    args.device = args.device.strip() or get_device()
    data = prepare_data(args)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = [run_seed(seed, data, args) for seed in seeds]
    summary = summarize(rows)
    passed = (
        summary["structure_commit_rate"] == 1.0
        and summary["noise_commit_rate"] == 0.0
        and summary["structure_reducibility"]["mean"] >= args.min_reducibility
        and summary["noise_reducibility"]["mean"] < args.min_reducibility
        and summary["base_logit_max_delta"]["mean"] <= 1e-8
        and summary["base_route_accuracy"]["mean"] >= 0.90
        and summary["structure_route_accuracy"]["mean"] >= 0.70
        and summary["structure_consolidated_bpc"]["mean"]
            < summary["structure_before_bpc"]["mean"]
    )
    verdict = (
        "✅ 高能后的可约性闸门成立：结构变化在独立 held-out 上的降能超过固化门槛；"
        "等字符边际噪声只有有限的边际可约性、未过门槛，临时容量被丢弃，旧稳定态不变。"
        if passed else
        "🟡 当前可约性闸门尚未稳定分离结构与噪声；保留结果并修正 probe 预算或判据。"
    )
    result = {
        "task": "real English char stream: reducible reversed structure vs marginal-matched noise",
        "config": vars(args),
        "data": {"vocab_size": data["vocab_size"], "vocab": data["vocab"],
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
    ap = argparse.ArgumentParser(description="验证 held-out 自由能可约性可拒绝随机噪声生长。")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--device", default="")
    ap.add_argument("--seeds", default="61,62,63")
    ap.add_argument("--sentences", type=int, default=12000)
    ap.add_argument("--base-steps", type=int, default=350)
    ap.add_argument("--probe-steps", type=int, default=180)
    ap.add_argument("--readout-steps", type=int, default=180)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval-batch", type=int, default=384)
    ap.add_argument("--stream-batch", type=int, default=128)
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--relax-steps", type=int, default=5)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--transition-mult", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--probe-lr", type=float, default=2e-3)
    ap.add_argument("--readout-lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--free-energy-weight", type=float, default=2.0)
    ap.add_argument("--probe-ce-weight", type=float, default=0.0,
                    help="可约性 probe 默认只最小化 residual-F；CE 仅记录，不参与判定")
    ap.add_argument("--threshold-quantile", type=float, default=0.99)
    ap.add_argument("--min-growth-fraction", type=float, default=0.25,
                    help="高能流进入临时 probe 的比例门槛；是否固化由可约率二次裁决")
    ap.add_argument("--min-reducibility", type=float, default=0.30)
    ap.add_argument("--complexity-quantile", type=float, default=0.95,
                    help="新增通路 MDL 代价覆盖旧流虚假优势的分位数")
    ap.add_argument("--complexity-margin", type=float, default=1e-4)
    ap.add_argument("--probe-noise", type=float, default=1e-3)
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
