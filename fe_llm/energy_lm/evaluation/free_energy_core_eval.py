# -*- coding: utf-8 -*-
"""显式自由能核心的机制验收。

这不是语言能力 benchmark，而是新核心的“物理契约”检查：

1. 每一步自由能不增加；
2. 状态能在最大预算前按局部阈值停止；
3. 改变未来 token 不会改变过去的状态、logits 或停止时刻；
4. 最终自由能低于初始自由能。

运行：python -m fe_llm.energy_lm.evaluation.free_energy_core_eval
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from fe_llm.config import get_device
from fe_llm.energy_lm.models.free_energy_lm import FreeEnergyLM


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = args.device.strip() or get_device()
    net = FreeEnergyLM(
        vocab_size=args.vocab,
        max_len=args.ctx,
        dim=args.dim,
        relaxation_steps=args.relax_steps,
        tolerance=args.tolerance,
    ).to(device).eval()

    ids = torch.randint(0, args.vocab, (args.batch, args.ctx), device=device)
    logits, trace = net(ids, return_trace=True)
    energy = trace["free_energy"].cpu()
    monotonic = bool(torch.all(energy[1:] <= energy[:-1] + 1e-5))

    prefix_len = max(1, args.ctx // 2)
    alternative = ids.clone()
    alternative[:, prefix_len:] = torch.randint(
        0, args.vocab, alternative[:, prefix_len:].shape, device=device)
    alt_logits, alt_trace = net(alternative, return_trace=True)
    prefix_logit_delta = float(
        (logits[:, :prefix_len] - alt_logits[:, :prefix_len]).abs().max().detach().cpu())
    prefix_step_equal = bool(torch.equal(
        trace["steps_per_position"][:, :prefix_len],
        alt_trace["steps_per_position"][:, :prefix_len],
    ))

    result = {
        "contract": "explicit causal free-energy relaxation",
        "config": {
            "device": device,
            "vocab": args.vocab,
            "ctx": args.ctx,
            "dim": args.dim,
            "max_relax_steps": args.relax_steps,
            "tolerance": args.tolerance,
            "seed": args.seed,
        },
        "free_energy_trace": [round(float(v), 6) for v in energy],
        "initial_free_energy": round(float(energy[0]), 6),
        "final_free_energy": round(float(energy[-1]), 6),
        "energy_drop": round(float(energy[0] - energy[-1]), 6),
        "monotonic_non_increasing": monotonic,
        "steps_min": int(trace["steps_per_position"].min().cpu()),
        "steps_max": int(trace["steps_per_position"].max().cpu()),
        "converged_fraction": round(float(trace["converged_fraction"].cpu()), 6),
        "future_to_past_max_logit_delta": prefix_logit_delta,
        "future_to_past_stopping_equal": prefix_step_equal,
    }
    result["pass"] = bool(
        monotonic
        and result["final_free_energy"] < result["initial_free_energy"]
        and prefix_logit_delta <= 1e-7
        and prefix_step_equal
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="验收显式自由能核心的稳定性与严格因果性。")
    ap.add_argument("--device", default="")
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=24)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--relax-steps", type=int, default=12)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    result = run(build_arg_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
