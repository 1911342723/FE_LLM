# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_grow_dynamic_eval.py
=============================================
CAPCW Part 2：穷则变的**自动校准 + 动态按需分配**。见 `docs/FE-LLM核心引擎构想.md`。

阶段二b 证了"slot 数=绑定容量"，但 grow 钩子停得太晚（长到上限）。本实验给出**自校准**的生长准则
（相对边际增益：每加一个 slot 若不能把自由能再降 ≥ min_rel_gain 就停），并演示**按需分配**：
- 训练一个 max_slots 的 CAPCW；推理时对不同 K 用生长准则自动选 m；
- 期望：选出的 m 随 K 增长且 ≈ K+1（不再盲目到上限）；用 grow-m 的精度 ≈ 用 max 的精度
  （按需省 slot 而不掉精度）。

判据：grow-m 随 K 单调增长（自适应）且 |acc@grow-m − acc@max| 小（精度保持）→ 穷则变可自校准、按需分配。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_grow_dynamic_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from fe_llm.config import get_device
from fe_llm.world_model.capcw_binding_eval import CAPCWPCModel, gen_binding, train_eval

REPORT_JSON = os.path.join("docs", "reports", "capcw_grow_dynamic_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_grow_dynamic_eval.md")


def free_energy_curve(model, pairs_test, max_slots):
    """对 test pairs，给出 m=2..max 的批自由能。"""
    fe = {}
    with torch.no_grad():
        for m in range(2, max_slots + 1):
            fe[m] = float(model.ws(pairs_test, n_slots=m).free_energy)
    return fe


def pick_grow_m(fe: dict, min_rel_gain: float) -> int:
    """自校准生长：从 m=2 起，若 m→m+1 的自由能相对下降 ≥ min_rel_gain 则继续长，否则停。"""
    ms = sorted(fe)
    m = ms[0]
    for i in range(len(ms) - 1):
        cur, nxt = fe[ms[i]], fe[ms[i + 1]]
        gain = (cur - nxt) / max(cur, 1e-8)
        if gain >= min_rel_gain:
            m = ms[i + 1]
        else:
            break
    return m


def acc_at(model, test, device, n_slots):
    pk, pv, qk, y = (torch.tensor(t, device=device) for t in test)
    model.eval()
    with torch.no_grad():
        pred = model(pk, pv, qk, n_slots=n_slots).argmax(-1)
    return float((pred == y).float().mean())


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    ks = [int(x) for x in args.k_list.split(",")]
    print(f"[grow-dyn] device={device} K_list={ks} max_slots={args.max_slots} d={args.d} min_rel_gain={args.min_rel_gain}", flush=True)
    by_k = {}
    for k in ks:
        grow_ms, acc_grows, acc_maxs = [], [], []
        for si in range(args.seeds):
            seed = args.seed + si
            train = gen_binding(args.n_keys, args.n_vals, k, args.n_train, seed)
            test = gen_binding(args.n_keys, args.n_vals, k, args.n_test, seed + 5000)
            model = CAPCWPCModel(args.n_keys, args.n_vals, args.d, n_slots=args.max_slots, iters=args.iters)
            acc_max = train_eval(model, train, test, device=device, epochs=args.epochs, lr=args.lr, batch=args.batch, seed=seed)
            pk, pv = (torch.tensor(test[i], device=device) for i in range(2))
            pairs = model.enc(pk, pv)
            fe = free_energy_curve(model, pairs, args.max_slots)
            gm = pick_grow_m(fe, args.min_rel_gain)
            acc_grow = acc_at(model, test, device, gm)
            grow_ms.append(gm); acc_grows.append(acc_grow); acc_maxs.append(acc_max)
        by_k[k] = {
            "grow_m_mean": round(float(np.mean(grow_ms)), 2),
            "acc_grow_mean": round(float(np.mean(acc_grows)), 4),
            "acc_max_mean": round(float(np.mean(acc_maxs)), 4),
        }
        print(f"[grow-dyn] K={k} grow_m={by_k[k]['grow_m_mean']} acc@grow={by_k[k]['acc_grow_mean']:.3f} acc@max={by_k[k]['acc_max_mean']:.3f}", flush=True)

    grow_ms_ordered = [by_k[k]["grow_m_mean"] for k in ks]
    monotone = all(grow_ms_ordered[i] <= grow_ms_ordered[i + 1] + 0.5 for i in range(len(ks) - 1)) and grow_ms_ordered[-1] > grow_ms_ordered[0]
    max_acc_drop = max(by_k[k]["acc_max_mean"] - by_k[k]["acc_grow_mean"] for k in ks)
    if monotone and max_acc_drop <= 0.05:
        verdict = "PASS: 生长准则自校准——grow_m 随 K 增长且精度保持(按需分配 slot)"
    elif max_acc_drop <= 0.10:
        verdict = "PARTIAL: 精度基本保持但 grow_m 自适应偏弱"
    else:
        verdict = "FAIL: grow_m 选择掉精度或不随 K 自适应"

    result = {
        "task": "CAPCW dynamic growth: auto-calibrated slot allocation per binding load K",
        "config": {"k_list": ks, "max_slots": args.max_slots, "d": args.d, "min_rel_gain": args.min_rel_gain,
                   "n_keys": args.n_keys, "n_vals": args.n_vals, "epochs": args.epochs, "seeds": args.seeds},
        "by_k": by_k,
        "grow_m_monotone_in_k": bool(monotone),
        "max_acc_drop_grow_vs_max": round(max_acc_drop, 4),
        "verdict": verdict,
        "note": "自校准生长=相对边际增益(加 slot 降自由能 <min_rel_gain 即停)；训练 max_slots、推理按 K 用 grow_m。"
                "grow_m 随 K 增长=按需分配；acc@grow≈acc@max=省 slot 不掉精度。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# CAPCW Part2 · 穷则变自校准 + 按需动态分配",
        "",
        f"- 判定：**{verdict}**",
        f"- 自校准生长准则：相对边际增益 ≥ {args.min_rel_gain} 才继续加 slot；训练 max_slots={args.max_slots}，推理按 K 选 grow_m。",
        "",
        "| K（绑定负载） | 自选 grow_m | acc@grow_m | acc@max |",
        "|---:|---:|---:|---:|",
    ]
    for k in ks:
        b = by_k[k]
        lines.append(f"| {k} | {b['grow_m_mean']} | {b['acc_grow_mean']:.3f} | {b['acc_max_mean']:.3f} |")
    lines += [
        "",
        f"- grow_m 随 K 单调增长：{monotone}；最大精度损失(max−grow)：{max_acc_drop:.4f}",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[grow-dyn] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW dynamic growth auto-calibration & on-demand slot allocation.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--k-list", default="2,4,6")
    ap.add_argument("--max-slots", type=int, default=8)
    ap.add_argument("--min-rel-gain", type=float, default=0.15)
    ap.add_argument("--n-keys", type=int, default=12)
    ap.add_argument("--n-vals", type=int, default=12)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-json", default=REPORT_JSON)
    ap.add_argument("--report-md", default=REPORT_MD)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print("[grow-dyn] dry-run：未训练。自校准生长 + 按需分配：训练 max_slots、推理按 K 选 grow_m。")
        print("[grow-dyn] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
