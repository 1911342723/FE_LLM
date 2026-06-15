# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_growth_eval.py
=======================================
CAPCW 阶段二b：穷则变（结构成长）判定。见 `docs/FE-LLM核心引擎构想.md`。

穷则变的逻辑链：「自由能长期高 → 加 slot」。要让它成立，前提是 **slot 数 = 绑定容量**——
slot 不够时装不下、自由能高、绑定失败；slot 够了就能解。本实验验证这条前提，并确认 grow 钩子
能从自由能检测出"需要更多 slot"。

实验（固定绑定数 K，容量受限 d）：
- 扫 slot 数 M：训练 CAPCW_PC（PCWorkspace，固定 M slot）在 K-绑定任务上，看 accuracy vs M。
  预期：**M<K 明显低、M≥K 跳升**（绑定容量随 M 增长）。
- grow 钩子检测：对 K-绑定输入，PCWorkspace.grow_if_unexplained 在低阈值下应建议 m ≥ K
  （自由能驱动的"该长到多少 slot"判断）。

判定：accuracy(M≥K) − accuracy(M<K) 明显（≥ +0.15）→ slot 数即绑定容量，穷则变（长到 M≥K）
是对的机制；grow 钩子建议 m 随 K 增长。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_growth_eval --run
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
from fe_llm.world_model.capcw import PCWorkspace
from fe_llm.world_model.capcw_binding_eval import CAPCWPCModel, PairEncoder, gen_binding, train_eval  # noqa: F401

REPORT_JSON = os.path.join("docs", "reports", "capcw_growth_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_growth_eval.md")


def grow_suggestion(n_keys, n_vals, k, d, m_init, max_slots, device, seed):
    """对 K-绑定输入，用一个（未训练但生成模型可弛豫）PCWorkspace 看 grow 钩子建议的 slot 数。

    说明：grow 钩子靠"弛豫后自由能是否仍高"判断；这里用 pair 向量作为输入，阈值取相对值。
    """
    torch.manual_seed(seed)
    enc = PairEncoder(n_keys, n_vals, d).to(device)
    ws = PCWorkspace(dim=d, n_slots=m_init, iters=5).to(device)
    pk, pv, _, _ = gen_binding(n_keys, n_vals, k, 256, seed)
    pk = torch.tensor(pk, device=device)
    pv = torch.tensor(pv, device=device)
    with torch.no_grad():
        pairs = enc(pk, pv)
        # 以 m_init 时的自由能为基准，阈值设为它的一半：要求继续生长直到自由能明显下降。
        base_f = float(ws(pairs, n_slots=m_init).free_energy)
        threshold = base_f * 0.5
        m = ws.grow_if_unexplained(pairs, threshold=threshold, max_slots=max_slots)
    return m, round(base_f, 4), round(threshold, 4)


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    k = args.k
    ms = [int(x) for x in args.m_list.split(",")]
    print(f"[grow] device={device} K={k} M_list={ms} d={args.d} n_keys={args.n_keys} n_vals={args.n_vals} seeds={args.seeds}", flush=True)

    acc_by_m = {}
    for m in ms:
        accs = []
        for si in range(args.seeds):
            seed = args.seed + si
            train = gen_binding(args.n_keys, args.n_vals, k, args.n_train, seed)
            test = gen_binding(args.n_keys, args.n_vals, k, args.n_test, seed + 5000)
            model = CAPCWPCModel(args.n_keys, args.n_vals, args.d, n_slots=m, iters=args.iters)
            accs.append(train_eval(model, train, test, device=device, epochs=args.epochs, lr=args.lr, batch=args.batch, seed=seed))
        acc_by_m[m] = {"mean": round(float(np.mean(accs)), 4), "std": round(float(np.std(accs)), 4)}
        print(f"[grow] M={m} acc={acc_by_m[m]['mean']:.3f}±{acc_by_m[m]['std']:.3f} (K={k}, random={1.0/args.n_vals:.3f})", flush=True)

    under = [m for m in ms if m < k]
    over = [m for m in ms if m >= k]
    acc_under = float(np.mean([acc_by_m[m]["mean"] for m in under])) if under else 0.0
    acc_over = float(np.mean([acc_by_m[m]["mean"] for m in over])) if over else 0.0
    transition = round(acc_over - acc_under, 4)

    # grow 钩子：建议的 slot 数应随 K 增长、且对 K-绑定 ≥ K（在低阈值下）。
    grow_m, base_f, thr = grow_suggestion(args.n_keys, args.n_vals, k, args.d, m_init=2, max_slots=k + 3, device=device, seed=args.seed)

    if transition >= 0.15:
        verdict = ("PASS: slot 数即绑定容量（M<K 低、M≥K 跳升），穷则变（自由能高→长到 M≥K）机制成立")
    elif transition >= 0.05:
        verdict = "PARTIAL: 容量随 M 增长但转折偏弱"
    else:
        verdict = "FAIL: slot 数与绑定容量关系不明显，穷则变前提不成立"

    result = {
        "task": "binding capacity vs slot count (CAPCW_PC), fixed K",
        "config": {"k": k, "m_list": ms, "d": args.d, "n_keys": args.n_keys, "n_vals": args.n_vals,
                   "epochs": args.epochs, "seeds": args.seeds, "random_baseline": round(1.0 / args.n_vals, 4)},
        "acc_by_m": acc_by_m,
        "acc_under_k_mean": round(acc_under, 4),
        "acc_over_k_mean": round(acc_over, 4),
        "transition_over_minus_under": transition,
        "grow_hook": {"suggested_m_for_K": grow_m, "k": k, "base_free_energy_at_m2": base_f, "threshold": thr},
        "verdict": verdict,
        "note": "固定 K 扫 slot 数 M；M<K 装不下→自由能高→绑定失败，M≥K 能解。这条前提成立，"
                "穷则变（自由能高就加 slot 长到 M≥K）才是对的结构成长机制。grow 钩子从自由能检测所需 slot 数。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# CAPCW 阶段二b · 穷则变（结构成长）判定",
        "",
        f"- 判定：**{verdict}**",
        f"- 任务：固定绑定数 K={k}，扫 slot 数 M，看 CAPCW_PC accuracy vs M（d={args.d} 容量受限；随机 {1.0/args.n_vals:.3f}）。",
        "",
        "| slot 数 M | accuracy | 备注 |",
        "|---:|---:|---|",
    ]
    for m in ms:
        tag = "M<K 容量不足" if m < k else "M≥K 容量充足"
        lines.append(f"| {m} | {acc_by_m[m]['mean']:.3f}±{acc_by_m[m]['std']:.3f} | {tag} |")
    lines += [
        "",
        f"- M<K 平均 {acc_under:.3f} → M≥K 平均 {acc_over:.3f}（转折 **{transition:+.4f}**）",
        f"- grow 钩子：对 K={k} 绑定，m_init=2、阈值={thr}（=m2 自由能 {base_f} 的一半）→ 建议 slot 数 = **{grow_m}**（应 ≥ K 或随 K 增长）",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[grow] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    print(f"[grow] grow 钩子建议 m={grow_m}（K={k}）；报告 {args.report_json}", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW growth (穷则变) adjudication: binding capacity vs slot count.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--m-list", default="2,3,4,5,7")
    ap.add_argument("--n-keys", type=int, default=10)
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
        print("[grow] dry-run：未训练。固定 K 扫 slot 数 M，验证绑定容量随 M 增长 + grow 钩子检测所需 slot。")
        print("[grow] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
