# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_controller_integration_eval.py
=======================================================
CAPCW 接回 controller：用引擎 surprise 驱动"知道何时不该答 + 内容取回"。
见 `fe_llm/active_inference/capcw_memory.py`、`docs/FE-LLM核心引擎构想.md`。

把已验证的 CAPCW（内容寻址 slot 工作空间 + query 路由 surprise，Part3）做成 controller 兼容的工作
记忆组件 `CAPCWWorkingMemory`，在 controller 的决策框架（ActionType.ANSWER / ASK_CLARIFICATION）里
端到端演示：

对话 = 现场给出 K 个 in-context 绑定（key→value）+ 一个查询键（bound 或 unbound，各半）：
- FE-agent（CAPCW 工作记忆）：用引擎 surprise 裁决——bound(低 surprise)→ANSWER + 取回 value；
  unbound(高 surprise)→ASK_CLARIFICATION（不胡答）。**无动作监督，决策从引擎涌现。**
- baseline（无工作记忆、永远直答）：无 in-context 记忆→bound 只能瞎猜(≈随机)、unbound 也直答(胡编)。

判据（实验 C 同口径）
---------------------
- balacc_ask_answer ≥ 0.80：引擎 surprise 能正确分开"该答(bound)"与"该问(unbound)"；
- value_acc ≥ 0.60：bound 且选择回答时，取回的 value 正确（内容取回）；
- 且 FE 任务成功率显著高于无记忆基线、unbound 胡答率显著低于基线。
满足 → CAPCW 引擎接回 controller 决策成立（知道何时不该答 + 内容取回，从引擎 surprise 涌现）。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_controller_integration_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from fe_llm.active_inference.capcw_memory import CAPCWWorkingMemory
from fe_llm.active_inference.policy import ActionType
from fe_llm.config import get_device

REPORT_JSON = os.path.join("docs", "reports", "capcw_controller_integration_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_controller_integration_eval.md")
CKPT = os.path.join("checkpoints", "world_model", "capcw_wm.pt")


def gen_dialogues(n_keys, n_vals, k, n, seed, p_bound=0.5):
    """每段对话：K 个不同 (key→value) in-context 绑定 + 一个查询键（bound/unbound）。"""
    rng = np.random.default_rng(seed)
    dialogues = []
    for _ in range(n):
        keys = rng.choice(n_keys, size=k, replace=False)
        vals = rng.choice(n_vals, size=k, replace=False)
        bindings = list(zip((int(x) for x in keys), (int(x) for x in vals)))
        if rng.random() < p_bound:
            qi = int(rng.integers(k))
            qk, is_bound, true_val = int(keys[qi]), True, int(vals[qi])
        else:
            not_in = [x for x in range(n_keys) if x not in set(int(z) for z in keys)]
            qk, is_bound, true_val = int(rng.choice(not_in)), False, None
        dialogues.append((bindings, qk, is_bound, true_val))
    return dialogues


def _balanced_acc(pred_answer, is_bound):
    """ASK/ANSWER 决策的 balanced accuracy（类=bound/unbound，预测=是否选 ANSWER）。"""
    pred = np.asarray(pred_answer, dtype=bool)
    truth = np.asarray(is_bound, dtype=bool)
    accs = []
    for cls in (True, False):
        m = truth == cls
        if m.any():
            # bound 类正确=选了 ANSWER；unbound 类正确=没选 ANSWER（即 ASK）。
            correct = (pred[m] == cls)
            accs.append(float(correct.mean()))
    return float(np.mean(accs)) if accs else 0.0


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    print(f"[capcw-int] device={device} n_keys={args.n_keys} n_vals={args.n_vals} k={args.k} "
          f"ask_threshold={args.ask_threshold}", flush=True)
    wm = CAPCWWorkingMemory(n_keys=args.n_keys, n_vals=args.n_vals, d=args.d, n_slots=max(args.n_slots, args.k + 1),
                            iters=args.iters, ask_threshold=args.ask_threshold, device=device)
    bind_acc = wm.train_on_binding(k_pairs=args.k, n_train=args.n_train, epochs=args.epochs, seed=args.seed)
    print(f"[capcw-int] 工作空间绑定训练准确率={bind_acc:.3f}", flush=True)
    try:
        wm.save(CKPT)
    except Exception:
        pass

    dialogues = gen_dialogues(args.n_keys, args.n_vals, args.k, args.n_eval, args.seed + 7000, p_bound=args.p_bound)
    rng = np.random.default_rng(args.seed + 99)

    pred_answer, is_bound_l = [], []
    fe_success, base_success = [], []
    fe_unbound_halluc, n_unbound = 0, 0
    value_correct, value_total = 0, 0
    for bindings, qk, is_bound, true_val in dialogues:
        wm.reset()
        for key, val in bindings:
            wm.bind(key, val)
        dec = wm.decide(qk)
        answered = dec.action == ActionType.ANSWER
        pred_answer.append(answered)
        is_bound_l.append(is_bound)

        # FE-agent 任务成功：bound→须答且 value 对；unbound→须问。
        if is_bound:
            ok = answered and (dec.value == true_val)
            if answered:
                value_total += 1
                value_correct += int(dec.value == true_val)
            fe_success.append(float(ok))
        else:
            n_unbound += 1
            fe_unbound_halluc += int(answered)
            fe_success.append(float(not answered))

        # baseline：无工作记忆、永远直答。bound 只能随机猜 value；unbound 直答=胡编(失败)。
        if is_bound:
            guess = int(rng.integers(args.n_vals))
            base_success.append(float(guess == true_val))
        else:
            base_success.append(0.0)

    balacc = _balanced_acc(pred_answer, is_bound_l)
    value_acc = (value_correct / value_total) if value_total else 0.0
    fe_rate = float(np.mean(fe_success))
    base_rate = float(np.mean(base_success))
    fe_halluc = (fe_unbound_halluc / n_unbound) if n_unbound else 0.0

    h_decision = balacc >= 0.80
    h_value = value_acc >= 0.60
    if h_decision and h_value and fe_rate > base_rate + 0.2:
        verdict = ("PASS: CAPCW 引擎接回 controller 决策成立——引擎 surprise 无动作监督即正确分开"
                   "该答(bound)/该问(unbound)，且取回 value 正确；任务成功率远超无记忆基线、几乎不胡答。")
    elif h_decision:
        verdict = ("PARTIAL: 引擎 surprise 能分开 ASK/ANSWER（决策成立），但内容取回或相对基线优势偏弱。")
    else:
        verdict = "FAIL: 引擎 surprise 未能稳定驱动 controller 的 ASK/ANSWER 决策。"

    result = {
        "task": "CAPCW working-memory wired into controller decision: engine surprise -> ASK/ANSWER + value retrieval",
        "config": {"n_keys": args.n_keys, "n_vals": args.n_vals, "k": args.k, "d": args.d,
                   "ask_threshold": args.ask_threshold, "n_eval": args.n_eval, "p_bound": args.p_bound,
                   "epochs": args.epochs, "random_value_baseline": round(1.0 / args.n_vals, 4)},
        "binding_train_acc": round(bind_acc, 4),
        "ask_answer_balanced_acc": round(balacc, 4),
        "value_retrieval_acc_bound": round(value_acc, 4),
        "fe_task_success": round(fe_rate, 4),
        "baseline_task_success": round(base_rate, 4),
        "fe_unbound_hallucination_rate": round(fe_halluc, 4),
        "baseline_unbound_hallucination_rate": 1.0,
        "verdict": verdict,
        "note": "FE-agent=CAPCW 工作记忆(引擎 surprise 裁决 ASK/ANSWER + 内容取回，无动作监督)；"
                "baseline=无 in-context 记忆、永远直答(bound 随机猜、unbound 胡编)。决策从引擎 surprise 涌现，"
                "对应 controller 招牌'知道何时不该答'，且 value 取回对应 B2c 的'内容/grounding'价值。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# CAPCW 接回 controller · 引擎 surprise 驱动 ASK/ANSWER + 内容取回",
        "",
        f"- 判定：**{verdict}**",
        f"- 设置：n_keys={args.n_keys}, n_vals={args.n_vals}, K={args.k}, ask_threshold={args.ask_threshold}, "
        f"n_eval={args.n_eval}；value 随机基线 {1.0/args.n_vals:.3f}",
        f"- 工作空间绑定训练准确率：{bind_acc:.4f}",
        "",
        "| 指标 | FE-agent（CAPCW 工作记忆） | baseline（无记忆·永远直答） |",
        "|---|---:|---:|",
        f"| ASK/ANSWER balanced acc（引擎 surprise，无动作监督） | {balacc:.4f} | — |",
        f"| 内容取回 value 准确率（bound 且回答时） | {value_acc:.4f} | {1.0/args.n_vals:.4f}（随机猜） |",
        f"| 任务成功率（bound 答对 / unbound 该问） | {fe_rate:.4f} | {base_rate:.4f} |",
        f"| unbound 胡答率（越低越好） | {fe_halluc:.4f} | 1.0000 |",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[capcw-int] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    print(f"[capcw-int] balacc(ASK/ANSWER)={balacc:.3f} value_acc={value_acc:.3f} "
          f"FE成功={fe_rate:.3f} vs baseline={base_rate:.3f} | FE胡答={fe_halluc:.3f} vs 1.0", flush=True)
    print(f"[capcw-int] 报告：{args.report_json} / {args.report_md}", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW working-memory wired into controller decision (engine surprise -> action).")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-keys", type=int, default=10)
    ap.add_argument("--n-vals", type=int, default=12)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--n-slots", type=int, default=6)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--ask-threshold", type=float, default=0.5)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--p-bound", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-json", default=REPORT_JSON)
    ap.add_argument("--report-md", default=REPORT_MD)
    return ap


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print("[capcw-int] dry-run：未训练。CAPCW 工作记忆接回 controller 决策：引擎 surprise→ASK/ANSWER+内容取回。")
        print("[capcw-int] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
