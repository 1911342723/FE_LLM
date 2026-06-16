# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_multihop_controller_eval.py
====================================================
CAPCW **多跳链式** 工作记忆接回 controller：把已验证的「decode→re-embed + 中间监督（潜在 CoT）」
机制（第 23 节 / `capcw_multihop_cot_eval.py`）从受控合成 eval 推进到 **controller 决策框架**里的
多步推理（蓝图「对内推理」）。见 `fe_llm/active_inference/capcw_chain_memory.py`。

对话 = 现场给出一条链 c0→c1→…→cH（+ 干扰边）+ 一个 start 查询键（bound=c0 / unbound 各半）：
- FE-agent（CAPCW 链式工作记忆，cot=decode→re-embed）：从 start 链式取回 H 跳——start 绑定→低
  surprise→ANSWER + 链尾 value（多跳取回）；start 未绑定→高 surprise→ASK（多跳版"知道何时不该答"）。
- baseline（单跳，只读 1 跳）：H≥2 时只会取回 c1（≠cH）→ 多跳任务必失败（**没有链式组合就够不到链尾**）。
- latent 消融（cot=False，潜读出直接当下一跳 query + 仅末跳监督）：上轮失败形态，作"decode→re-embed
  是否必要"的对照（held-out 上报增益）。

判据（先写死；与 capcw_controller_integration_eval 同口径，多跳更难故 value 阈值放宽）
----------------------------------------------------------------------------------
- balacc_ask_answer ≥ 0.80：首跳匹配度的 surprise 能正确分开 start 绑定(该答)/未绑定(该问)；
- 多跳(H≥2) cot 链尾 value 取回 ≥ 0.50（受 d=32 容量限制，不强求高）；
- 多跳(H≥2) cot 任务成功率 − baseline(单跳) ≥ +0.20：**链式组合带来的增量**（核心）。
满足 → PASS：多跳 CoT 链式取回接回 controller 成立（对内多步推理 + 知道何时不该答 + 可溯源 CoT trace）。
同时报告 cot − latent(held-out 多跳)：再确认"显式解码中间符号"对链式的作用（capacity caveat）。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_multihop_controller_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from fe_llm.active_inference.capcw_chain_memory import CAPCWChainMemory
from fe_llm.active_inference.policy import ActionType
from fe_llm.config import get_device

REPORT_JSON = os.path.join("docs", "reports", "capcw_multihop_controller_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_multihop_controller_eval.md")
CKPT = os.path.join("checkpoints", "world_model", "capcw_chain_wm.pt")


def gen_chain_dialogues(n_sym, n_hops, n_pairs, n, seed, p_bound=0.5):
    """每段对话：链 c0→…→cH（+干扰边凑满 n_pairs）+ 一个 start 查询键（bound=c0 / unbound）。

    无歧义口径：链符号与干扰符号互不相同；unbound start 取一个未作任何边 key 的符号。
    返回 [(bindings:list[(k,v)], start, is_bound, answer=cH)]。
    """
    rng = np.random.default_rng(seed)
    H = n_hops
    n_distract = max(0, n_pairs - H)
    need = (H + 1) + 2 * n_distract
    dialogues = []
    for _ in range(n):
        picks = rng.choice(n_sym, size=need, replace=False)
        c = picks[: H + 1]
        ds = picks[H + 1:]
        d_keys, d_vals = ds[:n_distract], ds[n_distract:]
        keys = [int(c[h]) for h in range(H)] + [int(k) for k in d_keys]
        vals = [int(c[h + 1]) for h in range(H)] + [int(v) for v in d_vals]
        bindings = list(zip(keys, vals))
        answer = int(c[H])
        if rng.random() < p_bound:
            start, is_bound = int(c[0]), True
        else:
            used_keys = set(keys)
            not_key = [s for s in range(n_sym) if s not in used_keys]
            start, is_bound = int(rng.choice(not_key)), False
        dialogues.append((bindings, start, is_bound, answer))
    return dialogues


def _balanced_acc(pred_answer, is_bound):
    pred = np.asarray(pred_answer, dtype=bool)
    truth = np.asarray(is_bound, dtype=bool)
    accs = []
    for cls in (True, False):
        m = truth == cls
        if m.any():
            accs.append(float((pred[m] == cls).mean()))
    return float(np.mean(accs)) if accs else 0.0


def _eval_agent(mem, dialogues, n_hops, *, single_hop=False):
    """对一批对话跑链式裁决。single_hop=True 时强制只读 1 跳（baseline：无链式组合）。

    返回 dict：决策预测(answered)、is_bound、bound 任务的链尾 value 取回、整体任务成功。
    """
    pred_answer, is_bound_l, success, value_correct, value_total = [], [], [], 0, 0
    for bindings, start, is_bound, answer in dialogues:
        mem.reset()
        for k, v in bindings:
            mem.bind(k, v)
        dec = mem.decide_chain(start, 1 if single_hop else n_hops)
        answered = dec.action == ActionType.ANSWER
        pred_answer.append(answered)
        is_bound_l.append(is_bound)
        if is_bound:
            ok = answered and (dec.value == answer)
            if answered:
                value_total += 1
                value_correct += int(dec.value == answer)
            success.append(float(ok))
        else:
            success.append(float(not answered))      # unbound→该问（不胡答）
    return {
        "balacc": _balanced_acc(pred_answer, is_bound_l),
        "task_success": float(np.mean(success)),
        "value_acc_bound": (value_correct / value_total) if value_total else 0.0,
    }


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    hop_list = [int(x) for x in args.hop_list.split(",")]
    max_h = max(hop_list)
    print(f"[mh-ctrl] device={device} n_sym={args.n_sym} n_pairs={args.n_pairs} hop_list={hop_list} "
          f"ask_threshold={args.ask_threshold}", flush=True)

    # 训练 cot（decode→re-embed + 中间监督，生产形态）与 latent（潜读出 + 仅末跳监督，消融对照）。
    cot = CAPCWChainMemory(n_sym=args.n_sym, d=args.d, n_slots=max(args.n_slots, args.n_pairs + 1),
                           iters=args.iters, ask_threshold=args.ask_threshold, cot=True, device=device)
    cot_acc = cot.train_on_chain(max_hops=max_h, n_pairs=args.n_pairs, n_train=args.n_train,
                                 epochs=args.epochs, seed=args.seed)
    print(f"[mh-ctrl] cot 链式训练准确率(链尾,H={max_h})={cot_acc:.3f}", flush=True)
    latent = CAPCWChainMemory(n_sym=args.n_sym, d=args.d, n_slots=max(args.n_slots, args.n_pairs + 1),
                             iters=args.iters, ask_threshold=args.ask_threshold, cot=False, device=device)
    lat_acc = latent.train_on_chain(max_hops=max_h, n_pairs=args.n_pairs, n_train=args.n_train,
                                    epochs=args.epochs, seed=args.seed)
    print(f"[mh-ctrl] latent 链式训练准确率(链尾,H={max_h})={lat_acc:.3f}", flush=True)
    try:
        cot.save(CKPT)
    except Exception:
        pass

    by_h = {}
    for h in hop_list:
        dials = gen_chain_dialogues(args.n_sym, h, args.n_pairs, args.n_eval, args.seed + 7000 + h, p_bound=args.p_bound)
        fe = _eval_agent(cot, dials, h)
        base = _eval_agent(cot, dials, h, single_hop=True)       # baseline=同模型只读 1 跳（无链式组合）
        lat = _eval_agent(latent, dials, h)
        by_h[h] = {
            "cot": {k: round(v, 4) for k, v in fe.items()},
            "baseline_single_hop": {k: round(v, 4) for k, v in base.items()},
            "latent": {k: round(v, 4) for k, v in lat.items()},
        }
        print(f"[mh-ctrl] H={h} cot[succ={fe['task_success']:.3f} val={fe['value_acc_bound']:.3f} "
              f"balacc={fe['balacc']:.3f}] base[succ={base['task_success']:.3f}] "
              f"latent[val={lat['value_acc_bound']:.3f}]", flush=True)

    multi = [h for h in hop_list if h >= 2]
    cot_multi_succ = float(np.mean([by_h[h]["cot"]["task_success"] for h in multi])) if multi else 0.0
    cot_multi_val = float(np.mean([by_h[h]["cot"]["value_acc_bound"] for h in multi])) if multi else 0.0
    base_multi_succ = float(np.mean([by_h[h]["baseline_single_hop"]["task_success"] for h in multi])) if multi else 0.0
    lat_multi_val = float(np.mean([by_h[h]["latent"]["value_acc_bound"] for h in multi])) if multi else 0.0
    balacc_mean = float(np.mean([by_h[h]["cot"]["balacc"] for h in hop_list]))
    cot_minus_base = round(cot_multi_succ - base_multi_succ, 4)
    cot_minus_latent = round(cot_multi_val - lat_multi_val, 4)
    rnd = round(1.0 / args.n_sym, 4)

    h_decision = balacc_mean >= 0.80
    h_value = cot_multi_val >= 0.50
    h_chain = cot_minus_base >= 0.20
    if h_decision and h_value and h_chain:
        cap = "" if cot_multi_val >= 0.6 else "（链尾绝对值受 d=32 容量限制：高跳数下滑，与小 d 容量结论一致）"
        verdict = (f"PASS: 多跳 CoT 链式取回接回 controller 成立——start 绑定→ANSWER + 链式取回链尾 value、"
                   f"未绑定→ASK（balacc {balacc_mean:.3f}）；多跳链尾取回 {cot_multi_val:.3f}，比只读 1 跳的"
                   f"baseline 任务成功率高 {cot_minus_base:+.4f}（**链式组合的增量**）。对内多步推理 + 知道何时"
                   f"不该答 + 可溯源 CoT trace 在 controller 决策框架内成立{cap}。cot−latent(held-out 多跳取回)"
                   f"={cot_minus_latent:+.4f}（显式解码中间符号的作用）。")
    elif h_decision and h_chain:
        verdict = (f"PARTIAL: 链式组合带来增量（cot−base {cot_minus_base:+.4f}）且决策成立（balacc "
                   f"{balacc_mean:.3f}），但链尾取回偏弱（{cot_multi_val:.3f}<0.50，d=32 容量）。")
    else:
        verdict = (f"FAIL: 多跳 CoT 未在 controller 框架内稳定成立（balacc {balacc_mean:.3f}/链尾 "
                   f"{cot_multi_val:.3f}/链式增量 {cot_minus_base:+.4f}）。诚实记录。")

    result = {
        "task": "multi-hop chained retrieval (decode->re-embed CoT) wired into controller decision framework",
        "design": "FE-agent=CAPCW chain memory(cot, decode->re-embed + intermediate sup); "
                  "baseline=same model read 1 hop only(no chaining); latent=cot=False ablation(final-only sup).",
        "config": {"n_sym": args.n_sym, "n_pairs": args.n_pairs, "hop_list": hop_list, "d": args.d,
                   "ask_threshold": args.ask_threshold, "n_eval": args.n_eval, "p_bound": args.p_bound,
                   "epochs": args.epochs, "random_baseline": rnd,
                   "cot_train_acc": round(cot_acc, 4), "latent_train_acc": round(lat_acc, 4)},
        "by_n_hops": by_h,
        "ask_answer_balanced_acc_mean": round(balacc_mean, 4),
        "cot_multihop_task_success": round(cot_multi_succ, 4),
        "cot_multihop_value_acc": round(cot_multi_val, 4),
        "baseline_multihop_task_success": round(base_multi_succ, 4),
        "cot_minus_baseline_multihop": cot_minus_base,
        "cot_minus_latent_multihop_value": cot_minus_latent,
        "verdict": verdict,
        "note": "决策从引擎首跳路由 surprise 涌现（无动作监督）；链式由 decode→re-embed（潜在 CoT）实现，"
                "每跳解码的中间符号=可溯源 CoT trace；baseline 只读 1 跳故够不到链尾（H≥2），凸显链式组合的增量。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# CAPCW 多跳链式工作记忆接回 controller · decode→re-embed（潜在 CoT）",
        "",
        f"- 判定：**{verdict}**",
        f"- 设置：n_sym={args.n_sym}, n_pairs={args.n_pairs}, d={args.d}, ask_threshold={args.ask_threshold}, "
        f"n_eval={args.n_eval}；随机基线 {rnd:.3f}",
        f"- 训练准确率（链尾,H={max_h}）：cot={cot_acc:.4f} / latent={lat_acc:.4f}",
        "",
        "| n_hops | cot 任务成功 | cot 链尾取回 | cot balacc | baseline(单跳) 成功 | latent 链尾取回 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for h in hop_list:
        b = by_h[h]
        lines.append(f"| {h} | {b['cot']['task_success']:.3f} | {b['cot']['value_acc_bound']:.3f} | "
                     f"{b['cot']['balacc']:.3f} | {b['baseline_single_hop']['task_success']:.3f} | "
                     f"{b['latent']['value_acc_bound']:.3f} |")
    lines += [
        "",
        f"- 多跳(H≥2)：cot 链尾取回 **{cot_multi_val:.3f}**；cot−baseline 任务成功 **{cot_minus_base:+.4f}**"
        f"（链式组合增量）；cot−latent 链尾取回 **{cot_minus_latent:+.4f}**（显式解码中间符号的作用）。",
        f"- ASK/ANSWER balanced acc（均值，首跳 surprise 驱动，无动作监督）：**{balacc_mean:.4f}**",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[mh-ctrl] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    print(f"[mh-ctrl] balacc={balacc_mean:.3f} cot多跳取回={cot_multi_val:.3f} "
          f"cot−base={cot_minus_base:+.4f} cot−latent={cot_minus_latent:+.4f}", flush=True)
    print(f"[mh-ctrl] 报告：{args.report_json} / {args.report_md}", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW multi-hop chain memory wired into controller (decode->re-embed CoT).")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-sym", type=int, default=20)
    ap.add_argument("--n-pairs", type=int, default=5)
    ap.add_argument("--hop-list", default="1,2,3")
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--n-slots", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--ask-threshold", type=float, default=0.5)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--p-bound", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=60)
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
        print("[mh-ctrl] dry-run：未训练。多跳 CoT 链式工作记忆接回 controller：decode→re-embed + 首跳 surprise→ASK/ANSWER。")
        print("[mh-ctrl] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
