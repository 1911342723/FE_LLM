# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_surprise_action_eval.py
================================================
CAPCW Part 3：surprise→动作 闭环。见 `docs/FE-LLM核心引擎构想.md`。

FE-LLM 的招牌是"知道何时不该答"。本实验证明这个动作可以**直接从引擎的 surprise 自然涌现**，
而非靠单独训练的动作分类头：
- 在**绑定任务**（取回 value）上训练 CAPCW（完全没教它 ASK/ANSWER）；
- 推理时对 bound / unbound 的 query，用 query→slot 最大路由权重作为"匹配度"，其补值=surprise；
- 仅用一个**阈值**就把 unbound（高 surprise=匹配不到=该追问 ASK）与 bound（低 surprise=该 ANSWER）分开。

判据：仅凭 surprise 阈值（无动作监督）就能高 balanced accuracy 分开 bound/unbound（≥ 0.8）→
"自由能/surprise → 何时不该答" 的闭环在引擎层自然成立。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_surprise_action_eval --run
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

REPORT_JSON = os.path.join("docs", "reports", "capcw_surprise_action_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_surprise_action_eval.md")


def gen_mixed(n_keys, n_vals, k, n, seed, p_bound=0.5):
    """K 对随机绑定 + query；bound=1(query 在场) / unbound=0(query 不在场)。用于 surprise→动作评测。"""
    rng = np.random.default_rng(seed)
    pk = np.zeros((n, k), dtype=np.int64)
    pv = np.zeros((n, k), dtype=np.int64)
    qk = np.zeros((n,), dtype=np.int64)
    bound = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        keys = rng.choice(n_keys, size=k, replace=False)
        vals = rng.choice(n_vals, size=k, replace=False)
        pk[i] = keys
        pv[i] = vals
        if rng.random() < p_bound:
            qk[i] = int(keys[rng.integers(k)]); bound[i] = 1
        else:
            not_in = [x for x in range(n_keys) if x not in set(keys.tolist())]
            qk[i] = int(rng.choice(not_in)); bound[i] = 0
    return pk, pv, qk, bound


def best_threshold_balacc(score, label):
    """阈值扫描：score≥thr 判 bound(1)。返回最佳 balanced accuracy 与阈值。"""
    order = np.unique(score)
    best, best_thr = 0.0, 0.0
    for thr in order:
        pred = (score >= thr).astype(np.int64)
        accs = []
        for c in (0, 1):
            m = label == c
            if m.any():
                accs.append(float((pred[m] == label[m]).mean()))
        bal = float(np.mean(accs)) if accs else 0.0
        if bal > best:
            best, best_thr = bal, float(thr)
    return best, best_thr


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    print(f"[surp-act] device={device} K={args.k} d={args.d} seeds={args.seeds}", flush=True)
    bals, seps, sb_l, su_l = [], [], [], []
    for si in range(args.seeds):
        seed = args.seed + si
        # 训练：只教绑定取值（完全没教 ASK/ANSWER）。
        train = gen_binding(args.n_keys, args.n_vals, args.k, args.n_train, seed)
        bind_test = gen_binding(args.n_keys, args.n_vals, args.k, 1000, seed + 100)
        n_slots = max(args.n_slots, args.k + 1)
        model = CAPCWPCModel(args.n_keys, args.n_vals, args.d, n_slots=n_slots, iters=args.iters)
        bind_acc = train_eval(model, train, bind_test, device=device, epochs=args.epochs, lr=args.lr, batch=args.batch, seed=seed)
        # 评测：bound/unbound 混合，仅用 surprise(=1-匹配度) 阈值判 ASK/ANSWER。
        mix = gen_mixed(args.n_keys, args.n_vals, args.k, args.n_test, seed + 5000)
        pk, pv, qk = (torch.tensor(mix[i], device=device) for i in range(3))
        match = model.query_match(pk, pv, qk).cpu().numpy()
        label = np.asarray(mix[3])
        bal, thr = best_threshold_balacc(match, label)
        bals.append(bal)
        sb = float(match[label == 1].mean()); su = float(match[label == 0].mean())
        sb_l.append(sb); su_l.append(su); seps.append(sb - su)
        print(f"[surp-act] seed={seed} bind_acc={bind_acc:.3f} | surprise→动作 balacc={bal:.3f} (match bound={sb:.3f} unbound={su:.3f})", flush=True)

    bal_m = float(np.mean(bals)); sep_m = float(np.mean(seps))
    verdict = ("PASS: 仅凭 surprise 阈值(无动作监督)即可分开 bound/unbound——'自由能→何时不该答'闭环在引擎层自然成立"
               if bal_m >= 0.80 else ("PARTIAL: surprise 有区分力但偏弱" if bal_m >= 0.65 else "FAIL: surprise 不足以驱动动作"))

    result = {
        "task": "surprise->action: threshold CAPCW query-routing to decide ASK(unbound)/ANSWER(bound), no action supervision",
        "config": {"k": args.k, "d": args.d, "n_keys": args.n_keys, "n_vals": args.n_vals,
                   "epochs": args.epochs, "seeds": args.seeds},
        "surprise_action_balanced_acc": round(bal_m, 4),
        "match_bound_mean": round(float(np.mean(sb_l)), 4),
        "match_unbound_mean": round(float(np.mean(su_l)), 4),
        "separation": round(sep_m, 4),
        "verdict": verdict,
        "note": "模型只在绑定取值任务上训练，从未见 ASK/ANSWER 标签；用 query→slot 最大路由权重作匹配度、"
                "其补=surprise，单阈值判动作。bound 匹配高(低 surprise=该答)、unbound 匹配低(高 surprise=该问)。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# CAPCW Part3 · surprise→动作 闭环（无动作监督）",
        "",
        f"- 判定：**{verdict}**",
        "- 任务：模型只学绑定取值（没教 ASK/ANSWER）；推理时仅用 query→slot 路由 surprise 阈值判 ASK/ANSWER。",
        f"- K={args.k}, d={args.d}, seeds={args.seeds}",
        "",
        f"- **surprise→动作 balanced accuracy（无监督）：{bal_m:.4f}**",
        f"- query 匹配度：bound（该答）{float(np.mean(sb_l)):.4f} vs unbound（该问）{float(np.mean(su_l)):.4f}，分离 {sep_m:+.4f}",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[surp-act] === 裁决 ===", flush=True)
    print(f"{verdict}  surprise→动作 balacc={bal_m:.3f} 分离={sep_m:+.3f}", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW phase: surprise->action closed loop (no action supervision).")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-keys", type=int, default=10)
    ap.add_argument("--n-vals", type=int, default=12)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--n-slots", type=int, default=6)
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
        print("[surp-act] dry-run：未训练。只学绑定取值，用 surprise 阈值无监督判 ASK/ANSWER。")
        print("[surp-act] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
