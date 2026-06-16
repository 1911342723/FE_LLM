# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_crosswoz_eval.py
=========================================
CAPCW Part 1：把内容寻址工作空间推向**真实语言任务**——CrossWOZ 真实对话状态绑定检索。

合成绑定（capcw_binding_eval）已证 CAPCW>单向量。本实验用**真实 CrossWOZ** 的 (领域·槽位 → 值)
inform 绑定：每个对话其实就是一组在线绑定的状态。任务=给一组该对话真实 informed 的 (slot,value) +
一个 query slot，检索它的 value。真实槽/真实值/真实共现/真实在线绑定（同一槽在不同对话取不同值，
故必须 in-context 检索，不能记忆先验）；按**对话切分** train/test（test 对话未见）。

对照（唯一变量=世界状态结构）：flat(单向量) vs CAPCW_PC(自由能 slot 工作空间)。容量受限 d。
判据：CAPCW − flat ≥ +0.10 → CAPCW 在真实语言绑定上同样胜单向量。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_crosswoz_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from fe_llm.active_inference.experiments.crosswoz_headroom_eval import load_dialogues
from fe_llm.config import get_device
from fe_llm.world_model.capcw_binding_eval import CAPCWPCModel, FlatModel, train_eval

REPORT_JSON = os.path.join("docs", "reports", "capcw_crosswoz_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_crosswoz_eval.md")
DATA = os.path.join("data", "crosswoz", "train.json.zip")
NON_DOMAIN = {"General", "", None}


def dialogue_informs(d) -> dict[str, str]:
    """一个对话里用户 Inform 的 (领域·槽位 → 值)（同槽取最后一次）。"""
    slots: dict[str, str] = {}
    for m in d.get("messages", []):
        if m.get("role") != "usr":
            continue
        for act in m.get("dialog_act") or []:
            if act and len(act) >= 4 and act[0] == "Inform" and act[1] not in NON_DOMAIN:
                key, val = f"{act[1]}·{act[2]}", act[3]
                if val:
                    slots[key] = val
    return slots


def build_examples(dialogue_slots, key2id, val2id, k, per_dialogue, rng, shuffle=False, n_vals=0):
    """每个对话采若干 example：取其 K 个 value 在词表内的 (slot,value)，query 一个、label 其 value。

    shuffle=True：把每例 K 个槽位的 value 随机重指派（破坏"槽→典型值"先验=变成真 in-context 绑定，
    值不可由键预测，必须读本例的对应关系），用于对照诊断"真实槽值可记忆 vs 真 in-context"。
    """
    pk, pv, qk, y = [], [], [], []
    for slots in dialogue_slots:
        usable = [(s, v) for s, v in slots.items() if v in val2id and s in key2id]
        if len(usable) < k:
            continue
        for _ in range(per_dialogue):
            picks = [usable[i] for i in rng.choice(len(usable), size=k, replace=False)]
            keys = [key2id[s] for s, _ in picks]
            if shuffle:
                vals = [int(x) for x in rng.choice(n_vals, size=k, replace=False)]  # 随机在线绑定
            else:
                vals = [val2id[v] for _, v in picks]
            qi = int(rng.integers(k))
            pk.append(keys)
            pv.append(vals)
            qk.append(keys[qi])
            y.append(vals[qi])
    return (np.array(pk, dtype=np.int64), np.array(pv, dtype=np.int64),
            np.array(qk, dtype=np.int64), np.array(y, dtype=np.int64))


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    dialogues = load_dialogues(args.data)
    all_slots = [dialogue_informs(d) for d in dialogues]
    all_slots = [s for s in all_slots if len(s) >= args.k]

    # 词表：slot 全收；value 取高频前 n_vals（保证分类可解 + 真实在线绑定）。
    key_counter = Counter(s for slots in all_slots for s in slots)
    val_counter = Counter(v for slots in all_slots for v in slots.values())
    key2id = {k: i for i, (k, _) in enumerate(key_counter.most_common())}
    val2id = {v: i for i, (v, _) in enumerate(val_counter.most_common(args.n_vals))}
    n_keys, n_vals = len(key2id), len(val2id)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(all_slots))
    cut = int(len(idx) * 0.8)
    train_d = [all_slots[i] for i in idx[:cut]]
    test_d = [all_slots[i] for i in idx[cut:]]
    train = build_examples(train_d, key2id, val2id, args.k, args.per_dialogue, np.random.default_rng(args.seed), shuffle=args.shuffle, n_vals=n_vals)
    test = build_examples(test_d, key2id, val2id, args.k, args.per_dialogue, np.random.default_rng(args.seed + 1), shuffle=args.shuffle, n_vals=n_vals)
    mode = "shuffled(真in-context)" if args.shuffle else "real(真实可记忆)"
    print(
        f"[crosswoz-capcw] device={device} mode={mode} keys={n_keys} vals={n_vals} K={args.k} "
        f"train={len(train[0])} test={len(test[0])} (random={1.0/max(n_vals,1):.3f})",
        flush=True,
    )
    if len(train[0]) < 100 or len(test[0]) < 50:
        raise RuntimeError("真实样本太少，调小 K 或 n_vals")

    flat_accs, capcw_accs = [], []
    for si in range(args.seeds):
        seed = args.seed + si
        n_slots = max(args.n_slots, args.k + 1)
        flat = FlatModel(n_keys, n_vals, args.d)
        capcw = CAPCWPCModel(n_keys, n_vals, args.d, n_slots=n_slots, iters=args.iters)
        common = dict(device=device, epochs=args.epochs, lr=args.lr, batch=args.batch, seed=seed)
        flat_accs.append(train_eval(flat, train, test, **common))
        capcw_accs.append(train_eval(capcw, train, test, **common))
        print(f"[crosswoz-capcw] seed={seed} flat={flat_accs[-1]:.3f} capcw={capcw_accs[-1]:.3f}", flush=True)

    flat_m, capcw_m = float(np.mean(flat_accs)), float(np.mean(capcw_accs))
    delta = round(capcw_m - flat_m, 4)
    verdict = ("PASS: CAPCW 在真实语言(CrossWOZ)绑定检索上同样明显胜单向量" if delta >= 0.10
               else ("PARTIAL: 正向但偏弱" if delta >= 0.03 else "FAIL: 真实数据上未明显胜"))

    result = {
        "task": "CrossWOZ real dialogue-state binding retrieval (query slot -> value)",
        "config": {"k": args.k, "d": args.d, "n_keys": n_keys, "n_vals": n_vals,
                   "epochs": args.epochs, "seeds": args.seeds, "random_baseline": round(1.0 / max(n_vals, 1), 4)},
        "mode": mode,
        "flat_acc": round(flat_m, 4), "capcw_acc": round(capcw_m, 4), "delta_capcw_minus_flat": delta,
        "verdict": verdict,
        "note": "真实 CrossWOZ informed (领域·槽位→值)；同槽在不同对话取不同值→必须 in-context 检索；"
                "按对话切分 test 未见。唯一变量=世界状态结构(单向量 vs slot 工作空间)。",
    }
    rj = args.report_json.replace(".json", "_shuffle.json") if args.shuffle else args.report_json
    rm = args.report_md.replace(".md", "_shuffle.md") if args.shuffle else args.report_md
    os.makedirs(os.path.dirname(rj), exist_ok=True)
    with open(rj, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        f"# CAPCW Part1 · 真实语言(CrossWOZ)对话状态绑定检索 [{mode}]",
        "",
        f"- 判定：**{verdict}**",
        f"- 任务：真实 CrossWOZ informed (领域·槽位→值) 绑定，query 槽位检索其值；按对话切分。",
        f"- 词表：slot {n_keys} / value {n_vals}；K={args.k}, d={args.d}（容量受限）；随机基线 {1.0/max(n_vals,1):.3f}",
        "",
        "| 世界状态结构 | 检索 accuracy |",
        "|---|---:|",
        f"| flat（单向量） | {flat_m:.4f} |",
        f"| CAPCW_PC（slot 工作空间） | {capcw_m:.4f} |",
        "",
        f"- delta（CAPCW − flat）= **{delta:+.4f}**",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(rm, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[crosswoz-capcw] === 裁决 ===", flush=True)
    print(f"{verdict}  flat={flat_m:.3f} capcw={capcw_m:.3f} (delta {delta:+.4f})", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW on real CrossWOZ dialogue-state binding retrieval.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--shuffle", action="store_true", help="随机重指派每例槽值=破坏先验=真 in-context 绑定（诊断对照）")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-vals", type=int, default=40)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--n-slots", type=int, default=6)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--per-dialogue", type=int, default=6)
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
        print("[crosswoz-capcw] dry-run：未训练。真实 CrossWOZ 绑定检索上 flat vs CAPCW。")
        print("[crosswoz-capcw] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
