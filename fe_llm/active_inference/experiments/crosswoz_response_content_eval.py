# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/crosswoz_response_content_eval.py
====================================================================
B2c：在真实任务对话 CrossWOZ 上验证 belief 对「生成正确回复内容」的 headroom（生成内容代理）。

承接 B2（动作类型 belief 无 headroom）与 B2b（领域追踪 belief 有强 headroom）。本实验把
belief 价值延伸到**回复内容**这一维：预测系统下一轮**主告知的 (领域·槽位)**（例如 餐馆·营业时间），
这是"系统该说什么"的内容代理。

要点：跟进句里**槽位词常在 utterance 中**（"营业时间呢"含"营业时间"），但**领域不在**（不含"餐馆"），
领域只能由对话状态（belief）决定。所以预测完整 (领域·槽位) 时：
- blind：拿得到槽位、却拿不准领域 → 在领域未明示子集上弱；
- context(+belief)：领域由活跃 belief 补上 → 应明显更强。
若 belief 在领域未明示子集上显著提升，即说明 belief 不仅帮"理解"，也帮"说对内容"（grounding 生成）。

唯一变量=是否加 belief；同 MLP/split/train（口径同 teacher_corpus_eval）。
数据：`data/crosswoz/train.json.zip`。默认 dry-run；--run 真跑。
运行：python -m fe_llm.active_inference.experiments.crosswoz_response_content_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from fe_llm.active_inference.experiments.crosswoz_headroom_eval import load_dialogues
from fe_llm.active_inference.experiments.crosswoz_domain_tracking_eval import _turn_domain
from fe_llm.active_inference.experiments.teacher_corpus_eval import stratified_split, train_eval
from fe_llm.config import get_device

DATA = os.path.join("data", "crosswoz", "train.json.zip")
REPORT_JSON = os.path.join("docs", "reports", "crosswoz_response_content_eval.json")
REPORT_MD = os.path.join("docs", "reports", "crosswoz_response_content_eval.md")

DOMAINS = ["餐馆", "酒店", "景点", "地铁", "出租"]


def _primary_inform(dialog_act: list) -> tuple[str, str] | None:
    """系统主告知的 (领域, 槽位)：第一个 Inform 动作的 (domain, slot)。"""
    for act in dialog_act or []:
        if act and len(act) >= 3 and act[0] == "Inform" and act[1] in DOMAINS:
            return (act[1], act[2])
    return None


def extract_samples(dialogues: list[dict]) -> list[dict]:
    """抽 (utterance, label=系统主告知 领域·槽位, belief=决策前活跃领域)。"""
    samples = []
    for d in dialogues:
        msgs = d.get("messages", [])
        seen: set[str] = set()
        last: str | None = None
        i = 0
        while i < len(msgs) - 1:
            cur, nxt = msgs[i], msgs[i + 1]
            if cur.get("role") == "usr" and nxt.get("role") == "sys":
                inf = _primary_inform(nxt.get("dialog_act"))
                if inf is not None:
                    samples.append(
                        {
                            "utterance": (cur.get("content") or "").strip(),
                            "domain": inf[0],
                            "label": f"{inf[0]}·{inf[1]}",
                            "seen": sorted(seen),
                            "last": last,
                        }
                    )
                dom = _turn_domain(cur.get("dialog_act"))
                if dom is not None:
                    seen.add(dom)
                    last = dom
                i += 2
            else:
                dom = _turn_domain(cur.get("dialog_act"))
                if dom is not None:
                    seen.add(dom)
                    last = dom
                i += 1
    return [s for s in samples if s["utterance"]]


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    dialogues = load_dialogues(args.data)
    if args.max_dialogues > 0:
        dialogues = dialogues[: args.max_dialogues]
    samples = extract_samples(dialogues)

    # 过滤过稀疏标签，保证 balanced accuracy 有意义。
    label_count: dict[str, int] = defaultdict(int)
    for s in samples:
        label_count[s["label"]] += 1
    keep = {lab for lab, n in label_count.items() if n >= args.min_label_freq}
    samples = [s for s in samples if s["label"] in keep]
    if len(samples) < 50:
        raise RuntimeError(f"样本太少：{len(samples)}（检查数据 {args.data}）")

    labels = sorted({s["label"] for s in samples})
    lidx = {lab: i for i, lab in enumerate(labels)}
    n_classes = len(labels)
    y = np.array([lidx[s["label"]] for s in samples], dtype=np.int64)

    # 领域未明示子集：utterance 不含目标领域名（跟进句，领域只能靠 belief）。
    underspec = np.array([s["domain"] not in s["utterance"] for s in samples], dtype=bool)

    char_count: dict[str, int] = defaultdict(int)
    for s in samples:
        for c in s["utterance"]:
            char_count[c] += 1
    vocab = sorted([c for c, n in char_count.items() if n >= args.min_char_freq])
    cidx = {c: i for i, c in enumerate(vocab)}

    def bow(t: str) -> np.ndarray:
        v = np.zeros(len(vocab), dtype=np.float32)
        for c in t:
            j = cidx.get(c)
            if j is not None:
                v[j] += 1.0
        return v / max(len(t), 1)

    dom_idx = {dmn: i for i, dmn in enumerate(DOMAINS)}

    def belief_vec(seen: list[str], last: str | None) -> np.ndarray:
        v = np.zeros(2 * len(DOMAINS) + 1, dtype=np.float32)
        for dmn in seen:
            if dmn in dom_idx:
                v[dom_idx[dmn]] = 1.0
        if last in dom_idx:
            v[len(DOMAINS) + dom_idx[last]] = 1.0
        else:
            v[-1] = 1.0
        return v

    U = np.stack([bow(s["utterance"]) for s in samples])
    B = np.stack([belief_vec(s["seen"], s["last"]) for s in samples])
    X_blind = U
    X_ctx = np.concatenate([U, B], axis=1)

    tr, va = stratified_split(y, n_classes, args.seed)
    us_val = underspec[va]

    print(
        f"[content] device={device} samples={len(samples)} labels={n_classes} "
        f"vocab={len(vocab)} underspec_frac={float(underspec.mean()):.3f}",
        flush=True,
    )
    print("[content] 训练 blind 臂 ...", flush=True)
    b_o, b_us = train_eval(X_blind, y, tr, va, us_val, n_classes, device, args.seed, args.epochs)
    print(f"[content] blind overall={b_o:.4f} underspec={b_us:.4f}", flush=True)
    print("[content] 训练 context-aware 臂 ...", flush=True)
    c_o, c_us = train_eval(X_ctx, y, tr, va, us_val, n_classes, device, args.seed, args.epochs)
    print(f"[content] ctx   overall={c_o:.4f} underspec={c_us:.4f}", flush=True)

    us_delta = c_us - b_us
    overall_delta = c_o - b_o
    if us_delta > 0.2:
        verdict = "PASS: belief 对回复内容(领域·槽位)在跟进句上有强 headroom（grounding 生成）"
    elif us_delta > 0.05:
        verdict = "PARTIAL: belief 对回复内容有正向 headroom"
    else:
        verdict = "WEAK: belief 对回复内容 headroom 不明显"

    result = {
        "dataset": "CrossWOZ (real human-annotated task dialogue)",
        "task": "response content proxy: predict system primary informed (domain·slot)",
        "data": args.data,
        "n_dialogues": len(dialogues),
        "n_samples": len(samples),
        "n_labels": n_classes,
        "underspecified_frac": round(float(underspec.mean()), 4),
        "underspecified_count_val": int(us_val.sum()),
        "overall": {"context_blind": round(b_o, 4), "context_aware": round(c_o, 4), "delta": round(overall_delta, 4)},
        "underspecified_subset": {"context_blind": round(b_us, 4), "context_aware": round(c_us, 4), "delta": round(us_delta, 4)},
        "verdict": verdict,
        "note": (
            "回复内容代理=系统主告知 (领域·槽位)；唯一变量=belief(活跃领域)。跟进句里槽位常在 utterance、"
            "领域不在，故 belief 补领域→帮系统说对内容。与 B2/B2b 合成 belief 价值地图：动作类型(无)、"
            "状态/领域追踪(强)、回复内容 grounding(本实验)。"
        ),
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# B2c · 真实任务对话(CrossWOZ) belief 对回复内容的 headroom",
        "",
        f"- 判定：**{verdict}**",
        f"- 数据：`{args.data}`（{result['n_dialogues']} 对话 / {result['n_samples']} 样本 / {n_classes} 个 领域·槽位 标签）",
        f"- 领域未明示子集占比：{result['underspecified_frac']}（验证集 {result['underspecified_count_val']} 条）",
        "",
        "> 任务=预测系统主告知 (领域·槽位)；唯一变量=是否加 belief（活跃领域）；同 MLP/split/train。",
        "",
        "## 总体 balanced accuracy",
        f"- 盲：{result['overall']['context_blind']} → 感知：{result['overall']['context_aware']}（delta {result['overall']['delta']:+.4f}）",
        "",
        "## 领域未明示子集（跟进句，headroom 关键）",
        f"- 盲：{result['underspecified_subset']['context_blind']} → 感知：{result['underspecified_subset']['context_aware']}（delta **{result['underspecified_subset']['delta']:+.4f}**）",
        "",
        "## belief 价值地图（B2 系列合并结论）",
        "- 动作类型选择（B2）：belief 无 headroom（−0.02）",
        "- 状态/领域追踪（B2b）：belief 强 headroom（未明示子集 +0.19）",
        "- 回复内容 grounding（B2c）：见上",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CrossWOZ response-content belief headroom eval.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-char-freq", type=int, default=3)
    parser.add_argument("--min-label-freq", type=int, default=30)
    parser.add_argument("--max-dialogues", type=int, default=0)
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print("[content] dry-run：未训练。")
        print(f"[content] 在 {args.data} 上预测系统主告知 (领域·槽位)：盲 vs +belief（重点领域未明示子集）。")
        print("[content] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
