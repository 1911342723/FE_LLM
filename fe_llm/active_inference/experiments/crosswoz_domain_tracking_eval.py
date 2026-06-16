# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/crosswoz_domain_tracking_eval.py
===================================================================
B2b：在真实任务对话 CrossWOZ 上，给 belief 找它**真正有 headroom 的环节**——领域追踪。

B2 的发现：真实数据里系统**动作类型**几乎由当前 utterance 决定，belief 无 headroom。
但真实对话里大量**跟进句**（"营业时间呢""电话多少""怎么去"）的 utterance 本身不含领域词，
其正确领域由对话状态（belief 的当前活跃领域）决定——这正是 belief 在真实数据里真正决定性
的地方。本实验验证：

任务：预测用户当前轮的领域（餐馆/酒店/景点/地铁/出租）。
- blind：只看 utterance 字符袋；
- context-aware：+ belief（已出现领域 multi-hot + 上一活跃领域 one-hot）；
- 唯一变量=是否加 belief；同 MLP/split/train（口径同 teacher_corpus_eval）。

判定（balanced accuracy）：
- 关键看「领域未明示子集」=utterance 不含目标领域名（跟进句）——blind 应明显弱、context 应明显强；
- 若 context 在该子集上显著高于 blind（delta ≥ +0.2），即证明 belief 在真实数据上**确有强 headroom**
  （只是在领域/状态追踪环节，而非 B2 测的动作类型环节）。

数据：`data/crosswoz/train.json.zip`（同 B2）。
默认 dry-run；--run 真跑。
运行：python -m fe_llm.active_inference.experiments.crosswoz_domain_tracking_eval --run
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
from fe_llm.active_inference.experiments.teacher_corpus_eval import stratified_split, train_eval
from fe_llm.config import get_device

DATA = os.path.join("data", "crosswoz", "train.json.zip")
REPORT_JSON = os.path.join("docs", "reports", "crosswoz_domain_tracking_eval.json")
REPORT_MD = os.path.join("docs", "reports", "crosswoz_domain_tracking_eval.md")

NON_DOMAIN = {"General", "", None}


def _turn_domain(dialog_act: list) -> str | None:
    """取一轮 dialog_act 的主任务领域（跳过 General 意图；只认真任务领域）。"""
    for act in dialog_act or []:
        # act = [intent, domain, slot, value]；General(greet/thank/bye) 的 act[1] 不是任务领域。
        if act and len(act) >= 2 and act[0] != "General" and act[1] not in NON_DOMAIN:
            return act[1]
    return None


def extract_samples(dialogues: list[dict]) -> list[dict]:
    """抽用户轮的 (utterance, label_domain, belief)；belief=决策前已出现领域 + 上一活跃领域。"""
    samples = []
    for d in dialogues:
        seen_domains: set[str] = set()
        last_domain: str | None = None
        for m in d.get("messages", []):
            dom = _turn_domain(m.get("dialog_act"))
            if m.get("role") == "usr":
                label = dom
                if label is not None:
                    text = (m.get("content") or "").strip()
                    if text:
                        samples.append(
                            {
                                "utterance": text,
                                "label": label,
                                "seen": sorted(seen_domains),   # 决策前已出现的领域
                                "last": last_domain,            # 决策前的上一活跃领域
                            }
                        )
            # 更新状态（当前轮之后）
            if dom is not None:
                seen_domains.add(dom)
                last_domain = dom
    return samples


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    dialogues = load_dialogues(args.data)
    if args.max_dialogues > 0:
        dialogues = dialogues[: args.max_dialogues]
    samples = extract_samples(dialogues)
    if len(samples) < 50:
        raise RuntimeError(f"样本太少：{len(samples)}（检查数据 {args.data}）")

    domains = sorted({s["label"] for s in samples})
    didx = {dmn: i for i, dmn in enumerate(domains)}
    n_classes = len(domains)
    y = np.array([didx[s["label"]] for s in samples], dtype=np.int64)

    # 领域未明示子集：utterance 文本里不含「目标领域名」(餐馆/酒店/景点/地铁/出租)——跟进句。
    underspec = np.array([s["label"] not in s["utterance"] for s in samples], dtype=bool)

    # utterance 字符袋
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

    # belief：已出现领域 multi-hot + 上一活跃领域 one-hot(+none)
    def belief_vec(seen: list[str], last: str | None) -> np.ndarray:
        v = np.zeros(2 * n_classes + 1, dtype=np.float32)
        for dmn in seen:
            if dmn in didx:
                v[didx[dmn]] = 1.0
        if last is not None and last in didx:
            v[n_classes + didx[last]] = 1.0
        else:
            v[-1] = 1.0  # 无上一活跃领域（对话开头）
        return v

    U = np.stack([bow(s["utterance"]) for s in samples])
    B = np.stack([belief_vec(s["seen"], s["last"]) for s in samples])
    X_blind = U
    X_ctx = np.concatenate([U, B], axis=1)

    tr, va = stratified_split(y, n_classes, args.seed)
    us_val = underspec[va]
    dist = {dmn: int((y == i).sum()) for dmn, i in didx.items()}

    print(
        f"[domain] device={device} samples={len(samples)} domains={domains} "
        f"vocab={len(vocab)} underspec_frac={float(underspec.mean()):.3f}",
        flush=True,
    )
    print("[domain] 训练 blind 臂 ...", flush=True)
    b_overall, b_us = train_eval(X_blind, y, tr, va, us_val, n_classes, device, args.seed, args.epochs)
    print(f"[domain] blind overall={b_overall:.4f} underspec={b_us:.4f}", flush=True)
    print("[domain] 训练 context-aware 臂 ...", flush=True)
    c_overall, c_us = train_eval(X_ctx, y, tr, va, us_val, n_classes, device, args.seed, args.epochs)
    print(f"[domain] ctx   overall={c_overall:.4f} underspec={c_us:.4f}", flush=True)

    us_delta = c_us - b_us
    overall_delta = c_overall - b_overall
    if us_delta > 0.2:
        verdict = "PASS: 真实数据上 belief 在领域追踪环节有强 headroom（跟进句的领域由状态决定）"
    elif us_delta > 0.05:
        verdict = "PARTIAL: belief 在领域追踪上有正向 headroom"
    else:
        verdict = "WEAK: belief 在领域追踪上 headroom 不明显"

    result = {
        "dataset": "CrossWOZ (real human-annotated task dialogue)",
        "task": "domain tracking (predict user-turn domain)",
        "data": args.data,
        "n_dialogues": len(dialogues),
        "n_samples": len(samples),
        "domains": domains,
        "class_dist": dist,
        "underspecified_frac": round(float(underspec.mean()), 4),
        "underspecified_count_val": int(us_val.sum()),
        "overall": {"context_blind": round(b_overall, 4), "context_aware": round(c_overall, 4), "delta": round(overall_delta, 4)},
        "underspecified_subset": {"context_blind": round(b_us, 4), "context_aware": round(c_us, 4), "delta": round(us_delta, 4)},
        "verdict": verdict,
        "note": (
            "唯一变量=belief(已出现领域 multi-hot + 上一活跃领域 one-hot)。领域未明示子集=utterance "
            "不含目标领域名（餐馆/酒店/景点/地铁/出租），即跟进句，其领域只能由对话状态决定。"
            "与 B2(动作类型 belief 无 headroom) 互补：belief 在真实数据上的价值在状态/领域追踪，不在动作类型。"
        ),
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# B2b · 真实任务对话(CrossWOZ)领域追踪：belief 的真实 headroom",
        "",
        f"- 判定：**{verdict}**",
        f"- 数据：`{args.data}`（{result['n_dialogues']} 对话 / {result['n_samples']} 用户轮）",
        f"- 领域：{domains}；类别分布：{dist}",
        f"- 领域未明示子集占比：{result['underspecified_frac']}（验证集 {result['underspecified_count_val']} 条）",
        "",
        "> 唯一变量=是否加 belief（已出现领域 + 上一活跃领域）；同 MLP/split/train（口径同 teacher_corpus_eval）。",
        "",
        "## 总体 balanced accuracy",
        f"- 盲（只看句子）：{result['overall']['context_blind']}",
        f"- 上下文感知（句子+belief）：{result['overall']['context_aware']}",
        f"- delta：{result['overall']['delta']}",
        "",
        "## 领域未明示子集（跟进句，headroom 关键）",
        f"- 盲：{result['underspecified_subset']['context_blind']}",
        f"- 上下文感知：{result['underspecified_subset']['context_aware']}",
        f"- delta：**{result['underspecified_subset']['delta']}**",
        "",
        "## 与 B2 互补",
        "- B2（动作类型 offer/nooffer）：belief 无 headroom（−0.02），真实数据动作几乎由 utterance 决定。",
        "- B2b（领域追踪）：见上——belief 在状态/领域追踪环节才是真正决定性的。",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CrossWOZ domain-tracking belief headroom eval.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-char-freq", type=int, default=3)
    parser.add_argument("--max-dialogues", type=int, default=0, help="0=全部；调试用上限")
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print("[domain] dry-run：未训练。")
        print(f"[domain] 在真实任务对话 {args.data} 上做领域追踪：盲 vs +belief（重点领域未明示子集）。")
        print("[domain] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
