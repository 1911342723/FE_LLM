# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/crosswoz_headroom_eval.py
============================================================
B2：在**真实人标任务型对话** CrossWOZ（清华 thu-coai）上验证控制层 headroom——
把 `teacher_corpus_eval` 的「上下文感知(句子+belief) vs 盲(只看句子)」方法论原样搬到
真实数据，回答关键开放问题：belief/槽位机制在真实任务对话上是否也带来 headroom？

对照锚点：
- 教师合成任务语料：歧义子集 belief 0.49→1.0（delta +0.51，强）；
- 真实开放闲聊 LCCC：belief pairwise 0.655（弱，开放闲聊高熵）；
- 本实验：真实**任务型**对话，预期介于两者、且明显强于开放闲聊。

口径（与 `teacher_corpus_eval` 完全一致，保证可比）：
- 复用其 MLP / balanced_accuracy / stratified_split / train_eval（唯一变量=是否加 belief）；
- utterance = 用户当前轮文本；
- action（系统该 ask 还是 answer）= 系统轮 dialog_act：含 Request → ask_clarification，否则 answer；
- belief = 当前轮**之前**累积的用户已告知 (domain·slot)（= 决策前 known_slots）；
- 歧义子集 = 同一 utterance 文本对应了多于一种 action（只看句子无法区分）。

数据：`data/crosswoz/train.json.zip`（CrossWOZ，Apache-2.0）。不在仓库则先下载：
    curl -L -o data/crosswoz/train.json.zip \
      https://raw.githubusercontent.com/thu-coai/CrossWOZ/master/data/crosswoz/train.json.zip
默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.active_inference.experiments.crosswoz_headroom_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from fe_llm.active_inference.experiments.teacher_corpus_eval import (
    stratified_split,
    train_eval,
)
from fe_llm.config import get_device

DATA = os.path.join("data", "crosswoz", "train.json.zip")
REPORT_JSON = os.path.join("docs", "reports", "crosswoz_headroom_eval.json")
REPORT_MD = os.path.join("docs", "reports", "crosswoz_headroom_eval.md")

# CrossWOZ 里系统从不 Request（用户向系统提问、系统只 Inform/Recommend/NoOffer）。
# 所以测两种系统决策方案（都只在实质轮上，纯 General/空 跳过）：
#   - binary：offer(Inform/Recommend) vs nooffer(NoOffer)——能否满足累积约束；
#   - triple：inform vs recommend vs nooffer——recommend 偏早(首次推荐)、inform 偏晚(追问细节)，
#             可能有 belief/轮深依赖，用作 belief 价值的鲁棒性对照。
# 唯一变量始终=是否加 belief；两方案共用同一份 utterance/belief 特征。
SUBSTANTIVE_INTENTS = {"Inform", "Recommend", "NoOffer"}
ACTION_SCHEMES = {
    "binary_offer_nooffer": ["offer", "nooffer"],
    "triple_inform_recommend_nooffer": ["inform", "recommend", "nooffer"],
}


def _label_binary(intents: set[str]) -> str:
    return "nooffer" if "NoOffer" in intents else "offer"


def _label_triple(intents: set[str]) -> str:
    if "NoOffer" in intents:
        return "nooffer"
    if "Recommend" in intents:
        return "recommend"
    return "inform"


SCHEME_LABELERS = {
    "binary_offer_nooffer": _label_binary,
    "triple_inform_recommend_nooffer": _label_triple,
}


def load_dialogues(path: str) -> list[dict]:
    """从 CrossWOZ 的 .json 或 .json.zip 读出对话字典列表。"""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if n.endswith(".json")][0]
            data = json.loads(z.read(name).decode("utf-8"))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return list(data.values())


def _sys_substantive_intents(dialog_act: list) -> set[str] | None:
    """系统轮的实质 intents（Inform/Recommend/NoOffer 的子集）；纯 General/空→None(跳过)。"""
    intents = {act[0] for act in (dialog_act or []) if act}
    sub = intents & SUBSTANTIVE_INTENTS
    return sub or None


def _user_informed_slots(dialog_act: list) -> list[str]:
    """用户本轮告知的 (domain·slot) 键（Inform 动作）。"""
    keys = []
    for act in dialog_act or []:
        # act = [intent, domain, slot, value]
        if act and act[0] == "Inform" and len(act) >= 3:
            keys.append(f"{act[1]}·{act[2]}")
    return keys


def extract_samples(dialogues: list[dict]) -> list[dict]:
    """遍历对话，抽 (utterance, belief_keys_before, action) 样本。

    belief = 当前用户轮**之前**累积的用户已告知槽位（决策前 known_slots）。
    """
    samples = []
    for d in dialogues:
        msgs = d.get("messages", [])
        belief: set[str] = set()
        i = 0
        while i < len(msgs) - 1:
            cur, nxt = msgs[i], msgs[i + 1]
            if cur.get("role") == "usr" and nxt.get("role") == "sys":
                intents = _sys_substantive_intents(nxt.get("dialog_act"))
                if intents is not None:  # 跳过纯寒暄/收尾的非控制决策轮
                    samples.append(
                        {
                            "utterance": (cur.get("content") or "").strip(),
                            "belief": sorted(belief),  # 决策前（不含当前轮 informs）
                            "intents": sorted(intents),
                        }
                    )
                # 记录后再把当前用户轮的 informs 并入 belief（供后续轮使用）。
                belief.update(_user_informed_slots(cur.get("dialog_act")))
                i += 2
            else:
                # 非 usr→sys 相邻（如连续同角色），仍累积用户 informs。
                if cur.get("role") == "usr":
                    belief.update(_user_informed_slots(cur.get("dialog_act")))
                i += 1
    return [s for s in samples if s["utterance"]]


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    dialogues = load_dialogues(args.data)
    if args.max_dialogues > 0:
        dialogues = dialogues[: args.max_dialogues]
    samples = extract_samples(dialogues)
    if len(samples) < 50:
        raise RuntimeError(f"样本太少：{len(samples)}（检查数据 {args.data}）")

    # ── 共享特征：utterance 字符袋 + belief 向量（两方案唯一变量都只是是否加 belief）──
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

    slot_keys = sorted({k for s in samples for k in s["belief"]})
    skidx = {k: i for i, k in enumerate(slot_keys)}

    def belief_vec(keys: list[str]) -> np.ndarray:
        v = np.zeros(len(slot_keys) + 1, dtype=np.float32)
        for k in keys:
            if k in skidx:
                v[skidx[k]] = 1.0
        v[-1] = min(len(keys), 8) / 8.0
        return v

    U = np.stack([bow(s["utterance"]) for s in samples])
    B = np.stack([belief_vec(s["belief"]) for s in samples])
    X_blind = U
    X_ctx = np.concatenate([U, B], axis=1)
    print(
        f"[crosswoz] device={device} samples={len(samples)} vocab={len(vocab)} slot_keys={len(slot_keys)}",
        flush=True,
    )

    # ── 两种动作方案各跑 blind vs context-aware ──
    schemes_out: dict[str, dict] = {}
    for scheme, classes in ACTION_SCHEMES.items():
        labeler = SCHEME_LABELERS[scheme]
        labels = [labeler(set(s["intents"])) for s in samples]
        cls_idx = {c: i for i, c in enumerate(classes)}
        n_classes = len(classes)
        y = np.array([cls_idx[lab] for lab in labels], dtype=np.int64)
        text_actions = defaultdict(set)
        for s, lab in zip(samples, labels):
            text_actions[s["utterance"]].add(lab)
        ambiguous = np.array([len(text_actions[s["utterance"]]) > 1 for s in samples], dtype=bool)
        tr, va = stratified_split(y, n_classes, args.seed)
        amb_val = ambiguous[va]
        dist = {c: int((y == i).sum()) for c, i in cls_idx.items()}
        b_o, b_a = train_eval(X_blind, y, tr, va, amb_val, n_classes, device, args.seed, args.epochs)
        c_o, c_a = train_eval(X_ctx, y, tr, va, amb_val, n_classes, device, args.seed, args.epochs)
        print(
            f"[crosswoz][{scheme}] dist={dist} amb_frac={float(ambiguous.mean()):.4f} | "
            f"blind o={b_o:.4f} a={b_a:.4f} | ctx o={c_o:.4f} a={c_a:.4f} | overall_delta={c_o - b_o:+.4f}",
            flush=True,
        )
        schemes_out[scheme] = {
            "classes": classes,
            "class_dist": dist,
            "ambiguous_frac": round(float(ambiguous.mean()), 4),
            "ambiguous_count_val": int(amb_val.sum()),
            "overall": {"context_blind": round(b_o, 4), "context_aware": round(c_o, 4), "delta": round(c_o - b_o, 4)},
            "ambiguous_subset": {"context_blind": round(b_a, 4), "context_aware": round(c_a, 4), "delta": round(c_a - b_a, 4)},
        }

    best_delta = max(s["overall"]["delta"] for s in schemes_out.values())
    if best_delta > 0.05:
        verdict = "PARTIAL/PASS: 真实任务对话上 belief 有正向 headroom"
    elif best_delta > 0.01:
        verdict = "WEAK+: belief 仅微弱正向"
    else:
        verdict = (
            "WEAK: 真实任务对话上 belief 对系统动作预测无 headroom"
            "（同句多动作≈0，系统动作几乎由当前 utterance 决定）"
        )

    result = {
        "dataset": "CrossWOZ (real human-annotated task dialogue)",
        "data": args.data,
        "n_dialogues": len(dialogues),
        "n_samples": len(samples),
        "vocab": len(vocab),
        "slot_keys": len(slot_keys),
        "schemes": schemes_out,
        "verdict": verdict,
        "anchors": {"teacher_synthetic_amb_delta": 0.51, "lccc_open_chat_belief": 0.655},
        "note": (
            "真实人标任务对话；唯一变量=belief(决策前累积用户 informed (domain·slot) multi-hot)。"
            "口径与 teacher_corpus_eval 一致（同 MLP/split/train）。CrossWOZ 系统从不 Request，故测"
            " offer/nooffer（binary）与 inform/recommend/nooffer（triple）。对照：教师合成歧义子集 +0.51、"
            "开放闲聊 LCCC 0.655。"
        ),
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    lines = [
        "# B2 · 真实任务对话(CrossWOZ)控制层 headroom（上下文感知 vs 盲）",
        "",
        f"- 判定：**{verdict}**",
        f"- 数据：`{args.data}`（{result['n_dialogues']} 对话 / {result['n_samples']} usr→sys 实质轮）",
        f"- vocab={result['vocab']}，belief 槽位键={result['slot_keys']}",
        "",
        "> 唯一变量=是否加 belief（决策前累积用户 informed (domain·slot)）；同 MLP/split/train（口径同 teacher_corpus_eval）。",
        "> CrossWOZ 系统从不 Request，故测 offer/nooffer 与 inform/recommend/nooffer 两种动作方案。",
        "",
    ]
    for scheme, out in schemes_out.items():
        lines += [
            f"## 方案 {scheme}",
            f"- 类别分布：{out['class_dist']}，歧义占比 {out['ambiguous_frac']}",
            f"- 总体 balanced_acc：盲 {out['overall']['context_blind']} → 感知 {out['overall']['context_aware']}（delta **{out['overall']['delta']:+.4f}**）",
            f"- 歧义子集：盲 {out['ambiguous_subset']['context_blind']} → 感知 {out['ambiguous_subset']['context_aware']}（delta {out['ambiguous_subset']['delta']:+.4f}）",
            "",
        ]
    lines += [
        "## 对照锚点",
        "- 教师合成任务语料歧义子集 belief delta：+0.51（强，但属构造特性）",
        "- 真实开放闲聊 LCCC belief：0.655（弱）",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CrossWOZ real-data control-layer headroom eval.")
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
        print("[crosswoz] dry-run：未训练。")
        print(f"[crosswoz] 在真实任务对话 {args.data} 上：上下文感知(句子+belief) vs 盲(只看句子)。")
        print("[crosswoz] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
