# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/nlu_intent_eval.py
======================================================
证明"学习式槽位意图 NLU"优于"关键词表"：在 held-out 改写（刻意不含关键词子串）上，
keyword 基线会判错（命中不到固定词→none），学习式模型能从训练里的多样表达泛化识别。

判定：学习式在 held-out 改写上的意图准确率显著高于 keyword 基线。
默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.active_inference.experiments.nlu_intent_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.active_inference.nlu.slot_intent_nlu import SlotIntentNLU
from fe_llm.active_inference.observation import BOOKING_TERMS, HOTEL_TERMS, REMINDER_TERMS

REPORT_JSON = os.path.join("docs", "reports", "nlu_intent_eval.json")
REPORT_MD = os.path.join("docs", "reports", "nlu_intent_eval.md")

CITIES = ["北京", "上海", "广州", "深圳", "杭州"]
TIMES = ["明天8点", "周末", "下午3点", "后天", "晚上9点"]

# 训练表达（含关键词形式 + 多样化）
TRAIN = {
    "booking": ["帮我订票", "订一下票", "买票去{c}", "预订车票", "订张机票", "帮我订{c}的票", "买张到{c}的票"],
    "hotel": ["帮我订酒店", "订间房", "预订旅馆", "订个民宿", "帮我订{c}的酒店", "订{c}的房"],
    "reminder": ["提醒我{t}开会", "设个提醒", "定个闹钟{t}", "提醒一下{t}", "加个提醒{t}"],
    "none": ["今天心情不错", "讲个笑话", "你好啊", "谢谢你的帮助", "陪我聊聊天", "解释一下原理"],
}
# held-out 改写（刻意不含上面任何关键词子串，keyword 基线会漏判）
HELDOUT = {
    "booking": ["想坐高铁去{c}", "来一张到{c}的车", "搞张去{c}的高铁", "怎么购到{c}的车次", "去{c}的火车帮我安排"],
    "hotel": ["在{c}住一晚", "找个{c}的住处", "{c}有合适的客栈吗", "安排个{c}的宾馆", "{c}过夜的地方"],
    "reminder": ["{t}叫我起床", "别让我忘了{t}的事", "到{t}喊我一声", "记得{t}通知我", "{t}催我一下"],
    "none": ["天气真好", "随便聊聊", "你在干嘛", "早上好呀", "今天过得不错"],
}


def _expand(spec: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for intent, templates in spec.items():
        for tpl in templates:
            if "{c}" in tpl:
                for c in CITIES:
                    texts.append(tpl.format(c=c)); labels.append(intent)
            elif "{t}" in tpl:
                for t in TIMES:
                    texts.append(tpl.format(t=t)); labels.append(intent)
            else:
                texts.append(tpl); labels.append(intent)
    return texts, labels


def keyword_intent(text: str) -> str:
    if any(t in text for t in REMINDER_TERMS):
        return "reminder"
    if any(t in text for t in HOTEL_TERMS):
        return "hotel"
    if any(t in text for t in BOOKING_TERMS):
        return "booking"
    return "none"


def _accuracy(preds: list[str], labels: list[str]) -> float:
    return sum(p == y for p, y in zip(preds, labels)) / max(len(labels), 1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Learned slot-intent NLU vs keyword baseline on held-out paraphrases.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-nlu", default="", help="可选：把训练好的 NLU 存到该路径")
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[nlu-eval] dry-run：未训练。")
    print("[nlu-eval] 对照 学习式NLU vs keyword 基线，比 held-out 改写（不含关键词）的意图准确率。")
    print("[nlu-eval] 真正运行请显式追加 --run。")


def run(args: argparse.Namespace) -> dict:
    train_texts, train_labels = _expand(TRAIN)
    held_texts, held_labels = _expand(HELDOUT)

    nlu = SlotIntentNLU().fit(train_texts, train_labels, epochs=args.epochs, seed=args.seed)
    learned_preds = [nlu.predict(t) for t in held_texts]
    keyword_preds = [keyword_intent(t) for t in held_texts]
    learned_acc = _accuracy(learned_preds, held_labels)
    keyword_acc = _accuracy(keyword_preds, held_labels)

    # 同时确认学习式没有牺牲训练域（应≈1.0）
    train_acc = _accuracy([nlu.predict(t) for t in train_texts], train_labels)

    if args.save_nlu:
        nlu.save(args.save_nlu)

    delta = learned_acc - keyword_acc
    verdict = "PASS: 学习式 NLU 在改写上显著优于关键词表" if delta > 0.2 else "FAIL: 学习式未明显优于关键词"
    result = {
        "n_train": len(train_texts),
        "n_heldout": len(held_texts),
        "train_acc_learned": round(train_acc, 4),
        "heldout_acc_learned": round(learned_acc, 4),
        "heldout_acc_keyword": round(keyword_acc, 4),
        "delta": round(delta, 4),
        "verdict": verdict,
        "note": "held-out 改写刻意不含关键词子串；keyword 基线只能命中固定词→漏判，学习式从训练多样表达泛化。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# FE-LLM 学习式槽位意图 NLU vs 关键词表（held-out 改写）",
        "",
        f"- 判定：**{verdict}**",
        f"- held-out 改写意图准确率：学习式 {result['heldout_acc_learned']} vs 关键词 {result['heldout_acc_keyword']}（delta {result['delta']}）",
        f"- 学习式训练域准确率：{result['train_acc_learned']}（确认未牺牲已知表达）",
        f"- 训练 {result['n_train']} / held-out {result['n_heldout']}",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print_dry_run(args)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
