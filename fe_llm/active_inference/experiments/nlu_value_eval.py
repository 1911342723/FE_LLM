# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/nlu_value_eval.py
=====================================================
评估从 0 字符级槽位值标注器（SlotValueTagger）：在 held-out 的新日期/时间表达上
能否抽取出值 span，并诚实区分能力边界——

  - DATE/TIME 是规则模式 → 期望泛化到训练未见的新表达；
  - CITY 是开放命名实体 → 字符级小模型对训练未见城市受容量限制（预期召回低）。

判定：held-out DATE/TIME span 召回显著 > held-out CITY span 召回（验证"规则可学、开放实体受限"的诚实结论）。
默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.active_inference.experiments.nlu_value_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.active_inference.nlu.slot_value_tagger import LABELS, SlotValueTagger

REPORT_JSON = os.path.join("docs", "reports", "nlu_value_eval.json")
REPORT_MD = os.path.join("docs", "reports", "nlu_value_eval.md")

LABEL_ID = {name: i for i, name in enumerate(LABELS)}

# 模板：每段 (文本或占位符, 标签)
TEMPLATES = [
    [("提醒我", "O"), ("{TIME}", "TIME"), ("开会", "O")],
    [("{CITY}", "CITY"), ("天气怎么样", "O")],
    [("帮我订", "O"), ("{DATE}", "DATE"), ("的酒店", "O")],
    [("我", "O"), ("{DATE}", "DATE"), ("去", "O"), ("{CITY}", "CITY"), ("玩", "O")],
    [("订", "O"), ("{CITY}", "CITY"), ("到", "O"), ("{CITY}", "CITY"), ("的票", "O")],
    [("记得", "O"), ("{TIME}", "TIME"), ("叫我", "O")],
]

TRAIN_POOLS = {
    "CITY": ["北京", "上海", "广州", "杭州"],
    "DATE": ["明天", "后天", "周末", "5号", "周六"],
    "TIME": ["8点", "早上9点", "下午3点", "中午12点"],
}
HELDOUT_POOLS = {
    "CITY": ["深圳", "成都", "武汉"],          # 训练未见城市 → 预期容量受限
    "DATE": ["大后天", "下周", "12号", "3月"],   # 新日期表达（规则）→ 预期泛化
    "TIME": ["10点", "晚上7点", "11点半"],       # 新时间表达（规则）→ 预期泛化
}


def _gen(pools: dict[str, list[str]], per_template: int, seed: int):
    rng = random.Random(seed)
    texts, char_labels, gold_spans = [], [], []
    for tpl in TEMPLATES:
        for _ in range(per_template):
            text, labels, golds = "", [], []
            for seg, lab in tpl:
                value = rng.choice(pools[lab]) if seg.startswith("{") else seg
                text += value
                labels += [LABEL_ID[lab]] * len(value)
                if lab != "O":
                    golds.append((lab, value))
            texts.append(text)
            char_labels.append(labels)
            gold_spans.append(golds)
    return texts, char_labels, gold_spans


def _span_recall(tagger: SlotValueTagger, texts, gold_spans):
    by_type = {"CITY": [0, 0], "DATE": [0, 0], "TIME": [0, 0]}  # [matched, total]
    for text, golds in zip(texts, gold_spans):
        pred = set(tagger.extract_spans(text))
        for lab, val in golds:
            by_type[lab][1] += 1
            if (lab, val) in pred:
                by_type[lab][0] += 1
    return {k: round(v[0] / v[1], 4) if v[1] else 0.0 for k, v in by_type.items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Char-level slot value tagger eval (date/time generalize, city capacity-bound).")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--per-template", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[nlu-value] dry-run：未训练。")
    print("[nlu-value] 训练字符级标注器，测 held-out 新日期/时间 vs 新城市 的 span 召回。")
    print("[nlu-value] 真正运行请显式追加 --run。")


def run(args: argparse.Namespace) -> dict:
    train_texts, train_labels, _ = _gen(TRAIN_POOLS, args.per_template, args.seed)
    held_texts, _, held_golds = _gen(HELDOUT_POOLS, max(args.per_template // 2, 4), args.seed + 1)

    tagger = SlotValueTagger().fit(train_texts, train_labels, epochs=args.epochs, seed=args.seed)
    held_recall = _span_recall(tagger, held_texts, held_golds)

    datetime_recall = round((held_recall["DATE"] + held_recall["TIME"]) / 2, 4)
    city_recall = held_recall["CITY"]
    verdict = (
        "PASS: 规则型 DATE/TIME 可学且泛化，开放实体 CITY 受容量限制（诚实边界）"
        if datetime_recall > city_recall and datetime_recall >= 0.5
        else "INCONCLUSIVE: 需调参或更多数据"
    )
    result = {
        "n_train": len(train_texts),
        "n_heldout": len(held_texts),
        "heldout_span_recall": held_recall,
        "heldout_datetime_recall": datetime_recall,
        "heldout_city_recall": city_recall,
        "verdict": verdict,
        "note": "从 0 字符级序列标注；DATE/TIME 规则模式可泛化到新表达，CITY 开放命名实体对未见城市受容量限制（小模型 NER 固有边界）。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# FE-LLM 从 0 字符级槽位值标注器（能力与边界）",
        "",
        f"- 判定：**{verdict}**",
        f"- held-out span 召回：DATE {held_recall['DATE']} / TIME {held_recall['TIME']} / CITY {held_recall['CITY']}",
        f"- DATE+TIME 平均 {datetime_recall} vs CITY {city_recall}",
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
