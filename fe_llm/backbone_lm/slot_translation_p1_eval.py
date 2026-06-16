"""P1 evaluator for frozen-backbone FE-LLM experiments.

The evaluator consumes prediction JSONL produced by A/B/C runs and applies the
same word-F1 / char-F1口径 as M2 translation reports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fe_llm.energy_lm.evaluation.translation_eval import char_f1, word_f1

DEFAULT_PRED_PATH = os.path.join("docs", "reports", "slot_translation_p1_predictions.jsonl")
REPORT_JSON = os.path.join("docs", "reports", "slot_translation_p1_eval.json")
REPORT_MD = os.path.join("docs", "reports", "slot_translation_p1_eval.md")
PASS_WORD_F1 = 0.3


@dataclass(frozen=True)
class P1Prediction:
    group: str
    zh: str
    ref: str
    pred: str
    residual_start: float | None = None
    residual_end: float | None = None
    coverage_start: float | None = None
    coverage_end: float | None = None
    disagreement_rate: float | None = None


def _optional_float(item: dict[str, Any], key: str) -> float | None:
    value = item.get(key)
    if value is None:
        return None
    return float(value)


def load_predictions(path: str) -> list[P1Prediction]:
    rows: list[P1Prediction] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                rows.append(
                    P1Prediction(
                        group=item["group"],
                        zh=item["zh"],
                        ref=item["ref"],
                        pred=item["pred"],
                        residual_start=_optional_float(item, "residual_start"),
                        residual_end=_optional_float(item, "residual_end"),
                        coverage_start=_optional_float(item, "coverage_start"),
                        coverage_end=_optional_float(item, "coverage_end"),
                        disagreement_rate=_optional_float(item, "disagreement_rate"),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return rows


def score_prediction(row: P1Prediction) -> dict[str, Any]:
    residual_descent = (
        row.residual_start is not None
        and row.residual_end is not None
        and row.residual_end < row.residual_start
    )
    coverage_descent = (
        row.coverage_start is not None
        and row.coverage_end is not None
        and row.coverage_end < row.coverage_start
    )
    return {
        "group": row.group,
        "zh": row.zh,
        "ref": row.ref,
        "pred": row.pred,
        "word_f1": round(word_f1(row.pred, row.ref), 4),
        "char_f1": round(char_f1(row.pred, row.ref), 4),
        "exact": row.pred.strip() == row.ref.strip(),
        "residual_descent": residual_descent,
        "coverage_descent": coverage_descent,
        "disagreement_rate": row.disagreement_rate,
    }


def summarize(scored_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups = sorted({row["group"] for row in scored_rows})
    summary: dict[str, dict[str, Any]] = {}
    for group in groups:
        rows = [row for row in scored_rows if row["group"] == group]
        disagreement = [row["disagreement_rate"] for row in rows if row["disagreement_rate"] is not None]
        summary[group] = {
            "n": len(rows),
            "mean_word_f1": round(mean(row["word_f1"] for row in rows), 4),
            "mean_char_f1": round(mean(row["char_f1"] for row in rows), 4),
            "exact_match": sum(1 for row in rows if row["exact"]),
            "unique_outputs": len({row["pred"] for row in rows}),
            "word_f1_ge_0_5": sum(1 for row in rows if row["word_f1"] >= 0.5),
            "word_f1_eq_0": sum(1 for row in rows if row["word_f1"] == 0.0),
            "residual_descent_rate": round(
                sum(1 for row in rows if row["residual_descent"]) / max(len(rows), 1),
                4,
            ),
            "coverage_descent_rate": round(
                sum(1 for row in rows if row["coverage_descent"]) / max(len(rows), 1),
                4,
            ),
            "mean_disagreement_rate": round(mean(disagreement), 4) if disagreement else None,
        }
    return summary


def n1_verdict(summary: dict[str, dict[str, Any]]) -> str:
    if "B" not in summary:
        return "INCOMPLETE: missing B group"
    b = summary["B"]["mean_word_f1"]
    a = summary.get("A", {}).get("mean_word_f1")
    c = summary.get("C", {}).get("mean_word_f1")
    if b < PASS_WORD_F1:
        return "FAIL: B below 0.3 word-F1"
    if a is None or c is None:
        return "INCOMPLETE: missing A or C control"
    if b <= a or b <= c:
        return "FAIL: B does not beat A/C controls"
    return "PASS: B reaches threshold and beats A/C"


def render_markdown(summary: dict[str, dict[str, Any]], rows: list[dict[str, Any]], verdict: str) -> str:
    lines = [
        "# FE-LLM N1 P1 评估：冻结底座 + FE 机制层",
        "",
        f"- 判定：**{verdict}**",
        f"- 通过标准：B 组 mean word-F1 ≥ {PASS_WORD_F1}，且高于 A/C 对照。",
        "",
        "## 分组汇总",
        "",
        "| 组别 | n | word-F1 | char-F1 | exact | unique | residual↓ | coverage↓ | disagreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in sorted(summary):
        item = summary[group]
        disagreement = item["mean_disagreement_rate"]
        disagreement_text = "-" if disagreement is None else f"{disagreement:.1%}"
        lines.append(
            f"| {group} | {item['n']} | {item['mean_word_f1']:.3f} | "
            f"{item['mean_char_f1']:.3f} | {item['exact_match']} | {item['unique_outputs']} | "
            f"{item['residual_descent_rate']:.0%} | {item['coverage_descent_rate']:.0%} | "
            f"{disagreement_text} |"
        )
    lines.extend(
        [
            "",
            "## 样例",
            "",
            "| 组别 | 中文 | 参考 | 生成 | word-F1 |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in rows[:30]:
        lines.append(
            f"| {row['group']} | {row['zh']} | {row['ref']} | {row['pred']} | {row['word_f1']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FE-LLM P1 A/B/C prediction JSONL.")
    parser.add_argument("--run", action="store_true", help="真正读取预测文件并生成报告；默认只 dry-run。")
    parser.add_argument("--pred-path", default=DEFAULT_PRED_PATH)
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[p1-eval] dry-run：未读取预测文件，未生成报告。")
    print(f"[p1-eval] pred_path = {args.pred_path}")
    print("[p1-eval] 预测 JSONL 每行需要字段：group, zh, ref, pred。")
    print("[p1-eval] 可选字段：residual_start/end, coverage_start/end, disagreement_rate。")
    print("[p1-eval] 真正评估请显式追加 --run。")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    predictions = load_predictions(args.pred_path)
    scored = [score_prediction(row) for row in predictions]
    summary = summarize(scored)
    verdict = n1_verdict(summary)
    report = render_markdown(summary, scored, verdict)

    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": scored, "verdict": verdict}, f, ensure_ascii=False, indent=2)
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    return {"summary": summary, "rows": scored, "verdict": verdict}


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.run:
        print_dry_run(args)
        return 0
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
