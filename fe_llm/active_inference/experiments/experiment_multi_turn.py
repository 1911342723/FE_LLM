"""实验 B：多轮闭环验证——对外行动（追问/记忆）能否降低后续自由能。

这是 FE-LLM 路线的核心证据实验：
- 模型在模糊输入下选择 ask_clarification（对外行动）；
- 用户补充信息后，prediction error 应被预期验证所压低，surprise 显著下降；
- 负对照：用户继续模糊输入时，surprise 不应下降；
- 记忆闭环：update_memory 后的轮次应能召回记忆并影响回答文本。

输出 docs/reports/multi_turn_surprise.{md,json}。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.active_inference import ActiveInferenceController, ActionType


# 每条用例：多轮 prompt + 每轮期望动作 + 期望的 surprise 走向。
# expectation: "drop" 表示该轮 surprise 应比上一轮明显下降；"stay_high" 表示不应明显下降。
DIALOGUE_CASES: list[dict[str, Any]] = [
    {
        "name": "vague_then_clarified_report",
        "turns": [
            {"text": "帮我写一下", "expected_action": ActionType.ASK_CLARIFICATION},
            {
                "text": "帮我写一份给老板的项目周报，大约两百字",
                "expected_action": ActionType.ANSWER,
                "expectation": "drop",
            },
        ],
    },
    {
        "name": "vague_then_clarified_translation",
        "turns": [
            {"text": "帮我弄一下", "expected_action": ActionType.ASK_CLARIFICATION},
            {
                "text": "把这段话翻译成英文：早上好，今天的会议改到三点",
                "expected_action": ActionType.ANSWER,
                "expectation": "drop",
            },
        ],
    },
    {
        "name": "conflict_then_corrected",
        "turns": [
            {"text": "我昨天明天去了北京", "expected_action": ActionType.ASK_CLARIFICATION},
            {
                "text": "抱歉说错了，我是昨天去了北京，想问当地有什么好玩的",
                "expected_action": ActionType.ANSWER,
                "expectation": "drop",
            },
        ],
    },
    {
        # 负对照：持续模糊，surprise 不应下降。
        "name": "vague_then_still_vague",
        "turns": [
            {"text": "帮我写一下", "expected_action": ActionType.ASK_CLARIFICATION},
            {
                "text": "帮我弄一下",
                "expected_action": ActionType.ASK_CLARIFICATION,
                "expectation": "stay_high",
            },
        ],
    },
    {
        # 记忆闭环：第二轮应召回偏好并体现在回答里。
        "name": "memory_then_recalled",
        "turns": [
            {"text": "记住我喜欢简短回答", "expected_action": ActionType.UPDATE_MEMORY},
            {
                "text": "给我讲讲什么是自由能原理",
                "expected_action": ActionType.ANSWER,
                "expects_memory_recall": True,
            },
        ],
    },
]

# surprise 下降的判定阈值：相对降幅至少 25% 才算有效平复。
DROP_RATIO_THRESHOLD = 0.25


def run_case(controller: ActiveInferenceController, case: dict[str, Any], session_id: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    prev_surprise: float | None = None
    checks: list[dict[str, Any]] = []
    for index, turn in enumerate(case["turns"]):
        response = controller.respond(turn["text"], session_id=session_id)
        surprise = float(response.surprise_score.total)
        row = {
            "turn": index + 1,
            "text": turn["text"],
            "selected_action": response.selected_action_type.value,
            "expected_action": turn["expected_action"].value,
            "surprise": surprise,
            "prediction_error": response.prediction_error.to_dict(),
            "recalled_memories": [item.text for item in response.recalled_memories or []],
            "response_text": response.text,
        }
        rows.append(row)

        checks.append(
            {
                "kind": "action",
                "turn": index + 1,
                "passed": response.selected_action_type == turn["expected_action"],
                "detail": f"expected={turn['expected_action'].value} selected={response.selected_action_type.value}",
            }
        )
        expectation = turn.get("expectation")
        if expectation and prev_surprise is not None:
            drop_ratio = (prev_surprise - surprise) / max(prev_surprise, 1e-8)
            if expectation == "drop":
                passed = drop_ratio >= DROP_RATIO_THRESHOLD
            else:  # stay_high
                passed = drop_ratio < DROP_RATIO_THRESHOLD
            checks.append(
                {
                    "kind": f"surprise_{expectation}",
                    "turn": index + 1,
                    "passed": passed,
                    "detail": f"prev={prev_surprise:.3f} now={surprise:.3f} drop_ratio={drop_ratio:.2%}",
                }
            )
        if turn.get("expects_memory_recall"):
            recalled = bool(response.recalled_memories)
            applied = "偏好" in response.text
            checks.append(
                {
                    "kind": "memory_recall",
                    "turn": index + 1,
                    "passed": recalled and applied,
                    "detail": f"recalled={recalled} applied_in_text={applied}",
                }
            )
        prev_surprise = surprise
    return {
        "name": case["name"],
        "rows": rows,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


def make_report(results: list[dict[str, Any]]) -> str:
    total_checks = sum(len(item["checks"]) for item in results)
    passed_checks = sum(1 for item in results for check in item["checks"] if check["passed"])
    drop_checks = [
        check for item in results for check in item["checks"] if check["kind"] == "surprise_drop"
    ]
    lines = [
        "# FE-LLM 实验 B：多轮 surprise 平复评测",
        "",
        "验证主动推理核心命题：对外行动（追问/记忆更新）改变环境后，后续观测的自由能应当下降。",
        "",
        f"- 用例数：{len(results)}（通过 {sum(1 for item in results if item['passed'])}）",
        f"- 检查点：{passed_checks}/{total_checks}",
        f"- surprise 下降判定阈值：相对降幅 ≥ {DROP_RATIO_THRESHOLD:.0%}",
        f"- 澄清满足后 surprise 下降通过：{sum(1 for c in drop_checks if c['passed'])}/{len(drop_checks)}",
        "",
        "## 各用例轨迹",
        "",
    ]
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(f"### {item['name']} [{status}]")
        lines.append("")
        lines.append("| turn | prompt | action | surprise | 召回记忆 |")
        lines.append("|---:|---|---|---:|---|")
        for row in item["rows"]:
            memories = "; ".join(row["recalled_memories"]) or "-"
            lines.append(
                f"| {row['turn']} | {row['text']} | `{row['selected_action']}` | "
                f"{row['surprise']:.3f} | {memories} |"
            )
        lines.append("")
        for check in item["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(f"- [{mark}] {check['kind']} (turn {check['turn']}): {check['detail']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=os.path.join("docs", "reports", "multi_turn_surprise.json"))
    ap.add_argument("--markdown-out", default=os.path.join("docs", "reports", "multi_turn_surprise.md"))
    ap.add_argument("--no-intent-model", action="store_true")
    args = ap.parse_args()

    results = []
    for index, case in enumerate(DIALOGUE_CASES):
        # 每条用例独立 controller + 独立 session，避免跨用例信念污染。
        controller = ActiveInferenceController(
            use_intent_model=not args.no_intent_model,
            use_energy_decoder=False,
            memory_candidate_path=None,
        )
        results.append(run_case(controller, case, session_id=f"exp-b-{index}"))

    report = make_report(results)
    print(report)
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    if args.markdown_out:
        os.makedirs(os.path.dirname(args.markdown_out), exist_ok=True)
        with open(args.markdown_out, "w", encoding="utf-8") as f:
            f.write(report)


if __name__ == "__main__":
    main()
