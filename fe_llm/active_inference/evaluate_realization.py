"""生成层评测：EnergyDecoder 接入 answer 动作后的能量轨迹质量。

度量三件事：
1. 接入率：answer 动作里有多少真正由 energy_decoder 生成（而非规则/模板回退）；
2. 能量收敛：逐字残余能量是否整体下降、单调下降步占比、平均降幅；
3. 信念贯通：意图来源是否为 belief_mixed（控制层信念真实进入了生成目标）。

输出 docs/reports/realization_eval.{md,json}。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fe_llm.active_inference import ActiveInferenceController, ActionType


# 评测用 answer 类 prompt：避开寒暄白名单（那部分走规则快速路径），
# 选与 LCCC 训练分布相近的闲聊/陈述句。
ANSWER_PROMPTS = [
    "我有点累",
    "晚安",
    "我今天很开心",
    "你吃饭了吗",
    "周末打算去爬山",
    "我喜欢这个故事的开头",
    "最近工作压力好大",
    "我想休息一下",
    "今天加班到很晚",
    "我朋友过生日送什么好",
    "刚看完一部电影",
    "外面好像下雨了",
]


def evaluate(controller: ActiveInferenceController) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, prompt in enumerate(ANSWER_PROMPTS):
        response = controller.respond(prompt, session_id=f"real-{index}")
        realization = dict(response.trace.realization or {})
        trace = realization.pop("energy_trace", [])
        row: dict[str, Any] = {
            "prompt": prompt,
            "selected_action": response.selected_action_type.value,
            "text": response.text,
            "engine": realization.get("engine"),
            "intent_source": realization.get("intent_source"),
            "decode_mode": realization.get("decode_mode"),
            "decode_disagreement_steps": realization.get("decode_disagreement_steps"),
            "decode_total_steps": realization.get("decode_total_steps"),
            "rejected_generation": realization.get("rejected_generation", {}).get("rejected")
            if isinstance(realization.get("rejected_generation"), dict)
            else None,
        }
        if trace:
            energies = [item["residual_energy"] for item in trace]
            descent_steps = sum(1 for item in trace if item["energy_drop"] > 0)
            row.update(
                {
                    "energy_start": energies[0],
                    "energy_end": energies[-1],
                    "energy_drop_ratio": round(1.0 - energies[-1] / max(energies[0], 1e-8), 4),
                    "monotonic_step_ratio": round(descent_steps / max(len(trace) - 1, 1), 4),
                    "n_chars": realization.get("n_chars"),
                }
            )
        rows.append(row)

    answer_rows = [row for row in rows if row["selected_action"] == ActionType.ANSWER.value]
    decoder_rows = [row for row in answer_rows if row["engine"] == "energy_decoder"]
    descent_rows = [row for row in decoder_rows if row.get("energy_drop_ratio", 0) > 0]
    summary = {
        "total_prompts": len(rows),
        "answer_actions": len(answer_rows),
        "energy_decoder_used": len(decoder_rows),
        "decoder_usage_rate": round(len(decoder_rows) / max(len(answer_rows), 1), 4),
        "belief_mixed_count": sum(1 for row in decoder_rows if row["intent_source"] == "belief_mixed"),
        "energy_descent_pass": len(descent_rows),
        "mean_energy_drop_ratio": round(
            sum(row["energy_drop_ratio"] for row in decoder_rows) / max(len(decoder_rows), 1), 4
        ),
        "mean_monotonic_step_ratio": round(
            sum(row["monotonic_step_ratio"] for row in decoder_rows) / max(len(decoder_rows), 1), 4
        ),
        "fallback_count": len(answer_rows) - len(decoder_rows),
        # 决策分歧率：hybrid 选字与纯 argmax logit 不同的步数占比，
        # 这是"能量信号真实参与选字"（而非概率机器）的直接证据。
        "mean_decode_disagreement_rate": round(
            sum(
                (row.get("decode_disagreement_steps") or 0) / max(row.get("decode_total_steps") or 1, 1)
                for row in decoder_rows
            )
            / max(len(decoder_rows), 1),
            4,
        ),
    }
    return {"summary": summary, "rows": rows}


def make_report(results: dict[str, Any]) -> str:
    summary = results["summary"]
    lines = [
        "# FE-LLM 生成层评测：EnergyDecoder 接入 answer 动作",
        "",
        "可溯源生成的证据：answer 由能量递减解码产生，控制层信念意图注入为生成目标，",
        "逐字残余能量轨迹随 trace 返回，未收敛的生成被门控拒绝并可溯源回退。",
        "",
        f"- answer 动作数：{summary['answer_actions']}/{summary['total_prompts']}",
        f"- energy_decoder 接入率：{summary['decoder_usage_rate']:.0%}（{summary['energy_decoder_used']}/{summary['answer_actions']}，回退 {summary['fallback_count']}）",
        f"- 信念意图注入（belief_mixed）：{summary['belief_mixed_count']}/{summary['energy_decoder_used']}",
        f"- 能量整体下降通过：{summary['energy_descent_pass']}/{summary['energy_decoder_used']}",
        f"- 平均能量降幅：{summary['mean_energy_drop_ratio']:.1%}",
        f"- 单调下降步占比：{summary['mean_monotonic_step_ratio']:.1%}",
        f"- 选字决策分歧率（hybrid vs 纯 logit argmax）：{summary['mean_decode_disagreement_rate']:.1%}",
        "",
        "## 样例明细",
        "",
        "| prompt | action | engine | 意图来源 | 能量降幅 | 单调步比 | 输出 |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in results["rows"]:
        drop = f"{row.get('energy_drop_ratio', 0):.1%}" if row.get("energy_drop_ratio") is not None else "-"
        mono = f"{row.get('monotonic_step_ratio', 0):.0%}" if row.get("monotonic_step_ratio") is not None else "-"
        lines.append(
            f"| {row['prompt']} | `{row['selected_action']}` | `{row['engine']}` | "
            f"{row.get('intent_source') or '-'} | {drop} | {mono} | {row['text']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=os.path.join("docs", "reports", "realization_eval.json"))
    ap.add_argument("--markdown-out", default=os.path.join("docs", "reports", "realization_eval.md"))
    args = ap.parse_args()

    controller = ActiveInferenceController(memory_candidate_path=None)
    results = evaluate(controller)
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
