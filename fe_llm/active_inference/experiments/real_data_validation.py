# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/real_data_validation.py
===========================================================
在真实多样对话数据（LCCC 真人中文对话）上验证控制层的 belief / surprise 是否仍有
区分力（headroom），而不是只在我们设计的合成任务上成立。

LCCC 每行 {prompt, response} 是一对真实相邻轮（A→B）。两个自动、无需人工标注的测试：

测试1（belief 预测降低连贯续接的 surprise）：
    处理真实上文 A（形成 belief）后，对"真实的下一句 B"的 surprise 应低于对"随机真实句 B'"。
    指标：mean surprise(true) < mean surprise(random) 且 pairwise 胜率 > 0.5。

测试2（surprise 检测真实表面异常）：
    真实句 vs 其字符打乱版（同字不同序），打乱版 surprise 应更高。

诚实：阳/阴结果都如实落盘——这是为了知道控制层在真实输入上到底有没有 headroom。
默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.active_inference.experiments.real_data_validation --run --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.active_inference.controller import ActiveInferenceController
from fe_llm.energy_lm.training.intent_train import load_extra_dialogues

REPORT_JSON = os.path.join("docs", "reports", "real_data_validation.json")
REPORT_MD = os.path.join("docs", "reports", "real_data_validation.md")
LCCC_PATH = os.path.join("data", "dialogue", "dialogues_lccc_highentropy.jsonl")


def _shuffle_chars(text: str, rng: random.Random) -> str:
    chars = list(text)
    rng.shuffle(chars)
    return "".join(chars)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate control layer belief/surprise on real LCCC dialogue.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--n", type=int, default=200, help="真实对话对数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default=LCCC_PATH)
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[real-val] dry-run：未运行。")
    print(f"[real-val] 在真实 LCCC（{args.data}）上验证 belief/surprise 的 headroom。")
    print("[real-val] 测试1：连贯续接 surprise < 随机；测试2：真实 vs 字符打乱。")
    print("[real-val] 真正运行请显式追加 --run。")


def run(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    pairs = load_extra_dialogues(args.data)
    if not pairs:
        raise SystemExit(f"未找到真实数据：{args.data}")
    rng.shuffle(pairs)
    pairs = pairs[: args.n]
    responses = [r for _, r in pairs]

    controller = ActiveInferenceController(memory_candidate_path=None)

    # 测试1：belief 预测降低连贯续接 surprise
    s_true, s_rand, wins = [], [], 0
    for i, (a, b_true) in enumerate(pairs):
        b_rand = responses[rng.randrange(len(responses))]
        sid_t = f"rt-{i}"
        controller.respond(a, session_id=sid_t)
        st = controller.respond(b_true, session_id=sid_t).surprise_score.total
        sid_r = f"rr-{i}"
        controller.respond(a, session_id=sid_r)
        sr = controller.respond(b_rand, session_id=sid_r).surprise_score.total
        s_true.append(st)
        s_rand.append(sr)
        if st < sr:
            wins += 1
    mean_true = sum(s_true) / len(s_true)
    mean_rand = sum(s_rand) / len(s_rand)
    win_rate = wins / len(pairs)

    # 测试2：surprise 检测真实表面异常（字符打乱）
    s_norm, s_shuf, wins2 = [], [], 0
    for i, (a, _) in enumerate(pairs):
        sn = controller.respond(a, session_id=f"n-{i}").surprise_score.total
        ss = controller.respond(_shuffle_chars(a, rng), session_id=f"s-{i}").surprise_score.total
        s_norm.append(sn)
        s_shuf.append(ss)
        if ss > sn:
            wins2 += 1
    mean_norm = sum(s_norm) / len(s_norm)
    mean_shuf = sum(s_shuf) / len(s_norm)
    shuf_win = wins2 / len(pairs)

    t1_pass = mean_true < mean_rand and win_rate > 0.5
    verdict1 = "PASS" if t1_pass else "FAIL/INCONCLUSIVE"
    result = {
        "n_pairs": len(pairs),
        "test1_belief_continuation": {
            "mean_surprise_true": round(mean_true, 4),
            "mean_surprise_random": round(mean_rand, 4),
            "pairwise_win_rate": round(win_rate, 4),
            "verdict": verdict1,
        },
        "test2_surface_anomaly": {
            "mean_surprise_normal": round(mean_norm, 4),
            "mean_surprise_shuffled": round(mean_shuf, 4),
            "shuffle_higher_rate": round(shuf_win, 4),
        },
        "note": "真实 LCCC 对话；test1 验证 belief 预测让连贯续接更不惊奇，test2 验证 surprise 对真实表面异常敏感。阳/阴均如实记录。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# FE-LLM 控制层真实数据验证（LCCC 真人对话）",
        "",
        "## 测试1：belief 预测降低连贯续接的 surprise",
        f"- 真实下一句 surprise 均值：{result['test1_belief_continuation']['mean_surprise_true']}",
        f"- 随机句 surprise 均值：{result['test1_belief_continuation']['mean_surprise_random']}",
        f"- pairwise 胜率（真实<随机）：{result['test1_belief_continuation']['pairwise_win_rate']}",
        f"- 判定：**{verdict1}**",
        "",
        "## 测试2：surprise 检测真实表面异常",
        f"- 正常 surprise 均值：{result['test2_surface_anomaly']['mean_surprise_normal']}",
        f"- 字符打乱 surprise 均值：{result['test2_surface_anomaly']['mean_surprise_shuffled']}",
        f"- 打乱更高比例：{result['test2_surface_anomaly']['shuffle_higher_rate']}",
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
