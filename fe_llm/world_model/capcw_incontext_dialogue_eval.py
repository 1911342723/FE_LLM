# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_incontext_dialogue_eval.py
===================================================
CAPCW in-context 工作记忆的**活文本闭环**：绑定 NLU → 工作记忆 → 引擎 surprise 驱动 controller 决策。
见 `active_inference/incontext_binding_nlu.py`、`capcw_memory.py`、`docs/FE-LLM核心引擎构想.md` 第 19 节。

把"接回 controller"从"显式 API"推进到"**活文本自动**"：用户用自然语言陈述/查询现场关联，
controller 经绑定 NLU 把关联喂进 CAPCW 工作记忆，并用**引擎 surprise** 裁决 ASK/ANSWER + 取回 value。

演示 + 判定（一段会话）：
- 陈述绑定（记住X是Y / X对应Y / X的工号是Y）→ 存入工作记忆；
- 查询已绑定（X是多少）→ 引擎低 surprise → ANSWER + 取回正确 value；
- 查询未绑定（从未提过的 key）→ 引擎高 surprise → ASK_CLARIFICATION（知道何时不该答）；
- 寒暄/闲聊（你好 / 我有点累）→ 不被劫持（绑定 NLU 高精度不误触）。

判据：① 已绑定查询 ANSWER 且 value 取回正确；② 未绑定查询 ASK；③ 寒暄不被劫持（incontext_value 为空且
动作不被工作记忆改写）。聚合多段随机会话上 决策 balanced acc ≥ 0.8、value 取回 acc ≥ 0.6。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_incontext_dialogue_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from fe_llm.active_inference.capcw_memory import CAPCWWorkingMemory
from fe_llm.active_inference.controller import ActiveInferenceController
from fe_llm.active_inference.policy import ActionType

REPORT_JSON = os.path.join("docs", "reports", "capcw_incontext_dialogue_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_incontext_dialogue_eval.md")

# 受控字符串池（key 不含属性触发词，故绑定用"记住X是Y"标记式；value 为编号/短串）。
KEYS = ["会议室", "项目代号", "服务器", "门禁卡", "停车位", "工位", "储物柜", "会议时间"]
VALS = ["B302", "X9", "10.0.0.7", "8821", "P17", "A05", "C13", "周三下午", "404", "Z42", "L6", "T1"]
CHITCHAT = ["你好", "我有点累", "今天天气不错", "谢谢你", "随便聊聊"]


def _train_wm(path: str, *, n_keys: int, n_vals: int, k: int, epochs: int, seed: int) -> float:
    wm = CAPCWWorkingMemory(n_keys=n_keys, n_vals=n_vals, d=32, n_slots=max(6, k + 1), ask_threshold=0.5)
    acc = wm.train_on_binding(k_pairs=k, n_train=8000, epochs=epochs, seed=seed)
    wm.save(path)
    return acc


def _scripted_demo(controller: ActiveInferenceController) -> list[dict]:
    """一段可读会话实录，串起 存绑定 / 答已绑定 / 问未绑定 / 不劫持寒暄。"""
    controller.reset_working_memory()
    sid = "incontext-demo"
    turns = [
        ("记住会议室是B302", "bind"),
        ("项目代号对应X9", "bind"),
        ("会议室是多少", "query_bound"),       # 期望 ANSWER + B302
        ("项目代号是什么", "query_bound"),     # 期望 ANSWER + X9
        ("门禁卡是多少", "query_unbound"),     # 从未提过 → 期望 ASK
        ("你好", "chitchat"),                  # 不劫持
    ]
    log = []
    for text, kind in turns:
        resp = controller.respond(text, session_id=sid)
        log.append({"text": text, "kind": kind,
                    "action": resp.selected_action_type.value,
                    "reply": resp.text,                       # grounded 回答（取回内容入生成）
                    "incontext_value": resp.incontext_value})
    return log


def _aggregate(controller: ActiveInferenceController, n: int, k: int, seed: int) -> dict:
    """聚合多段随机会话：每段 k 个 in-context 绑定 + 1 个查询（bound/unbound 各半）+ 1 句寒暄。"""
    rng = np.random.default_rng(seed)
    pred_answer, is_bound = [], []
    value_correct, value_total = 0, 0
    hijack = 0
    for _ in range(n):
        controller.reset_working_memory()
        sid = f"agg-{rng.integers(1_000_000)}"
        keys = list(rng.choice(KEYS, size=k, replace=False))
        vals = list(rng.choice(VALS, size=k, replace=False))
        for key, val in zip(keys, vals):
            controller.respond(f"记住{key}是{val}", session_id=sid)
        if rng.random() < 0.5:
            qi = int(rng.integers(k))
            qkey, bound, true_val = keys[qi], True, vals[qi]
        else:
            unseen = [x for x in KEYS if x not in set(keys)]
            qkey, bound, true_val = str(rng.choice(unseen)), False, None
        resp = controller.respond(f"{qkey}是多少", session_id=sid)
        answered = resp.selected_action_type == ActionType.ANSWER
        pred_answer.append(answered)
        is_bound.append(bound)
        if bound and answered:
            value_total += 1
            value_correct += int(resp.incontext_value == true_val)
        # 寒暄不被劫持：incontext_value 应为空。
        chit = controller.respond(str(rng.choice(CHITCHAT)), session_id=sid)
        hijack += int(chit.incontext_value is not None)

    pred = np.asarray(pred_answer, dtype=bool)
    truth = np.asarray(is_bound, dtype=bool)
    accs = []
    for cls in (True, False):
        m = truth == cls
        if m.any():
            accs.append(float((pred[m] == cls).mean()))
    balacc = float(np.mean(accs)) if accs else 0.0
    value_acc = (value_correct / value_total) if value_total else 0.0
    hijack_rate = hijack / n
    return {"decision_balanced_acc": round(balacc, 4), "value_retrieval_acc": round(value_acc, 4),
            "chitchat_hijack_rate": round(hijack_rate, 4), "n": n}


def run(args: argparse.Namespace) -> dict:
    print(f"[incontext] n_keys={args.n_keys} n_vals={args.n_vals} k={args.k} n_eval={args.n_eval}", flush=True)
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "wm.pt")
        bind_acc = _train_wm(ckpt, n_keys=args.n_keys, n_vals=args.n_vals, k=args.k, epochs=args.epochs, seed=args.seed)
        print(f"[incontext] 工作空间绑定训练准确率={bind_acc:.3f}", flush=True)
        controller = ActiveInferenceController(capcw_memory_path=ckpt)
        demo = _scripted_demo(controller)
        agg = _aggregate(controller, n=args.n_eval, k=args.k, seed=args.seed + 7000)

    for t in demo:
        print(f"[incontext] 用户「{t['text']}」 → 动作={t['action']} 回答「{t['reply']}」 取回={t['incontext_value']}", flush=True)

    # 判定（含 grounded 生成：回答文本须扎根于取回的 value）
    demo_ok = (
        demo[2]["action"] == "answer" and demo[2]["incontext_value"] == "B302" and "B302" in (demo[2]["reply"] or "")
        and demo[3]["action"] == "answer" and demo[3]["incontext_value"] == "X9" and "X9" in (demo[3]["reply"] or "")
        and demo[4]["action"] == "ask_clarification"
        and demo[5]["incontext_value"] is None
    )
    h_decision = agg["decision_balanced_acc"] >= 0.80
    h_value = agg["value_retrieval_acc"] >= 0.60
    h_nohijack = agg["chitchat_hijack_rate"] <= 0.02
    if demo_ok and h_decision and h_value and h_nohijack:
        verdict = ("PASS: 活文本闭环成立——绑定 NLU 把现场关联喂工作记忆，引擎 surprise 在 controller 里"
                   "正确驱动 ANSWER(取回 value)/ASK，且不劫持寒暄。CAPCW 引擎接回真实 controller 活文本路径。")
    elif h_decision and h_nohijack:
        verdict = "PARTIAL: 决策与不劫持成立，但内容取回或脚本演示有偏差。"
    else:
        verdict = "FAIL: 活文本闭环未稳定（决策/取回/劫持其一未达标）。"

    result = {
        "task": "in-context binding NLU -> CAPCW working memory -> engine surprise -> controller ASK/ANSWER (live text)",
        "config": {"n_keys": args.n_keys, "n_vals": args.n_vals, "k": args.k, "n_eval": args.n_eval,
                   "binding_train_acc": round(bind_acc, 4)},
        "scripted_demo": demo,
        "aggregate": agg,
        "demo_ok": demo_ok,
        "verdict": verdict,
        "note": "绑定 NLU 高精度规则触发（记住/对应/设为/等于/的{属性}是 + 查询词），裸'X是Y'与寒暄不触发；"
                "工作记忆决策由引擎 query 路由 surprise 涌现（无动作监督）；value 由取回的符号 id 映回字符串。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# CAPCW 活文本闭环 · 绑定 NLU → 工作记忆 → 引擎 surprise 驱动 controller",
        "",
        f"- 判定：**{verdict}**",
        f"- 工作空间绑定训练准确率：{bind_acc:.4f}",
        "",
        "## 脚本会话实录（grounded 回答=取回内容入生成）",
        "",
        "| 用户输入 | 动作 | 回答（grounded） | in-context 取回 |",
        "|---|---|---|---|",
    ]
    for t in demo:
        lines.append(f"| {t['text']} | {t['action']} | {t['reply'] or '—'} | {t['incontext_value'] or '—'} |")
    lines += [
        "",
        "## 聚合指标（多段随机会话）",
        "",
        f"- 决策 balanced acc（bound→ANSWER / unbound→ASK，引擎 surprise）：**{agg['decision_balanced_acc']:.4f}**",
        f"- 内容取回 value 准确率：**{agg['value_retrieval_acc']:.4f}**",
        f"- 寒暄劫持率（越低越好）：**{agg['chitchat_hijack_rate']:.4f}**（n={agg['n']}）",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[incontext] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    print(f"[incontext] 决策 balacc={agg['decision_balanced_acc']:.3f} value_acc={agg['value_retrieval_acc']:.3f} "
          f"劫持率={agg['chitchat_hijack_rate']:.3f} demo_ok={demo_ok}", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW in-context binding NLU -> working memory -> controller (live text).")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-keys", type=int, default=10)
    ap.add_argument("--n-vals", type=int, default=12)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-json", default=REPORT_JSON)
    ap.add_argument("--report-md", default=REPORT_MD)
    return ap


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print("[incontext] dry-run：未训练。活文本：绑定 NLU→CAPCW 工作记忆→引擎 surprise→controller ASK/ANSWER。")
        print("[incontext] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
