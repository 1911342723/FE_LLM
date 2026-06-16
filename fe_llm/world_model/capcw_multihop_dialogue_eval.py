# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_multihop_dialogue_eval.py
==================================================
CAPCW 多跳链式工作记忆的**活文本闭环**：多跳绑定 NLU → 链式工作记忆 → decode→re-embed 逐跳取回。
见 `active_inference/incontext_binding_nlu.py::MultiHopBindingNLU`、`capcw_chain_memory.py`、
`docs/FE-LLM核心引擎构想.md` 第 24/25 节。

把第 24 节的 controller API（显式 `chain_working_memory_decision`）推进到**活文本自动多步推理**：用户用
自然语言陈述多条关联（"记住A的经理是B"/"记住B的工位是C302"），再用复合所有格提问（"A的经理的工位是
多少"），controller 经多跳 NLU 把它解析成 base+关系链，链式工作记忆**逐跳"解码中间符号→拼下一属性→
再检索"**（潜在 CoT 的字符串层落地）取回链尾 value，并把中间链作为**可溯源 CoT trace** 输出。

演示 + 判定（一段会话）：
- 陈述多条边（记住X的R是Y）→ 存入链式工作记忆；
- 复合所有格查询（X的R1的R2是多少，已绑定）→ 逐跳链式取回链尾 + grounded 回答 + CoT trace；
- 复合所有格查询（链中某跳未绑定）→ 断链→ASK（知道何时不该答，可溯源到断点）；
- 寒暄（你好）→ 不被劫持。

判据：① 已绑定多跳查询 ANSWER 且链尾 value 正确、CoT trace 完整；② 断链查询 ASK；③ 寒暄不劫持。
聚合多段随机会话：决策 balanced acc ≥ 0.80、链尾 value 取回 acc ≥ 0.60、寒暄劫持率 ≤ 0.02。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_multihop_dialogue_eval --run
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

from fe_llm.active_inference.capcw_chain_memory import CAPCWChainMemory
from fe_llm.active_inference.controller import ActiveInferenceController
from fe_llm.active_inference.policy import ActionType

REPORT_JSON = os.path.join("docs", "reports", "capcw_multihop_dialogue_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_multihop_dialogue_eval.md")

# 受控字符串池（2 跳复合所有格：org→role→person、person→attr→val）。
ORGS = ["项目甲", "部门乙", "小组丙", "客户丁", "团队戊", "分队己"]
ROLES = ["经理", "组长", "负责人", "队长"]
PERSONS = ["张三", "李四", "王五", "赵六", "钱七", "孙八"]
ATTRS = ["工位", "电话", "房间", "工号"]
VALS = ["B302", "C13", "8821", "A05", "X9", "T1", "P17", "Z42", "L6", "D7"]
CHITCHAT = ["你好", "我有点累", "今天天气不错", "谢谢你", "随便聊聊"]


def _train_chain_wm(path: str, *, n_sym: int, n_slots: int, max_hops: int, n_pairs: int,
                    epochs: int, seed: int) -> float:
    mem = CAPCWChainMemory(n_sym=n_sym, d=32, n_slots=n_slots, ask_threshold=0.5, cot=True)
    acc = mem.train_on_chain(max_hops=max_hops, n_pairs=n_pairs, n_train=8000, epochs=epochs, seed=seed)
    mem.save(path)
    return acc


def _scripted_demo(controller: ActiveInferenceController) -> list[dict]:
    """可读会话实录：存多条边 → 复合所有格多跳查询（含 grounded 回答 + CoT trace）→ 断链该问 → 不劫持寒暄。"""
    sid = "mh-demo"
    controller.reset_chain_working_memory(session_id=sid)
    turns = [
        ("记住项目甲的经理是张三", "bind"),
        ("记住张三的工位是B302", "bind"),
        ("记住部门乙的组长是李四", "bind"),
        ("项目甲的经理的工位是多少", "query_bound"),      # 2 跳：项目甲→张三→B302
        ("部门乙的组长的工位是多少", "query_unbound"),    # 张三的工位有、李四的工位没绑 → 断链该问
        ("记住李四的工位是C13", "bind"),                  # 补绑定（主动推理：满足追问）
        ("部门乙的组长的工位是多少", "query_bound"),      # 再问 → 链式取回 C13（surprise 下降）
        ("你好", "chitchat"),
    ]
    log = []
    for text, kind in turns:
        resp = controller.respond(text, session_id=sid)
        log.append({"text": text, "kind": kind,
                    "action": resp.selected_action_type.value,
                    "reply": resp.text,
                    "incontext_value": resp.incontext_value,
                    "incontext_chain": resp.incontext_chain,
                    "incontext_surprise": resp.incontext_surprise})
    return log


def _aggregate(controller: ActiveInferenceController, n: int, seed: int) -> dict:
    """聚合多段随机会话：每段 2 跳链 + 1~2 干扰边 + 一个复合所有格查询（bound/unbound 各半）+ 1 句寒暄。"""
    rng = np.random.default_rng(seed)
    pred_answer, is_bound = [], []
    value_correct, value_total = 0, 0
    hijack = 0
    sid = "mh-agg"
    for _ in range(n):
        controller.reset_chain_working_memory(session_id=sid)
        org = str(rng.choice(ORGS))
        role = str(rng.choice(ROLES))
        person = str(rng.choice(PERSONS))
        attr = str(rng.choice(ATTRS))
        val = str(rng.choice(VALS))
        controller.respond(f"记住{org}的{role}是{person}", session_id=sid)      # 第 1 跳
        bound = rng.random() < 0.5
        if bound:
            controller.respond(f"记住{person}的{attr}是{val}", session_id=sid)   # 第 2 跳（bound 才绑）
        # 一条干扰边（不同人/属性，避免与目标链相交）。
        dperson = str(rng.choice([p for p in PERSONS if p != person]))
        dval = str(rng.choice([v for v in VALS if v != val]))
        controller.respond(f"记住{dperson}的{attr}是{dval}", session_id=sid)
        resp = controller.respond(f"{org}的{role}的{attr}是多少", session_id=sid)
        answered = resp.selected_action_type == ActionType.ANSWER
        pred_answer.append(answered)
        is_bound.append(bound)
        if bound and answered:
            value_total += 1
            value_correct += int(resp.incontext_value == val)
        chit = controller.respond(str(rng.choice(CHITCHAT)), session_id=sid)
        hijack += int(chit.incontext_value is not None or chit.incontext_chain is not None)

    pred = np.asarray(pred_answer, dtype=bool)
    truth = np.asarray(is_bound, dtype=bool)
    accs = []
    for cls in (True, False):
        m = truth == cls
        if m.any():
            accs.append(float((pred[m] == cls).mean()))
    balacc = float(np.mean(accs)) if accs else 0.0
    value_acc = (value_correct / value_total) if value_total else 0.0
    return {"decision_balanced_acc": round(balacc, 4), "value_retrieval_acc": round(value_acc, 4),
            "chitchat_hijack_rate": round(hijack / n, 4), "n": n}


def run(args: argparse.Namespace) -> dict:
    print(f"[mh-dlg] n_sym={args.n_sym} n_slots={args.n_slots} n_eval={args.n_eval}", flush=True)
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "chain_wm.pt")
        acc = _train_chain_wm(ckpt, n_sym=args.n_sym, n_slots=args.n_slots, max_hops=args.max_hops,
                              n_pairs=args.n_pairs, epochs=args.epochs, seed=args.seed)
        print(f"[mh-dlg] 链式工作空间训练准确率(链尾)={acc:.3f}", flush=True)
        controller = ActiveInferenceController(capcw_chain_memory_path=ckpt)
        demo = _scripted_demo(controller)
        agg = _aggregate(controller, n=args.n_eval, seed=args.seed + 7000)

    for t in demo:
        s = t["incontext_surprise"]
        s_str = f"{s:.3f}" if isinstance(s, (int, float)) else "—"
        print(f"[mh-dlg] 用户「{t['text']}」 → 动作={t['action']} 回答「{t['reply']}」 "
              f"链={t['incontext_chain']} surprise={s_str}", flush=True)

    # 判定（含 grounded 多跳生成 + 主动推理 surprise 下降）
    closure_ok = (
        demo[4]["action"] == "ask_clarification"                       # 断链→该问
        and demo[6]["action"] == "answer" and demo[6]["incontext_value"] == "C13"  # 补绑定后→该答
        and isinstance(demo[4]["incontext_surprise"], (int, float))
        and isinstance(demo[6]["incontext_surprise"], (int, float))
        and demo[6]["incontext_surprise"] < demo[4]["incontext_surprise"]          # surprise 下降
    )
    demo_ok = (
        demo[3]["action"] == "answer" and demo[3]["incontext_value"] == "B302"
        and demo[3]["incontext_chain"] == ["张三", "B302"]            # 可溯源 CoT trace
        and "B302" in (demo[3]["reply"] or "")
        and closure_ok
        and demo[7]["incontext_value"] is None and demo[7]["incontext_chain"] is None  # 寒暄不劫持
    )
    h_decision = agg["decision_balanced_acc"] >= 0.80
    h_value = agg["value_retrieval_acc"] >= 0.60
    h_nohijack = agg["chitchat_hijack_rate"] <= 0.02
    if demo_ok and h_decision and h_value and h_nohijack:
        verdict = ("PASS: 活文本多跳闭环成立——复合所有格经多跳 NLU + 链式工作记忆 decode→re-embed 逐跳取回，"
                   "引擎在 controller 里正确驱动 ANSWER(链尾 value + 可溯源 CoT trace)/ASK(断链)，不劫持寒暄。"
                   "「对内多步推理」在真实 controller 活文本路径成立。")
    elif h_decision and h_nohijack:
        verdict = "PARTIAL: 决策与不劫持成立，但链尾取回或脚本演示有偏差（d=32 容量/NLU 覆盖）。"
    else:
        verdict = "FAIL: 活文本多跳闭环未稳定（决策/取回/劫持其一未达标）。"

    result = {
        "task": "multi-hop binding NLU -> CAPCW chain memory -> decode->re-embed path retrieval -> controller (live text)",
        "config": {"n_sym": args.n_sym, "n_slots": args.n_slots, "max_hops": args.max_hops,
                   "n_pairs": args.n_pairs, "n_eval": args.n_eval, "chain_train_acc": round(acc, 4)},
        "scripted_demo": demo,
        "aggregate": agg,
        "demo_ok": demo_ok,
        "verdict": verdict,
        "note": "多跳 NLU 高精度结构匹配复合所有格（X的R1的R2…是多少，≥2 跳）+ 复用单跳绑定（记住X的R是Y）；"
                "链式取回=字符串层 decode→re-embed（每跳解码中间符号串→拼下一属性→再内容寻址取回）；"
                "CoT trace=各跳中间符号（可溯源）；断链（某跳未绑定）→ASK；主动推理：断链追问→用户补绑定→再问→"
                "surprise 下降→grounded 链式回答（对外行动降未来自由能，与单跳同构）。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# CAPCW 活文本多跳闭环 · 复合所有格 → 链式工作记忆 → decode→re-embed 逐跳取回",
        "",
        f"- 判定：**{verdict}**",
        f"- 链式工作空间训练准确率（链尾）：{acc:.4f}",
        "",
        "## 脚本会话实录（grounded 回答=链尾取回入生成；链=可溯源 CoT trace；surprise=引擎路由）",
        "",
        "| 用户输入 | 动作 | 回答（grounded） | CoT trace | surprise |",
        "|---|---|---|---|---:|",
    ]
    for t in demo:
        s = t["incontext_surprise"]
        s_str = f"{s:.3f}" if isinstance(s, (int, float)) else "—"
        chain_str = "→".join(t["incontext_chain"]) if t["incontext_chain"] else "—"
        lines.append(f"| {t['text']} | {t['action']} | {t['reply'] or '—'} | {chain_str} | {s_str} |")
    lines += [
        "",
        "## 聚合指标（多段随机 2 跳会话）",
        "",
        f"- 决策 balanced acc（bound→ANSWER / 断链→ASK）：**{agg['decision_balanced_acc']:.4f}**",
        f"- 链尾 value 取回准确率：**{agg['value_retrieval_acc']:.4f}**",
        f"- 寒暄劫持率（越低越好）：**{agg['chitchat_hijack_rate']:.4f}**（n={agg['n']}）",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[mh-dlg] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    print(f"[mh-dlg] 决策 balacc={agg['decision_balanced_acc']:.3f} value_acc={agg['value_retrieval_acc']:.3f} "
          f"劫持率={agg['chitchat_hijack_rate']:.3f} demo_ok={demo_ok}", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW multi-hop binding NLU -> chain memory -> controller (live text).")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-sym", type=int, default=24)
    ap.add_argument("--n-slots", type=int, default=8)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--n-pairs", type=int, default=5)
    ap.add_argument("--n-eval", type=int, default=80)
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
        print("[mh-dlg] dry-run：未训练。活文本多跳：复合所有格 NLU→链式工作记忆→decode→re-embed 逐跳取回→controller。")
        print("[mh-dlg] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
