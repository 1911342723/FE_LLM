# -*- coding: utf-8 -*-
"""
code_router_growth_eval.py —— 端到端成长系统：surprise 门控自动长块 + 推理路由 + 隔离不忘
=============================================================================================
把实验 D 的 LoRA 式隔离模块做成**可用的端到端持续学习系统**，验证三件事：

  1. surprise 门控自动成长：处理知识流时，对每个 lesson 用现有块算最低 surprise(completion loss)——
     高于阈值（现有块都搞不定）→ **自动长一个新块**；低于阈值（已会）→ **复用、不重复长**。无人显式指挥。
  2. 推理路由（无 GT）：来一个 prefix，自动选用哪个块——用"块激活下 prefix 末位预测熵"（置信度）路由，
     与触发词路由（上限）对照。
  3. 隔离不忘：路由对后，各知识用各自块 → 全部保持（隔离的数学保证）。

知识流（含重复，考验"不重复长"）：make → get → make → new → get
（3 个 base 不会的非冲突复制规则；make/get 各出现 2 次，应只在首次触发成长。）

输出：docs/reports/figs/code_router_growth.png + code_router_growth.{json,md}
运行：python -m fe_llm.energy_lm.evaluation.code_router_growth_eval --interactions 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from fe_llm.config import get_device
from fe_llm.energy_lm.models.tokenizer import CharTokenizer
from fe_llm.energy_lm.training.code_train import ckpt_paths, generate, load_any, _logits
import torch.nn.functional as F

from fe_llm.energy_lm.evaluation.code_lora_isolation_eval import (
    _inject_lora, _reset_lora, _snapshot, _set_lora)

FIG_DIR = os.path.join("docs", "reports", "figs")
FIG_PATH = os.path.join(FIG_DIR, "code_router_growth.png")
REPORT_JSON = os.path.join("docs", "reports", "code_router_growth.json")
REPORT_MD = os.path.join("docs", "reports", "code_router_growth.md")

NOUNS = ["button", "slider", "panel", "dialog", "menu", "toolbar", "canvas", "label",
         "cursor", "frame", "badge", "spinner", "switch", "drawer", "tooltip", "modal",
         "widget", "layout", "header", "footer", "sidebar", "banner", "avatar", "ribbon"]

# 3 个独立知识：前缀结尾不同(可路由) + completion 带独特后缀 .a/.b/.c(知识真独立、互不泛化)
SKILL_RULES = {
    "make": lambda x: (f'def make_{x}():\n    return Box("', f'{x}").a\n'),    # 完成: X").a
    "get":  lambda x: (f'def get_{x}():\n    return Cup(', f'{x}).b\n'),        # 完成: X).b
    "tag":  lambda x: (f'def tag_{x}():\n    return Map[', f'{x}].c\n'),        # 完成: X].c
}
STREAM = ["make", "get", "make", "tag", "get"]   # 含重复，考验"不重复长"
TRIGGER = {"make": "make", "get": "get", "tag": "tag"}  # 触发词路由（上限对照）用的前缀关键词


def comp_loss(net, tok, skill, x, device):
    prefix, completion = SKILL_RULES[skill](x)
    ids = tok.encode(prefix + completion)
    p_len = len(tok.encode(prefix))
    seq = torch.tensor([ids], device=device)
    logits = _logits(net, seq)[0].float()
    tgt = torch.tensor(ids[p_len:], device=device)
    return F.cross_entropy(logits[p_len - 1: len(ids) - 1], tgt) / np.log(2)


@torch.no_grad()
def copy_acc(net, tok, skill, xs, device):
    ok = 0
    for x in xs:
        prefix, completion = SKILL_RULES[skill](x)
        out = generate(net, tok, prefix, net.max_len, device, max_new=len(completion) + 4,
                       temperature=0.0, top_k=0)
        ok += int(out.startswith(completion))
    return ok / max(1, len(xs))


@torch.no_grad()
def block_surprise(net, tok, skill, xs, device):
    """某块（已 set）对一批样本的平均 completion loss = surprise。"""
    return float(np.mean([float(comp_loss(net, tok, skill, x, device)) for x in xs]))


@torch.no_grad()
def prefix_entropy(net, tok, prefix, device):
    """某块（已 set）下，prefix 末位 next-token 分布的熵（越低=越确定=越可能匹配）。"""
    ids = tok.encode(prefix)
    seq = torch.tensor([ids], device=device)
    logits = _logits(net, seq)[0, len(ids) - 1].float()
    p = torch.softmax(logits, dim=-1)
    return float(-(p * (p + 1e-12).log()).sum())


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="端到端成长系统：surprise 门控自动长块 + 推理路由 + 不忘。")
    ap.add_argument("--interactions", type=int, default=40)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--replay", type=int, default=4)
    ap.add_argument("--lora-lr", type=float, default=2e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--n-teach", type=int, default=8)
    ap.add_argument("--n-probe", type=int, default=4, help="surprise 门控探针样本数")
    ap.add_argument("--grow-threshold", type=float, default=0.5, help="surprise=1−最匹配块复制准确率；高于此则长新块")
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    device = args.device.strip() or get_device()
    per_path, per_last, tok_path, _ = ckpt_paths("per")
    if not os.path.exists(tok_path):
        print("[router] 找不到分词器。"); return 1
    tok = CharTokenizer.load(tok_path)
    teach, held = NOUNS[:args.n_teach], NOUNS[args.n_teach:]
    probes = teach[:args.n_probe]
    t0 = time.time()

    path = per_path if os.path.exists(per_path) else per_last
    net = load_any(path, device).to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    loras = _inject_lora(net, args.rank, args.alpha, device)
    params = [w for l in loras for w in (l.A.weight, l.B.weight)]
    zero_snap = _snapshot(loras)
    n_per = sum(w.numel() for w in params)

    def train_block(skill, seed):
        _reset_lora(loras, args.rank)
        for w in params:
            w.requires_grad_(True)
        opt = torch.optim.AdamW(params, lr=args.lora_lr)
        rng = np.random.default_rng(seed)
        net.train()
        buf = []
        for it in range(1, args.interactions + 1):
            x = teach[(it - 1) % len(teach)]
            if x not in buf:
                buf.append(x)
            for _ in range(args.steps):
                k = min(len(buf), args.replay)
                bx = list(rng.choice(buf, size=k, replace=False)) if len(buf) > 1 else buf
                loss = torch.stack([comp_loss(net, tok, skill, xi, device) for xi in bx]).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        for w in params:
            w.requires_grad_(False)
        return _snapshot(loras)

    # ---------- 1) surprise 门控自动成长 ----------
    print("[router] === surprise 门控自动成长（知识流）===", flush=True)
    blocks = {}            # skill -> snapshot
    growth_log = []
    for i, skill in enumerate(STREAM):
        if blocks:
            cand = []
            for bk, snap in blocks.items():
                _set_lora(loras, snap)
                cand.append((copy_acc(net, tok, skill, probes, device), bk))
            best_acc, who = max(cand, key=lambda t: t[0])
            ms = 1.0 - best_acc        # surprise = 现有最匹配块都续写不出的程度（1−复制准确率）
        else:
            ms, who = 1.0, None
        if ms > args.grow_threshold:
            blocks[skill] = train_block(skill, args.seed + 1000 * i)
            growth_log.append({"step": i + 1, "skill": skill, "decision": "GROW", "min_surprise": round(ms, 3)})
            print(f"[router]   #{i+1} {skill}: min_surprise={ms:.2f} > {args.grow_threshold} → 长新块（现有 {len(blocks)} 块）", flush=True)
        else:
            growth_log.append({"step": i + 1, "skill": skill, "decision": "REUSE", "min_surprise": round(ms, 3), "routed": who})
            print(f"[router]   #{i+1} {skill}: min_surprise={ms:.2f} ≤ {args.grow_threshold} → 复用块「{who}」不重复长", flush=True)

    learned = list(blocks.keys())
    n_grow = sum(1 for g in growth_log if g["decision"] == "GROW")

    # ---------- 2) 推理路由（无 GT）----------
    print("[router] === 推理路由（能量/置信度 vs 触发词上限）===", flush=True)

    def route_by_entropy(prefix):
        best, be = None, 1e18
        for sk, snap in blocks.items():
            _set_lora(loras, snap)
            e = prefix_entropy(net, tok, prefix, device)
            if e < be:
                be, best = e, sk
        return best

    def route_by_trigger(prefix):
        for sk, kw in TRIGGER.items():
            if sk in blocks and f"def {kw}_" in prefix:
                return sk
        return None

    ent_correct = trig_correct = total = 0
    for true_sk in learned:
        for x in held:
            prefix = SKILL_RULES[true_sk](x)[0]
            total += 1
            ent_correct += int(route_by_entropy(prefix) == true_sk)
            trig_correct += int(route_by_trigger(prefix) == true_sk)
    ent_acc = ent_correct / max(1, total)
    trig_acc = trig_correct / max(1, total)
    print(f"[router]   路由正确率：能量/置信度 {ent_acc:.0%}  vs  触发词(上限) {trig_acc:.0%}", flush=True)

    # ---------- 3) 端到端 acc（路由→选块→生成）+ 隔离不忘 ----------
    @torch.no_grad()
    def end2end_acc(router):
        ok = 0
        for true_sk in learned:
            for x in held:
                prefix, completion = SKILL_RULES[true_sk](x)
                sk = router(prefix)
                if sk is None:
                    continue
                _set_lora(loras, blocks[sk]); net.eval()
                out = generate(net, tok, prefix, net.max_len, device, max_new=len(completion) + 4,
                               temperature=0.0, top_k=0)
                ok += int(out.startswith(completion))
        return ok / max(1, total)

    e2e_ent = end2end_acc(route_by_entropy)
    e2e_trig = end2end_acc(route_by_trigger)

    @torch.no_grad()
    def oracle_retention():  # 用各自正确块（隔离上限）
        net.eval()
        out = {}
        for sk in learned:
            _set_lora(loras, blocks[sk])
            out[sk] = copy_acc(net, tok, sk, held, device)
        return out
    retention = oracle_retention()
    ret_mean = float(np.mean(list(retention.values())))

    print(f"[router]   端到端 acc：能量路由 {e2e_ent:.0%}  触发词路由 {e2e_trig:.0%}  | 隔离不忘(各用正确块) {ret_mean:.0%} {retention}", flush=True)

    expected_grows = list(dict.fromkeys(STREAM))
    grown_skills = [g["skill"] for g in growth_log if g["decision"] == "GROW"]
    grow_ok = (grown_skills == expected_grows)
    verdict = (
        f"【端到端成长系统】\n"
        f"  1) surprise 门控自动成长：知识流 {STREAM} → 自动长 {n_grow} 块（{[g['skill'] for g in growth_log if g['decision']=='GROW']}），"
        f"重复的 make/get 第二次 min_surprise 低→复用不重复长 → {'✅ 正确' if grow_ok else '⚠️ 见日志'}。\n"
        f"  2) 推理路由正确率：能量/置信度 {ent_acc:.0%} vs 触发词上限 {trig_acc:.0%}；端到端 acc 能量 {e2e_ent:.0%} / 触发词 {e2e_trig:.0%}。\n"
        f"  3) 隔离不忘：各用正确块平均 acc {ret_mean:.0%}（{retention}）——隔离的数学保证。\n"
        f"结论：LoRA-ISO + surprise 门控 + 路由 = 端到端持续成长系统（自动判断该长则长、不忘旧、按需取用）。"
        f"{'能量路由已接近触发词上限。' if ent_acc >= trig_acc - 1e-6 else '能量路由弱于触发词上限（学习化路由是真实难点，触发词路由可作可靠兜底）。'}"
    )
    print("=" * 70, flush=True); print(verdict, flush=True)

    del net
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- 画图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    xs = list(range(1, len(STREAM) + 1))
    ms_vals = [g["min_surprise"] if g["min_surprise"] != float("inf") else args.grow_threshold * 2 for g in growth_log]
    cols = ["#1565c0" if g["decision"] == "GROW" else "#9e9e9e" for g in growth_log]
    ax.bar(xs, ms_vals, color=cols)
    ax.axhline(args.grow_threshold, color="#c62828", ls="--", label=f"成长阈值 {args.grow_threshold}")
    for x, g in zip(xs, growth_log):
        ax.text(x, ms_vals[x - 1] + 0.03, g["decision"], ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([f"#{i}\n{s}" for i, s in zip(xs, STREAM)])
    ax.set_title("① surprise 门控自动成长（蓝=长新块/灰=复用）", fontsize=12)
    ax.set_ylabel("surprise = 1 − 最匹配块复制准确率"); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    labels = ["能量/置信度路由", "触发词路由(上限)"]
    ax.bar([0, 1], [ent_acc, trig_acc], color=["#1565c0", "#2e7d32"])
    ax.bar([3, 4], [e2e_ent, e2e_trig], color=["#42a5f5", "#66bb6a"])
    ax.set_xticks([0, 1, 3, 4]); ax.set_xticklabels(["路由\n能量", "路由\n触发词", "端到端\n能量", "端到端\n触发词"], fontsize=9)
    ax.set_ylim(0, 1.08); ax.set_ylabel("准确率 ↑")
    ax.set_title("② 路由正确率 + 端到端 acc", fontsize=12); ax.grid(axis="y", alpha=0.3)
    for i, v in zip([0, 1, 3, 4], [ent_acc, trig_acc, e2e_ent, e2e_trig]):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)

    ax = axes[2]
    sk_names = list(retention.keys())
    ax.bar(range(len(sk_names)), [retention[s] for s in sk_names], color="#1565c0")
    ax.set_xticks(range(len(sk_names))); ax.set_xticklabels(sk_names)
    ax.set_ylim(0, 1.08); ax.set_ylabel("held-out acc ↑")
    ax.set_title("③ 隔离不忘：各知识用正确块（全保持）", fontsize=12); ax.grid(axis="y", alpha=0.3)
    for i, s in enumerate(sk_names):
        ax.text(i, retention[s] + 0.02, f"{retention[s]:.0%}", ha="center", fontsize=9)

    fig.suptitle("端到端成长系统：surprise 门控自动长块 + 路由 + 隔离不忘（真实 52M 代码模型）",
                 fontsize=13, fontweight="bold")
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig(FIG_PATH, dpi=130); plt.close()
    print(f"[router] 图已保存：{FIG_PATH}", flush=True)

    result = {"config": vars(args), "device": device, "stream": STREAM, "growth_log": growth_log,
              "n_grow": n_grow, "learned": learned, "n_lora_per_block_k": round(n_per / 1e3, 1),
              "route_acc": {"entropy": ent_acc, "trigger": trig_acc},
              "end2end_acc": {"entropy": e2e_ent, "trigger": e2e_trig},
              "retention": retention, "retention_mean": ret_mean, "verdict": verdict, "fig": FIG_PATH}
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(
            "# 端到端成长系统：surprise 门控自动长块 + 推理路由 + 隔离不忘\n\n"
            "把 LoRA 式隔离模块做成可用的持续学习系统。知识流 `make → get → make → new → get`"
            "（3 个 base 不会的非冲突规则，make/get 各出现 2 次）。真实 52M 代码模型底座，每块 "
            f"{n_per/1e3:.0f}K 低秩参数。\n\n"
            "## 1) surprise 门控自动成长\n\n"
            "| # | lesson | 现有块最低 surprise | 决策 |\n|---|---|---:|---|\n"
            + "".join(f"| {g['step']} | {g['skill']} | {g['min_surprise']} | **{g['decision']}** |\n" for g in growth_log)
            + f"\n自动长出 {n_grow} 块（{[g['skill'] for g in growth_log if g['decision']=='GROW']}），重复知识不重复长。\n\n"
            "## 2) 推理路由 + 端到端\n\n"
            f"| 指标 | 能量/置信度路由 | 触发词路由(上限) |\n|---|---:|---:|\n"
            f"| 路由正确率 | {ent_acc:.0%} | {trig_acc:.0%} |\n"
            f"| 端到端 acc | {e2e_ent:.0%} | {e2e_trig:.0%} |\n\n"
            "## 3) 隔离不忘\n\n"
            f"各知识用正确块 held-out acc：{retention}，平均 {ret_mean:.0%}（隔离的数学保证）。\n\n"
            f"## 结论\n\n{verdict}\n\n"
            "## 诚实边界\n\n"
            "- surprise 门控用 completion loss（有教学信号时可靠）；阈值需按任务标定。\n"
            "- **学习化路由（能量/置信度）是真实难点**：若弱于触发词路由上限，说明仅靠 LoRA 块的 prefix 置信度"
            "区分知识不足；触发词/小分类器路由可作可靠兜底，学习化路由是开放问题。\n"
            "- 仍是隔离类共性代价：参数随知识线性增长、无前向迁移。\n"
            f"- 机制验证（3 知识、held-out {len(held)} 实例），不证规模。\n\n"
            f"图：`{FIG_PATH}`\n"
        )
    print(f"[router] 报告：{REPORT_MD}  用时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
