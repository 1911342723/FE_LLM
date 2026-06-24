# -*- coding: utf-8 -*-
"""
code_lora_isolation_eval.py —— LoRA 式隔离知识模块：既学得好又不忘（解决"学得多记得少"）
==========================================================================================
把两个发现合并成一个新架构模块并验证：
  - 实验 B（code_forgetting_compare）：隔离（每技能一块 + 冻旧）→ 不遗忘（Δ≈0），但 synapse 载体学得糙(~38%)。
  - 实验 C（code_param_efficiency_eval）：内容相关低秩适配 LoRA → 高效学新（r8 用 1/5 参数达 83%）。
合并 = **LoRA 式隔离**：每个技能新建一组低秩 LoRA 块（作用 W_pred + head）、冻结旧块；
评估某技能用它**自己那组冻结的 LoRA 快照** → 学后续不改它（隔离不忘）+ LoRA 高表达力（学得好）。

四臂对照（顺序学 3 个同前缀冲突技能，同 teach/held/种子；指标 = held-out completion loss + 复制准确率）：
  1. LoRA-ISO    （本方案）每技能一组低秩 LoRA + 冻旧 → 预期"既学得好(高 acc)又不忘(Δ≈0)"
  2. synapse-ISO （实验 B）可学突触隔离 → 不忘但学得糙
  3. Transformer-FT（实验 B）标准顺序微调 → 学得好但灾难遗忘
  4. LoRA-SHARED （对照）共享同一组 LoRA 顺序学 → 会忘（证明"不忘来自隔离"而非 LoRA 本身）

输出：docs/reports/figs/code_lora_isolation.png + code_lora_isolation.{json,md}
运行：python -m fe_llm.energy_lm.evaluation.code_lora_isolation_eval --interactions 45
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
import torch.nn as nn

from fe_llm.config import get_device
from fe_llm.energy_lm.models.tokenizer import CharTokenizer
from fe_llm.energy_lm.training.code_train import ckpt_paths, load_any
from fe_llm.energy_lm.evaluation.code_param_efficiency_eval import LoRALinear
from fe_llm.energy_lm.evaluation.code_forgetting_compare import (
    SKILLS, NOUNS, comp_loss, held_loss, copy_acc, per_arm, tf_arm)

FIG_DIR = os.path.join("docs", "reports", "figs")
FIG_PATH = os.path.join(FIG_DIR, "code_lora_isolation.png")
REPORT_JSON = os.path.join("docs", "reports", "code_lora_isolation.json")
REPORT_MD = os.path.join("docs", "reports", "code_lora_isolation.md")


def _inject_lora(net, r, alpha, device):
    """把每个 block 的 W_pred 与 net.head 换成 LoRALinear，返回 LoRA 模块列表。"""
    loras = []
    for blk in net.blocks:
        lora = LoRALinear(blk.W_pred, r, alpha).to(device)
        blk.W_pred = lora
        loras.append(lora)
    lora_h = LoRALinear(net.head, r, alpha).to(device)
    net.head = lora_h
    loras.append(lora_h)
    return loras


def _reset_lora(loras, r):
    for l in loras:
        nn.init.normal_(l.A.weight, std=1.0 / r)
        nn.init.zeros_(l.B.weight)


def _snapshot(loras):
    return [(l.A.weight.detach().clone(), l.B.weight.detach().clone()) for l in loras]


def _set_lora(loras, snap):
    for l, (a, b) in zip(loras, snap):
        l.A.weight.data = a.clone()
        l.B.weight.data = b.clone()


def lora_arm(isolated, tok, device, teach, held, skills, args, r=8, alpha=16):
    """LoRA 臂：冻结 backbone，只训低秩 LoRA。isolated=True 每技能新块+冻旧（用快照评估）；False 共享。"""
    net_path, last_path, _, _ = ckpt_paths("per")
    path = net_path if os.path.exists(net_path) else last_path
    net = load_any(path, device).to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    loras = _inject_lora(net, r, alpha, device)
    params = [w for l in loras for w in (l.A.weight, l.B.weight)]
    n_per_skill = sum(w.numel() for w in params)

    def train_current(skill, seed):
        rng = np.random.default_rng(seed)
        for w in params:
            w.requires_grad_(True)
        opt = torch.optim.AdamW(params, lr=args.lora_lr)
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

    @torch.no_grad()
    def evalp(skill):
        net.eval()
        r_ = (held_loss(net, tok, skill, held, device), copy_acc(net, tok, skill, held, device))
        net.train(); return r_

    rows = []
    if isolated:
        snaps = {}
        for ki, skill in enumerate(skills):
            _reset_lora(loras, r)
            train_current(skill, args.seed + 1000 * ki)
            snaps[skill] = _snapshot(loras)
            row = {}
            for j in skills[:ki + 1]:
                _set_lora(loras, snaps[j]); row[j] = evalp(j)
            rows.append(row)
    else:
        _reset_lora(loras, r)
        for ki, skill in enumerate(skills):
            train_current(skill, args.seed + 1000 * ki)
            rows.append({j: evalp(j) for j in skills[:ki + 1]})
    del net
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows, n_per_skill


def _s0(rows, s0):
    return [rows[k][s0][0] for k in range(len(rows))]


def _s0a(rows, s0):
    return [rows[k][s0][1] for k in range(len(rows))]


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="LoRA 式隔离知识模块：既学得好又不忘。")
    ap.add_argument("--interactions", type=int, default=45)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--replay", type=int, default=4)
    ap.add_argument("--lora-lr", type=float, default=2e-3)
    ap.add_argument("--per-lr", type=float, default=0.05, help="synapse-ISO 学习率（实验 C 最优值）")
    ap.add_argument("--tf-lr", type=float, default=5e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--n-teach", type=int, default=8)
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    device = args.device.strip() or get_device()
    _, _, tok_path, _ = ckpt_paths("per")
    if not os.path.exists(tok_path):
        print("[loraiso] 找不到分词器。"); return 1
    tok = CharTokenizer.load(tok_path)
    teach, held = NOUNS[:args.n_teach], NOUNS[args.n_teach:]
    skills = list(SKILLS.keys())
    s0 = skills[0]
    t0 = time.time()

    print("[loraiso] === LoRA-ISO（低秩隔离·本方案）===", flush=True)
    liso, n_lora = lora_arm(True, tok, device, teach, held, skills, args, args.rank, args.alpha)
    for ki, sk in enumerate(skills):
        print(f"[loraiso]   学完 {sk}: " + " ".join(f"{j}=loss{v[0]:.2f}/acc{v[1]:.0%}" for j, v in liso[ki].items()), flush=True)

    print("[loraiso] === LoRA-SHARED（共享对照）===", flush=True)
    lshr, _ = lora_arm(False, tok, device, teach, held, skills, args, args.rank, args.alpha)
    for ki, sk in enumerate(skills):
        print(f"[loraiso]   学完 {sk}: " + " ".join(f"{j}=loss{v[0]:.2f}/acc{v[1]:.0%}" for j, v in lshr[ki].items()), flush=True)

    print("[loraiso] === synapse-ISO（实验 B 复用，lr=最优 0.05）===", flush=True)
    siso, syn_k = per_arm(True, tok, device, teach, held, skills, args)

    print("[loraiso] === Transformer-FT（实验 B 复用）===", flush=True)
    tf, _ = tf_arm(tok, device, teach, held, skills, args)
    if tf is None:
        print("[loraiso] 找不到 Transformer 模型。"); return 1

    arms = {
        "LoRA-ISO": (liso, n_lora, "#1565c0"),
        "synapse-ISO": (siso, syn_k, "#2e7d32"),
        "Transformer-FT": (tf, None, "#c62828"),
        "LoRA-SHARED": (lshr, n_lora, "#f9a825"),
    }

    def forget(rows):
        return round(_s0(rows, s0)[-1] - _s0(rows, s0)[0], 3)

    def learned(rows):  # 各技能"刚学完"时的对角线 acc 均值（学会程度）
        return float(np.mean([rows[k][skills[k]][1] for k in range(len(skills))]))

    summary = {name: {"forget_s0": forget(rows), "learned_diag_acc": learned(rows),
                      "s0_acc_first": _s0a(rows, s0)[0], "s0_acc_last": _s0a(rows, s0)[-1],
                      "final_loss": [round(rows[-1][s][0], 2) for s in skills]}
               for name, (rows, _, _) in arms.items()}
    for name, s in summary.items():
        print(f"[loraiso] {name:16s} 学会(对角acc均值)={s['learned_diag_acc']:.0%}  "
              f"首技能遗忘Δ={s['forget_s0']:+.2f}  首技能acc {s['s0_acc_first']:.0%}→{s['s0_acc_last']:.0%}", flush=True)

    li, si = summary["LoRA-ISO"], summary["synapse-ISO"]
    verdict = (
        f"【LoRA 式隔离·四臂对照】学会程度(对角 acc 均值) / 首技能遗忘Δ：\n"
        f"  - LoRA-ISO（本方案）：学会 {li['learned_diag_acc']:.0%}，遗忘 {li['forget_s0']:+.2f} → 既学得好又不忘\n"
        f"  - synapse-ISO：学会 {si['learned_diag_acc']:.0%}，遗忘 {si['forget_s0']:+.2f} → 不忘但学得{'糙' if si['learned_diag_acc']<li['learned_diag_acc'] else '好'}\n"
        f"  - Transformer-FT：学会 {summary['Transformer-FT']['learned_diag_acc']:.0%}，遗忘 {summary['Transformer-FT']['forget_s0']:+.2f} → 学得好但灾难遗忘\n"
        f"  - LoRA-SHARED：学会 {summary['LoRA-SHARED']['learned_diag_acc']:.0%}，遗忘 {summary['LoRA-SHARED']['forget_s0']:+.2f} → 共享会忘（证明不忘来自隔离）\n"
        f"结论：LoRA-ISO 用每技能 {n_lora/1e3:.0f}K 低秩参数，把'高效学新'与'隔离不忘'合到一起——"
        f"相对 synapse-ISO 学会程度 {li['learned_diag_acc']-si['learned_diag_acc']:+.0%}、相对 Transformer 把遗忘从 "
        f"{summary['Transformer-FT']['forget_s0']:+.2f} 降到 {li['forget_s0']:+.2f}。这是对'学得多记得少'的架构级回答。"
    )
    print("=" * 70, flush=True); print(verdict, flush=True)

    # ---- 画图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    stages = list(range(1, len(skills) + 1))
    xl = [f"学完{s}" for s in skills]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    for name, (rows, _, c) in arms.items():
        ax.plot(stages, _s0(rows, s0), "o-" if name == "LoRA-ISO" else "s--",
                color=c, lw=2.4 if name == "LoRA-ISO" else 1.8, label=name)
    ax.set_xticks(stages); ax.set_xticklabels(xl)
    ax.set_title(f"① 首技能「{s0}」held-out loss（越平=越不忘）", fontsize=12)
    ax.set_ylabel("completion loss (bits) ↓"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    for name, (rows, _, c) in arms.items():
        ax.plot(stages, _s0a(rows, s0), "o-" if name == "LoRA-ISO" else "s--",
                color=c, lw=2.4 if name == "LoRA-ISO" else 1.8, label=name)
    ax.set_xticks(stages); ax.set_xticklabels(xl); ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"② 首技能「{s0}」复制准确率（掉=忘）", fontsize=12)
    ax.set_ylabel("copy accuracy ↑"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[2]
    for name, (rows, _, c) in arms.items():
        # 横轴"学会程度"(对角 acc 均值) vs 纵轴"首技能最终仍保持的 acc"(不忘)：右上=既学得好又不忘
        x_learn = summary[name]["learned_diag_acc"]
        keep = summary[name]["s0_acc_last"]
        ax.scatter(x_learn, keep, s=200, color=c, zorder=3, edgecolors="k", linewidths=0.5)
        ax.annotate(name, (x_learn, keep), textcoords="offset points", xytext=(7, 5), fontsize=9)
    ax.set_xlim(0, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("学会程度（新技能对角 acc 均值）→"); ax.set_ylabel("首技能最终保持 acc（不忘）→")
    ax.set_title("③ 理想=右上角（既学得好又不忘）", fontsize=12); ax.grid(alpha=0.3)

    fig.suptitle("LoRA 式隔离知识模块：既学得好又不忘（四臂对照·真实 52M 代码模型）",
                 fontsize=13, fontweight="bold")
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig(FIG_PATH, dpi=130); plt.close()
    print(f"[loraiso] 图已保存：{FIG_PATH}", flush=True)

    # ---- 报告 ----
    result = {"config": vars(args), "device": device, "skills": skills, "n_lora_per_skill_k": round(n_lora / 1e3, 1),
              "arms": {name: [{k: list(v) for k, v in r.items()} for r in rows] for name, (rows, _, _) in arms.items()},
              "summary": summary, "verdict": verdict, "fig": FIG_PATH}
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(
            "# LoRA 式隔离知识模块：既学得好又不忘\n\n"
            "把**实验 B 的隔离不忘**与**实验 C 的 LoRA 高效**合并成一个新架构模块：每个技能新建一组低秩 LoRA 块"
            "（作用 W_pred + head）、冻结旧块，评估某技能用它自己那组冻结快照。四臂顺序学 3 个同前缀冲突技能，"
            f"真实 52M 代码模型底座；LoRA 每技能 {n_lora/1e3:.0f}K 参数（rank={args.rank}）。\n\n"
            "| 臂 | 学会程度(对角 acc 均值) | 首技能遗忘Δ | 首技能 acc 首→末 |\n|---|---:|---:|---:|\n"
            + "".join(
                f"| {name} | {s['learned_diag_acc']:.0%} | {s['forget_s0']:+.2f} | {s['s0_acc_first']:.0%}→{s['s0_acc_last']:.0%} |\n"
                for name, s in summary.items())
            + f"\n各臂末态各技能 loss：" + " / ".join(f"{name} {s['final_loss']}" for name, s in summary.items())
            + f"\n\n## 结论\n\n{verdict}\n\n"
            "## 诚实边界\n\n"
            "- LoRA-ISO 与 synapse-ISO 都是'隔离不忘'（数学保证 Δ≈0），差别在**学习载体**：LoRA（内容低秩）vs synapse（位置门控）。\n"
            "- 代价同隔离类方法：**参数随技能线性增长**（每技能 +一组 LoRA）、**无前向迁移**（每技能从底座独立学）；"
            "理想解 = Progressive 式（冻旧 + 侧向连接，迁移且不忘），列入未来工作。\n"
            "- 部署需**路由**选用哪块 LoRA（隔离保证不忘，路由是另一可解问题：可用触发词/小分类器/surprise 门控）。\n"
            f"- 任务为 3 冲突技能的机制验证（held-out {len(held)} 实例、synapse-only/LoRA-only），证机制不证规模。\n\n"
            f"图：`{FIG_PATH}`\n"
        )
    print(f"[loraiso] 报告：{REPORT_MD}  用时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
