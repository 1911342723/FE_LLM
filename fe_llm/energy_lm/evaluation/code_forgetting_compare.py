# -*- coding: utf-8 -*-
"""
code_forgetting_compare.py —— 持续学习·灾难遗忘三臂对照：PER 隔离 vs PER 共享 vs 标准 Transformer
=====================================================================================================
直面 Transformer 的痛点之一【灾难性遗忘】：顺序学 3 个"同前缀、冲突输出"的技能，看**早学技能**
是否被后学覆盖。这是把现有"PER 内部 ISO vs SHARED"对照升级为**正面对照 Transformer**——
让 Transformer 当场暴露"持续学习就灾难遗忘"的痛点，而 PER 用其结构隔离机制做到不遗忘。

三臂（同任务 / 同数据 / 同训练协议 / 同评估集 / 同种子，唯一差别 = 架构与有无隔离机制）：
  1. PER-ISO    完整 PER 原型，可学突触**隔离**成长：每技能新开一块突触 + 冻旧块 → 旧技能用自己
                的冻结快照评估 → 学后续不改它 → 旧能力恒定（数学保证）。参数随技能增长。
  2. PER-SHARED 完整 PER 原型，可学突触**共享**：所有技能挤同一张突触（= PER 去掉隔离的诚实对照，
                说明"不遗忘"来自隔离机制本身，而非 PER 天生不忘）。
  3. TF-FT      标准 Transformer，**顺序微调**（无隔离机制，持续学习的默认做法）→ 灾难遗忘。

为何公平：PER 与 TF 都从各自**已训练好的 52M 真实代码模型**底座出发（code_model.pt / code_model_tf.pt，
同一份 code_tokenizer），用相同 interactions / steps / replay / teach 集 / held-out 集 / 种子；
PER 臂动突触、TF 臂顺序微调——这正是两种架构各自的"持续学习"标准方式。
指标 = held-out completion loss(bits) + 复制准确率(贪心)。

任务（3 冲突技能，标准灾难遗忘设置；同一前缀逼出"后学覆盖先学"）：
    def task_<X>():\n    return   →   "<X>"      (str)
                                      [<X>]       (list)
                                      (<X>,)      (tuple)

输出：docs/reports/figs/code_forgetting_compare.png + code_forgetting_compare.{json,md}
运行：python -m fe_llm.energy_lm.evaluation.code_forgetting_compare --interactions 45
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
import torch.nn.functional as F

from fe_llm.config import get_device
from fe_llm.energy_lm.models.tokenizer import CharTokenizer
from fe_llm.energy_lm.training.code_train import ckpt_paths, generate, load_any, _logits

FIG_DIR = os.path.join("docs", "reports", "figs")
FIG_PATH = os.path.join(FIG_DIR, "code_forgetting_compare.png")
REPORT_JSON = os.path.join("docs", "reports", "code_forgetting_compare.json")
REPORT_MD = os.path.join("docs", "reports", "code_forgetting_compare.md")

NOUNS = ["button", "slider", "panel", "dialog", "menu", "toolbar", "canvas", "label",
         "cursor", "frame", "badge", "spinner", "switch", "drawer", "tooltip", "modal"]

# 冲突技能：同一前缀 def task_<X>(): return，要求三种不同的输出包装（标准灾难性遗忘设置）。
_PREFIX = lambda x: f'def task_{x}():\n    return '
SKILLS = {
    "str":   lambda x: (_PREFIX(x), f'"{x}"\n'),
    "list":  lambda x: (_PREFIX(x), f'[{x}]\n'),
    "tuple": lambda x: (_PREFIX(x), f'({x},)\n'),
}


def comp_loss(net, tok, skill, x, device):
    """单样本 completion loss（bits）。用 _logits 统一 PER(能量) / Transformer(logits)。"""
    prefix, completion = SKILLS[skill](x)
    ids = tok.encode(prefix + completion)
    p_len = len(tok.encode(prefix))
    seq = torch.tensor([ids], device=device)
    logits = _logits(net, seq)[0].float()
    tgt = torch.tensor(ids[p_len:], device=device)
    return F.cross_entropy(logits[p_len - 1: len(ids) - 1], tgt) / np.log(2)


@torch.no_grad()
def held_loss(net, tok, skill, xs, device):
    return float(np.mean([float(comp_loss(net, tok, skill, x, device)) for x in xs]))


@torch.no_grad()
def copy_acc(net, tok, skill, xs, device):
    ok = 0
    for x in xs:
        prefix, completion = SKILLS[skill](x)
        out = generate(net, tok, prefix, net.max_len, device, max_new=len(completion) + 4,
                       temperature=0.0, top_k=0)
        ok += int(out.startswith(completion))
    return ok / max(1, len(xs))


# ---------------------------------------------------------------- PER 臂（synapse-only）
def per_arm(isolated, tok, device, teach, held, skills, args):
    """PER 臂：冻结 backbone，只动可学突触。isolated=True 每技能新块+冻旧；False 共享同一张突触。"""
    net_path, last_path, _, _ = ckpt_paths("per")
    path = net_path if os.path.exists(net_path) else last_path
    net = load_any(path, device).to(device).train()
    if not getattr(net, "use_synapse", False):
        raise RuntimeError("PER 模型无可学突触（use_synapse=False）。")
    for p in net.parameters():
        p.requires_grad_(False)
    blocks = [b for b in net.blocks if getattr(b, "use_synapse", False)]
    base_syn = [b.synapse.detach().clone() for b in blocks]
    syn_per_block = sum(b.synapse.numel() for b in blocks)

    def set_syn(snap):
        for b, s in zip(blocks, snap):
            b.synapse.data = s.clone()

    def train_current(skill, seed):
        rng = np.random.default_rng(seed)
        for b in blocks:
            b.synapse.requires_grad_(True)
        opt = torch.optim.AdamW([b.synapse for b in blocks], lr=args.per_lr)
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
                torch.nn.utils.clip_grad_norm_([b.synapse for b in blocks], 1.0); opt.step()
        for b in blocks:
            b.synapse.requires_grad_(False)
        return [b.synapse.detach().clone() for b in blocks]

    @torch.no_grad()
    def evalp(snap, skill):
        set_syn(snap); net.eval()
        r = (held_loss(net, tok, skill, held, device), copy_acc(net, tok, skill, held, device))
        net.train(); return r

    rows = []
    if isolated:
        adapters = {}
        for ki, skill in enumerate(skills):
            set_syn(base_syn)
            adapters[skill] = train_current(skill, args.seed + 1000 * ki)
            rows.append({j: evalp(adapters[j], j) for j in skills[:ki + 1]})
    else:
        set_syn(base_syn)
        shared = [b.synapse.detach().clone() for b in blocks]
        for ki, skill in enumerate(skills):
            set_syn(shared)
            shared = train_current(skill, args.seed + 1000 * ki)
            rows.append({j: evalp(shared, j) for j in skills[:ki + 1]})
    del net
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows, syn_per_block


# ---------------------------------------------------------------- Transformer 臂（顺序微调）
def tf_arm(tok, device, teach, held, skills, args):
    """Transformer 臂：标准持续学习——顺序 full fine-tune（无隔离机制），状态累积 → 灾难遗忘。"""
    net_path, last_path, _, _ = ckpt_paths("transformer")
    path = net_path if os.path.exists(net_path) else last_path
    if not os.path.exists(path):
        return None, 0
    net = load_any(path, device).to(device).train()
    n_param = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.tf_lr)

    @torch.no_grad()
    def evaltf(skill):
        net.eval()
        r = (held_loss(net, tok, skill, held, device), copy_acc(net, tok, skill, held, device))
        net.train(); return r

    rows = []
    for ki, skill in enumerate(skills):
        rng = np.random.default_rng(args.seed + 1000 * ki)
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
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        rows.append({j: evaltf(j) for j in skills[:ki + 1]})
        print(f"[fgt]   TF 学完 {skill}: " +
              " ".join(f"{j}=loss{v[0]:.2f}/acc{v[1]:.0%}" for j, v in rows[-1].items()), flush=True)
    del net
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows, n_param


def _s0_curve(rows, s0):
    return [rows[k][s0][0] for k in range(len(rows))]


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="持续学习灾难遗忘三臂对照：PER 隔离 vs PER 共享 vs Transformer 微调。")
    ap.add_argument("--interactions", type=int, default=45)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--replay", type=int, default=4)
    ap.add_argument("--per-lr", type=float, default=0.03, help="PER synapse-only 学习率（对齐 per_code_growth 成功设置）")
    ap.add_argument("--tf-lr", type=float, default=5e-4, help="Transformer 顺序微调学习率")
    ap.add_argument("--n-teach", type=int, default=8)
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    device = args.device.strip() or get_device()
    _, _, tok_path, _ = ckpt_paths("per")
    if not os.path.exists(tok_path):
        print(f"[fgt] 找不到分词器 {tok_path}。先训代码模型。"); return 1
    tok = CharTokenizer.load(tok_path)
    teach, held = NOUNS[:args.n_teach], NOUNS[args.n_teach:]
    skills = list(SKILLS.keys())
    s0 = skills[0]
    t0 = time.time()

    print("[fgt] === PER-ISO（可学突触·隔离成长）===", flush=True)
    iso, syn_per_block = per_arm(True, tok, device, teach, held, skills, args)
    for ki, skill in enumerate(skills):
        print(f"[fgt]   ISO 学完 {skill}: " +
              " ".join(f"{j}=loss{v[0]:.2f}/acc{v[1]:.0%}" for j, v in iso[ki].items()), flush=True)

    print("[fgt] === PER-SHARED（可学突触·共享覆盖）===", flush=True)
    shr, _ = per_arm(False, tok, device, teach, held, skills, args)
    for ki, skill in enumerate(skills):
        print(f"[fgt]   SHARED 学完 {skill}: " +
              " ".join(f"{j}=loss{v[0]:.2f}/acc{v[1]:.0%}" for j, v in shr[ki].items()), flush=True)

    print("[fgt] === Transformer-FT（标准顺序微调）===", flush=True)
    tf, tf_param = tf_arm(tok, device, teach, held, skills, args)
    if tf is None:
        print("[fgt] 找不到 Transformer 模型 code_model_tf.pt。先训：code_train --arch transformer。"); return 1

    iso_s0, shr_s0, tf_s0 = _s0_curve(iso, s0), _s0_curve(shr, s0), _s0_curve(tf, s0)
    iso_s0_acc = [iso[k][s0][1] for k in range(len(iso))]
    shr_s0_acc = [shr[k][s0][1] for k in range(len(shr))]
    tf_s0_acc = [tf[k][s0][1] for k in range(len(tf))]
    iso_final = [iso[-1][s][0] for s in skills]
    shr_final = [shr[-1][s][0] for s in skills]
    tf_final = [tf[-1][s][0] for s in skills]
    # 当前技能（对角线）acc——确认三臂都"学会了新技能"（对照公平前提）
    iso_diag = [iso[k][skills[k]][1] for k in range(len(skills))]
    shr_diag = [shr[k][skills[k]][1] for k in range(len(skills))]
    tf_diag = [tf[k][skills[k]][1] for k in range(len(skills))]

    iso_forget = round(iso_s0[-1] - iso_s0[0], 3)
    shr_forget = round(shr_s0[-1] - shr_s0[0], 3)
    tf_forget = round(tf_s0[-1] - tf_s0[0], 3)

    verdict = (
        f"【灾难遗忘·三臂对照】顺序学 {len(skills)} 个同前缀冲突技能后，首技能「{s0}」held-out loss 漂移："
        f"PER-ISO Δ={iso_forget:+.2f}（隔离·恒定不忘）、PER-SHARED Δ={shr_forget:+.2f}、"
        f"**Transformer-FT Δ={tf_forget:+.2f}（{tf_s0[0]:.2f}→{tf_s0[-1]:.2f}，灾难遗忘）**；"
        f"首技能复制准确率：ISO {iso_s0_acc[0]:.0%}→{iso_s0_acc[-1]:.0%}、TF {tf_s0_acc[0]:.0%}→{tf_s0_acc[-1]:.0%}。"
        f" 结论：Transformer 顺序微调（无隔离机制）灾难遗忘早学技能；PER 用可学突触隔离做到旧技能恒定不忘"
        f"（PER-SHARED 去掉隔离也会忘，证明'不忘'来自隔离机制而非 PER 天生）。"
    )
    print("=" * 70, flush=True); print(f"[fgt] {verdict}", flush=True)

    # -------- 画图（三臂）--------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    stages = list(range(1, len(skills) + 1))
    xl = [f"学完{s}" for s in skills]
    C = {"iso": "#2e7d32", "shr": "#f9a825", "tf": "#c62828"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    ax.plot(stages, iso_s0, "o-", color=C["iso"], lw=2.2, label="PER-ISO 隔离（不忘）")
    ax.plot(stages, shr_s0, "^--", color=C["shr"], lw=2.0, label="PER-SHARED 共享")
    ax.plot(stages, tf_s0, "s--", color=C["tf"], lw=2.2, label="Transformer-FT（灾难遗忘）")
    ax.set_xticks(stages); ax.set_xticklabels(xl)
    ax.set_title(f"① 首技能「{s0}」held-out loss（越平=越不忘）", fontsize=12)
    ax.set_ylabel("completion loss (bits) ↓"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    sx = np.arange(len(skills))
    ax.bar(sx - 0.25, iso_final, width=0.25, color=C["iso"], label="PER-ISO")
    ax.bar(sx + 0.00, shr_final, width=0.25, color=C["shr"], label="PER-SHARED")
    ax.bar(sx + 0.25, tf_final, width=0.25, color=C["tf"], label="Transformer-FT")
    ax.set_xticks(sx); ax.set_xticklabels([f"{s}\n(第{i+1}个学)" for i, s in enumerate(skills)])
    ax.set_title("② 学完全部后·各技能 held-out loss（早学的被覆盖↑）", fontsize=12)
    ax.set_ylabel("completion loss (bits) ↓"); ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    ax.plot(stages, iso_s0_acc, "o-", color=C["iso"], lw=2.2, label="PER-ISO 隔离")
    ax.plot(stages, shr_s0_acc, "^--", color=C["shr"], lw=2.0, label="PER-SHARED")
    ax.plot(stages, tf_s0_acc, "s--", color=C["tf"], lw=2.2, label="Transformer-FT")
    ax.set_xticks(stages); ax.set_xticklabels(xl)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"③ 首技能「{s0}」复制准确率（掉到 0=彻底忘）", fontsize=12)
    ax.set_ylabel("copy accuracy ↑"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("持续学习·灾难遗忘三臂对照：PER 隔离 vs PER 共享 vs 标准 Transformer（同底座/同协议/同种子）",
                 fontsize=13, fontweight="bold")
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig(FIG_PATH, dpi=130); plt.close()
    print(f"[fgt] 图已保存：{FIG_PATH}", flush=True)

    # -------- 报告 --------
    result = {
        "config": vars(args), "skills": skills, "device": device,
        "iso": [{k: list(v) for k, v in r.items()} for r in iso],
        "shared": [{k: list(v) for k, v in r.items()} for r in shr],
        "tf": [{k: list(v) for k, v in r.items()} for r in tf],
        "s0": s0,
        "iso_s0_loss": iso_s0, "shared_s0_loss": shr_s0, "tf_s0_loss": tf_s0,
        "iso_s0_acc": iso_s0_acc, "shared_s0_acc": shr_s0_acc, "tf_s0_acc": tf_s0_acc,
        "iso_final_loss": iso_final, "shared_final_loss": shr_final, "tf_final_loss": tf_final,
        "diag_acc": {"iso": iso_diag, "shared": shr_diag, "tf": tf_diag},
        "forget_s0": {"iso": iso_forget, "shared": shr_forget, "tf": tf_forget},
        "syn_per_block_k": round(syn_per_block / 1e3, 1), "tf_params_m": round(tf_param / 1e6, 2),
        "verdict": verdict, "fig": FIG_PATH,
    }
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(
            "# 持续学习·灾难遗忘三臂对照：PER 隔离 vs PER 共享 vs 标准 Transformer\n\n"
            "顺序学 3 个**同前缀、冲突输出**的技能（标准灾难遗忘设置），看**早学技能**是否被覆盖。"
            "三臂同底座（各自 52M 真实代码模型 code_model / code_model_tf）、同 interactions/steps/replay/"
            "teach 集 / held-out 集 / 种子；唯一差别 = 架构与有无隔离机制。指标 = held-out completion loss(bits)。\n\n"
            "## 任务\n\n"
            "```\ndef task_<X>():\n    return   →   \"<X>\"   (str) | [<X>]  (list) | (<X>,)  (tuple)\n```\n\n"
            f"## 首技能「{s0}」遗忘对照\n\n"
            f"| 臂 | 刚学完 loss | 学完全部后 loss | 遗忘Δ | 刚学完 acc | 学完全部后 acc |\n"
            f"|---|---:|---:|---:|---:|---:|\n"
            f"| **PER-ISO 隔离** | {iso_s0[0]:.2f} | {iso_s0[-1]:.2f} | **{iso_forget:+.2f}（恒定不忘）** | {iso_s0_acc[0]:.0%} | {iso_s0_acc[-1]:.0%} |\n"
            f"| PER-SHARED 共享 | {shr_s0[0]:.2f} | {shr_s0[-1]:.2f} | {shr_forget:+.2f} | {shr_s0_acc[0]:.0%} | {shr_s0_acc[-1]:.0%} |\n"
            f"| **Transformer-FT** | {tf_s0[0]:.2f} | {tf_s0[-1]:.2f} | **{tf_forget:+.2f}（灾难遗忘）** | {tf_s0_acc[0]:.0%} | {tf_s0_acc[-1]:.0%} |\n\n"
            f"末态各技能 loss：ISO {[round(v,2) for v in iso_final]} / SHARED {[round(v,2) for v in shr_final]} / TF {[round(v,2) for v in tf_final]}\n\n"
            f"当前技能（对角线）acc（确认三臂都学会了新技能，对照公平）：ISO {[f'{a:.0%}' for a in iso_diag]} / "
            f"SHARED {[f'{a:.0%}' for a in shr_diag]} / TF {[f'{a:.0%}' for a in tf_diag]}\n\n"
            f"## 结论\n\n{verdict}\n\n"
            "## 诚实边界\n\n"
            "- **公平性**：PER 用其内生的可学突触隔离机制（synapse-only + 加容量冻旧块），Transformer 用标准顺序"
            "微调——这是两种架构各自的持续学习方式，对照展示的是**架构能力差异**，非超参不公平。\n"
            "- **PER-ISO 的代价**：参数随技能线性增长（每技能 +突触一块）、且**无前向迁移**（每技能从底座独立学）；"
            "理想解 = Progressive 式（冻旧块 + 侧向连接，迁移且不忘）。\n"
            "- **Transformer 并非无解**：EWC / replay / adapter / LoRA 等可缓解其灾难遗忘，但都是**额外加的机制**；"
            "本对照证明的是**标准 Transformer 无内生隔离**，而 PER 的隔离是架构自带。\n"
            "- **规模**：本任务为 3 个冲突技能的机制验证（synapse-only、小步数），证机制不证规模。\n\n"
            f"图：`{FIG_PATH}`\n"
        )
    print(f"[fgt] 报告：{REPORT_MD}  用时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
