# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/evaluation/code_trace_viz.py —— PER 代码模型「可溯源 + 特性」可视化
=====================================================================================
把 PER（SeqEnergyNet）区别于黑箱注意力的**可溯源特性**，对一段代码前缀跑一次前向后渲染出来：

  1. 逐字惊奇/能量轨迹：每个字符 -log2 P(实际下一字)（bits），看模型在代码哪处确信/意外。
  2. 跨 depth 预测误差弛豫：每层平均 ||ε|| 随深度下降——预测编码"能量下降"的直接证据。
  3. 各层弛豫步长 η：可学的弛豫速率（attention 没有的动力系统量）。
  4. 可学突触 S 结构记忆：softplus(synapse) 热图（持久、可定位、可干预的位置依赖）。
  5. 内容寻址路由 g：某层的 g 矩阵（类注意力图），看实际生成时的路由。

依赖 CausalPERBlock 的 _capture 钩子（默认关闭、零开销）抓取 g 与 ||ε||。
输出图：docs/reports/figs/per_code_trace.png（并打印文本溯源摘要）。

运行：
    python -m fe_llm.energy_lm.evaluation.code_trace_viz --prompt "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr"
    python -m fe_llm.energy_lm.evaluation.code_trace_viz   # 用默认 prompt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn.functional as F

from fe_llm.config import get_device
from fe_llm.energy_lm.models.tokenizer import CharTokenizer
from fe_llm.energy_lm.training.code_train import CKPT_TOK, ckpt_paths, load_any

FIG_DIR = os.path.join("docs", "reports", "figs")
FIG_PATH = os.path.join(FIG_DIR, "per_code_trace.png")

DEFAULT_PROMPT = "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n"


def _disp(ch: str) -> str:
    # 用 ASCII 安全符号，避免 matplotlib 字体缺字
    return {"\n": "\\n", " ": "_", "\t": "\\t"}.get(ch, ch)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="PER 代码模型可溯源 + 特性可视化。")
    ap.add_argument("--arch", choices=["per", "transformer"], default="per")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-chars", type=int, default=56, help="可视化展示的最长字符数（太长热图会糊）")
    ap.add_argument("--routing-layer", type=int, default=-1, help="展示哪一层的路由 g（-1=最后一层）")
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default=FIG_PATH)
    args = ap.parse_args(argv)

    device = args.device.strip() or get_device()
    net_path, last_path, tok_path, _ = ckpt_paths(args.arch)
    path = net_path if os.path.exists(net_path) else last_path
    if not (os.path.exists(path) and os.path.exists(tok_path)):
        print(f"[viz] 找不到 {args.arch} 模型或分词器，请先训练。"); return 1

    tok = CharTokenizer.load(tok_path)
    net = load_any(path, device).to(device).eval()
    if not getattr(net, "use_synapse", False) and args.arch == "per":
        print("[viz] 警告：该 PER 模型未启用突触，突触面板将为空。", flush=True)

    # 截断到可视长度（命令行里的字面 \n / \t 还原成真换行/制表）
    prompt = args.prompt.replace("\\n", "\n").replace("\\t", "\t")
    ids = tok.encode(prompt)[: min(args.max_chars, net.max_len)]
    chars = [tok.id_to_tok[i] for i in ids]
    L = len(ids)
    if L < 3:
        print("[viz] prompt 太短。"); return 1
    seq = torch.tensor([ids], device=device)

    # 打开捕获，跑一次前向
    blocks = list(net.blocks)
    for b in blocks:
        b._capture = True
    with torch.no_grad():
        energy = net(seq)                      # (1,L,V)，energy=-logits
    logits = -energy[0]                          # (L,V)
    logp = F.log_softmax(logits.float(), dim=-1)

    # 1) 逐字惊奇（bits）：位置 i 预测下一字 ids[i+1]
    surprise = []
    for i in range(L - 1):
        surprise.append(float(-logp[i, ids[i + 1]] / np.log(2)))
    # 2) 跨层平均 ||ε||
    err_per_layer = [float(b.cap_err_norm[0].mean()) for b in blocks]
    # 3) η per layer
    eta_per_layer = [float(b.eta.detach()) for b in blocks]
    # 4) synapse 热图（首/末层）
    syn_mats = []
    if getattr(net, "use_synapse", False):
        for b in blocks:
            syn_mats.append(F.softplus(b.synapse.detach()[:L, :L]).cpu().numpy())
    # 5) routing g（指定层）
    li = args.routing_layer % len(blocks)
    g_mat = blocks[li].cap_g[0].cpu().numpy()    # (L,L)

    # ---- 文本溯源摘要 ----
    print("=" * 64, flush=True)
    print(f"[viz] arch={args.arch} 模型={os.path.basename(path)} 序列长度={L}", flush=True)
    order = np.argsort(surprise)[::-1]
    print("[viz] 最意外的 5 个下一字（高能量/高惊奇）：", flush=True)
    for k in order[:5]:
        ctx = "".join(_disp(c) for c in chars[max(0, k - 6):k + 1])
        print(f"        '{ctx}' →预测下一字 '{_disp(chars[k+1])}'  惊奇={surprise[k]:.2f} bits", flush=True)
    print(f"[viz] 跨层平均 ||ε||（弛豫）：{[round(x,3) for x in err_per_layer]}", flush=True)
    print(f"[viz]   首层 {err_per_layer[0]:.3f} → 末层 {err_per_layer[-1]:.3f} "
          f"({'下降' if err_per_layer[-1] < err_per_layer[0] else '未降'})", flush=True)
    print(f"[viz] 各层 η：{[round(x,3) for x in eta_per_layer]}", flush=True)

    # ---- 渲染 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"PER 代码模型 · 可溯源 + 特性可视化（{os.path.basename(path)}, 序列 {L} 字符）",
                 fontsize=14, fontweight="bold")

    # A 逐字惊奇
    ax = axes[0, 0]
    xs = np.arange(L - 1)
    cols = plt.cm.RdYlGn_r(np.clip(np.array(surprise) / max(1e-6, max(surprise)), 0, 1))
    ax.bar(xs, surprise, color=cols)
    ax.set_xticks(xs)
    ax.set_xticklabels([_disp(chars[i + 1]) for i in range(L - 1)], fontsize=7, rotation=0)
    ax.set_title("① 逐字惊奇 / 能量轨迹（bits，越高越意外）", fontsize=11)
    ax.set_ylabel("-log2 P(实际下一字)")
    ax.grid(axis="y", alpha=0.3)

    # B 跨层误差弛豫
    ax = axes[0, 1]
    ax.plot(range(1, len(err_per_layer) + 1), err_per_layer, "o-", color="#d2691e", lw=2)
    ax.set_title("② 各层预测误差 ||ε||（逐层弛豫量·每层重归一,非全局单调）", fontsize=11)
    ax.set_xlabel("PER 层（=弛豫迭代）"); ax.set_ylabel("平均 ||ε||")
    ax.grid(alpha=0.3)

    # C η per layer
    ax = axes[0, 2]
    ax.bar(range(1, len(eta_per_layer) + 1), eta_per_layer, color="#4a90d9")
    ax.set_title("③ 各层弛豫步长 η（可学，attention 无此量）", fontsize=11)
    ax.set_xlabel("PER 层"); ax.set_ylabel("η")
    ax.grid(axis="y", alpha=0.3)

    # D synapse 首层
    ax = axes[1, 0]
    if syn_mats:
        im = ax.imshow(syn_mats[0], cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("④ 可学突触 S · 首层（持久结构记忆）", fontsize=11)
        ax.set_xlabel("源位置 i"); ax.set_ylabel("目标位置 j")
    else:
        ax.text(0.5, 0.5, "无突触（阉割/Transformer）", ha="center", va="center"); ax.axis("off")

    # E synapse 末层
    ax = axes[1, 1]
    if syn_mats:
        im = ax.imshow(syn_mats[-1], cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("⑤ 可学突触 S · 末层（经验刻成的低阻通路）", fontsize=11)
        ax.set_xlabel("源位置 i"); ax.set_ylabel("目标位置 j")
    else:
        ax.text(0.5, 0.5, "无突触", ha="center", va="center"); ax.axis("off")

    # F routing g
    ax = axes[1, 2]
    im = ax.imshow(g_mat, cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"⑥ 内容寻址路由 g · 第 {li+1} 层（因果，类注意力）", fontsize=11)
    ax.set_xlabel("被汇聚位置 i"); ax.set_ylabel("当前位置 j")

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(args.out, dpi=130)
    plt.close()
    print(f"[viz] 图已保存：{args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
