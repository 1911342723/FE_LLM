# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/evaluation/code_growth_eval.py —— "随交互成长"证明（synapse-only 在线学习）
================================================================================================
证明 PER 能**随交互次数逐步成长**：冻结 backbone、**只更新可学突触 S**（辨识点 #2，
"经验刻进结构记忆"），把一个**复制规则**任务的示例一条条喂进去（=一次次交互），每次做几步
synapse-only 梯度更新，量它在三组上的表现随交互次数的变化：

  - teach（教过的实例）   ：loss 应快速下降（学到了）
  - held-out（没教过、同规则的新实例）：loss 下降 = **真成长/泛化**，不是死记
  - control（无关代码）   ：loss 基本不变 = 没把旧能力忘掉（升高=灾难性遗忘，如实呈现）

任务（复制规则，正是路由/突触的拿手活）：
    def make_<X>():
        return Widget("<X>")
  模型需把函数名 <X> 复制进字符串。teach 用一批 <X>，held-out 用另一批没见过的 <X>。

输出：docs/reports/figs/per_code_growth.png（三条成长曲线 + 复制准确率 + 突触Δ热图）
      docs/reports/per_code_growth.{json,md}

运行：
    python -m fe_llm.energy_lm.evaluation.code_growth_eval --interactions 40 --steps 4 --lr 0.03
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
from fe_llm.energy_lm.training.code_train import CORPUS, ckpt_paths, generate, load_any

FIG_DIR = os.path.join("docs", "reports", "figs")
FIG_PATH = os.path.join(FIG_DIR, "per_code_growth.png")
REPORT_JSON = os.path.join("docs", "reports", "per_code_growth.json")
REPORT_MD = os.path.join("docs", "reports", "per_code_growth.md")

NOUNS = ["button", "slider", "panel", "dialog", "menu", "toolbar", "canvas", "label",
         "cursor", "frame", "badge", "spinner", "switch", "drawer", "tooltip", "modal"]


def split_ex(x: str):
    prefix = f'def make_{x}():\n    return Widget("'
    completion = f'{x}")\n'
    return prefix, completion


def comp_loss(net, tok, x, device):
    """completion 区（复制出 <X> 那段）的 next-char 交叉熵（bits）。"""
    prefix, completion = split_ex(x)
    ids = tok.encode(prefix + completion)
    p_len = len(tok.encode(prefix))
    seq = torch.tensor([ids], device=device)
    out = net(seq)[0]                       # energy=-logits
    logits = (-out).float()
    tgt = torch.tensor(ids[p_len:], device=device)
    pred = logits[p_len - 1: len(ids) - 1]
    return F.cross_entropy(pred, tgt) / np.log(2)


def control_loss(net, tok, snippets, device):
    tot = 0.0
    for s in snippets:
        ids = tok.encode(s)
        seq = torch.tensor([ids], device=device)
        logits = (-net(seq)[0]).float()
        tgt = torch.tensor(ids[1:], device=device)
        tot += float(F.cross_entropy(logits[:-1], tgt) / np.log(2))
    return tot / max(1, len(snippets))


@torch.no_grad()
def copy_acc(net, tok, xs, device):
    ok = 0
    for x in xs:
        prefix, completion = split_ex(x)
        out = generate(net, tok, prefix, net.max_len, device, max_new=len(completion) + 4,
                       temperature=0.0, top_k=0)
        if out.startswith(x + '")'):
            ok += 1
    return ok / max(1, len(xs))


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="PER 随交互成长证明（synapse-only）。")
    ap.add_argument("--arch", default="per")
    ap.add_argument("--interactions", type=int, default=40)
    ap.add_argument("--steps", type=int, default=4, help="每次交互的 synapse 梯度步数")
    ap.add_argument("--replay", type=int, default=4, help="经验回放每步采样的已见样本数")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--n-teach", type=int, default=8)
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    device = args.device.strip() or get_device()
    net_path, last_path, tok_path, _ = ckpt_paths(args.arch)
    path = net_path if os.path.exists(net_path) else last_path
    if not (os.path.exists(path) and os.path.exists(tok_path)):
        print(f"[grow] 找不到 {args.arch} 模型/分词器。"); return 1
    tok = CharTokenizer.load(tok_path)
    net = load_any(path, device).to(device).train()
    if not getattr(net, "use_synapse", False):
        print("[grow] 该模型无可学突触，无法做 synapse-only 成长。"); return 1

    # 冻结除突触外的一切
    for p in net.parameters():
        p.requires_grad_(False)
    syn_params = []
    for b in net.blocks:
        if getattr(b, "use_synapse", False):
            b.synapse.requires_grad_(True)
            syn_params.append(b.synapse)
    n_syn = sum(p.numel() for p in syn_params)
    print(f"[grow] 设备={device} 仅训突触参数={n_syn/1e3:.0f}k（backbone 冻结）", flush=True)

    teach = NOUNS[:args.n_teach]
    held = NOUNS[args.n_teach:]
    # control：从训练语料取几小段无关代码
    control = []
    if os.path.exists(CORPUS):
        with open(CORPUS, "r", encoding="utf-8") as f:
            blob = f.read(200000)
        parts = [p for p in blob.split("\n\n") if 60 < len(p) < 240][:4]
        control = parts
    if not control:
        control = ["import os\nimport sys\n\ndef read(path):\n    with open(path) as f:\n        return f.read()\n"]

    syn0 = F.softplus(net.blocks[-1].synapse.detach()).cpu().numpy().copy()
    opt = torch.optim.AdamW(syn_params, lr=args.lr)

    rng = np.random.default_rng(args.seed)
    hist = {"interaction": [], "teach": [], "held": [], "control": [], "copy_held": []}

    def snapshot(it):
        net.eval()
        with torch.no_grad():
            tl = float(np.mean([float(comp_loss(net, tok, x, device)) for x in teach]))
            hl = float(np.mean([float(comp_loss(net, tok, x, device)) for x in held]))
            cl = control_loss(net, tok, control, device)
        ca = copy_acc(net, tok, held, device)
        net.train()
        hist["interaction"].append(it); hist["teach"].append(round(tl, 4))
        hist["held"].append(round(hl, 4)); hist["control"].append(round(cl, 4))
        hist["copy_held"].append(round(ca, 3))
        print(f"[grow] it {it:3d} | teach {tl:.3f} | held {hl:.3f} | control {cl:.3f} | copy_held {ca:.0%}", flush=True)

    t0 = time.time()
    snapshot(0)
    buffer: list[str] = []   # 经验回放缓冲：只含"已交互过"的样本（在线、不偷看未来）
    for it in range(1, args.interactions + 1):
        x = teach[(it - 1) % len(teach)]
        if x not in buffer:
            buffer.append(x)
        for _ in range(args.steps):
            k = min(len(buffer), args.replay)
            bx = list(rng.choice(buffer, size=k, replace=False)) if len(buffer) > 1 else buffer
            loss = torch.stack([comp_loss(net, tok, xi, device) for xi in bx]).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(syn_params, 1.0); opt.step()
        if it % 2 == 0 or it == args.interactions:
            snapshot(it)

    syn1 = F.softplus(net.blocks[-1].synapse.detach()).cpu().numpy()
    dsyn = syn1 - syn0

    # 结论
    teach_drop = round(hist["teach"][0] - hist["teach"][-1], 3)
    held_drop = round(hist["held"][0] - hist["held"][-1], 3)
    ctrl_delta = round(hist["control"][-1] - hist["control"][0], 3)
    copy0, copy1 = hist["copy_held"][0], hist["copy_held"][-1]
    if held_drop > 0.3 and copy1 > copy0:
        verdict = (f"✅ 成长成立：held-out（没教过的同规则实例）loss 随交互下降 {held_drop:+.2f} bits、"
                   f"复制准确率 {copy0:.0%}→{copy1:.0%}——只动突触就把规则刻进结构记忆并泛化到新实例。")
    elif teach_drop > 0.3:
        verdict = (f"🟡 部分成长：教过的实例 loss 降 {teach_drop:+.2f}（学到了），但 held-out 泛化有限"
                   f"（{held_drop:+.2f}、复制 {copy0:.0%}→{copy1:.0%}）——synapse-only 偏记忆、泛化弱（#2 边界）。")
    else:
        verdict = (f"⛔ 未见明显成长：synapse-only 在该任务上 loss 几乎不降（teach {teach_drop:+.2f}）——"
                   f"该规则可能超出突触路由可表达范围（诚实负）。")
    ctrl_msg = (f"对照（无关代码）loss 变化 {ctrl_delta:+.3f} bits："
                f"{'基本无遗忘' if abs(ctrl_delta) < 0.1 else '有一定遗忘（#2 抗遗忘弱，如实）' if ctrl_delta > 0 else '意外下降'}。")
    print("=" * 64, flush=True)
    print(f"[grow] {verdict}", flush=True)
    print(f"[grow] {ctrl_msg}", flush=True)

    # 画图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    its = hist["interaction"]
    ax = axes[0]
    ax.plot(its, hist["teach"], "o-", label="teach（教过）", color="#2e7d32")
    ax.plot(its, hist["held"], "s-", label="held-out（没教过·同规则）", color="#1565c0")
    ax.plot(its, hist["control"], "^--", label="control（无关代码）", color="#9e9e9e")
    ax.set_title("① 成长曲线：completion loss 随交互次数", fontsize=12)
    ax.set_xlabel("交互次数"); ax.set_ylabel("bits/char ↓"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(its, [c * 100 for c in hist["copy_held"]], "D-", color="#6a1b9a")
    ax.set_title("② held-out 复制准确率随交互（泛化）", fontsize=12)
    ax.set_xlabel("交互次数"); ax.set_ylabel("复制正确率 %"); ax.set_ylim(-5, 105); ax.grid(alpha=0.3)
    ax = axes[2]
    crop = min(56, dsyn.shape[0])          # 样本短，变化集中在前若干位置，裁剪到活跃区
    dview = dsyn[:crop, :crop]
    vmax = float(np.abs(dview).max()) or 1.0
    im = ax.imshow(dview, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"③ 突触 S 变化量 Δ(末层·前{crop}位)：经验刻进通路", fontsize=12)
    ax.set_xlabel("源位置 i"); ax.set_ylabel("目标位置 j")
    fig.suptitle("PER 随交互成长（synapse-only · 冻结 backbone）", fontsize=13, fontweight="bold")
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig(FIG_PATH, dpi=130); plt.close()
    print(f"[grow] 图已保存：{FIG_PATH}", flush=True)

    result = {"config": {"arch": args.arch, "interactions": args.interactions, "steps": args.steps,
                         "lr": args.lr, "n_teach": len(teach), "n_held": len(held), "syn_params": n_syn},
              "history": hist, "teach_drop": teach_drop, "held_drop": held_drop,
              "control_delta": ctrl_delta, "copy_held_start": copy0, "copy_held_end": copy1,
              "verdict": verdict, "control_note": ctrl_msg, "fig": FIG_PATH}
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(f"# PER 随交互成长证明（synapse-only · 冻结 backbone）\n\n"
                f"任务：复制规则 `def make_<X>(): return Widget(\"<X>\")`，teach {len(teach)} 个 X、"
                f"held-out {len(held)} 个没教过的 X。只更新可学突触 S（{n_syn/1e3:.0f}k 参数）。\n\n"
                f"- teach loss {hist['teach'][0]}→{hist['teach'][-1]}（降 {teach_drop}）\n"
                f"- **held-out loss {hist['held'][0]}→{hist['held'][-1]}（降 {held_drop}）**\n"
                f"- held-out 复制准确率 {copy0:.0%}→{copy1:.0%}\n"
                f"- control（无关代码）{hist['control'][0]}→{hist['control'][-1]}（Δ{ctrl_delta:+.3f}）\n\n"
                f"## 结论\n\n{verdict}\n\n{ctrl_msg}\n\n图：`{FIG_PATH}`\n")
    print(f"[grow] 报告：{REPORT_MD}  用时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
