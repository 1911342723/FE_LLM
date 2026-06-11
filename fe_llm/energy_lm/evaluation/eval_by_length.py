# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/eval_by_length.py —— 按回应长度分桶评测（找字序天花板）
======================================================================
这次规模化实验的核心观测工具。非自回归并行去掩码的已知结构性难点是：
**回应越长，整句坍缩越容易字序错乱/重复/漂移。** 本脚本量化这一点：

把验证对话按"真实回应字数"分桶（1-4 / 5-8 / 9-12 / 13-16 / 17+），每桶分别测：
    1. 去掩码填字准确率：随机掩回应区，模型填回真字的比例（判别能力）。
    2. 整句生成精确匹配：从全 [MASK] 退火坍缩，生成回应与真值完全一致的比例（生成能力）。
    3. 生成字序错乱率：生成回应里的字虽多来自真值但顺序错的比例（字序专项）。

若准确率随桶长单调下降、长桶骤降，即定位了"长回应字序崩"的临界长度。

运行：
    python -m fe_llm.energy_lm.evaluation.eval_by_length --n 600
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.nn.functional as F

from fe_llm.config import get_device
from fe_llm.energy_lm.diagnostics.collapse import EnergyCollapseChat
from fe_llm.energy_lm.data.corpus import load_dialogues
from fe_llm.energy_lm.models.energy_net import DialogueEnergyNet
from fe_llm.energy_lm.models.tokenizer import CharTokenizer
from fe_llm.energy_lm.training.train import CKPT_NET, CKPT_TOK, build_example

BUCKETS = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 99)]


def _bucket(n: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi if hi < 99 else '∞'}"
    return "?"


@torch.no_grad()
def fill_accuracy(net, tok, device, pairs):
    """去掩码填字准确率（按桶）。掩回应区 60%，看填回真字比例。"""
    rng = random.Random(0)
    hit = Counter(); tot = Counter()
    for p, r in pairs:
        seq, is_resp = build_example(tok, p, r)
        seq_t = torch.tensor([seq], device=device)
        resp_pos = [i for i, b in enumerate(is_resp) if b]
        # 只评真实内容字位置（非 EOS/PAD），更贴近"字序对不对"
        content = [i for i in resp_pos
                   if not tok.is_special(seq[i])]
        if not content:
            continue
        k = max(1, int(len(content) * 0.6))
        masked = rng.sample(content, k)
        inp = seq_t.clone()
        for i in masked:
            inp[0, i] = tok.mask_id
        logits = -net(inp)[0]
        b = _bucket(len(r))
        for i in masked:
            pred = int(logits[i].argmax())
            hit[b] += int(pred == seq[i]); tot[b] += 1
    return hit, tot


@torch.no_grad()
def gen_match(chat, tok, pairs, cap: int = 120):
    """整句生成评测（按桶）：精确匹配率 + 字序错乱率。
    退火坍缩较慢，最多评 cap 条（按桶均衡抽样）。"""
    exact = Counter(); reorder = Counter(); tot = Counter()
    use = pairs[:cap]
    for p, r in use:
        out, _ = chat.respond(p)
        b = _bucket(len(r))
        tot[b] += 1
        if out == r:
            exact[b] += 1
        elif Counter(out) == Counter(r):
            # 字相同但顺序不同 = 纯字序错乱
            reorder[b] += 1
    return exact, reorder, tot


def _show(title, hit, tot):
    print(f"\n  {title}")
    for lo, hi in BUCKETS:
        b = f"{lo}-{hi if hi < 99 else '∞'}"
        if tot[b]:
            print(f"    回应{b:>5}字：{hit[b]/tot[b]:6.1%}  (n={tot[b]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="评测样本数")
    args = ap.parse_args()

    if not os.path.exists(CKPT_NET):
        print("未找到权重，请先训练")
        return
    device = get_device()
    net = DialogueEnergyNet.load(CKPT_NET, map_location=device).to(device).eval()
    tok = CharTokenizer.load(CKPT_TOK)
    chat = EnergyCollapseChat(net, tok, device=device)

    dias = load_dialogues()
    rng = random.Random(42)
    sample = rng.sample(dias, min(args.n, len(dias)))

    print("=" * 64)
    print(f"按回应长度分桶评测（n={len(sample)}）—— 定位非自回归字序天花板")
    print("=" * 64)

    hit, tot = fill_accuracy(net, tok, device, sample)
    _show("① 去掩码填字准确率（判别：掩了能不能填回）", hit, tot)

    exact, reorder, tot2 = gen_match(chat, tok, sample)
    _show("② 整句生成精确匹配率（生成：从全MASK坍缩=真值）", exact, tot2)
    _show("③ 字序错乱率（字对但顺序错／生成专项）", reorder, tot2)

    print("\n" + "=" * 64)
    print("读法：①高②低说明'会判别但不会组装'，差距随长度拉大即字序天花板；")
    print("      ③随长度上升直接量化字序错乱。")


if __name__ == "__main__":
    main()
