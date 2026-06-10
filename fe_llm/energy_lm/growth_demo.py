# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/growth_demo.py —— 规模版上的零重训成长
========================================================
验证：在 1502 条对话训练的能量对话模型上，**不动任何权重**，只往外挂经验记忆库
加几条新对话，模型立刻就能正确应答这些原本答错的输入。

这是"经验=能量地貌上的沟"哲学的直接体现：新经验 = 新刻一条低能沟（外挂记忆），
推理时坍缩被引导走向它，无需重训网络。对照 Transformer：学新对话必须重训权重。

运行：
    python -m fe_llm.energy_lm.growth_demo
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fe_llm.config import get_device
from fe_llm.energy_lm.collapse import EnergyCollapseChat, MemoryBank
from fe_llm.energy_lm.energy_net import DialogueEnergyNet
from fe_llm.energy_lm.tokenizer import CharTokenizer
from fe_llm.energy_lm.train import CKPT_NET, CKPT_TOK

# 训练语料未覆盖的输入（模型大概率答非所问），及我们要"教"给它的正确回应
NEW_KNOWLEDGE = [
    ("一加一等于几", "等于二"),
    ("二加二等于几", "等于四"),
    ("地球是什么形状", "是圆的"),
    ("水的沸点是多少", "一百度"),
    ("中国的首都是哪", "是北京"),
]


def main():
    if not os.path.exists(CKPT_NET):
        print("未找到权重，请先 python -m fe_llm.energy_lm.train")
        return
    device = get_device()
    net = DialogueEnergyNet.load(CKPT_NET, map_location=device)
    tok = CharTokenizer.load(CKPT_TOK)

    print("=" * 64)
    print("规模版零重训成长：冻结权重，只往经验记忆库加新对话")
    print("=" * 64)

    # —— A. 注入前：用未挂记忆的模型回答（大概率答非所问）——
    chat_before = EnergyCollapseChat(net, tok, device=device)
    print("\n— A. 注入前（权重已训练，但没见过这些问答）—")
    before = {}
    for p, _ in NEW_KNOWLEDGE:
        r, _info = chat_before.respond(p)
        before[p] = r
        print(f"  你：{p}　→　模型：{r}")

    # —— B. 零重训注入：把新对话挂进经验记忆库（不动一个权重）——
    print("\n— B. 把新知识挂进经验记忆库（零重训，权重未变）—")
    mem = MemoryBank()
    for p, r in NEW_KNOWLEDGE:
        mem.add(p, r)
    print(f"  记忆库现有 {len(mem)} 条经验（外挂，不在网络权重里）")

    # —— C. 注入后：同一冻结网络 + 记忆库 ——
    chat_after = EnergyCollapseChat(net, tok, device=device, memory=mem)
    print("\n— C. 注入后（同一冻结网络，挂上记忆库）—")
    n_ok = 0
    for p, gold in NEW_KNOWLEDGE:
        r, _info = chat_after.respond(p)
        ok = (gold in r) or (r == gold)
        n_ok += int(ok)
        print(f"  你：{p}　→　模型：{r}　{'[对]' if ok else '[错]'}")

    # —— D. 不破坏原能力：原训练对话仍正常 ——
    print("\n— D. 原有能力未被破坏（训练过的对话仍正常）—")
    for p in ["你好", "谢谢", "今天天气怎么样"]:
        r, _ = chat_after.respond(p)
        print(f"  你：{p}　→　模型：{r}")

    print("\n" + "=" * 64)
    print("【裁决】")
    print(f"  注入前答对：{sum(1 for p,g in NEW_KNOWLEDGE if g in before[p])}/{len(NEW_KNOWLEDGE)}")
    print(f"  注入后答对：{n_ok}/{len(NEW_KNOWLEDGE)}")
    print(f"  网络权重改动：0（仅往外挂记忆库加 {len(mem)} 条）")
    grew = n_ok > sum(1 for p, g in NEW_KNOWLEDGE if g in before[p])
    print("\n  结论：" + (
        "[通过] 零重训成长成立——加几条经验记忆，冻结网络立刻会用。\n"
        "        新经验 = 能量地貌上新刻的低能沟，不必重训权重。\n"
        "        这是 Transformer 结构上做不到的（学新知识须重训）。"
        if grew else "[未通过] 记忆引导未生效，需调 mem_bonus/阈值。"))


if __name__ == "__main__":
    main()
