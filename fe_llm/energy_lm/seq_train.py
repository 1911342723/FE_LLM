# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/seq_train.py —— 训练双轴自由能生成网络（v4）
==============================================================
对应 设计v4 第四节。训练目标 = 前缀条件的下一字能量（与顺序生成一致）：

    序列布局：[上文 c] [SEP] [BOS] r₁ r₂ … rₙ [EOS]
    对每个回应位置 i（含末尾 EOS），以 (c, r_{<i}) 为条件，让真字 rᵢ 能量最低。
    因为 SeqEnergyNet 带因果掩码，位置 i 的输出天然只依赖 c 与 r_{≤i}，
    所以"位置 i 预测 i+1 个字"就是标准的 teacher-forcing 下一字训练。

    损失 = Σ_i CrossEntropy(-E(·|c,r_{<i}), rᵢ)   —— 只在回应区(含EOS)计损失。

与 v3 的根本区别：链式分解 ∏P(rᵢ|r_{<i}) 精确无独立假设，学的是"推进一个字"的
可泛化映射，故能随数据规模变好（对照 v3 的"越多越崩"）。

运行：python -m fe_llm.energy_lm.seq_train --n 0   (n=0 用全部数据)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

from fe_llm.config import get_device
from fe_llm.energy_lm.corpus import load_dialogues
from fe_llm.energy_lm.seq_net import SeqEnergyNet
from fe_llm.energy_lm.tokenizer import build_tokenizer, CharTokenizer

CKPT_DIR = os.path.join("checkpoints", "energy_lm")
CKPT_NET = os.path.join(CKPT_DIR, "seq_energy.pt")
CKPT_TOK = os.path.join(CKPT_DIR, "seq_tokenizer.json")

MAX_LEN = 48            # [上文][SEP][BOS][回应...][EOS] 的定长上限
DIM, DEPTH, NHEADS = 256, 6, 8


def build_example(tok: CharTokenizer, prompt: str, response: str):
    """拼 [c][SEP][BOS] r... [EOS]，返回 ids 与回应区(含EOS)的标记。
    回应区位置 i 的监督目标 = ids[i+1]（下一字）。"""
    c = tok.encode(prompt)
    r = tok.encode(response)
    seq = c + [tok.sep_id, tok.bos_id] + r + [tok.eos_id]
    if len(seq) > MAX_LEN:
        seq = seq[:MAX_LEN]
    # 监督位置：从 BOS 开始的每个位置都要预测"下一字"（BOS→r₁, r₁→r₂, …, rₙ→EOS）
    bos_pos = len(c) + 1                     # BOS 的下标
    sup = [False] * len(seq)
    for i in range(bos_pos, len(seq) - 1):   # 到倒数第二位（最后一位 EOS 无下一字）
        sup[i] = True
    # padding
    seq = seq + [tok.pad_id] * (MAX_LEN - len(seq))
    sup = sup + [False] * (MAX_LEN - len(sup))
    return seq, sup


def make_dataset(tok, dias):
    seqs, sups = [], []
    for p, r in dias:
        if len(p) + len(r) + 3 > MAX_LEN:
            continue
        s, sup = build_example(tok, p, r)
        seqs.append(s); sups.append(sup)
    return np.array(seqs), np.array(sups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="取前 n 条；0=全部")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    device = get_device()
    print(f"[seq] 设备：{device}")
    dias = load_dialogues()
    if args.n > 0:
        dias = dias[:args.n]
    tok = build_tokenizer()
    seqs, sups = make_dataset(tok, dias)
    print(f"[seq] 对话 {len(seqs)} 条，字表 {tok.vocab_size}，定长 {MAX_LEN}")

    net = SeqEnergyNet(tok.vocab_size, MAX_LEN, dim=DIM, depth=DEPTH,
                       n_heads=NHEADS).to(device)
    print(f"[seq] 参数量 {sum(p.numel() for p in net.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = random.Random(0)
    n_steps = (len(seqs) + args.batch - 1) // args.batch

    best = 1e9
    for ep in range(1, args.epochs + 1):
        net.train()
        idx = list(range(len(seqs))); rng.shuffle(idx)
        tot, nc, nt = 0.0, 0, 0
        t0 = time.time()
        for si, i in enumerate(range(0, len(idx), args.batch)):
            ch = idx[i:i + args.batch]
            seq = torch.tensor(seqs[ch], device=device, dtype=torch.long)
            sup = torch.tensor(sups[ch], device=device)            # (B,L) bool
            logits = -net(seq)                                     # (B,L,V)
            # 位置 i 预测 ids[i+1]：对齐 logits[:, :-1] 与 target seq[:, 1:]
            pred_logits = logits[:, :-1, :]
            target = seq[:, 1:]
            mask = sup[:, :-1]                                      # 监督位（BOS..rₙ）
            sl = pred_logits[mask]; tg = target[mask]
            loss = F.cross_entropy(sl, tg)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * tg.numel()
            nc += (sl.argmax(-1) == tg).sum().item(); nt += tg.numel()
            if si % 10 == 0:
                spd = (si + 1) / max(1e-6, time.time() - t0)
                print(f"[seq]   epoch {ep:3d} step {si:3d}/{n_steps} "
                      f"loss={float(loss.detach()):.4f} {spd:.1f}it/s", flush=True)
        sched.step()
        tl = tot / max(1, nt); acc = nc / max(1, nt)
        print(f"[seq] epoch {ep:3d} | loss={tl:.4f} 下一字准确率={acc:.1%} "
              f"用时={time.time()-t0:.0f}s", flush=True)
        if tl < best:
            best = tl
            os.makedirs(CKPT_DIR, exist_ok=True)
            net.save(CKPT_NET); tok.save(CKPT_TOK)
    print(f"\n[seq] 完成，最佳 loss={best:.4f} → {CKPT_NET}")


if __name__ == "__main__":
    main()
