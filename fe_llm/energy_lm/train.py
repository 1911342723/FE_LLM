# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/train.py —— 训练对话能量网络（去掩码 = 刻低能深沟）
====================================================================
训练目标（去掩码去噪）：随机把回应里的字替换成 [MASK]，让 E_θ 在这些位置
给**真实字**最低能量（= 最高 logit）。这等价于"经验就是省电"——
反复见过的对话被刻成能量地貌上的低能深沟，以后输入熟悉上文，回应就顺势滚出来。

序列布局（定长 seq_len）：
    [上文字...] [SEP] [BOS] [回应字...] [EOS] [PAD...]
只在**回应区**做掩码与计损失（上文是条件，不预测）。

运行：
    python -m fe_llm.energy_lm.train
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

from fe_llm.config import get_device
from fe_llm.energy_lm.corpus import load_dialogues
from fe_llm.energy_lm.energy_net import DialogueEnergyNet
from fe_llm.energy_lm.tokenizer import build_tokenizer

CKPT_DIR = os.path.join("checkpoints", "energy_lm")
CKPT_NET = os.path.join(CKPT_DIR, "dialogue_energy.pt")
CKPT_TOK = os.path.join(CKPT_DIR, "tokenizer.json")

SEQ_LEN = 32            # 定长：上文+SEP+BOS+[回应窗口 RESP_MAX]
RESP_MAX = 12           # 回应窗口固定长度：真实字 + EOS + PAD 都在窗口内、都受监督
DIM, DEPTH, NHEADS = 256, 6, 8   # 小模型收敛快（吃满GPU与收敛速度矛盾，优先收敛）


def build_example(tok, prompt: str, response: str):
    """
    拼成定长序列。关键改动（根因解决）：
    回应窗口固定 RESP_MAX 位，布局 = [真实字...][EOS][PAD...]，
    **整个窗口都算回应区、都参与掩码与监督**——让模型学会"内容说完后该填 EOS/空"，
    于是长度由能量自然决定（多余位置塌缩成 EOS/PAD 这个低能稳态），不再硬塞字。
    """
    p = tok.encode(prompt)
    r = tok.encode(response)[:RESP_MAX - 1]       # 给 EOS 留位
    # 回应窗口：真实字 + EOS + PAD 补满到 RESP_MAX
    resp_win = r + [tok.eos_id] + [tok.pad_id] * (RESP_MAX - len(r) - 1)
    seq = p + [tok.sep_id, tok.bos_id] + resp_win
    resp_start = len(p) + 2
    if len(seq) > SEQ_LEN:
        seq = seq[:SEQ_LEN]
    seq = seq + [tok.pad_id] * (SEQ_LEN - len(seq))
    # 回应区 = 整个窗口（含 EOS 与其后的 PAD），都要学
    is_resp = [resp_start <= i < resp_start + RESP_MAX for i in range(SEQ_LEN)]
    return seq, is_resp


def make_dataset(tok):
    data = []
    for p, r in load_dialogues():
        # 只保留回应能放进窗口的（≤RESP_MAX-1，给EOS留位），与1502条配置一致
        if len(r) > RESP_MAX - 1:
            continue
        seq, is_resp = build_example(tok, p, r)
        data.append((np.array(seq), np.array(is_resp)))
    return data


def main():
    device = get_device()
    print(f"[energy_lm] 设备：{device}")
    print(f"[energy_lm] 第一性原理：去掩码训练 = 把对话经验刻成能量地貌的低能深沟。")

    tok = build_tokenizer()
    data = make_dataset(tok)
    print(f"[energy_lm] 对话 {len(data)} 对，字表 {tok.vocab_size}，定长 {SEQ_LEN}")

    net = DialogueEnergyNet(tok.vocab_size, SEQ_LEN, dim=DIM, depth=DEPTH,
                            n_heads=NHEADS).to(device)
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"[energy_lm] 能量网络参数量：{n_params:.2f}M")

    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    epochs, batch = 600, 128
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = random.Random(0)
    seqs = np.stack([d[0] for d in data])
    resps = np.stack([d[1] for d in data])
    n_steps = (len(data) + batch - 1) // batch
    import time as _time

    best = 1e9
    for ep in range(1, epochs + 1):
        net.train()
        idx = list(range(len(data))); rng.shuffle(idx)
        tot, ncorrect, ntok = 0.0, 0, 0
        ep_t0 = _time.time()
        for si, i in enumerate(range(0, len(idx), batch)):
            chunk = idx[i:i + batch]
            seq = torch.tensor(seqs[chunk], device=device, dtype=torch.long)   # (B, L)
            resp = torch.tensor(resps[chunk], device=device)       # (B, L) bool
            # —— 向量化随机掩码（去掉逐样本 Python 循环，大幅提速）——
            # 回应区每位以概率 p 独立掩码；p 每条在 [0.4,0.9] 间随机
            inp = seq.clone()
            B, L = seq.shape
            p = torch.empty(B, 1, device=device).uniform_(0.4, 0.9)
            rand = torch.rand(B, L, device=device)
            mask_pos = (rand < p) & resp                        # (B,L) bool
            # 保证每条至少掩 1 个回应位：若某行全 False，强制掩它第一个回应位
            none_row = (mask_pos & resp).sum(1) == 0
            if none_row.any():
                first_resp = resp.float().argmax(1)             # 每行第一个回应位
                rows = none_row.nonzero(as_tuple=True)[0]
                mask_pos[rows, first_resp[rows]] = True
            inp[mask_pos] = tok.mask_id
            energy = net(inp)                                      # (B,L,V) = -logits
            logits = -energy
            tgt = seq[mask_pos]                                    # 被掩位真值
            sel_logits = logits[mask_pos]
            # —— PAD 降权（根因解决）——回应窗口含大量 PAD 占位，若等权，
            # 准确率/梯度被"填 PAD"灌水，内容字学不会。给 PAD 位权重 0.1，
            # 内容字与 EOS 保持 1.0，让梯度聚焦真正的内容。
            w = torch.ones_like(tgt, dtype=torch.float)
            w[tgt == tok.pad_id] = 0.1
            ce = F.cross_entropy(sel_logits, tgt, reduction="none")
            loss = (ce * w).sum() / w.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            pred = sel_logits.argmax(-1)
            # 准确率/loss 只统计内容字+EOS（剔除 PAD），真实反映"内容学没学会"
            non_pad = tgt != tok.pad_id
            tot += float(loss.detach()) * int(non_pad.sum().item())
            ncorrect += (pred[non_pad] == tgt[non_pad]).sum().item()
            ntok += int(non_pad.sum().item())
            # —— step 级进度（让人看得到在动，不是干等）——
            if si % 5 == 0:
                spd = (si + 1) / max(1e-6, _time.time() - ep_t0)
                print(f"[energy_lm]   epoch {ep:3d} step {si:3d}/{n_steps} "
                      f"loss={float(loss.detach()):.4f} {spd:.1f}it/s", flush=True)
        sched.step()
        train_loss = tot / max(1, ntok)
        acc = ncorrect / max(1, ntok)
        print(f"[energy_lm] epoch {ep:3d} | 去掩码loss={train_loss:.4f} "
              f"填字准确率={acc:.1%} 用时={_time.time()-ep_t0:.0f}s", flush=True)
        if train_loss < best:
            best = train_loss
            os.makedirs(CKPT_DIR, exist_ok=True)
            net.save(CKPT_NET); tok.save(CKPT_TOK)

    print(f"\n[energy_lm] 训练完成，最佳去掩码loss={best:.4f}")
    print(f"[energy_lm] 权重：{CKPT_NET}")
    print(f"[energy_lm] 判据：填字准确率高 → 对话经验已刻成低能深沟。")


if __name__ == "__main__":
    main()
