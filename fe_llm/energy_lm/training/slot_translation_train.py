# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/slot_translation_train.py —— 槽位化翻译训练（M1+M2 判定实验）
==============================================================================
与 translation_train.py 完全同数据、同规模、同训练预算，唯一变量是
意图表示：单 128 维向量 → global_intent + K 槽位 + salience。

判定标准（草案第 5 节 M2）：未见句 word-F1 显著超过单向量版的 0.07
（目标 ≥0.3）说明瓶颈在表达结构；若改善微弱，转预训练底座路线。

损失（草案 3.3 节）：
    total = L_decode                  # CE，条件于 (global, slots)
          + 2.0  · L_intent          # 全局 InfoNCE（zh/en 意图互为正例）
          + 0.1  · L_approach        # 全局能量递减（detach 防作弊）
          + 0.1  · L_slot_div        # 槽间正交，防塌缩
          + 0.05 · L_slot_coverage   # 解码隐状态应覆盖高显著槽位（detach 槽位）

运行：python fe_llm/energy_lm/slot_translation_train.py [--epochs 80]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.nn.functional as F

from fe_llm.config import get_device
from fe_llm.energy_lm.models.slot_intent_model import SlotIntentLM
from fe_llm.energy_lm.models.tokenizer import CharTokenizer
from fe_llm.energy_lm.training.translation_train import (
    DEC_MAX,
    ENC_MAX,
    TRAIN_PATH,
    load_pairs,
    make_dataset,
)

CKPT_DIR = os.path.join("checkpoints", "energy_lm")
CKPT_PATH = os.path.join(CKPT_DIR, "slot_translation_lm.pt")
CKPT_TOK = os.path.join(CKPT_DIR, "slot_translation_tok.json")

DIM = 256
ENC_DEPTH = 4
DEC_DEPTH = 6
NHEADS = 8
INTENT_DIM = 128
N_SLOTS = 8
LAMBDA_APPROACH = 0.1
LAMBDA_SLOT_DIV = 0.1
LAMBDA_COVERAGE = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    device = get_device()
    print(f"[slot-trans] 设备：{device}")

    pairs = load_pairs(TRAIN_PATH)
    if args.n > 0:
        pairs = pairs[: args.n]
    chars = sorted({c for zh, en in pairs for c in zh + en})
    tok = CharTokenizer(chars)
    import numpy as np

    prompts, resp_in, resp_tgt = make_dataset(tok, pairs)
    print(f"[slot-trans] 翻译对 {len(prompts)} 条，字表 {tok.vocab_size}")

    model = SlotIntentLM(
        vocab_size=tok.vocab_size, enc_max=ENC_MAX, dec_max=DEC_MAX,
        dim=DIM, enc_depth=ENC_DEPTH, dec_depth=DEC_DEPTH,
        n_heads=NHEADS, intent_dim=INTENT_DIM, n_slots=N_SLOTS,
    ).to(device)
    print(f"[slot-trans] 参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = random.Random(42)
    n_steps = (len(prompts) + args.batch - 1) // args.batch
    eye = torch.eye(N_SLOTS, device=device)

    best = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        idx = list(range(len(prompts)))
        rng.shuffle(idx)
        tot_l, nc, nt = 0.0, 0, 0
        t0 = time.time()
        for si, i in enumerate(range(0, len(idx), args.batch)):
            ch = idx[i : i + args.batch]
            p_ids = torch.tensor(prompts[ch], device=device, dtype=torch.long)
            ri = torch.tensor(resp_in[ch], device=device, dtype=torch.long)
            rt = torch.tensor(resp_tgt[ch], device=device, dtype=torch.long)

            # 铁律：解码器只条件于 prompt 侧意图（训练/推理同分布）。
            logits, h_intent, z_global, slots, salience = model(p_ids, ri)
            with torch.no_grad():
                z_target, _, _ = model.encoder(ri)

            mask = rt != tok.pad_id
            l_decode = F.cross_entropy(logits[mask], rt[mask])

            # 全局 InfoNCE：zh 全局意图与 en 全局意图互为正例。
            sim = F.cosine_similarity(z_global.unsqueeze(1), z_target.unsqueeze(0), dim=-1)
            labels = torch.arange(sim.size(0), device=device)
            l_intent = F.cross_entropy(sim / 0.07, labels)

            # 全局能量递减（detach 防止编码器把目标拉向隐状态作弊）。
            dist = torch.norm(h_intent - z_global.detach().unsqueeze(1), dim=-1)
            violations = F.relu(dist[:, 1:] - dist[:, :-1])
            l_approach = violations[mask[:, 1:]].mean() if mask[:, 1:].any() else torch.tensor(0.0, device=device)

            # 槽间正交：防止 K 个槽位塌缩成同一个向量。
            s_norm = F.normalize(slots, dim=-1)                       # (B,K,d)
            gram = s_norm @ s_norm.transpose(1, 2)                    # (B,K,K)
            l_slot_div = ((gram - eye) ** 2).mean()

            # 槽位覆盖：每个高显著槽位都应被某个解码位置接近（detach 槽位）。
            # dists_ks: (B,K,L) = 每槽位到每个解码位置的距离
            dists_ks = torch.cdist(slots.detach(), h_intent)          # (B,K,L)
            # 无效位置距离置大，避免 PAD 充当覆盖者
            big = torch.full_like(dists_ks, 1e4)
            dists_ks = torch.where(mask.unsqueeze(1), dists_ks, big)
            min_dist = dists_ks.min(dim=-1).values                    # (B,K)
            l_coverage = (salience.detach() * min_dist).sum(dim=-1).mean()

            loss = (l_decode + 2.0 * l_intent + LAMBDA_APPROACH * l_approach
                    + LAMBDA_SLOT_DIV * l_slot_div + LAMBDA_COVERAGE * l_coverage)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            pred = logits[mask].argmax(-1)
            nc += (pred == rt[mask]).sum().item()
            nt += int(mask.sum().item())
            tot_l += float(l_decode.detach()) * int(mask.sum())

            if si % 50 == 0:
                spd = (si + 1) / max(1e-6, time.time() - t0)
                print(f"[slot-trans]   ep {ep:3d} step {si}/{n_steps} "
                      f"L_dec={float(l_decode):.3f} L_int={float(l_intent):.3f} "
                      f"L_div={float(l_slot_div):.3f} L_cov={float(l_coverage):.3f} "
                      f"{spd:.1f}it/s", flush=True)

        sched.step()
        acc = nc / max(1, nt)
        avg_l = tot_l / max(1, nt)
        print(f"[slot-trans] ep {ep:3d} | decode_loss={avg_l:.4f} 下一字准确率={acc:.1%} "
              f"用时={time.time() - t0:.0f}s", flush=True)
        if avg_l < best:
            best = avg_l
            os.makedirs(CKPT_DIR, exist_ok=True)
            model.save(CKPT_PATH)
            tok.save(CKPT_TOK)

    print(f"\n[slot-trans] 完成，最佳 decode_loss={best:.4f} -> {CKPT_PATH}")


if __name__ == "__main__":
    main()
