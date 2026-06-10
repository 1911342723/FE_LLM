# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/translation_train.py —— 中→英翻译版 IntentLM（泛化能力检验）
==============================================================================
完全复用对话版的三阶段架构与损失：
    IntentEncoder ：中文句 → PER 弛豫 → 意图向量 z*（语义压缩）
    EnergyDecoder ：意图 z* + 英文前缀 → 逐字生成（跨语言重建）
    L_intent（InfoNCE 对齐 zh/en 意图）+ L_decode（CE）+ λ·L_approach（能量递减）

这是对架构假设的关键检验：如果"意图空间"真的承载语义而非表面模式，
那么同一套机制应当能把中文意图重建成英文——并对未见句子泛化。

运行：python fe_llm/energy_lm/translation_train.py [--epochs 80] [--batch 256]
"""

from __future__ import annotations

import argparse
import json
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
from fe_llm.energy_lm.intent_model import IntentLM
from fe_llm.energy_lm.tokenizer import CharTokenizer

CKPT_DIR = os.path.join("checkpoints", "energy_lm")
CKPT_PATH = os.path.join(CKPT_DIR, "translation_lm.pt")
CKPT_TOK = os.path.join(CKPT_DIR, "translation_tok.json")
TRAIN_PATH = os.path.join("data", "translation", "opus100_train.jsonl")

# 翻译版序列长度：中文 ≤28 字 + SEP；英文 ≤46 字符 + BOS/EOS。
# 注意：IntentLM 训练时用同一个编码器从 response 弛豫目标意图（z_target），
# 因此编码器位置表长度必须 ≥ DEC_MAX，这里取两者相同。
ENC_MAX = 48
DEC_MAX = 48
DIM = 256
ENC_DEPTH = 4
DEC_DEPTH = 6
NHEADS = 8
INTENT_DIM = 128
LAMBDA_APPROACH = 0.1


def load_pairs(path: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                out.append((item["zh"], item["en"]))
            except (ValueError, KeyError):
                continue
    return out


def pad_seq(ids: list[int], max_len: int, pad_id: int) -> list[int]:
    return (ids + [pad_id] * max_len)[:max_len]


def make_dataset(tok: CharTokenizer, pairs: list[tuple[str, str]]):
    prompts, resp_in, resp_tgt = [], [], []
    for zh, en in pairs:
        p_ids = tok.encode(zh)[: ENC_MAX - 1] + [tok.sep_id]
        r_enc = tok.encode(en)[: DEC_MAX - 2]
        prompts.append(pad_seq(p_ids, ENC_MAX, tok.pad_id))
        resp_in.append(pad_seq([tok.bos_id] + r_enc, DEC_MAX, tok.pad_id))
        resp_tgt.append(pad_seq(r_enc + [tok.eos_id], DEC_MAX, tok.pad_id))
    return np.array(prompts), np.array(resp_in), np.array(resp_tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n", type=int, default=0, help="取前 n 对，0=全部")
    args = ap.parse_args()

    device = get_device()
    print(f"[trans] 设备：{device}")

    pairs = load_pairs(TRAIN_PATH)
    if args.n > 0:
        pairs = pairs[: args.n]
    chars = sorted({c for zh, en in pairs for c in zh + en})
    tok = CharTokenizer(chars)
    prompts, resp_in, resp_tgt = make_dataset(tok, pairs)
    print(f"[trans] 翻译对 {len(prompts)} 条，字表 {tok.vocab_size}")

    model = IntentLM(
        vocab_size=tok.vocab_size, enc_max=ENC_MAX, dec_max=DEC_MAX,
        dim=DIM, enc_depth=ENC_DEPTH, dec_depth=DEC_DEPTH,
        n_heads=NHEADS, intent_dim=INTENT_DIM,
    ).to(device)
    print(f"[trans] 参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = random.Random(42)
    n_steps = (len(prompts) + args.batch - 1) // args.batch

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

            # 关键修复（对照 IntentLM.forward 的原始写法）：
            # 原始版解码器条件于 z_target=encoder(英文)，推理时却只有 z_pred=encoder(中文)。
            # InfoNCE 没收敛时两者根本不对齐 -> 解码器面对分布外意图，塌缩成固定句。
            # 这里改为解码器直接条件于 z_pred：
            #   1) 训练/推理同分布（推理拿到的就是训练见过的条件）；
            #   2) decode loss 的梯度经 z_pred 直接训练编码器（不再只靠 InfoNCE）。
            z_pred = model.encoder(p_ids)
            with torch.no_grad():
                z_target = model.encoder(ri)
            logits, h_intent = model.decoder(ri, z_pred)

            mask = rt != tok.pad_id
            l_decode = F.cross_entropy(logits[mask], rt[mask])

            # InfoNCE 保留为辅助对齐项（zh 意图与 en 意图互为正例）。
            sim = F.cosine_similarity(z_pred.unsqueeze(1), z_target.unsqueeze(0), dim=-1)
            labels = torch.arange(sim.size(0), device=device)
            l_intent = F.cross_entropy(sim / 0.07, labels)

            # 能量递减正则改为朝 z_pred 收敛（detach 防止编码器把意图拉向隐状态作弊）。
            dist = torch.norm(h_intent - z_pred.detach().unsqueeze(1), dim=-1)
            violations = F.relu(dist[:, 1:] - dist[:, :-1])
            l_approach = violations[mask[:, 1:]].mean() if mask[:, 1:].any() else torch.tensor(0.0, device=device)

            loss = l_decode + 2.0 * l_intent + LAMBDA_APPROACH * l_approach
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
                print(f"[trans]   ep {ep:3d} step {si}/{n_steps} "
                      f"L_dec={float(l_decode):.3f} L_int={float(l_intent):.3f} "
                      f"L_app={float(l_approach):.3f} {spd:.1f}it/s", flush=True)

        sched.step()
        acc = nc / max(1, nt)
        avg_l = tot_l / max(1, nt)
        print(f"[trans] ep {ep:3d} | decode_loss={avg_l:.4f} 下一字准确率={acc:.1%} "
              f"用时={time.time() - t0:.0f}s", flush=True)
        if avg_l < best:
            best = avg_l
            os.makedirs(CKPT_DIR, exist_ok=True)
            model.save(CKPT_PATH)
            tok.save(CKPT_TOK)

    print(f"\n[trans] 完成，最佳 decode_loss={best:.4f} -> {CKPT_PATH}")


if __name__ == "__main__":
    main()
