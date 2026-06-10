# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/intent_train.py —— 训练意图驱动自由能模型（v5）
================================================================
对应 设计v5：

联合训练 IntentEncoder + EnergyDecoder：
    L_intent ：编码器从 prompt 弛豫出的意图 z_pred 要接近从 response 弛豫出的 z_target
    L_decode ：解码器在 z_target 引导下逐字预测 response
    L_approach：解码器隐状态逐字接近意图（能量递减正则——朝目标走）

总 loss = L_intent + L_decode + λ·L_approach

运行：python -m fe_llm.energy_lm.intent_train [--n N] [--epochs E]
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
from fe_llm.energy_lm.intent_model import IntentLM
from fe_llm.energy_lm.tokenizer import build_tokenizer, CharTokenizer

CKPT_DIR = os.path.join("checkpoints", "energy_lm")
CKPT_PATH = os.path.join(CKPT_DIR, "intent_lm.pt")
CKPT_TOK = os.path.join(CKPT_DIR, "intent_tok.json")

ENC_MAX = 24       # prompt 最大字数（含特殊符）
DEC_MAX = 24       # response 最大字数（含 BOS/EOS）
DIM = 256
ENC_DEPTH = 4
DEC_DEPTH = 6          # 解码器加深（需更强容量恢复具体文字）
NHEADS = 8
INTENT_DIM = 128       # 意图空间缩到128（信息瓶颈：逼迫编码器压缩出核心意图）
LAMBDA_APPROACH = 0.1     # 能量递减正则权重


def pad_seq(ids: list[int], max_len: int, pad_id: int) -> list[int]:
    return (ids + [pad_id] * max_len)[:max_len]


def make_dataset(tok: CharTokenizer, dias, enc_max: int, dec_max: int):
    """每条 (prompt, response) → (prompt_ids, resp_ids_input, resp_ids_target)
    resp_ids_input = [BOS] + response   （解码器输入）
    resp_ids_target = response + [EOS]  （解码器预测目标）
    """
    prompts, resp_in, resp_tgt = [], [], []
    for p, r in dias:
        p_ids = tok.encode(p)[:enc_max - 1] + [tok.sep_id]
        r_enc = tok.encode(r)[:dec_max - 2]
        ri = [tok.bos_id] + r_enc
        rt = r_enc + [tok.eos_id]
        prompts.append(pad_seq(p_ids, enc_max, tok.pad_id))
        resp_in.append(pad_seq(ri, dec_max, tok.pad_id))
        resp_tgt.append(pad_seq(rt, dec_max, tok.pad_id))
    return np.array(prompts), np.array(resp_in), np.array(resp_tgt)


def load_extra_dialogues(path: str) -> list[tuple[str, str]]:
    """加载额外的 jsonl 对话语料（如 LCCC 高熵子集），用于扩大训练规模。"""

    import json

    out: list[tuple[str, str]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                p, r = item["prompt"].strip(), item["response"].strip()
                if p and r:
                    out.append((p, r))
            except (ValueError, KeyError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="取前 n 条，0=全部")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--extra-data", default="", help="额外 jsonl 语料路径（与主语料合并训练）")
    args = ap.parse_args()

    device = get_device()
    print(f"[intent] 设备：{device}")

    dias = load_dialogues()
    if args.extra_data:
        extra = load_extra_dialogues(args.extra_data)
        print(f"[intent] 额外语料 {len(extra)} 条（{args.extra_data}）")
        dias = dias + extra
    if args.n > 0:
        dias = dias[:args.n]
    if args.extra_data:
        # 合并语料后字表必须覆盖全部字符，否则额外语料的字全部变 UNK。
        chars = sorted({c for p, r in dias for c in p + r})
        tok = CharTokenizer(chars)
    else:
        tok = build_tokenizer()
    prompts, resp_in, resp_tgt = make_dataset(tok, dias, ENC_MAX, DEC_MAX)
    print(f"[intent] 对话 {len(prompts)} 条，字表 {tok.vocab_size}")

    model = IntentLM(
        vocab_size=tok.vocab_size, enc_max=ENC_MAX, dec_max=DEC_MAX,
        dim=DIM, enc_depth=ENC_DEPTH, dec_depth=DEC_DEPTH,
        n_heads=NHEADS, intent_dim=INTENT_DIM
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[intent] 参数量 {n_params:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = random.Random(42)
    n_steps = (len(prompts) + args.batch - 1) // args.batch

    best = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        idx = list(range(len(prompts))); rng.shuffle(idx)
        tot_l, nc, nt = 0.0, 0, 0
        t0 = time.time()
        for si, i in enumerate(range(0, len(idx), args.batch)):
            ch = idx[i:i + args.batch]
            p_ids = torch.tensor(prompts[ch], device=device, dtype=torch.long)
            ri = torch.tensor(resp_in[ch], device=device, dtype=torch.long)
            rt = torch.tensor(resp_tgt[ch], device=device, dtype=torch.long)

            logits, h_intent, z_pred, z_target = model(p_ids, ri)

            # --- L_decode：解码器逐字交叉熵（条件于意图）---
            # 只在非 PAD 位置计损失
            mask = rt != tok.pad_id                                  # (B, L)
            l_decode = F.cross_entropy(
                logits[mask], rt[mask])

            # --- L_intent：对比学习（逼意图编码器区分不同prompt→不同意图）---
            # InfoNCE：z_pred[i] 应该跟 z_target[i] 近，跟 z_target[j≠i] 远
            sim = F.cosine_similarity(z_pred.unsqueeze(1), z_target.detach().unsqueeze(0), dim=-1)  # (B,B)
            labels = torch.arange(sim.size(0), device=device)
            l_intent = F.cross_entropy(sim / 0.07, labels)     # temperature=0.07

            # --- L_approach：能量递减正则（隐状态逐字接近意图）---
            # 在每个有效位置，希望 dist(h_i, z_target) 逐步递减
            dist = torch.norm(h_intent - z_target.detach().unsqueeze(1), dim=-1)  # (B,L)
            # 惩罚 dist[i] > dist[i-1]（离意图更远了）
            violations = F.relu(dist[:, 1:] - dist[:, :-1])          # (B, L-1)
            l_approach = violations[mask[:, 1:]].mean() if mask[:, 1:].any() else torch.tensor(0.0, device=device)

            loss = l_decode + 2.0 * l_intent + LAMBDA_APPROACH * l_approach
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            pred = logits[mask].argmax(-1)
            nc += (pred == rt[mask]).sum().item()
            nt += int(mask.sum().item())
            tot_l += float(l_decode.detach()) * int(mask.sum())

            if si % 10 == 0:
                spd = (si + 1) / max(1e-6, time.time() - t0)
                print(f"[intent]   ep {ep:3d} step {si}/{n_steps} "
                      f"L_dec={float(l_decode):.3f} L_int={float(l_intent):.3f} "
                      f"L_app={float(l_approach):.3f} {spd:.1f}it/s", flush=True)

        sched.step()
        acc = nc / max(1, nt)
        avg_l = tot_l / max(1, nt)
        print(f"[intent] ep {ep:3d} | decode_loss={avg_l:.4f} "
              f"下一字准确率={acc:.1%} 用时={time.time()-t0:.0f}s", flush=True)
        if avg_l < best:
            best = avg_l
            os.makedirs(CKPT_DIR, exist_ok=True)
            model.save(CKPT_PATH); tok.save(CKPT_TOK)

    print(f"\n[intent] 完成，最佳 decode_loss={best:.4f} → {CKPT_PATH}")


if __name__ == "__main__":
    main()
