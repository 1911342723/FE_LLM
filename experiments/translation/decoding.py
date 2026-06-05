# -*- coding: utf-8 -*-
"""
translation/decoding.py —— 能量递减解码（贪心 + 束搜索）
=========================================================
对应文档第四步「能量递减解码器」：把 token 的能量定义为 E = -logit，
每一步在词表上选能量最低（logit 最高）的 token「滚落」，逐字生成，
直到吐出 </s>（残余能量耗尽，意图表达完毕）。

提供两种解码：
    greedy_decode  : 每步取能量最低的单个 token（最快）。
    beam_decode    : 束搜索，维护累计能量最低的若干条路径（质量更好）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def greedy_decode(model, tokenizer, src_ids: list[int], device: str,
                  max_len: int = 128) -> list[int]:
    """贪心能量递减解码。返回生成的 token id 序列（不含 BOS/EOS）。"""
    model.eval()
    src = torch.tensor([src_ids], device=device)
    memory, src_pad = model.encode(src)

    ys = torch.tensor([[tokenizer.bos_id]], device=device)
    out: list[int] = []
    for _ in range(max_len):
        logits = model.decode_step(ys, memory, src_pad)   # (1, T, V)
        next_logit = logits[:, -1, :]                      # (1, V)
        # 能量最低 = logit 最大
        nxt = int(next_logit.argmax(dim=-1).item())
        if nxt == tokenizer.eos_id:
            break
        out.append(nxt)
        ys = torch.cat([ys, torch.tensor([[nxt]], device=device)], dim=1)
    return out


@torch.no_grad()
def beam_decode(model, tokenizer, src_ids: list[int], device: str,
                beam_size: int = 5, max_len: int = 128,
                length_penalty: float = 0.7) -> list[int]:
    """
    束搜索能量递减解码。维护累计能量(= 负对数概率之和)最低的 beam_size 条路径。
    length_penalty 缓解对短句的偏好。
    """
    model.eval()
    src = torch.tensor([src_ids], device=device)
    memory, src_pad = model.encode(src)
    V = model.out.out_features if hasattr(model.out, "out_features") else None

    # 每条 beam: (累计能量, token序列, 是否结束)
    beams = [(0.0, [tokenizer.bos_id], False)]

    for _ in range(max_len):
        # 全部结束则停止
        if all(done for _, _, done in beams):
            break
        candidates = []
        for energy, seq, done in beams:
            if done:
                candidates.append((energy, seq, True))
                continue
            ys = torch.tensor([seq], device=device)
            logits = model.decode_step(ys, memory, src_pad)[:, -1, :]  # (1, V)
            logprob = F.log_softmax(logits, dim=-1).squeeze(0)         # (V,)
            # token 能量 = -logprob；累计能量越低越好
            topk = torch.topk(logprob, beam_size)
            for lp, tok in zip(topk.values.tolist(), topk.indices.tolist()):
                new_seq = seq + [tok]
                new_energy = energy - lp  # 减去 logprob = 加上能量
                is_done = (tok == tokenizer.eos_id)
                candidates.append((new_energy, new_seq, is_done))

        # 按"长度惩罚归一化能量"选出最优的 beam_size 条
        def score(item):
            e, s, _ = item
            length = max(1, len(s) - 1)
            return e / (length ** length_penalty)

        candidates.sort(key=score)
        beams = candidates[:beam_size]

    # 取能量最低的完整路径
    best = min(beams, key=lambda x: x[0] / (max(1, len(x[1]) - 1) ** length_penalty))
    seq = best[1][1:]  # 去掉 BOS
    if seq and seq[-1] == tokenizer.eos_id:
        seq = seq[:-1]
    return seq


def translate(model, tokenizer, text: str, target_lang: str, device: str,
              beam_size: int = 5) -> str:
    """端到端翻译一句话。target_lang ∈ {'zh','en'}。"""
    src_ids = tokenizer.encode_source(text, target_lang)
    out_ids = beam_decode(model, tokenizer, src_ids, device, beam_size=beam_size)
    return tokenizer.decode(out_ids)
