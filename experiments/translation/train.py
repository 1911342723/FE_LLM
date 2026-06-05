# -*- coding: utf-8 -*-
"""
translation/train.py —— 训练中英双向翻译模型
=============================================
运行：python experiments/translation/train.py

流程：
    1) 载入 DeepSeek 生成的平行语料 pairs.jsonl
    2) 训练 SentencePiece 共享子词词表（若不存在）
    3) 构造双向训练样本（中译英 + 英译中）
    4) 训练 Transformer，最小化交叉熵（= 最小化期望惊奇度 -ln P）
    5) 固化权重到 checkpoints/translation/

设备：自动用 CUDA（RTX 5060）。
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from fe_llm.config import get_device
from experiments.translation.dataset import (TranslationDataset, load_pairs,
                                             make_collate, write_spm_corpus)
from experiments.translation.decoding import translate
from experiments.translation.model import TranslationModel
from experiments.translation.teacher import PAIRS_PATH
from experiments.translation.tokenizer import (MODEL_DIR, SPM_MODEL, Tokenizer,
                                               train_spm)

# 超参（放大模型 + batch 吃满 RTX 5060 显存，训练更久更充分）
VOCAB_SIZE = 8000
D_MODEL = 512
NHEAD = 8
LAYERS = 6
DIM_FF = 2048
DROPOUT = 0.1
MAX_LEN = 128
EPOCHS = 150
BATCH = 256
LR = 5e-4
WARMUP = 4000
LABEL_SMOOTH = 0.1
NUM_WORKERS = 0        # Windows 下多进程 DataLoader 开销大，样本已预 tokenize，0 即可
USE_AMP = True         # 混合精度：用上 5060 的 Tensor Core，显著提速并降显存
CKPT_PATH = os.path.join(MODEL_DIR, "translation.pt")


def build_tokenizer(pairs) -> Tokenizer:
    """确保 SentencePiece 词表存在（不存在则用语料训练）。"""
    if not os.path.exists(SPM_MODEL):
        corpus_txt = os.path.join(MODEL_DIR, "spm_corpus.txt")
        write_spm_corpus(pairs, corpus_txt)
        train_spm(corpus_txt, vocab_size=VOCAB_SIZE)
    return Tokenizer(SPM_MODEL)


def lr_lambda(step: int):
    """Transformer 经典 warmup 学习率调度。"""
    step = max(1, step)
    return (D_MODEL ** -0.5) * min(step ** -0.5, step * WARMUP ** -1.5)


def main():
    device = get_device()
    if device == "cuda":
        # 开启 TF32 与 cuDNN 自动调优，提升矩阵乘吞吐
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print(f"[trans] 设备：{device} "
          f"({torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'})")

    # 1) 语料
    if not os.path.exists(PAIRS_PATH):
        raise FileNotFoundError(
            f"未找到语料 {PAIRS_PATH}，请先运行 experiments/translation/teacher.py 生成。")
    pairs = load_pairs(PAIRS_PATH)
    print(f"[trans] 平行语料：{len(pairs)} 句对")

    # 2) 分词器
    tok = build_tokenizer(pairs)
    print(f"[trans] 词表大小：{tok.vocab_size}")

    # 3) 数据集（双向）
    ds = TranslationDataset(pairs, tok, max_len=MAX_LEN)
    n_val = max(1, int(len(ds) * 0.03))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    collate = make_collate(tok.pad_id)
    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              collate_fn=collate, num_workers=NUM_WORKERS,
                              pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                            collate_fn=collate, num_workers=NUM_WORKERS,
                            pin_memory=pin)
    print(f"[trans] 训练样本：{n_train}（双向展开），验证：{n_val}")

    # 4) 模型
    model = TranslationModel(
        vocab_size=tok.vocab_size, d_model=D_MODEL, nhead=NHEAD,
        num_layers=LAYERS, dim_ff=DIM_FF, dropout=DROPOUT,
        pad_id=tok.pad_id, max_len=MAX_LEN,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[trans] 模型参数量：{n_params:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    # 交叉熵 = -ln P(token|上下文)，正是惊奇度；忽略 pad，加标签平滑
    loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_id,
                                  label_smoothing=LABEL_SMOOTH)
    # 混合精度梯度缩放器（仅 CUDA 启用）
    use_amp = USE_AMP and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 5) 训练循环
    step = 0
    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, count = 0.0, 0
        for src, tgt_in, tgt_out in train_loader:
            src = src.to(device, non_blocking=True)
            tgt_in = tgt_in.to(device, non_blocking=True)
            tgt_out = tgt_out.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=use_amp):
                logits = model(src, tgt_in)                  # (B,T,V)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)),
                               tgt_out.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            total += loss.item() * src.size(0); count += src.size(0)

        train_loss = total / count
        val_loss = evaluate(model, val_loader, loss_fn, device)
        ppl = math.exp(min(20, val_loss))  # 困惑度 = exp(惊奇度)
        gpu = (f" | 显存={torch.cuda.max_memory_allocated()/1e9:.2f}GB"
               if device == "cuda" else "")
        print(f"[trans] epoch {epoch:3d} | 训练惊奇={train_loss:.4f} "
              f"验证惊奇={val_loss:.4f} 困惑度={ppl:.2f} "
              f"lr={sched.get_last_lr()[0]:.2e}{gpu}")

        if val_loss < best_val:
            best_val = val_loss
            save(model, tok)

        # 每若干轮抽样翻译，直观观察进步
        if epoch % 10 == 0 or epoch == 1:
            sample_translate(model, tok, device)

    print(f"[trans] 训练完成，最佳验证惊奇={best_val:.4f}，权重：{CKPT_PATH}")


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for src, tgt_in, tgt_out in loader:
        src = src.to(device); tgt_in = tgt_in.to(device); tgt_out = tgt_out.to(device)
        logits = model(src, tgt_in)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        total += loss.item() * src.size(0); count += src.size(0)
    return total / count


def sample_translate(model, tok, device):
    """抽几句固定例句，观察翻译质量随训练的变化。"""
    samples = [("今天天气很好", "en"), ("我想喝一杯咖啡", "en"),
               ("How are you today?", "zh"), ("Where is the train station?", "zh")]
    print("  --- 抽样翻译 ---")
    for text, lang in samples:
        out = translate(model, tok, text, lang, device, beam_size=4)
        print(f"    [{'中→英' if lang=='en' else '英→中'}] {text}  =>  {out}")


def save(model, tok):
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        "config": {
            "vocab_size": tok.vocab_size, "d_model": D_MODEL, "nhead": NHEAD,
            "num_layers": LAYERS, "dim_ff": DIM_FF, "dropout": DROPOUT,
            "pad_id": tok.pad_id, "max_len": MAX_LEN,
        },
        "state_dict": model.state_dict(),
    }, CKPT_PATH)


if __name__ == "__main__":
    main()
