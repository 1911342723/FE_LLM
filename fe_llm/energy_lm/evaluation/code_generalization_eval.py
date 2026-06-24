# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/evaluation/code_generalization_eval.py —— 代码模型泛化能力评测
================================================================================
回答"模型是真泛化还是死记训练集"：用**训练时保存的同一个分词器**，在两批数据上同口径算 bpc——

    A) 训练见过的数据（训练语料的一段，shard 1）          → bpc_seen
    B) 完全没见过的数据（codeparrot-clean 另一分片 shard 2）→ bpc_heldout

判据：
    - bpc_heldout 本身就低（接近 bpc_seen）→ **泛化好**（held-out 压缩=泛化的直接度量）。
    - gap = bpc_heldout − bpc_seen 小（<~0.15 bpc）→ 没有明显记忆/过拟合；gap 大 → 在背训练集。

外加：用几个**新颖 prompt**（训练里不大可能逐字出现的函数签名）做补全 + ast 语法合法率，
看定性泛化（会不会把学到的结构迁移到没见过的起手式）。

前置：先备好 held-out 语料（未见分片）：
    python -m fe_llm.energy_lm.data.prepare_code --start-shard 2 --target-mb 30 --max-shards 1 --out python_heldout.txt

运行：
    python -m fe_llm.energy_lm.evaluation.code_generalization_eval --arch per
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from fe_llm.config import get_device
from fe_llm.energy_lm.models.tokenizer import CharTokenizer
from fe_llm.energy_lm.training.code_train import (
    CKPT_TOK, CORPUS, ckpt_paths, encode_corpus, eval_bpc, generate, load_any,
    make_chunks, syntactic_valid,
)

HELDOUT = os.path.join("data", "code", "python_heldout.txt")

# 新颖 prompt：常见模式但具体组合，训练里不大可能逐字命中，考察结构迁移
NOVEL_PROMPTS = [
    "def is_palindrome(s):\n    ",
    "def merge_two_sorted_lists(l1, l2):\n    ",
    "class BinaryTree:\n    def insert(self, value):\n        ",
    "def count_words(text):\n    ",
    "def binary_search(nums, target):\n    ",
    "def remove_duplicates(items):\n    ",
]


def bpc_on_text(net, tok, text, ctx, device, batch, amp_dtype, max_batches, seed):
    ids = encode_corpus(text, tok)
    chunks = make_chunks(ids, ctx)
    rng = np.random.default_rng(seed)
    chunks = chunks[rng.permutation(len(chunks))]
    bpc, ppl = eval_bpc(net, chunks, device, batch, amp_dtype, max_batches=max_batches)
    return bpc, ppl, len(chunks)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="代码模型泛化能力评测（seen vs 未见分片）。")
    ap.add_argument("--arch", choices=["per", "transformer"], default="per")
    ap.add_argument("--corpus", default=CORPUS, help="训练语料（取一段作 seen 参照）")
    ap.add_argument("--heldout", default=HELDOUT, help="未见分片语料")
    ap.add_argument("--seen-mb", type=float, default=30.0, help="从训练语料取多少 MB 作 seen 参照")
    ap.add_argument("--device", default="")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batches", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep-penalty", type=float, default=1.15)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    args = ap.parse_args(argv)

    device = args.device.strip() or get_device()
    amp_dtype = torch.bfloat16 if (args.amp and device.startswith("cuda")) else None
    if not os.path.exists(CKPT_TOK):
        print(f"[gen] 找不到分词器 {CKPT_TOK}，请先训练。"); return 1
    if not os.path.exists(args.heldout):
        print(f"[gen] 找不到 held-out 语料 {args.heldout}。先跑："
              f"python -m fe_llm.energy_lm.data.prepare_code --start-shard 2 --target-mb 30 --max-shards 1 --out python_heldout.txt")
        return 1
    net_path, last_path, _, meta_path = ckpt_paths(args.arch)
    path = net_path if os.path.exists(net_path) else last_path
    if not os.path.exists(path):
        print(f"[gen] 找不到 {args.arch} 模型 {net_path}。"); return 1

    print(f"[gen] arch={args.arch} 设备={device} amp={'bf16' if amp_dtype else 'off'}", flush=True)
    tok = CharTokenizer.load(CKPT_TOK)            # 关键：用训练时的同一词表
    net = load_any(path, device).to(device)
    print(f"[gen] 模型 {os.path.basename(path)} 参数={sum(p.numel() for p in net.parameters())/1e6:.2f}M 词表={tok.vocab_size}", flush=True)

    # seen 参照：训练语料的一段（模型确实训练过）
    with open(args.corpus, "r", encoding="utf-8") as f:
        seen_text = f.read(int(args.seen_mb * 1_000_000))
    # held-out：未见分片
    with open(args.heldout, "r", encoding="utf-8") as f:
        held_text = f.read()

    t0 = time.time()
    bpc_seen, ppl_seen, n_seen = bpc_on_text(net, tok, seen_text, args.ctx, device, args.batch, amp_dtype, args.eval_batches, args.seed)
    bpc_held, ppl_held, n_held = bpc_on_text(net, tok, held_text, args.ctx, device, args.batch, amp_dtype, args.eval_batches, args.seed)
    gap = round(bpc_held - bpc_seen, 4)

    # 字符覆盖率（held-out 里多少字符落在训练词表内，越高说明分布越接近）
    held_chars = set(held_text)
    in_vocab = sum(1 for c in held_chars if c in tok.tok_to_id)
    cov = in_vocab / max(1, len(held_chars))

    print("=" * 64, flush=True)
    print(f"[gen] seen   (训练分片1 一段)  bpc {bpc_seen:.4f}  ppl {ppl_seen:.2f}  ({n_seen} 块)", flush=True)
    print(f"[gen] heldout(未见分片2)       bpc {bpc_held:.4f}  ppl {ppl_held:.2f}  ({n_held} 块)", flush=True)
    print(f"[gen] 泛化 gap = heldout − seen = {gap:+.4f} bpc  | held-out 字符覆盖率 {cov:.1%}", flush=True)
    if gap <= 0.15:
        verdict = "泛化好：未见数据 bpc 接近训练数据，没有明显记忆/过拟合。"
    elif gap <= 0.35:
        verdict = "泛化尚可：有一定 gap，但未见数据仍被有效压缩。"
    else:
        verdict = "警示：gap 偏大，可能在较强记忆训练分布。"
    print(f"[gen] 判定：{verdict}", flush=True)
    print(f"[gen] 参照：训练日志 in-dist val_bpc ≈ {_meta_val(meta_path)}", flush=True)

    print("-" * 64, flush=True)
    print("[gen] 新颖 prompt 补全（考察结构迁移）：", flush=True)
    n_ok = 0
    for p in NOVEL_PROMPTS:
        out = generate(net, tok, p, net.max_len, device, max_new=args.max_new,
                       temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                       repetition_penalty=args.rep_penalty, amp_dtype=amp_dtype)
        ok = syntactic_valid(p + out)
        n_ok += int(ok)
        shown = (p + out).replace("\n", "\\n")
        print(f"[gen]   [{'ok' if ok else '×'}] {shown[:150]}", flush=True)
    print(f"[gen] 新颖 prompt 语法合法率 {n_ok}/{len(NOVEL_PROMPTS)} = {n_ok/len(NOVEL_PROMPTS):.0%}", flush=True)
    print(f"[gen] 用时 {time.time()-t0:.0f}s", flush=True)
    return 0


def _meta_val(meta_path):
    import json
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f).get("val_bpc", "?")
        except Exception:
            return "?"
    return "?"


if __name__ == "__main__":
    raise SystemExit(main())
