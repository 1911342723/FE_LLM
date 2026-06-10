# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/prepare_opus100.py —— 下载并筛选 opus-100 en-zh 翻译对
======================================================================
用途：用 IntentLM 架构（意图弛豫 + 能量递减解码）训练一个小的中文→英文
翻译模型，检验该架构在"语义压缩→跨语言重建"任务上的泛化能力。

筛选口径（字符级小模型的能力边界）：
    - 中文侧：2~28 个字符，必须含 CJK；
    - 英文侧：4~46 个 ASCII 字符（统一小写，缩小字表）；
    - 排除网址/数字堆/乱码行。

输出：
    data/translation/opus100_train.jsonl  （默认 5 万对）
    data/translation/opus100_val.jsonl    （验证集全部过筛样本）

运行：python fe_llm/energy_lm/prepare_opus100.py [--n 50000]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT_DIR = os.path.join("data", "translation")
TRAIN_PATH = os.path.join(OUT_DIR, "opus100_train.jsonl")
VAL_PATH = os.path.join(OUT_DIR, "opus100_val.jsonl")

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
EN_OK_RE = re.compile(r"^[a-z0-9 ,.!?'\-:;()\"]+$")


def clean_pair(zh: str, en: str) -> tuple[str, str] | None:
    zh = zh.strip()
    en = en.strip().lower()
    if not (2 <= len(zh) <= 28):
        return None
    if not (4 <= len(en) <= 46):
        return None
    if not CJK_RE.search(zh):
        return None
    # 中文侧不允许夹杂大量字母（多为代码/混排噪声）
    if sum(c.isascii() and c.isalpha() for c in zh) > 4:
        return None
    if not EN_OK_RE.match(en):
        return None
    if "http" in en or "www" in en:
        return None
    return zh, en


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000, help="训练对数量上限")
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(OUT_DIR, exist_ok=True)

    kept = 0
    seen: set[str] = set()
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        ds = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="train", streaming=True)
        for ex in ds:
            pair = clean_pair(ex["translation"]["zh"], ex["translation"]["en"])
            if pair is None:
                continue
            zh, en = pair
            if zh in seen:  # 同一中文多译会让字符级小模型目标互相打架
                continue
            seen.add(zh)
            f.write(json.dumps({"zh": zh, "en": en}, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 10000 == 0:
                print(f"[opus100] 已筛选 {kept} 对", flush=True)
            if kept >= args.n:
                break
    print(f"[opus100] 训练集 {kept} 对 -> {TRAIN_PATH}")

    val_kept = 0
    with open(VAL_PATH, "w", encoding="utf-8") as f:
        ds_val = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="validation")
        for ex in ds_val:
            pair = clean_pair(ex["translation"]["zh"], ex["translation"]["en"])
            if pair is None:
                continue
            zh, en = pair
            if zh in seen:  # 确保验证集对训练集完全未见
                continue
            f.write(json.dumps({"zh": zh, "en": en}, ensure_ascii=False) + "\n")
            val_kept += 1
    print(f"[opus100] 验证集 {val_kept} 对 -> {VAL_PATH}")


if __name__ == "__main__":
    main()
