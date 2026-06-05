# -*- coding: utf-8 -*-
"""
translation/dataset.py —— 平行语料加载与双向样本构造
=====================================================
把 data/translation/pairs.jsonl 加工成训练样本。

双向：每个中英对 (zh, en) 同时产生两条训练样本：
    - 中译英： 源=<2en> zh   目标=<s> en </s>
    - 英译中： 源=<2zh> en   目标=<s> zh </s>
这样一个模型就能学会双向翻译。
"""

from __future__ import annotations

import json
import os

import torch
from torch.utils.data import Dataset

from .tokenizer import Tokenizer


def load_pairs(path: str) -> list[dict]:
    """读取 jsonl 平行语料。"""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("zh") and obj.get("en"):
                    pairs.append(obj)
            except json.JSONDecodeError:
                continue
    return pairs


def write_spm_corpus(pairs: list[dict], out_txt: str) -> str:
    """把所有中英句子写成纯文本（每行一句），供 SentencePiece 训练词表。"""
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(p["zh"].strip() + "\n")
            f.write(p["en"].strip() + "\n")
    return out_txt


class TranslationDataset(Dataset):
    """双向翻译数据集。每个原始句对展开为两条样本。"""

    def __init__(self, pairs: list[dict], tokenizer: Tokenizer, max_len: int = 128):
        self.tok = tokenizer
        self.max_len = max_len
        self.samples: list[tuple[list[int], list[int]]] = []
        for p in pairs:
            zh, en = p["zh"].strip(), p["en"].strip()
            # 中译英
            self._add(self.tok.encode_source(zh, "en"), self.tok.encode_target(en))
            # 英译中
            self._add(self.tok.encode_source(en, "zh"), self.tok.encode_target(zh))

    def _add(self, src: list[int], tgt: list[int]) -> None:
        if 2 <= len(src) <= self.max_len and 2 <= len(tgt) <= self.max_len:
            self.samples.append((src, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def make_collate(pad_id: int):
    """构造 collate_fn：把变长样本 pad 成同长批次。"""

    def collate(batch):
        srcs, tgts = zip(*batch)
        src_max = max(len(s) for s in srcs)
        tgt_max = max(len(t) for t in tgts)

        def pad(seqs, maxlen):
            return torch.tensor(
                [s + [pad_id] * (maxlen - len(s)) for s in seqs], dtype=torch.long)

        src = pad(srcs, src_max)
        tgt = pad(tgts, tgt_max)
        # teacher forcing：decoder 输入是目标去掉最后一位，标签是目标去掉第一位
        return src, tgt[:, :-1], tgt[:, 1:]

    return collate
