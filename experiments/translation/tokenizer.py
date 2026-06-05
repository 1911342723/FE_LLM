# -*- coding: utf-8 -*-
"""
translation/tokenizer.py —— 共享子词分词器（SentencePiece）
============================================================
对应文档第四步「概念分词器」的精神：把连续文本切成离散语义单元。
这里用 SentencePiece 训练一个**中英共享**的子词词表（unigram 模型），
中英文混在一起训练，使两种语言共享一套 token 空间，便于单模型双向翻译。

特殊 token：
    <pad>=0  <unk>=1  <s>=2(BOS)  </s>=3(EOS)
    <2zh>    翻译方向标签：目标语言为中文
    <2en>    翻译方向标签：目标语言为英文

源句格式： <2en> 今天天气不错      （要翻成英文）
目标句格式：<s> The weather is nice today </s>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sentencepiece as spm

MODEL_DIR = os.path.join("checkpoints", "translation")
SPM_PREFIX = os.path.join(MODEL_DIR, "spm")
SPM_MODEL = SPM_PREFIX + ".model"

# 方向标签（作为 user_defined_symbols 加入词表，保证不被切碎）
TAG_ZH = "<2zh>"
TAG_EN = "<2en>"

# 特殊 id（与训练参数保持一致）
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def train_spm(corpus_txt: str, vocab_size: int = 8000) -> str:
    """
    用纯文本语料训练 SentencePiece 词表。
    corpus_txt 每行一句（中英混合）。返回模型路径。

    hard_vocab_limit=False：语料较小、可用子词不足以撑满目标词表时，
    自动缩减到可行的词表大小而不报错（小语料也能跑通）。
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=corpus_txt,
        model_prefix=SPM_PREFIX,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,   # 覆盖绝大多数中文字符
        pad_id=PAD_ID, unk_id=UNK_ID, bos_id=BOS_ID, eos_id=EOS_ID,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
        user_defined_symbols=[TAG_ZH, TAG_EN],
        normalization_rule_name="nmt_nfkc_cf",
        hard_vocab_limit=False,
    )
    print(f"[spm] 词表训练完成：{SPM_MODEL}")
    return SPM_MODEL


class Tokenizer:
    """SentencePiece 分词器封装。"""

    def __init__(self, model_path: str = SPM_MODEL):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到分词器模型：{model_path}，请先训练。")
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.pad_id = PAD_ID
        self.unk_id = UNK_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.tag_zh_id = self.sp.piece_to_id(TAG_ZH)
        self.tag_en_id = self.sp.piece_to_id(TAG_EN)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode_source(self, text: str, target_lang: str) -> list[int]:
        """源句编码：开头加方向标签。target_lang ∈ {'zh','en'}。"""
        tag = self.tag_en_id if target_lang == "en" else self.tag_zh_id
        return [tag] + self.sp.encode(text, out_type=int)

    def encode_target(self, text: str) -> list[int]:
        """目标句编码：<s> ... </s>。"""
        return [self.bos_id] + self.sp.encode(text, out_type=int) + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        """把 id 序列还原为文本（自动跳过特殊 token）。"""
        clean = [i for i in ids
                 if i not in (self.pad_id, self.bos_id, self.eos_id,
                              self.tag_zh_id, self.tag_en_id)]
        return self.sp.decode(clean)
