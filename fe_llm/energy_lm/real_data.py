# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/real_data.py —— 真实中文语料（规模化测试用）
=============================================================
从本地翻译语料的中文侧抽取**干净、长度适中**的真实句子，用于检验能量坍缩
架构离开几十句玩具、上到几千条真实句子时是否仍成立。

任务设定：句子重建（masked reconstruction）——给被部分遮盖的真实句，
能量坍缩把它复原。这是检验"能量地貌能否承载大量多样模式"的干净任务，
不依赖人工对话对。
"""

from __future__ import annotations

import json
import os
import re

_CJK = re.compile(r"[\u4e00-\u9fff]")
PAIRS = os.path.join("data", "translation", "pairs.jsonl")


def load_clean_sentences(max_n: int = 4000, min_len: int = 4, max_len: int = 14):
    """
    抽取干净中文句：长度适中、以中文为主、无明显噪声标点。
    返回去重后的句子列表。
    """
    out, seen = [], set()
    with open(PAIRS, "r", encoding="utf-8") as f:
        for line in f:
            try:
                zh = json.loads(line)["zh"].strip()
            except (json.JSONDecodeError, KeyError):
                continue
            # 清洗：去首尾标点/空白，只保留中文为主的句子
            zh = zh.strip(" .。,，!！?？\"'-—…·、")
            if not (min_len <= len(zh) <= max_len):
                continue
            cjk = sum(1 for c in zh if _CJK.match(c))
            if cjk / len(zh) < 0.85:               # 至少 85% 是中文
                continue
            if zh in seen:
                continue
            seen.add(zh)
            out.append(zh)
            if len(out) >= max_n:
                break
    return out


def char_set(sentences):
    s = set()
    for sent in sentences:
        s.update(sent)
    return sorted(s)


if __name__ == "__main__":
    sents = load_clean_sentences()
    chars = char_set(sents)
    print(f"干净中文句：{len(sents)}")
    print(f"字表规模：{len(chars)}")
    for s in sents[:10]:
        print("  ", s)
