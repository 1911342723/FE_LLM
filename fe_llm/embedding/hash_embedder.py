# -*- coding: utf-8 -*-
"""
embedding/hash_embedder.py —— 确定性哈希嵌入（降级后端）
========================================================
当没有网络或未配置 API 密钥时使用。同一段文字永远得到同一个向量，
语义无关的文字在高维空间近似正交。纯 numpy，无外部依赖，用于离线开发与单测。
"""

from __future__ import annotations

import hashlib

import numpy as np

from .base import Embedder, tokenize_words, unit


class HashEmbedder(Embedder):
    """基于 md5 种子 + 伪随机的确定性嵌入。"""

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    @staticmethod
    def _seed(text: str) -> int:
        """用 md5 把字符串转成稳定的 32 位整数种子。"""
        return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)

    def _embed_token(self, token: str) -> np.ndarray:
        """单个词元 → 单位向量（确定性）。"""
        rng = np.random.default_rng(self._seed("tok::" + token))
        return unit(rng.standard_normal(self.dimension))

    def embed_one(self, text: str) -> np.ndarray:
        """一段文本 → 词袋平均向量。空输入返回由原文哈希决定的随机向量。"""
        tokens = tokenize_words(text)
        if not tokens:
            rng = np.random.default_rng(self._seed("empty::" + text))
            return unit(rng.standard_normal(self.dimension))
        acc = np.zeros(self.dimension, dtype=np.float32)
        for tok in tokens:
            acc += self._embed_token(tok)
        return unit(acc)
