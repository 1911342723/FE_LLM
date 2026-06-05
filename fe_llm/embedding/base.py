# -*- coding: utf-8 -*-
"""
embedding/base.py —— 嵌入器抽象接口与几何工具
==============================================
定义所有嵌入后端必须实现的统一接口 Embedder，以及向量空间的几何度量工具。
上层（世界模型、自由能引擎）只依赖这个抽象，不关心具体是真实 API 还是哈希降级。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import numpy as np


def unit(vec: np.ndarray) -> np.ndarray:
    """把向量归一化为单位向量（长度 1），便于用点积直接当余弦相似度。"""
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度：1=方向一致，0=正交，-1=完全相反。"""
    return float(np.dot(unit(a), unit(b)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """余弦距离 = 1 - 相似度，范围 [0,2]。这是浅层惊奇(语义距离)的核心度量。"""
    return 1.0 - cosine_similarity(a, b)


def tokenize_words(text: str) -> list[str]:
    """极简分词：中文按单字、英文/数字按连续串切。供哈希嵌入与因果检测复用。"""
    pattern = r"[\u4e00-\u9fff]|[a-zA-Z]+|[0-9]+"
    return re.findall(pattern, text)


class Embedder(ABC):
    """
    嵌入器统一接口。

    任何后端只要实现 embed_one()，即可获得批量 embed() 与几何工具能力。
    """

    #: 该后端输出向量的维度（真实模型 1536，哈希降级可自定）
    dimension: int

    @abstractmethod
    def embed_one(self, text: str) -> np.ndarray:
        """把单段文本编码为一维 numpy 单位向量。"""
        raise NotImplementedError

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        批量编码，返回形状 (N, dimension) 的矩阵。
        默认逐条调用 embed_one；真实 API 后端会重写为一次网络请求批处理。
        """
        return np.vstack([self.embed_one(t) for t in texts])

    # —— 几何工具透传，方便外部直接通过 embedder 调用 ——
    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        return cosine_distance(a, b)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return cosine_similarity(a, b)
