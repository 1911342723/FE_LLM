# -*- coding: utf-8 -*-
"""
world_model/memory_store.py —— 内存向量存储（降级后端）
========================================================
纯内存 numpy 矩阵实现，无数据库依赖。语义与 PgVectorStore 完全一致，
用于离线开发、单元测试，或数据库不可用时的自动降级。
"""

from __future__ import annotations

import numpy as np

from ..embedding.base import cosine_distance
from .concept import Concept
from .store import VectorStore


class MemoryStore(VectorStore):
    """基于内存字典 + numpy 的向量存储。"""

    def __init__(self):
        self._concepts: dict[str, Concept] = {}
        self._matrix: np.ndarray | None = None
        self._names: list[str] = []

    def upsert(self, concept: Concept) -> None:
        if concept.vector is None:
            raise ValueError(f"概念「{concept.name}」缺少向量，无法写入。")
        self._concepts[concept.name] = concept
        self._matrix = None  # 失效缓存

    def _rebuild(self) -> None:
        self._names = list(self._concepts.keys())
        if self._names:
            self._matrix = np.vstack([self._concepts[n].vector for n in self._names])
        else:
            self._matrix = np.zeros((0, 1))

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[Concept, float]]:
        if self._matrix is None:
            self._rebuild()
        if not self._names:
            return []
        dists = np.array([cosine_distance(query, v) for v in self._matrix])
        order = np.argsort(dists)[:top_k]
        return [(self._concepts[self._names[i]], float(dists[i])) for i in order]

    def get(self, name: str) -> Concept | None:
        return self._concepts.get(name)

    def all(self) -> list[Concept]:
        return list(self._concepts.values())

    def count(self) -> int:
        return len(self._concepts)
