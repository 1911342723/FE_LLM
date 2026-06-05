# -*- coding: utf-8 -*-
"""
world_model/store.py —— 向量存储抽象接口
==========================================
定义世界模型底层存储必须实现的能力。上层 WorldModel 只依赖此抽象，
因此可在 PgVectorStore(正式) 与 MemoryStore(降级) 之间无缝切换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .concept import Concept


class VectorStore(ABC):
    """向量存储后端接口。"""

    @abstractmethod
    def upsert(self, concept: Concept) -> None:
        """插入或更新一个概念（按 name 主键）。"""

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int) -> list[tuple[Concept, float]]:
        """
        近邻检索：返回与 query 最接近的 top_k 个概念及其「余弦距离」。
        这是能量地貌「滚落」操作的物理实现。
        """

    @abstractmethod
    def get(self, name: str) -> Concept | None:
        """按名称取概念。"""

    @abstractmethod
    def all(self) -> list[Concept]:
        """返回全部概念。"""

    @abstractmethod
    def count(self) -> int:
        """概念总数。"""
