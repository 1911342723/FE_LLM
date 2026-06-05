# -*- coding: utf-8 -*-
"""
world_model/world_model.py —— 世界模型门面
============================================
封装能量地貌的核心操作，屏蔽底层存储（pgvector / 内存）与嵌入后端差异：
    - settle()  : 「滚落」——把认知状态抛入地貌，返回最近的吸引子
    - nearest() : 最近的单个吸引子
    - min_surprise_distance() : 浅层惊奇的基准距离
    - learn()   : 把新认知作为浅吸引子写回（知识演化，零微调）
"""

from __future__ import annotations

import numpy as np

from ..embedding.base import Embedder
from .concept import Concept
from .memory_store import MemoryStore
from .store import VectorStore


class WorldModel:
    """能量地貌门面 = 向量存储 + 嵌入器 + 滚落逻辑。"""

    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    # ---------------------- 知识灌入 ----------------------
    def add(self, name: str, text: str, category: str = "常识",
            depth: float = 1.0, relations: dict | None = None) -> Concept:
        """添加一个概念：自动用嵌入器生成向量后写入存储。"""
        vec = self.embedder.embed_one(text)
        concept = Concept(name=name, text=text, category=category,
                          depth=depth, relations=relations or {}, vector=vec)
        self.store.upsert(concept)
        return concept

    # ---------------------- 能量地貌核心操作 ----------------------
    def settle(self, state: np.ndarray, top_k: int = 1) -> list[tuple[Concept, float]]:
        """
        「滚落」：把状态向量抛入地貌，返回它最可能落入的 top_k 个吸引子及余弦距离。

        深度只做**轻微**折扣（让深公理的谷略宽），严格限幅，不会让方向无关的
        深公理抢占最近邻。返回的距离始终是诚实的真实余弦距离，作惊奇度基准。
        """
        results = self.store.search(state, top_k=max(top_k, 8))
        if not results:
            return []

        rescored = []
        for concept, dist in results:
            discount = 1.0 - 0.08 * np.tanh(concept.depth)
            rescored.append((concept, dist, dist * discount))
        rescored.sort(key=lambda x: x[2])  # 按折扣后的有效距离排序
        # 但对外返回真实距离
        return [(c, d) for c, d, _ in rescored[:top_k]]

    def nearest(self, state: np.ndarray) -> tuple[Concept, float] | None:
        res = self.settle(state, top_k=1)
        return res[0] if res else None

    def min_surprise_distance(self, state: np.ndarray) -> float:
        """状态到最近吸引子的余弦距离。无知识时返回最大惊奇 2.0。"""
        n = self.nearest(state)
        return n[1] if n else 2.0

    def get(self, name: str) -> Concept | None:
        return self.store.get(name)

    def all_concepts(self) -> list[Concept]:
        return self.store.all()

    # ---------------------- 知识演化 ----------------------
    def learn(self, name: str, text: str, depth: float = 0.6) -> Concept:
        """把新认知作为一个浅吸引子写回（depth 小 → 不撼动核心公理）。"""
        return self.add(name=name, text=text, category="经验", depth=depth)

    def __len__(self) -> int:
        return self.store.count()


def build_world_model(embedder: Embedder, backend: str = "auto") -> WorldModel:
    """
    构建世界模型，自动选择存储后端。

    backend:
        "pg"     → 强制使用 pgvector（失败即报错）
        "memory" → 强制内存
        "auto"   → 优先 pgvector，连接失败自动降级内存
    """
    if backend in ("pg", "auto"):
        try:
            from .pgvector_store import PgVectorStore

            store: VectorStore = PgVectorStore(dimension=embedder.dimension)
            return WorldModel(store, embedder)
        except Exception as exc:
            if backend == "pg":
                raise
            print(f"[world_model] pgvector 不可用，降级为内存存储：{exc}")

    return WorldModel(MemoryStore(), embedder)
