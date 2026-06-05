# -*- coding: utf-8 -*-
"""
scripts/seed_db.py —— 灌入出厂世界观种子知识
=============================================
运行：python scripts/seed_db.py

把 data/seed_knowledge.py 中的所有概念，用真实向量模型编码后写入 pgvector。
这一步等价于给系统"出厂烧录世界观"。幂等：重复运行只会覆盖更新同名概念。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.seed_knowledge import SEED_CONCEPTS
from fe_llm.embedding import get_embedder
from fe_llm.world_model.concept import Concept
from fe_llm.world_model.pgvector_store import PgVectorStore


def main() -> None:
    embedder = get_embedder()  # 自动选 DashScope；无密钥则降级哈希
    print(f"[seed_db] 嵌入后端：{type(embedder).__name__}  维度={embedder.dimension}")

    store = PgVectorStore(dimension=embedder.dimension)

    # 批量编码（一次网络请求，命中缓存则更快）
    texts = [c["text"] for c in SEED_CONCEPTS]
    print(f"[seed_db] 正在编码 {len(texts)} 条种子知识 ...")
    vectors = embedder.embed(texts)

    concepts = []
    for spec, vec in zip(SEED_CONCEPTS, vectors):
        concepts.append(Concept(
            name=spec["name"],
            text=spec["text"],
            category=spec.get("category", "常识"),
            depth=float(spec.get("depth", 1.0)),
            relations=spec.get("relations", {}),
            vector=vec,
        ))

    store.upsert_many(concepts)
    print(f"[seed_db] 已写入/更新 {len(concepts)} 条概念，当前库中共 {store.count()} 条。")
    store.close()


if __name__ == "__main__":
    main()
