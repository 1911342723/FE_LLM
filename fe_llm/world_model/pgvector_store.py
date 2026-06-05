# -*- coding: utf-8 -*-
"""
world_model/pgvector_store.py —— PostgreSQL + pgvector 存储（正式后端）
=======================================================================
对应文档：「底层知识基座：挂载 pgvector 的 PostgreSQL，沉淀静态的高维概念与公理」。

表结构（建表见 scripts/init_db.py）：
    concepts(
        name      TEXT PRIMARY KEY,   -- 概念名（唯一）
        text      TEXT,               -- 自然语言描述
        metadata  JSONB,              -- {category, depth, relations}
        embedding VECTOR(dim)         -- 高维向量（pgvector 类型）
    )

近邻检索用 pgvector 的余弦距离算子 `<=>`，由 ivfflat/hnsw 索引加速。
"""

from __future__ import annotations

import json

import numpy as np

from ..config import PgConfig, get_pg_config
from .concept import Concept
from .store import VectorStore


class PgVectorStore(VectorStore):
    """pgvector 持久化后端。"""

    def __init__(self, dimension: int, config: PgConfig | None = None):
        import psycopg
        from pgvector.psycopg import register_vector

        self.config = config or get_pg_config()
        self.table = self.config.table
        self.dimension = dimension

        # 建立长连接（PoC 用单连接；生产可换连接池）
        self._conn = psycopg.connect(**self.config.conninfo(), autocommit=True)
        register_vector(self._conn)  # 注册 vector 类型，使 numpy 数组可直接传参

        # 提高 ivfflat 探针数以保证召回率。
        # 知识库较小时单探针只搜一个聚类簇会严重漏召回，导致最近邻判断错误，
        # 进而破坏因果冲突检测。这里设为较大值近似精确检索；库变大后可调小以换速度。
        try:
            self._conn.execute("SET ivfflat.probes = 100")
        except Exception:
            pass  # 无 ivfflat 索引时忽略

    # ---------------------- 写入 ----------------------
    def upsert(self, concept: Concept) -> None:
        if concept.vector is None:
            raise ValueError(f"概念「{concept.name}」缺少向量，无法写入。")
        vec = np.asarray(concept.vector, dtype=np.float32)
        self._conn.execute(
            f"""
            INSERT INTO {self.table} (name, text, metadata, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE
              SET text = EXCLUDED.text,
                  metadata = EXCLUDED.metadata,
                  embedding = EXCLUDED.embedding
            """,
            (concept.name, concept.text,
             json.dumps(concept.to_metadata(), ensure_ascii=False), vec),
        )

    def upsert_many(self, concepts: list[Concept]) -> None:
        """批量写入（建库灌种子时用）。"""
        for c in concepts:
            self.upsert(c)

    # ---------------------- 检索 ----------------------
    def search(self, query: np.ndarray, top_k: int) -> list[tuple[Concept, float]]:
        """用 pgvector 余弦距离算子 `<=>` 做近邻检索。"""
        vec = np.asarray(query, dtype=np.float32)
        rows = self._conn.execute(
            f"""
            SELECT name, text, metadata, embedding,
                   embedding <=> %s AS distance
            FROM {self.table}
            ORDER BY distance ASC
            LIMIT %s
            """,
            (vec, top_k),
        ).fetchall()

        results: list[tuple[Concept, float]] = []
        for name, text, metadata, embedding, distance in rows:
            meta = metadata if isinstance(metadata, dict) else json.loads(metadata)
            concept = Concept.from_row(
                name, text, meta, np.asarray(embedding, dtype=np.float32)
            )
            results.append((concept, float(distance)))
        return results

    def get(self, name: str) -> Concept | None:
        row = self._conn.execute(
            f"SELECT name, text, metadata, embedding FROM {self.table} WHERE name = %s",
            (name,),
        ).fetchone()
        if row is None:
            return None
        name, text, metadata, embedding = row
        meta = metadata if isinstance(metadata, dict) else json.loads(metadata)
        return Concept.from_row(name, text, meta,
                                np.asarray(embedding, dtype=np.float32))

    def all(self) -> list[Concept]:
        rows = self._conn.execute(
            f"SELECT name, text, metadata, embedding FROM {self.table}"
        ).fetchall()
        out = []
        for name, text, metadata, embedding in rows:
            meta = metadata if isinstance(metadata, dict) else json.loads(metadata)
            out.append(Concept.from_row(name, text, meta,
                                        np.asarray(embedding, dtype=np.float32)))
        return out

    def count(self) -> int:
        return self._conn.execute(
            f"SELECT COUNT(*) FROM {self.table}"
        ).fetchone()[0]

    def close(self) -> None:
        self._conn.close()
