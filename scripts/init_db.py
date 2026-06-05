# -*- coding: utf-8 -*-
"""
scripts/init_db.py —— 初始化 pgvector 数据库
=============================================
运行：python scripts/init_db.py

幂等操作：
    1) 确保目标数据库存在（不存在则创建）。
    2) 启用 vector 扩展。
    3) 建立 concepts 表（向量维度取自 .env 的 EMBEDDING_DIMENSION）。
    4) 建立 ivfflat 余弦距离索引以加速近邻检索。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 保证可从项目根导入 fe_llm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg

from fe_llm.config import get_embedding_config, get_pg_config


def ensure_database(pg, dbname: str) -> None:
    """连接到 postgres 维护库，确保目标库存在。"""
    admin = psycopg.connect(**pg.conninfo(dbname="postgres"), autocommit=True)
    try:
        exists = admin.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)
        ).fetchone()
        if not exists:
            admin.execute(f'CREATE DATABASE "{dbname}"')
            print(f"[init_db] 已创建数据库 {dbname}")
        else:
            print(f"[init_db] 数据库 {dbname} 已存在")
    finally:
        admin.close()


def init_schema(pg, table: str, dim: int) -> None:
    """在目标库中建表与索引。"""
    conn = psycopg.connect(**pg.conninfo(), autocommit=True)
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                name      TEXT PRIMARY KEY,
                text      TEXT NOT NULL,
                metadata  JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding VECTOR({dim}) NOT NULL
            )
            """
        )
        # ivfflat 余弦索引（lists 数量按数据量调整；种子量小时 lists=10 足够）
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {table}_embedding_cos_idx
            ON {table} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 10)
            """
        )
        print(f"[init_db] 表 {table}(VECTOR({dim})) 与余弦索引已就绪")
    finally:
        conn.close()


def main() -> None:
    pg = get_pg_config()
    emb = get_embedding_config()
    print(f"[init_db] 目标：{pg.host}:{pg.port}/{pg.database} 表={pg.table} 维度={emb.dimension}")
    ensure_database(pg, pg.database)
    init_schema(pg, pg.table, emb.dimension)
    print("[init_db] 完成。")


if __name__ == "__main__":
    main()
