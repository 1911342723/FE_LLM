# -*- coding: utf-8 -*-
"""
scripts/clean_memory.py —— 清理学习产生的经验记忆
==================================================
运行：python scripts/clean_memory.py

只删除 category='经验' 的概念（系统运行时通过知识演化写回的浅吸引子），
保留所有出厂世界观（公理/常识/领域）。用于把世界模型重置回纯净出厂状态，
便于演示与回归测试的可复现。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg

from fe_llm.config import get_pg_config


def main() -> None:
    pg = get_pg_config()
    conn = psycopg.connect(**pg.conninfo(), autocommit=True)
    try:
        deleted = conn.execute(
            f"DELETE FROM {pg.table} WHERE metadata->>'category' = '经验'"
        ).rowcount
        total = conn.execute(f"SELECT COUNT(*) FROM {pg.table}").fetchone()[0]
        print(f"[clean_memory] 已删除 {deleted} 条经验记忆，剩余 {total} 条出厂概念。")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
