# -*- coding: utf-8 -*-
"""
world_model 包 —— ① 世界模型（出厂世界观 / 能量地貌）
=====================================================
对应文档第一步：「先造脑干(数据库)」。

把核心公理/概念沉淀为高维向量「吸引子」，构成系统不可篡改的能量地貌。
信息提取不是查数据库，而是把当前认知状态抛入地貌，让它自然滚落到最近的深谷。

两种存储后端，对上层透明：
    - PgVectorStore : 挂载 pgvector 的 PostgreSQL，正式持久化后端。
    - MemoryStore   : 纯内存 numpy 矩阵，无数据库时的降级/测试后端。

WorldModel 是统一门面，封装「滚落(settle)」「最近邻」「写回学习」等能量地貌操作。
"""

from .concept import Concept
from .memory_store import MemoryStore
from .store import VectorStore
from .world_model import WorldModel, build_world_model

__all__ = [
    "Concept",
    "VectorStore",
    "MemoryStore",
    "WorldModel",
    "build_world_model",
]
