# -*- coding: utf-8 -*-
"""
embedding 包 —— 语义嵌入层
============================
把任意文本映射成高维向量，是整个系统感知外部世界的「视网膜」。

提供两种后端，由 get_embedder() 自动选择，对上层透明：
    - DashScopeEmbedder : 阿里云真实向量模型(text-embedding-v4, 1536维)，正式使用。
    - HashEmbedder      : 确定性哈希向量，无网络/无密钥时的降级方案，用于离线测试。

得益于马尔可夫毯式的接口隔离，更换后端时上层自由能逻辑完全不用改。
"""

from .base import Embedder, cosine_distance, cosine_similarity, unit
from .factory import get_embedder
from .hash_embedder import HashEmbedder

__all__ = [
    "Embedder",
    "HashEmbedder",
    "get_embedder",
    "cosine_distance",
    "cosine_similarity",
    "unit",
]
