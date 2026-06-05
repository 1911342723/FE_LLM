# -*- coding: utf-8 -*-
"""
embedding/factory.py —— 嵌入后端工厂
=====================================
根据配置与运行环境自动选择嵌入后端，并在进程内复用同一个实例（含缓存）。

选择策略：
    1) 显式指定 backend="dashscope"/"hash" → 按指定创建。
    2) backend="auto"（默认）：有密钥就用 DashScope 真实向量，否则降级哈希。
       真实后端初始化失败（无网络/密钥失效）时自动回退哈希，保证系统永远可跑。
"""

from __future__ import annotations

from ..config import get_embedding_config
from .base import Embedder
from .hash_embedder import HashEmbedder

# 进程级单例缓存：避免重复建客户端、重复读缓存
_INSTANCE: Embedder | None = None


def get_embedder(backend: str = "auto", force_new: bool = False) -> Embedder:
    """获取嵌入器实例（默认单例）。"""
    global _INSTANCE
    if _INSTANCE is not None and not force_new:
        return _INSTANCE

    embedder = _create(backend)
    if not force_new:
        _INSTANCE = embedder
    return embedder


def _create(backend: str) -> Embedder:
    cfg = get_embedding_config()

    if backend == "hash":
        return HashEmbedder()

    if backend in ("dashscope", "auto"):
        if cfg.api_key:
            try:
                from .dashscope_embedder import DashScopeEmbedder

                return DashScopeEmbedder(cfg)
            except Exception as exc:  # 网络/密钥问题 → 降级
                if backend == "dashscope":
                    raise
                print(f"[embedding] 真实后端不可用，降级为哈希嵌入：{exc}")
                return HashEmbedder()
        if backend == "dashscope":
            raise RuntimeError("backend=dashscope 但未配置 API 密钥。")
        # auto 且无密钥 → 哈希降级
        return HashEmbedder()

    raise ValueError(f"未知的嵌入后端：{backend}")
