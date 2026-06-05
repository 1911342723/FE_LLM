# -*- coding: utf-8 -*-
"""
embedding/dashscope_embedder.py —— 阿里云 DashScope 真实向量后端
=================================================================
通过 OpenAI 兼容协议调用 text-embedding-v4，输出 1536 维真实语义向量。
这是正式模型使用的嵌入后端。

特性：
    - 批量请求：一次网络调用编码多条文本，降低延迟。
    - 本地缓存：相同文本不重复请求（磁盘缓存），节省额度与时间。
    - 单位归一化：与全局几何约定一致，便于直接用点积算余弦相似度。
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from ..config import EmbeddingConfig, get_embedding_config
from .base import Embedder, unit


class DashScopeEmbedder(Embedder):
    """阿里云 DashScope 向量模型后端。"""

    def __init__(self, config: EmbeddingConfig | None = None,
                 cache_dir: str = ".cache/embeddings", batch_size: int = 10):
        from openai import OpenAI  # 延迟导入，未用真实后端时无需安装

        self.config = config or get_embedding_config()
        if not self.config.api_key:
            raise RuntimeError("未配置 DASHSCOPE_API_KEY，无法使用真实向量后端。")

        self.dimension = self.config.dimension
        # DashScope 单次请求最多 10 条，超出需自动分批
        self.batch_size = batch_size
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        # 磁盘缓存目录：相同文本的向量落盘，避免重复计费
        self._cache_dir = cache_dir
        os.makedirs(self._cache_dir, exist_ok=True)

    # ---------------------- 缓存读写 ----------------------
    def _cache_path(self, text: str) -> str:
        key = hashlib.md5(
            f"{self.config.model}:{self.dimension}:{text}".encode("utf-8")
        ).hexdigest()
        return os.path.join(self._cache_dir, key + ".json")

    def _load_cache(self, text: str) -> np.ndarray | None:
        path = self._cache_path(text)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return np.array(json.load(f), dtype=np.float32)
        return None

    def _save_cache(self, text: str, vec: np.ndarray) -> None:
        with open(self._cache_path(text), "w", encoding="utf-8") as f:
            json.dump(vec.tolist(), f)

    # ---------------------- 编码接口 ----------------------
    def _request(self, texts: list[str]) -> list[np.ndarray]:
        """向 DashScope 发起编码请求，自动按 batch_size 分批（上限 10 条/次）。"""
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            resp = self._client.embeddings.create(
                model=self.config.model,
                input=chunk,
                dimensions=self.dimension,
            )
            # 按 index 排序，确保与输入顺序一致
            ordered = sorted(resp.data, key=lambda d: d.index)
            out.extend(unit(np.array(d.embedding, dtype=np.float32)) for d in ordered)
        return out

    def embed_one(self, text: str) -> np.ndarray:
        """单条编码（优先命中缓存）。"""
        cached = self._load_cache(text)
        if cached is not None:
            return cached
        vec = self._request([text])[0]
        self._save_cache(text, vec)
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        批量编码：先查缓存，未命中的合并成一次网络请求，再回填缓存。
        """
        results: list[np.ndarray | None] = [None] * len(texts)
        missing_idx: list[int] = []
        missing_text: list[str] = []

        for i, t in enumerate(texts):
            cached = self._load_cache(t)
            if cached is not None:
                results[i] = cached
            else:
                missing_idx.append(i)
                missing_text.append(t)

        if missing_text:
            vecs = self._request(missing_text)
            for idx, t, v in zip(missing_idx, missing_text, vecs):
                results[idx] = v
                self._save_cache(t, v)

        return np.vstack(results)
