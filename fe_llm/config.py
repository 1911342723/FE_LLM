# -*- coding: utf-8 -*-
"""
config.py —— 统一配置中心
==========================
所有外部依赖（向量 API / PostgreSQL / 训练设备）的配置都集中从 .env 读取，
绝不在代码里硬编码密钥。其它模块只从这里取配置，便于统一管理与替换。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

try:
    from dotenv import load_dotenv

    load_dotenv()  # 自动加载项目根目录的 .env
except ImportError:  # pragma: no cover
    # 没装 python-dotenv 时也能用真实环境变量运行
    pass


@dataclass(frozen=True)
class EmbeddingConfig:
    """嵌入模型配置（阿里云 DashScope，OpenAI 兼容协议）。"""

    api_key: str
    base_url: str
    model: str
    dimension: int


@dataclass(frozen=True)
class TeacherConfig:
    """DeepSeek 教师模型配置（用于蒸馏生成训练数据）。"""

    api_key: str
    base_url: str
    model: str


@dataclass(frozen=True)
class PgConfig:
    """PostgreSQL + pgvector 连接配置。"""

    host: str
    port: int
    database: str
    user: str
    password: str
    table: str

    def conninfo(self, dbname: str | None = None) -> dict:
        """生成 psycopg.connect 所需的连接参数字典。"""
        return {
            "host": self.host,
            "port": self.port,
            "dbname": dbname or self.database,
            "user": self.user,
            "password": self.password,
        }


@lru_cache(maxsize=1)
def get_embedding_config() -> EmbeddingConfig:
    """读取嵌入配置（带缓存，进程内只解析一次）。"""
    return EmbeddingConfig(
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        base_url=os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        model=os.environ.get("EMBEDDING_MODEL", "text-embedding-v4"),
        dimension=int(os.environ.get("EMBEDDING_DIMENSION", "1536")),
    )


@lru_cache(maxsize=1)
def get_teacher_config() -> TeacherConfig:
    """读取 DeepSeek 教师模型配置。注意 .env 中可能含空格，统一 strip。"""
    return TeacherConfig(
        api_key=os.environ.get("DEEPSEEK_API_KEY", "").strip(),
        base_url=os.environ.get("DEEPSEEK_BASE_URL",
                                "https://api.deepseek.com").strip(),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip(),
    )


@lru_cache(maxsize=1)
def get_pg_config() -> PgConfig:
    """读取 PostgreSQL 配置。"""
    return PgConfig(
        host=os.environ.get("PG_HOST", "localhost"),
        port=int(os.environ.get("PG_PORT", "5432")),
        database=os.environ.get("PG_DATABASE", "fe_llm"),
        user=os.environ.get("PG_USER", "postgres"),
        password=os.environ.get("PG_PASSWORD", "postgres"),
        table=os.environ.get("PG_TABLE", "concepts"),
    )


def get_device() -> str:
    """返回训练/推理设备。优先用 .env 指定，其次自动探测 CUDA。"""
    want = os.environ.get("DEVICE", "auto")
    if want != "auto":
        return want
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# 权重文件默认目录（训练产出，git 忽略）
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")
