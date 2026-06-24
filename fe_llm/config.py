# -*- coding: utf-8 -*-
"""
config.py —— 统一配置中心
==========================
训练设备探测与教师模型（生成训练语料用）的配置都集中从 .env 读取，
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
class TeacherConfig:
    """DeepSeek 教师模型配置（用于蒸馏生成训练数据）。"""

    api_key: str
    base_url: str
    model: str


@lru_cache(maxsize=1)
def get_teacher_config() -> TeacherConfig:
    """读取 DeepSeek 教师模型配置。注意 .env 中可能含空格，统一 strip。"""
    return TeacherConfig(
        api_key=os.environ.get("DEEPSEEK_API_KEY", "").strip(),
        base_url=os.environ.get("DEEPSEEK_BASE_URL",
                                "https://api.deepseek.com").strip(),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip(),
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
