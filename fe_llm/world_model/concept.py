# -*- coding: utf-8 -*-
"""
world_model/concept.py —— 概念/吸引子数据结构
==============================================
一个 Concept 即能量地貌中的一个「吸引子(深谷)」。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Concept:
    """
    属性：
        name      : 概念名称（人类可读唯一标识）
        text      : 自然语言描述（用于生成向量 & 输出复述）
        category  : 分类（公理/常识/经验/...），便于按层管理深度与抗性
        depth     : 吸引子深度/稳固度。越深越是不可撼动的公理，越难被用户设定修改。
        relations : 因果/逻辑关系，如 {"互斥": ["地平说"], "蕴含": [...]}，
                    供「中层惊奇(因果冲突)」检测使用。
        vector    : 该概念在高维空间的位置（谷底坐标）。由嵌入层填充。
    """

    name: str
    text: str
    category: str = "常识"
    depth: float = 1.0
    relations: dict[str, list[str]] = field(default_factory=dict)
    vector: np.ndarray | None = field(default=None, repr=False)

    def to_metadata(self) -> dict:
        """序列化为可存入数据库 JSON 列的元数据（不含向量本身）。"""
        return {
            "category": self.category,
            "depth": self.depth,
            "relations": self.relations,
        }

    @classmethod
    def from_row(cls, name: str, text: str, metadata: dict,
                 vector: np.ndarray) -> "Concept":
        """从数据库行重建 Concept。"""
        return cls(
            name=name,
            text=text,
            category=metadata.get("category", "常识"),
            depth=float(metadata.get("depth", 1.0)),
            relations=metadata.get("relations", {}),
            vector=vector,
        )
