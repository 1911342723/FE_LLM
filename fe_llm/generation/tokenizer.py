# -*- coding: utf-8 -*-
"""
generation/tokenizer.py —— 概念分词器（替代 BPE）
==================================================
对应文档：「字典里映射的不是概率，而是能量梯度」。

每个「语义元(SemanticUnit)」= 一段可输出文字 + 其高维向量 + 语义角色标签。
解码器在这些语义元中寻路，挑选最能拉近"当前状态→目标意图"的那一个。

向量来源：用真实嵌入器编码语义元文本（与世界模型同一语义空间，保证可比）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..embedding.base import Embedder

EOS = "<EOS>"  # 句子结束符：残余能量耗散完毕时输出


@dataclass
class SemanticUnit:
    surface: str              # 输出文字片段
    tag: str = "通用"          # 语义角色（结论/追问/反驳/连接/阻断/问候）
    vector: np.ndarray | None = field(default=None, repr=False)


class ConceptTokenizer:
    """语义元词表：surface → vector 的能量映射。"""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._units: list[SemanticUnit] = []

    def add_unit(self, surface: str, tag: str = "通用") -> None:
        vec = self.embedder.embed_one(surface)
        self._units.append(SemanticUnit(surface=surface, tag=tag, vector=vec))

    def add_many(self, items: list[tuple[str, str]]) -> None:
        """批量添加 (surface, tag)，一次性编码以减少网络往返。"""
        surfaces = [s for s, _ in items]
        vecs = self.embedder.embed(surfaces)
        for (surface, tag), vec in zip(items, vecs):
            self._units.append(SemanticUnit(surface=surface, tag=tag, vector=vec))

    def candidates(self, tags: list[str] | None = None) -> list[SemanticUnit]:
        if tags is None:
            return list(self._units)
        return [u for u in self._units if u.tag in tags]

    def __len__(self) -> int:
        return len(self._units)


def build_default_tokenizer(embedder: Embedder) -> ConceptTokenizer:
    """构建默认语义元库，按行动策略分类。"""
    tk = ConceptTokenizer(embedder)
    tk.add_many([
        # 连接 / 结论 / 解释
        ("根据我的世界模型", "连接"),
        ("综合来看", "连接"),
        ("这个结论是成立的", "结论"),
        ("这是逻辑自洽的", "结论"),
        ("我可以确认这一点", "结论"),
        ("因为它符合已知的公理", "解释"),
        ("从因果关系上分析", "解释"),
        ("这与基础常识一致", "解释"),
        # 反驳
        ("但这违背了基本逻辑", "反驳"),
        ("这一点与公理相矛盾", "反驳"),
        ("我无法接受这个前提", "反驳"),
        ("现实中并非如此", "反驳"),
        # 追问
        ("能否补充更多约束条件", "追问"),
        ("你具体指的是哪个方面", "追问"),
        ("请提供进一步的信息", "追问"),
        # 阻断
        ("您的输入无法解析请重新表述", "阻断"),
        # 问候
        ("你好很高兴和你交流", "问候"),
        ("我在认真听你说", "问候"),
    ])
    return tk
