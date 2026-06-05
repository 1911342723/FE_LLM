# -*- coding: utf-8 -*-
"""
arithmetic/encoding.py —— 题目与答案空间编码
=============================================
把算术题和答案映射到向量空间，以契合"能量地貌 + 吸引子"的理论。

题目编码（连续特征，作为"感知信号"）：
    [a/scale, b/scale, op==+ , op==- , op==* ]  共 5 维
    （归一化是为了让网络训练稳定，不同量级的数落在相近范围）

答案空间（离散吸引子）：
    在 [min_answer, max_answer] 区间内的每个整数都是一个"答案吸引子"。
    用一张可学习的嵌入表(码本)表示它们在能量地貌中的坐标。
    - 生成 = 在所有吸引子里找能量最低的那个（滚落）。
    - 惊奇 = 错误答案吸引子对应的高能量。
"""

from __future__ import annotations

import numpy as np

# 与 teacher.generate_problems 的默认范围一致（加减、操作数 0~50）
OP_LIST = ["+", "-", "*"]
QUESTION_DIM = 2 + len(OP_LIST)  # a, b, 三个算子 one-hot
DEFAULT_SCALE = 50.0


def encode_question(a: int, b: int, op: str, scale: float = DEFAULT_SCALE) -> np.ndarray:
    """题目 → 5 维连续特征向量。"""
    vec = np.zeros(QUESTION_DIM, dtype=np.float32)
    vec[0] = a / scale
    vec[1] = b / scale
    vec[2 + OP_LIST.index(op)] = 1.0
    return vec


def answer_range(ops: tuple[str, ...], max_val: int = 50) -> tuple[int, int]:
    """根据算子集合推出答案的整数区间，决定吸引子数量。"""
    lo, hi = 0, 0
    if "+" in ops:
        hi = max(hi, max_val + max_val)
    if "-" in ops:
        lo = min(lo, -max_val)
        hi = max(hi, max_val)
    if "*" in ops:
        hi = max(hi, max_val * max_val)
    return lo, hi


class AnswerSpace:
    """离散答案空间：把整数答案与码本索引互相映射。"""

    def __init__(self, lo: int, hi: int):
        self.lo = lo
        self.hi = hi
        self.values = list(range(lo, hi + 1))      # 所有候选答案（吸引子）
        self.size = len(self.values)
        self._val2idx = {v: i for i, v in enumerate(self.values)}

    def to_index(self, value: int) -> int:
        """答案整数 → 码本索引。"""
        return self._val2idx[value]

    def to_value(self, index: int) -> int:
        """码本索引 → 答案整数。"""
        return self.values[index]

    def contains(self, value: int) -> bool:
        return value in self._val2idx
