"""独立规则判定器：完全不依赖任何模型。"""
from __future__ import annotations
import numpy as np

N_COLORS = 3


def is_valid(adj: np.ndarray, coloring: np.ndarray) -> bool:
    """整图是否完全合法：颜色取值合法 且 每条边两端颜色不同。"""
    if coloring.min() < 0 or coloring.max() >= N_COLORS:
        return False
    i, j = np.nonzero(np.triu(adj, 1))
    return bool(np.all(coloring[i] != coloring[j]))


def violation_count(adj: np.ndarray, coloring: np.ndarray) -> int:
    """诊断量：冲突边数（仅记录日志，不作为主指标）。"""
    i, j = np.nonzero(np.triu(adj, 1))
    return int(np.sum(coloring[i] == coloring[j]))


def edge_satisfy_rate(adj: np.ndarray, coloring: np.ndarray) -> float:
    """诊断量：平均边满足率（不得代替主指标）。"""
    i, j = np.nonzero(np.triu(adj, 1))
    if len(i) == 0:
        return 1.0
    return float(np.mean(coloring[i] != coloring[j]))
