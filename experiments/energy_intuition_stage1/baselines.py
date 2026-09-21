"""对照组：直接前向预测 / 零步 / 同预算采样择优。"""
from __future__ import annotations
import torch
import torch.nn as nn

from energy import GraphEncoder, N_COLORS, decode


class DirectPredictor(nn.Module):
    """主比较基线：图 -> 逐节点颜色 logit，一次前向。"""

    def __init__(self, d: int = 64, hidden: int = 128):
        super().__init__()
        self.enc = GraphEncoder(d)
        self.head = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(),
                                  nn.Linear(hidden, N_COLORS))

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(adj))


def zero_step(z0: torch.Tensor) -> torch.Tensor:
    """零步消融：直接解码初始点。"""
    return decode(z0)


def sample_best_of_n(model, adj: torch.Tensor, n: int, dim=(16, 3),
                     generator=None) -> torch.Tensor:
    """同预算采样择优：抽 n 个随机 z，取能量最低者解码。

    预算对齐：n 应设为与 K 步下降相同的能量网络前向次数。
    """
    best_z = None
    best_e = None
    for _ in range(n):
        z = torch.randn(adj.shape[0], *dim, generator=generator)
        with torch.no_grad():
            e = model(adj, z)
        if best_e is None:
            best_z, best_e = z, e
        else:
            better = e < best_e
            best_z = torch.where(better.view(-1, 1, 1), z, best_z)
            best_e = torch.where(better, e, best_e)
    return decode(best_z)
