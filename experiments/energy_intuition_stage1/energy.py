"""能量网络 E(h, z) 与图编码器。

z 形状 (B, 16, 3)，视作每节点三色 logit；解码 = 逐节点 argmax。
这样梯度下降在连续 relaxation 上进行，而轨迹上任意一点都可解码并由
独立判定器判定 —— 满足全轨迹监控要求。
"""
from __future__ import annotations
import torch
import torch.nn as nn

N_NODES = 16
N_COLORS = 3


class GraphEncoder(nn.Module):
    """简单消息传递编码器，输出每节点表示 h (B, N, d)。"""

    def __init__(self, d: int = 64, layers: int = 3):
        super().__init__()
        self.inp = nn.Linear(N_NODES, d)  # 用邻接行作为初始特征
        self.mp = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(layers)])
        self.act = nn.SiLU()

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        h = self.act(self.inp(adj))
        deg = adj.sum(-1, keepdim=True).clamp(min=1.0)
        for layer in self.mp:
            agg = torch.bmm(adj, h) / deg
            h = self.act(layer(torch.cat([h, agg], dim=-1)))
        return h


class EnergyNet(nn.Module):
    """E(h, z) -> 标量。非凸通用网络，故推理端必须配步长控制与下降检查。"""

    def __init__(self, d: int = 64, hidden: int = 128, spectral: bool = True):
        super().__init__()
        self.enc = GraphEncoder(d)
        lin = nn.Linear

        def mk(a, b):
            m = lin(a, b)
            return nn.utils.parametrizations.spectral_norm(m) if spectral else m

        self.node_mlp = nn.Sequential(
            mk(d + N_COLORS, hidden), nn.SiLU(), mk(hidden, hidden), nn.SiLU()
        )
        self.pair_mlp = nn.Sequential(
            mk(2 * hidden, hidden), nn.SiLU(), mk(hidden, 1)
        )
        self.readout = nn.Sequential(mk(hidden, hidden), nn.SiLU(), mk(hidden, 1))

    def forward(self, adj: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.enc(adj)                                   # (B,N,d)
        p = torch.softmax(z, dim=-1)                        # 连续松弛的颜色分布
        u = self.node_mlp(torch.cat([h, p], dim=-1))        # (B,N,hidden)
        B, N, H = u.shape
        ui = u.unsqueeze(2).expand(B, N, N, H)
        uj = u.unsqueeze(1).expand(B, N, N, H)
        pair = self.pair_mlp(torch.cat([ui, uj], dim=-1)).squeeze(-1)  # (B,N,N)
        edge_term = (pair * adj).sum((1, 2)) / 2.0
        node_term = self.readout(u).squeeze(-1).sum(-1)
        return edge_term + node_term


def decode(z: torch.Tensor) -> torch.Tensor:
    """确定性投影：逐节点 argmax。"""
    return z.argmax(dim=-1)
