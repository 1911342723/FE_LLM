# -*- coding: utf-8 -*-
"""低秩生成性动力学修正。

新增结构不复制完整转移网络，而是在冻结旧动力学上学习低维残差：

    T_new(z) = T_base(z) + alpha/r * U tanh(V z)

``V`` 把稳定信念投到少量新的自由度，``U`` 把修正写回原状态空间。基底通过弱引用
复用且不注册为该模块参数，因此优化器和参数统计只看到新增动力学容量。
"""

from __future__ import annotations

import weakref

import torch
import torch.nn as nn


class LowRankGenerativeTransition(nn.Module):
    """在冻结生成性转移上叠加低秩、非线性状态修正。"""

    def __init__(
        self,
        base_transition: nn.Module,
        dim: int,
        rank: int = 8,
        alpha: float | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0 or rank <= 0:
            raise ValueError("dim 与 rank 必须为正数。")
        self.dim = int(dim)
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self._base_ref = weakref.ref(base_transition)

        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / max(1, dim) ** 0.5)
        # 从旧稳定动力学精确出发；训练最初只打开写回方向。
        nn.init.zeros_(self.up.weight)
        self.register_buffer("scale", torch.tensor(self.alpha / rank))

    @property
    def base_transition(self) -> nn.Module:
        base = self._base_ref()
        if base is None:
            raise RuntimeError("基底生成性转移已被释放。")
        return base

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        correction = self.up(torch.tanh(self.down(state))) * self.scale
        return self.base_transition(state) + correction

    def added_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class LowRankReadout(nn.Module):
    """在冻结公共读出上学习结构专属的低秩解释映射。"""

    def __init__(
        self,
        base_readout: nn.Module,
        in_dim: int,
        out_dim: int,
        rank: int = 8,
        alpha: float | None = None,
    ) -> None:
        super().__init__()
        if min(in_dim, out_dim, rank) <= 0:
            raise ValueError("in_dim、out_dim 与 rank 必须为正数。")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self._base_ref = weakref.ref(base_readout)
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / max(1, in_dim) ** 0.5)
        nn.init.zeros_(self.up.weight)
        self.register_buffer("scale", torch.tensor(self.alpha / rank))

    @property
    def base_readout(self) -> nn.Module:
        base = self._base_ref()
        if base is None:
            raise RuntimeError("基础读出已被释放。")
        return base

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        correction = self.up(torch.tanh(self.down(state))) * self.scale
        return self.base_readout(state) + correction

    def added_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
