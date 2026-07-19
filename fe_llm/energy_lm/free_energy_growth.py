# -*- coding: utf-8 -*-
"""由残余自由能驱动的生成性容量生长。

这里的“生长”不是人为指定技能名后挂一个 adapter，而是模型选择过程：

1. 用已解释经验的 residual free-energy 分布校准稳定区间；
2. 新经验在所有现有生成性通路下仍长期高能，只获得临时 probe；
3. 临时通路只有在独立 held-out 上能泛化降低自由能才固化，不可约噪声直接丢弃；
4. 新通路支付由旧稳定流校准的结构复杂度代价，防止通用低能盆地抢走旧路由；
5. 冻结旧通路；推理时选择“残余自由能 + 结构代价”最低的解释。

通路只包含共享信念空间上的生成性转移 ``T(z_prev)``；embedding、观测模型与 readout
由核心共享。它不是 Q/K attention，也不是按关键词路由的 MoE。
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from fe_llm.energy_lm.models.free_energy_lm import FreeEnergyLM


class FreeEnergyGrowthSystem(nn.Module):
    """在共享 FreeEnergyLM 上按需增加生成性转移假设。"""

    def __init__(self, core: FreeEnergyLM) -> None:
        super().__init__()
        self.core = core
        self.grown_pathways = nn.ModuleList()
        self.grown_heads = nn.ModuleDict()
        self.register_buffer("pathway_costs", torch.zeros(1))
        self.growth_threshold: float | None = None

    @property
    def pathway_count(self) -> int:
        return 1 + len(self.grown_pathways)

    def transition_for(self, pathway: int) -> nn.Module:
        if pathway == 0:
            return self.core.transition
        if 1 <= pathway < self.pathway_count:
            return self.grown_pathways[pathway - 1]
        raise IndexError(f"pathway={pathway} 超出 [0,{self.pathway_count - 1}]。")

    def head_for(self, pathway: int) -> nn.Module:
        if pathway == 0:
            return self.core.head
        return self.grown_heads[str(pathway)] if str(pathway) in self.grown_heads else self.core.head

    def forward_pathway(self, ids: torch.Tensor, pathway: int, **kwargs):
        head = self.head_for(pathway)
        head_override = None if head is self.core.head else head
        return self.core(ids, transition_override=self.transition_for(pathway),
                         head_override=head_override, **kwargs)

    def add_pathway(self, source: int = 0, noise_std: float = 1e-3) -> int:
        """从已有假设复制并立即固化新容量。严肃生长应先走 provisional probe。"""
        return self.commit_pathway(self.create_provisional_pathway(source, noise_std))

    def create_provisional_pathway(
        self,
        source: int = 0,
        noise_std: float = 1e-3,
    ) -> nn.Module:
        """创建未注册的临时通路；只有 held-out 自由能可约才应固化。"""
        new_path = copy.deepcopy(self.transition_for(source))
        if noise_std > 0:
            with torch.no_grad():
                for parameter in new_path.parameters():
                    parameter.add_(torch.randn_like(parameter) * noise_std)
        return new_path

    def commit_pathway(
        self,
        provisional: nn.Module,
        complexity_cost: float = 0.0,
        head: nn.Module | None = None,
    ) -> int:
        """把通过可约性检验的临时通路转成持久容量。"""
        if complexity_cost < 0:
            raise ValueError("complexity_cost 不能为负。")
        self.grown_pathways.append(provisional)
        index = self.pathway_count - 1
        if head is not None:
            self.grown_heads[str(index)] = head
        cost = torch.tensor([complexity_cost], device=self.pathway_costs.device,
                            dtype=self.pathway_costs.dtype)
        self.pathway_costs = torch.cat((self.pathway_costs, cost))
        return index

    def train_only_provisional(self, provisional: nn.Module) -> list[nn.Parameter]:
        """冻结现有系统，只开放尚未固化的临时通路。"""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        params = list(provisional.parameters())
        for parameter in params:
            parameter.requires_grad_(True)
        return params

    def create_provisional_head(self, source: int = 0) -> nn.Module:
        return copy.deepcopy(self.head_for(source))

    def train_only_provisional_head(self, head: nn.Module) -> list[nn.Parameter]:
        """冻结稳定化动力学，只学习如何读出已通过可约性检验的新稳定态。"""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        params = list(head.parameters())
        for parameter in params:
            parameter.requires_grad_(True)
        return params

    def train_only_pathway(self, pathway: int) -> list[nn.Parameter]:
        """冻结共享核心与旧通路，只开放目标新通路。"""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        params = list(self.transition_for(pathway).parameters())
        for parameter in params:
            parameter.requires_grad_(True)
        return params

    @torch.no_grad()
    def residual_scores(self, ids: torch.Tensor, pathway: int, start: int = 1) -> torch.Tensor:
        """返回每个样本在指定通路下的归一化稳定后残余自由能。"""
        _, trace = self.forward_pathway(ids, pathway, return_trace=True)
        residual = trace["residual_free_energy_per_dim"]
        start = min(max(0, int(start)), residual.size(1) - 1)
        return residual[:, start:].mean(dim=1)

    @torch.no_grad()
    def provisional_scores(
        self,
        ids: torch.Tensor,
        provisional: nn.Module,
        start: int = 1,
    ) -> torch.Tensor:
        _, trace = self.core(ids, transition_override=provisional, return_trace=True)
        residual = trace["residual_free_energy_per_dim"]
        start = min(max(0, int(start)), residual.size(1) - 1)
        return residual[:, start:].mean(dim=1)

    @torch.no_grad()
    def calibrate_complexity_cost(
        self,
        stable_ids: torch.Tensor,
        provisional: nn.Module,
        *,
        start: int = 1,
        quantile: float = 0.99,
        margin: float = 1e-4,
    ) -> float:
        """用旧稳定流校准新增结构的 MDL 代价，防止通用低能盆地抢走旧路由。"""
        if not 0.5 < quantile < 1.0:
            raise ValueError("quantile 应在 (0.5,1.0) 内。")
        existing_best = self.score_all(stable_ids, start=start).min(dim=1).values
        provisional_raw = self.provisional_scores(stable_ids, provisional, start=start)
        false_advantage = existing_best - provisional_raw
        cost = torch.quantile(false_advantage, quantile).clamp_min(0) + max(0.0, margin)
        return float(cost.cpu())

    @torch.no_grad()
    def score_all(self, ids: torch.Tensor, start: int = 1) -> torch.Tensor:
        """返回 ``(B,K)``：每个样本在所有生成性通路下的残余自由能。"""
        raw = torch.stack(
            [self.residual_scores(ids, pathway, start=start)
             for pathway in range(self.pathway_count)],
            dim=1,
        )
        return raw + self.pathway_costs[:self.pathway_count].to(raw).unsqueeze(0)

    @torch.no_grad()
    def calibrate_threshold(
        self,
        stable_ids: torch.Tensor,
        *,
        pathway: int = 0,
        start: int = 1,
        quantile: float = 0.99,
    ) -> float:
        if not 0.5 < quantile < 1.0:
            raise ValueError("quantile 应在 (0.5,1.0) 内。")
        scores = self.residual_scores(stable_ids, pathway, start=start)
        self.growth_threshold = float(torch.quantile(scores, quantile).cpu())
        return self.growth_threshold

    @torch.no_grad()
    def growth_pressure(
        self,
        ids: torch.Tensor,
        *,
        start: int = 1,
        threshold: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回每样本是否所有现有通路都解释失败，以及其最低残余自由能。"""
        limit = self.growth_threshold if threshold is None else float(threshold)
        if limit is None:
            raise RuntimeError("尚未校准 growth_threshold。")
        best = self.score_all(ids, start=start).min(dim=1).values
        return best > limit, best

    @torch.no_grad()
    def should_grow(
        self,
        ids: torch.Tensor,
        *,
        start: int = 1,
        threshold: float | None = None,
        min_fraction: float = 0.5,
    ) -> tuple[bool, float, float]:
        if not 0.0 < min_fraction <= 1.0:
            raise ValueError("min_fraction 必须在 (0,1] 内。")
        pressure, best = self.growth_pressure(ids, start=start, threshold=threshold)
        fraction = float(pressure.float().mean().cpu())
        return fraction >= min_fraction, fraction, float(best.mean().cpu())

    @torch.no_grad()
    def route(self, ids: torch.Tensor, start: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score_all(ids, start=start)
        return scores.argmin(dim=1), scores

    @torch.no_grad()
    def routed_logits(self, ids: torch.Tensor, start: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        choices, _ = self.route(ids, start=start)
        candidates = torch.stack(
            [self.forward_pathway(ids, pathway)
             for pathway in range(self.pathway_count)],
            dim=1,
        )
        batch_index = torch.arange(ids.size(0), device=ids.device)
        return candidates[batch_index, choices], choices

    def added_parameter_count(self) -> int:
        transitions = sum(parameter.numel() for pathway in self.grown_pathways
                          for parameter in pathway.parameters())
        heads = sum(parameter.numel() for head in self.grown_heads.values()
                    for parameter in head.parameters())
        return transitions + heads
