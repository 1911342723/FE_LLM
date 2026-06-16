# -*- coding: utf-8 -*-
"""
fe_llm/world_model/hierarchical_encoder.py —— 分层预测编码世界模型（v2-M1 核心引擎）
=====================================================================================
对应 docs/FE-LLM从0自建v2架构设计.md 第 4、8 节。完全从 0，不依赖任何预训练底座。

把 v1 的"单层全局意图向量"升级为 2 层 latent：
    z_local : (B, L, intent_dim)  底层，承载词形/局部（"爻"）
    z_global: (B, intent_dim)     顶层，承载抽象意图/全局态势（"卦"）

机制（预测编码）：
    自上而下预测：z_local_hat = topdown(z_global)          # 高层预测低层
    自下而上误差：eps = z_local - z_local_hat
    精度加权自由能：F = 0.5 * precision * mean(eps^2)
    感知即弛豫：沿 -dF/dz_global 迭代更新 z_global 到 F 收敛

关键设计（保证 F 单调下降、可单测、可端到端训练）：
    topdown 为线性层 g(z)=W·z+b，则 dF/dz_global = -precision·mean_l(Wᵀ·eps)。
    弛豫用梯度下降 z_global += alpha·precision·mean_l(Wᵀ·eps)，
    即用 topdown 的权重转置做"自下而上反馈"（PC 的经典对称连接），
    因此无需额外反馈参数，且 F 在合适 alpha 下单调下降。
    展开的弛豫循环对所有参数可导，decode/aux 损失能反传训练整套权重。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from fe_llm.energy_lm.models.intent_model import PERBlock


@dataclass
class HierarchicalState:
    """分层预测编码的输出状态（全部为显式量，供 surprise/trace 直接使用）。"""

    z_global: torch.Tensor          # (B, intent_dim) 顶层抽象意图
    z_local: torch.Tensor           # (B, L, intent_dim) 底层局部 latent
    final_error: torch.Tensor       # (B, L, intent_dim) 弛豫后的残余预测误差
    free_energy: torch.Tensor       # 标量张量，弛豫后的自由能（可作 aux loss）
    free_energy_trace: list[float]  # 每个弛豫步的自由能（应单调下降）


class HierarchicalPredictiveEncoder(nn.Module):
    """2 层预测编码编码器：观象（bottom-up）+ 立卦（top-down 弛豫）。"""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        dim: int = 256,
        n_heads: int = 8,
        intent_dim: int = 128,
        depth: int = 4,
        relax_steps: int = 5,
        alpha: float = 0.3,
        precision: float = 1.0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos, std=0.02)
        # 序列内双向 PER 块：作为每层内部算子（保留 v1 已验证的预测-误差弛豫）。
        self.blocks = nn.ModuleList([PERBlock(dim, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.to_local = nn.Linear(dim, intent_dim)          # 观象：编码出底层 z_local
        self.topdown = nn.Linear(intent_dim, intent_dim)    # 自上而下预测 g：z_global → z_local
        self.intent_dim = intent_dim
        self.relax_steps = relax_steps
        self.alpha = alpha
        self.precision = precision

    def forward(
        self,
        ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        relax_steps: int | None = None,
    ) -> HierarchicalState:
        """ids:(B,L) → HierarchicalState。relax_steps 可覆盖弛豫步数（=0 即不弛豫）。"""

        if ids.ndim != 2:
            raise ValueError("ids must have shape (batch, seq_len)")
        batch, length = ids.shape
        x = self.embed(ids) + self.pos[:, :length]
        for block in self.blocks:
            x = block(x)
        h = self.norm(x)
        z_local = self.to_local(h)                          # (B, L, intent_dim)

        if attention_mask is None:
            attention_mask = torch.ones(batch, length, device=ids.device, dtype=z_local.dtype)
        # 注意：mask 仅作用于池化/误差聚合，不作用于上面的 PERBlock 注意力；
        # padding 级注意力屏蔽留待后续里程碑（v1 IntentEncoder 同样未做）。
        mask = attention_mask.unsqueeze(-1).to(z_local.dtype)   # (B, L, 1)
        token_count = mask.sum(dim=1).clamp_min(1.0)            # (B, 1) 每条有效 token 数
        total_valid = token_count.sum().clamp_min(1.0)          # 标量，整批有效 token 数

        # 初始 z_global = 有效位置的 masked mean（先验：局部均值即全局粗估）。
        z_global = (z_local * mask).sum(dim=1) / token_count    # (B, intent_dim)

        steps = self.relax_steps if relax_steps is None else relax_steps
        weight = self.topdown.weight                            # (intent_dim, intent_dim)
        trace: list[float] = []
        free_energy = z_local.new_zeros(())
        for step in range(steps + 1):
            pred = self.topdown(z_global).unsqueeze(1)          # (B, 1, intent_dim) 广播预测
            eps = (z_local - pred) * mask                       # (B, L, intent_dim) 自下而上误差
            free_energy = 0.5 * self.precision * (eps.pow(2).sum() / total_valid)
            trace.append(float(free_energy.detach()))
            if step == steps:
                break
            # 自下而上反馈 = Wᵀ·eps（topdown 权重转置），即 -dF/dz_global 方向。
            grad_dir = torch.matmul(eps, weight)               # (B, L, intent_dim)
            update = (grad_dir * mask).sum(dim=1) / token_count
            z_global = z_global + self.alpha * self.precision * update

        return HierarchicalState(
            z_global=z_global,
            z_local=z_local,
            final_error=eps,
            free_energy=free_energy,
            free_energy_trace=trace,
        )
