# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw.py —— 内容寻址预测编码工作空间（CAPCW 核心引擎）
========================================================================
见 `docs/FE-LLM核心引擎构想.md`。这是 CAPCW 阶段二：把"最小 slot 工作空间"（slot-attention
形态，阶段一已验证内容寻址 > 单向量）elaborate 成**显式预测编码 / 自由能**形态——让 slot 不再
是黑盒 attention，而是"解释输入的混合成分"，路由由重建误差导出（attention 即推理），弛豫沿
自由能下降，且全过程可溯源。

数学形式（生成模型：slot 工作空间生成/解释输入）：
    每个 slot s_m 经线性生成 g 预测一个输入：recon_m = g(s_m)
    内容路由（责任/attention，由重建误差导出）：r_{m,p} = softmax_m( -precision·||x_p - recon_m||² )
    混合重建：x_hat_p = Σ_m r_{m,p}·recon_m
    预测误差：eps_p = x_p - x_hat_p
    自由能：F = 0.5·precision·mean_p ||eps_p||²
    感知即弛豫：slot 沿 -dF/ds_m 下降（g 线性 → 反馈 = g.weightᵀ·eps，PC 经典对称连接）
        s_m ← s_m + α·precision·Σ_p r_{m,p}·(eps_p·Wg)

可溯源：responsibilities（谁解释了谁）、final_error、free_energy_trace 全部显式返回。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class WorkspaceState:
    """CAPCW 工作空间的显式输出（供读出 / surprise / trace 直接使用）。"""

    slots: torch.Tensor              # (B, M, d) 内容寻址 slot 工作空间（世界状态）
    free_energy: torch.Tensor        # 标量，弛豫后的自由能
    free_energy_trace: list[float]   # 每步自由能（应整体下降，自由能平复）
    responsibilities: torch.Tensor   # (B, M, P) 内容路由（哪个 slot 解释哪个输入；可溯源）
    final_error: torch.Tensor        # (B, P, d) 弛豫后的残余预测误差


class PCWorkspace(nn.Module):
    """内容寻址预测编码工作空间：slot 作为解释输入的混合成分，路由由重建误差导出，弛豫降自由能。"""

    def __init__(self, dim: int, n_slots: int, iters: int = 3, alpha: float = 0.5, precision: float = 1.0):
        super().__init__()
        if n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        self.dim = dim
        self.n_slots = n_slots
        self.iters = iters
        self.alpha = alpha
        self.precision = precision
        self.slots_mu = nn.Parameter(torch.randn(1, n_slots, dim) * 0.02)
        self.g = nn.Linear(dim, dim, bias=False)   # 生成模型 g：slot → 输入空间预测
        self.norm_in = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, n_slots: int | None = None, iters: int | None = None) -> WorkspaceState:
        """x:(B,P,d) → WorkspaceState。n_slots 可覆盖（穷则变生长用），iters 可覆盖。"""

        if x.ndim != 3:
            raise ValueError("x must have shape (batch, n_inputs, dim)")
        b, p, d = x.shape
        m = self.n_slots if n_slots is None else n_slots
        steps = self.iters if iters is None else iters
        x = self.norm_in(x)
        if m == self.n_slots:
            slots = self.slots_mu.expand(b, -1, -1).contiguous()
        else:
            # 生长/裁剪：复制前 m 个先验，多出的用首个先验加噪初始化。
            base = self.slots_mu.expand(b, -1, -1)
            if m <= self.n_slots:
                slots = base[:, :m].contiguous()
            else:
                extra = base[:, :1].expand(b, m - self.n_slots, d) + 0.02 * torch.randn(b, m - self.n_slots, d, device=x.device)
                slots = torch.cat([base, extra], dim=1).contiguous()

        weight = self.g.weight                                   # (d, d)
        trace: list[float] = []
        free_energy = x.new_zeros(())
        resp = x.new_zeros(b, m, p)
        eps = x.new_zeros(b, p, d)
        for step in range(steps + 1):
            recon = self.g(slots)                               # (B, M, d) 每 slot 的预测
            # 内容路由 r_{m,p}：输入 p 由哪个 slot 解释（按重建距离，softmax over slots）。
            dist = ((x.unsqueeze(1) - recon.unsqueeze(2)) ** 2).sum(dim=-1)   # (B, M, P)
            resp = (-self.precision * dist).softmax(dim=1)       # (B, M, P)
            x_hat = torch.einsum("bmp,bmd->bpd", resp, recon)   # (B, P, d) 混合重建
            eps = x - x_hat
            free_energy = 0.5 * self.precision * (eps.pow(2).sum() / (b * p))
            trace.append(float(free_energy.detach()))
            if step == steps:
                break
            # M 步：slot 沿 -dF/ds 下降（g 线性，反馈=g.weightᵀ·eps）。
            fb = torch.matmul(eps, weight)                      # (B, P, d) = eps·Wg = (Wgᵀ·eps)ᵀ
            upd = torch.einsum("bmp,bpd->bmd", resp, fb)        # (B, M, d)
            slots = slots + self.alpha * self.precision * upd

        return WorkspaceState(
            slots=slots,
            free_energy=free_energy,
            free_energy_trace=trace,
            responsibilities=resp,
            final_error=eps,
        )

    @torch.no_grad()
    def grow_if_unexplained(self, x: torch.Tensor, threshold: float, max_slots: int) -> int:
        """穷则变：若当前 slot 数下弛豫后自由能仍 > threshold，则建议增大 slot 数（返回建议 n_slots）。

        这是结构成长的判定钩子（不改参数，只给出建议 slot 数，供训练/推理外层决定）。
        """
        m = self.n_slots
        while m < max_slots:
            state = self.forward(x, n_slots=m)
            if float(state.free_energy) <= threshold:
                break
            m += 1
        return m


class SequenceAdjacency(nn.Module):
    """序列相邻算子（induction head 的 previous-token channel）。

    背景（见 `docs/FE-LLM核心引擎构想.md` 第 16/17 节）：集合式 slot 工作空间擅长"内容绑定"
    （pair 作为整体喂入），但**不自带"序列相邻"算子**——它把 token 独立嵌入后聚合，bigram a→b 的
    相邻信息在聚合时丢失，故无法做 induction（...A B ... A → 预测 B，in-context learning 基石）。
    Transformer 靠 attention+位置实现 induction；本模块给 CAPCW 显式补上这一缺失的横向序列算子。

    形式：把"独立 token 表示流"变成"(prev→cur) bigram 表示流"——
        out_t = proj([ x_{t-1} ; x_t ]),   x_{-1} = 可学 BOS 占位（位置 0 无前驱）
    放在 token → 工作空间写入**之前**，使每个位置在被聚成 slot 前已携带"前驱身份"（key，供 cue 匹配）
    与"当前身份"（value，供读出）。

    验证：`capcw_induction_seq_eval.py` 的 2×2 单变量析因（相邻算子 on/off × flat/CAPCW）证明
    induction 需**同时**具备"序列相邻算子"与"内容寻址 slot"——单向量池化即使加相邻算子也救不活
    （无法联想检索）、slot 无相邻算子也救不活；唯有 CAPCW+本算子可解（capcw_adj≈0.63，其余三格≈随机
    0.05；rescue≈+0.47、内容寻址优势≈+0.50）。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # 位置 0 的"前驱"占位（无前驱）。
        self.bos = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.bos, std=0.02)
        self.proj = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x:(B,L,d) → (B,L,d) 的 (prev→cur) bigram 表示（形状不变）。"""
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, seq_len, dim)")
        b = x.shape[0]
        # 右移一位得到"前驱"：prev[t] = x[t-1]，prev[0] = bos。
        prev = torch.cat([self.bos.expand(b, -1, -1), x[:, :-1]], dim=1)
        return self.proj(torch.cat([prev, x], dim=-1))
