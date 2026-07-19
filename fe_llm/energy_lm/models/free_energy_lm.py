# -*- coding: utf-8 -*-
"""显式自由能递归语言模型。

模型把语言推理实现为一串稳定化过程，而不是若干 attention 层：上一位置已经稳定的
信念状态 ``z[j-1]`` 通过共享转移模型产生先验 ``mu[j]``；当前 token 形成观测
``a[j]``；新的内部状态 ``z[j]`` 从观测扰动态出发，反复沿同一个显式自由能的负梯度
弛豫，稳定后再预测下一 token。

逐位置自由能为::

    F_j(z_j) = po/2 ||z_j-a_j||²
             + pp/2 ||z_j-T(z_{j-1})||²
             + pc/2 ||z_j||²

``T`` 可以是非线性生成性转移，但在位置 ``j`` 的内循环中先验保持固定，因此
``F_j`` 关于待推断状态 ``z_j`` 仍是凸二次函数。弛豫步长被约束在曲率倒数以内，
每一步都保证不增加自由能。

关键边界：

* 没有 Q/K、softmax attention 或逐层不同参数；
* 同一个转移模型和同一个弛豫定律用于所有位置、所有推理步；
* 每个位置只读取上一稳定态和当前观测，严格因果；
* 逐位置独立停止，未来 token 不能通过统一停止时刻泄漏到过去；
* 物理系数第一版固定，任务优化器不能通过把预测精度/步长压到 0 绕过动力学。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class FreeEnergyLM(nn.Module):
    """以显式自由能稳定化为计算本体的严格因果字符语言模型。"""

    returns_energy = False

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        dim: int = 256,
        relaxation_steps: int = 8,
        tolerance: float = 1e-4,
        observation_precision: float = 1.0,
        prediction_precision: float = 1.0,
        complexity_precision: float = 0.01,
        relaxation_fraction: float = 0.8,
        transition_mult: int = 2,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or max_len <= 0 or dim <= 0:
            raise ValueError("vocab_size、max_len 与 dim 必须为正数。")
        if relaxation_steps <= 0:
            raise ValueError("relaxation_steps 必须为正数。")
        if min(observation_precision, prediction_precision, complexity_precision) <= 0:
            raise ValueError("三项自由能精度必须为正数。")
        if not 0.0 < relaxation_fraction < 1.0:
            raise ValueError("relaxation_fraction 必须在 (0,1) 内。")
        if transition_mult <= 0:
            raise ValueError("transition_mult 必须为正数。")

        self.vocab_size = int(vocab_size)
        self.max_len = int(max_len)
        self.dim = int(dim)
        self.relaxation_steps = int(relaxation_steps)
        self.tolerance = float(tolerance)
        self.transition_mult = int(transition_mult)

        # 兼容现有训练/报告接口。depth 在这里表示最大弛豫步数，不是网络深度。
        self.depth = self.relaxation_steps
        self.n_heads = 0
        self.use_synapse = True

        self.embed = nn.Embedding(vocab_size, dim)
        self.observation_norm = nn.LayerNorm(dim)
        self.root_state = nn.Parameter(torch.zeros(1, dim))

        hidden = dim * transition_mult
        self.transition_norm = nn.LayerNorm(dim)
        self.transition = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )
        # 初始转移保持温和，避免未经训练的递归状态爆炸。
        nn.init.xavier_uniform_(self.transition[0].weight, gain=0.7)
        nn.init.xavier_uniform_(self.transition[2].weight, gain=0.7)
        nn.init.zeros_(self.transition[0].bias)
        nn.init.zeros_(self.transition[2].bias)

        # 固定物理系数：第一阶段先保证动力学不可被任务损失“学没”。
        self.register_buffer("observation_precision", torch.tensor(float(observation_precision)))
        self.register_buffer("prediction_precision", torch.tensor(float(prediction_precision)))
        self.register_buffer("complexity_precision", torch.tensor(float(complexity_precision)))
        self.register_buffer("relaxation_fraction", torch.tensor(float(relaxation_fraction)))

        self.readout_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.last_trace: dict[str, Any] | None = None
        self.last_free_energy_loss: torch.Tensor | None = None
        self.last_position_free_energy: torch.Tensor | None = None
        self.last_prediction_surprise: torch.Tensor | None = None

    def _precisions(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.observation_precision, self.prediction_precision, self.complexity_precision

    @staticmethod
    def _energy_components(
        z: torch.Tensor,
        observation: torch.Tensor,
        prior: torch.Tensor,
        precisions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回逐样本的总自由能与三个组成项，形状均为 ``(B,)``。"""
        po, pp, pc = precisions
        observation_error = 0.5 * po * (z - observation).square().sum(dim=-1)
        prediction_error = 0.5 * pp * (z - prior).square().sum(dim=-1)
        complexity = 0.5 * pc * z.square().sum(dim=-1)
        total = observation_error + prediction_error + complexity
        return total, observation_error, prediction_error, complexity

    def _relax_position(
        self,
        observation: torch.Tensor,
        prior: torch.Tensor,
        *,
        max_steps: int,
        adaptive: bool,
        return_trace: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """从当前观测扰动态出发，向先验与观测共同决定的稳定态弛豫。"""
        precisions = self._precisions()
        po, pp, pc = precisions
        curvature = po + pp + pc
        step_size = 0.99 * self.relaxation_fraction / curvature

        z = observation
        active = torch.ones(observation.size(0), dtype=torch.bool, device=observation.device)
        steps = torch.zeros(observation.size(0), dtype=torch.long, device=observation.device)
        total, obs, pred, comp = self._energy_components(z, observation, prior, precisions)

        initial_total = total
        initial_prediction = pred
        energy_trace = [total] if return_trace else []
        observation_trace = [obs] if return_trace else []
        prediction_trace = [pred] if return_trace else []
        complexity_trace = [comp] if return_trace else []

        for _ in range(max_steps):
            grad = po * (z - observation) + pp * (z - prior) + pc * z
            candidate = z - step_size * grad
            candidate = torch.where(active.unsqueeze(-1), candidate, z)
            next_total, next_obs, next_pred, next_comp = self._energy_components(
                candidate, observation, prior, precisions)

            decrease = total - next_total
            safe = decrease >= -1e-6
            candidate = torch.where(safe.unsqueeze(-1), candidate, z)
            next_total = torch.where(safe, next_total, total)
            next_obs = torch.where(safe, next_obs, obs)
            next_pred = torch.where(safe, next_pred, pred)
            next_comp = torch.where(safe, next_comp, comp)

            steps = steps + active.to(torch.long)
            relative_decrease = decrease.clamp_min(0) / total.abs().clamp_min(1e-8)
            if adaptive:
                active = active & safe & (relative_decrease.detach() > self.tolerance)

            z = candidate
            total, obs, pred, comp = next_total, next_obs, next_pred, next_comp
            if return_trace:
                energy_trace.append(total)
                observation_trace.append(obs)
                prediction_trace.append(pred)
                complexity_trace.append(comp)
            if adaptive and not bool(active.any()):
                break

        trace = None
        if return_trace:
            # 为跨位置聚合补齐到统一的物理时间轴；收敛后能量保持不变。
            while len(energy_trace) < max_steps + 1:
                energy_trace.append(energy_trace[-1])
                observation_trace.append(observation_trace[-1])
                prediction_trace.append(prediction_trace[-1])
                complexity_trace.append(complexity_trace[-1])
            trace = {
                "free_energy": torch.stack(energy_trace),       # (T+1,B)
                "observation": torch.stack(observation_trace),
                "prediction_error": torch.stack(prediction_trace),
                "complexity": torch.stack(complexity_trace),
                "initial_position_free_energy": initial_total,
                "final_position_free_energy": total,
                "initial_prediction_error": initial_prediction,
                "final_prediction_error": pred,
                "steps": steps,
                "step_size": step_size.expand_as(total),
            }
        return z, trace

    def relax_sequence(
        self,
        observations: torch.Tensor,
        *,
        adaptive: bool = True,
        max_relax_steps: int | None = None,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        """按因果顺序推断每个前缀的稳定信念状态。"""
        budget = self.relaxation_steps if max_relax_steps is None else int(max_relax_steps)
        if budget < 0 or budget > self.relaxation_steps:
            raise ValueError(f"max_relax_steps 必须在 [0,{self.relaxation_steps}] 内。")

        batch, length, _ = observations.shape
        previous = self.root_state.expand(batch, -1)
        states: list[torch.Tensor] = []
        final_energies: list[torch.Tensor] = []
        initial_surprises: list[torch.Tensor] = []
        position_traces: list[dict[str, torch.Tensor]] = []

        for position in range(length):
            observation = observations[:, position]
            prior = self.transition(self.transition_norm(previous))
            _, _, initial_prediction_error, _ = self._energy_components(
                observation, observation, prior, self._precisions())
            state, local_trace = self._relax_position(
                observation,
                prior,
                max_steps=budget,
                adaptive=adaptive,
                return_trace=return_trace,
            )
            states.append(state)
            final_total, _, _, _ = self._energy_components(
                state, observation, prior, self._precisions())
            final_energies.append(final_total)
            initial_surprises.append(initial_prediction_error)
            previous = state
            if local_trace is not None:
                position_traces.append(local_trace)

        stacked_states = torch.stack(states, dim=1)
        final_energy_matrix = torch.stack(final_energies, dim=1)
        initial_surprise_matrix = torch.stack(initial_surprises, dim=1)
        self.last_position_free_energy = final_energy_matrix / self.dim
        self.last_prediction_surprise = initial_surprise_matrix / self.dim
        # 根位置主要学习初始先验；语言结构的生成性目标从第二个位置开始。
        structural_energy = final_energy_matrix[:, 1:] if length > 1 else final_energy_matrix
        self.last_free_energy_loss = structural_energy.mean() / self.dim
        trace: dict[str, Any] | None = None
        if return_trace:
            # 每项由 L 个 (T+1,B) 组成。总能量沿物理时间聚合后仍应单调不增。
            def aggregate(name: str) -> torch.Tensor:
                cube = torch.stack([item[name] for item in position_traces], dim=1)
                return cube.sum(dim=1).mean(dim=-1).detach()

            def positions(name: str) -> torch.Tensor:
                return torch.stack([item[name] for item in position_traces], dim=1).detach()

            steps_per_position = positions("steps")
            trace = {
                "free_energy": aggregate("free_energy"),
                "observation": aggregate("observation"),
                "prediction_error": aggregate("prediction_error"),
                "complexity": aggregate("complexity"),
                "initial_position_free_energy": positions("initial_position_free_energy"),
                "final_position_free_energy": positions("final_position_free_energy"),
                "initial_prediction_error": positions("initial_prediction_error"),
                "final_prediction_error": positions("final_prediction_error"),
                "surprise_per_dim": positions("initial_prediction_error") / self.dim,
                "residual_free_energy_per_dim": positions("final_position_free_energy") / self.dim,
                "step_size": position_traces[0]["step_size"][0].detach(),
                "steps_per_position": steps_per_position,
                "converged_fraction": (
                    (steps_per_position < budget).float().mean().detach()
                    if budget > 0 else torch.tensor(0.0, device=observations.device)
                ),
                "max_relax_steps": budget,
            }
        return stacked_states, trace

    def forward(
        self,
        ids: torch.Tensor,
        *,
        return_trace: bool = False,
        adaptive: bool = True,
        max_relax_steps: int | None = None,
    ):
        if ids.ndim != 2:
            raise ValueError(f"ids 应为 (B,L)，收到 {tuple(ids.shape)}。")
        if ids.size(1) <= 0 or ids.size(1) > self.max_len:
            raise ValueError(f"序列长度必须在 [1,{self.max_len}] 内，收到 {ids.size(1)}。")

        observations = self.observation_norm(self.embed(ids)).float()
        states, trace = self.relax_sequence(
            observations,
            adaptive=adaptive,
            max_relax_steps=max_relax_steps,
            return_trace=return_trace,
        )
        logits = self.head(self.readout_norm(states))
        self.last_trace = trace
        if return_trace:
            return logits, trace
        return logits

    def checkpoint(self, step: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "arch": "free_energy",
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "dim": self.dim,
            "relaxation_steps": self.relaxation_steps,
            "tolerance": self.tolerance,
            "observation_precision": float(self.observation_precision),
            "prediction_precision": float(self.prediction_precision),
            "complexity_precision": float(self.complexity_precision),
            "relaxation_fraction": float(self.relaxation_fraction),
            "transition_mult": self.transition_mult,
            "state_dict": self.state_dict(),
        }
        if step is not None:
            payload["step"] = int(step)
        return payload

    def save(self, path: str, step: int | None = None) -> None:
        torch.save(self.checkpoint(step), path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "FreeEnergyLM":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        net = cls(
            vocab_size=ck["vocab_size"],
            max_len=ck["max_len"],
            dim=ck["dim"],
            relaxation_steps=ck.get("relaxation_steps", ck.get("depth", 8)),
            tolerance=ck.get("tolerance", 1e-4),
            observation_precision=ck.get("observation_precision", 1.0),
            prediction_precision=ck.get("prediction_precision", 1.0),
            complexity_precision=ck.get("complexity_precision", 0.01),
            relaxation_fraction=ck.get("relaxation_fraction", 0.8),
            transition_mult=ck.get("transition_mult", 2),
        )
        net.load_state_dict(ck["state_dict"])
        net.eval()
        return net
