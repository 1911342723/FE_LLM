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
import hashlib
import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from fe_llm.energy_lm.models.free_energy_lm import FreeEnergyLM

ARCHIVE_FORMAT_VERSION = 1


def _update_digest_with_tensor(digest, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())


def _state_digest(metadata: dict, state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name in sorted(state):
        _update_digest_with_tensor(digest, name, state[name])
    return digest.hexdigest()


def _state_dict_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


@dataclass
class ArchivedPathway:
    """从活动路由卸下的冻结盆地，可保存为受限、可校验的低秩归档。"""

    transition: nn.Module
    head: nn.Module | None
    complexity_cost: float
    original_index: int
    offload_device: str


class StructuralFreeEnergyStabilizer(nn.Module):
    """让跨窗口结构不稳定性本身也经历显式自由能弛豫。

    窗口 ``t`` 的高 residual-F 样本比例为 ``p_t``，慢状态为 ``s``：

        G_t(s) = po/2 (s-p_t)^2 + pp/2 (s-s_{t-1})^2 + pc/2 s^2

    ``pp`` 是结构惯性，阻止一次污染 burst 立刻改变模型结构；持续外部失稳则不断做功，
    最终让稳定后的 ``s`` 越过生长势垒。环境改变会让不同窗口的初始自由能升高，但每个
    窗口内部的弛豫仍严格单调不增。
    """

    def __init__(
        self,
        *,
        observation_precision: float = 1.0,
        persistence_precision: float = 4.0,
        complexity_precision: float = 0.01,
        relaxation_steps: int = 8,
        relaxation_fraction: float = 0.8,
        tolerance: float = 1e-6,
        activation_barrier: float = 0.25,
        reset_barrier: float = 0.10,
    ) -> None:
        super().__init__()
        if min(observation_precision, persistence_precision, complexity_precision) <= 0:
            raise ValueError("结构自由能的三个精度必须为正。")
        if relaxation_steps <= 0:
            raise ValueError("relaxation_steps 必须为正。")
        if not 0.0 < relaxation_fraction < 1.0:
            raise ValueError("relaxation_fraction 必须在 (0,1) 内。")
        if tolerance < 0:
            raise ValueError("tolerance 不能为负。")
        if not 0.0 <= reset_barrier < activation_barrier <= 1.0:
            raise ValueError("势垒必须满足 0 <= reset < activation <= 1。")

        self.relaxation_steps = int(relaxation_steps)
        self.tolerance = float(tolerance)
        self.activation_barrier = float(activation_barrier)
        self.reset_barrier = float(reset_barrier)
        self.register_buffer("observation_precision", torch.tensor(float(observation_precision)))
        self.register_buffer("persistence_precision", torch.tensor(float(persistence_precision)))
        self.register_buffer("complexity_precision", torch.tensor(float(complexity_precision)))
        self.register_buffer("relaxation_fraction", torch.tensor(float(relaxation_fraction)))
        self.register_buffer("state", torch.tensor(0.0))
        self.register_buffer("active", torch.tensor(False))

    def _energy(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        return (
            0.5 * self.observation_precision * (state - observation).square()
            + 0.5 * self.persistence_precision * (state - prior).square()
            + 0.5 * self.complexity_precision * state.square()
        )

    @torch.no_grad()
    def reset(self, state: float = 0.0, *, active: bool = False) -> None:
        if not 0.0 <= state <= 1.0:
            raise ValueError("state 必须在 [0,1] 内。")
        self.state.fill_(float(state))
        self.active.fill_(bool(active))

    @torch.no_grad()
    def observe(
        self,
        high_energy_fraction: float | torch.Tensor,
        *,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, bool, dict[str, torch.Tensor | bool] | None]:
        observation = torch.as_tensor(
            high_energy_fraction, device=self.state.device, dtype=self.state.dtype)
        if observation.numel() != 1:
            raise ValueError("high_energy_fraction 必须是标量。")
        observation = observation.reshape(())
        if not bool(((observation >= 0) & (observation <= 1)).item()):
            raise ValueError("high_energy_fraction 必须在 [0,1] 内。")

        prior = self.state.detach().clone()
        state = prior.clone()
        curvature = (
            self.observation_precision
            + self.persistence_precision
            + self.complexity_precision
        )
        step_size = 0.99 * self.relaxation_fraction / curvature
        energy = self._energy(state, observation, prior)
        energies = [energy.clone()]
        states = [state.clone()]

        for _ in range(self.relaxation_steps):
            gradient = (
                self.observation_precision * (state - observation)
                + self.persistence_precision * (state - prior)
                + self.complexity_precision * state
            )
            candidate = state - step_size * gradient
            next_energy = self._energy(candidate, observation, prior)
            if bool(next_energy > energy + 1e-8):
                candidate = state
                next_energy = energy
            decrease = energy - next_energy
            state, energy = candidate, next_energy
            energies.append(energy.clone())
            states.append(state.clone())
            if float(decrease.cpu()) <= self.tolerance:
                break

        self.state.copy_(state.clamp(0.0, 1.0))
        was_active = bool(self.active.item())
        if not was_active and float(self.state.cpu()) >= self.activation_barrier:
            self.active.fill_(True)
        elif was_active and float(self.state.cpu()) <= self.reset_barrier:
            self.active.fill_(False)
        is_active = bool(self.active.item())

        trace = None
        if return_trace:
            trace = {
                "free_energy": torch.stack(energies),
                "state": torch.stack(states),
                "observation": observation.detach().clone(),
                "prior": prior,
                "step_size": step_size.detach().clone(),
                "activated": (not was_active and is_active),
                "deactivated": (was_active and not is_active),
            }
        return self.state.detach().clone(), is_active, trace


class FreeEnergyGrowthSystem(nn.Module):
    """在共享 FreeEnergyLM 上按需增加生成性转移假设。"""

    def __init__(self, core: FreeEnergyLM) -> None:
        super().__init__()
        self.core = core
        self.grown_pathways = nn.ModuleList()
        self.grown_heads = nn.ModuleDict()
        # 普通 Python 容器刻意不注册为子模块：归档参数不随 system.to(cuda) 回到活动显存，
        # 也不进入活动参数统计。磁盘持久化需后续显式 archive checkpoint 协议。
        self.archived_pathways: list[ArchivedPathway | None] = []
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

    def remove_pathway(self, pathway: int) -> dict[int, int]:
        """回收一个已固化的新增盆地，并返回旧索引到新索引的映射。

        基础通路代表共享稳定态，不能被回收。新增通路删除后保持相对顺序，专属读出与
        MDL 代价同步重编号，避免生命周期操作让路由、动力学与读出错位。
        """
        if pathway == 0:
            raise ValueError("基础通路不能被回收。")
        if not 1 <= pathway < self.pathway_count:
            raise IndexError(f"pathway={pathway} 超出 [1,{self.pathway_count - 1}]。")

        old_count = self.pathway_count
        del self.grown_pathways[pathway - 1]

        reindexed_heads = nn.ModuleDict()
        mapping: dict[int, int] = {0: 0}
        for old_index in range(1, old_count):
            if old_index == pathway:
                continue
            new_index = old_index if old_index < pathway else old_index - 1
            mapping[old_index] = new_index
            key = str(old_index)
            if key in self.grown_heads:
                reindexed_heads[str(new_index)] = self.grown_heads[key]
        self.grown_heads = reindexed_heads

        keep = torch.ones(old_count, dtype=torch.bool, device=self.pathway_costs.device)
        keep[pathway] = False
        self.pathway_costs = self.pathway_costs[:old_count][keep]
        return mapping

    @property
    def archive_count(self) -> int:
        return sum(entry is not None for entry in self.archived_pathways)

    def archived_parameter_count(self) -> int:
        total = 0
        for entry in self.archived_pathways:
            if entry is None:
                continue
            total += sum(parameter.numel() for parameter in entry.transition.parameters())
            if entry.head is not None:
                total += sum(parameter.numel() for parameter in entry.head.parameters())
        return total

    def archive_pathway(
        self,
        pathway: int,
        *,
        offload_device: str = "cpu",
    ) -> tuple[int, dict[int, int]]:
        """冻结并从活动路由移除一个新增盆地，返回归档 id 与活动索引映射。"""
        if pathway == 0:
            raise ValueError("基础通路不能归档。")
        if not 1 <= pathway < self.pathway_count:
            raise IndexError(f"pathway={pathway} 超出 [1,{self.pathway_count - 1}]。")
        transition = self.transition_for(pathway)
        key = str(pathway)
        head = self.grown_heads[key] if key in self.grown_heads else None
        for parameter in transition.parameters():
            parameter.requires_grad_(False)
        if head is not None:
            for parameter in head.parameters():
                parameter.requires_grad_(False)
        cost = float(self.pathway_costs[pathway].detach().cpu())
        transition.to(offload_device)
        if head is not None:
            head.to(offload_device)
        entry = ArchivedPathway(
            transition=transition,
            head=head,
            complexity_cost=cost,
            original_index=pathway,
            offload_device=offload_device,
        )
        mapping = self.remove_pathway(pathway)
        self.archived_pathways.append(entry)
        return len(self.archived_pathways) - 1, mapping

    def _archive_entry(self, archive_id: int) -> ArchivedPathway:
        if not 0 <= archive_id < len(self.archived_pathways):
            raise IndexError(f"archive_id={archive_id} 越界。")
        entry = self.archived_pathways[archive_id]
        if entry is None:
            raise RuntimeError(f"archive_id={archive_id} 已恢复或不存在。")
        return entry

    @torch.no_grad()
    def archived_scores(
        self,
        ids: torch.Tensor,
        archive_id: int,
        *,
        start: int = 1,
    ) -> torch.Tensor:
        """临时载入归档动力学，用 residual-F + 原 MDL 代价复核是否值得恢复。"""
        entry = self._archive_entry(archive_id)
        active_device = next(self.core.parameters()).device
        entry.transition.to(active_device)
        try:
            raw = self.provisional_scores(ids, entry.transition, start=start)
            return raw + entry.complexity_cost
        finally:
            entry.transition.to(entry.offload_device)

    def restore_archived_pathway(self, archive_id: int) -> int:
        """把已通过返回能量复核的冷盆地无训练恢复到活动路由。"""
        entry = self._archive_entry(archive_id)
        active_device = next(self.core.parameters()).device
        entry.transition.to(active_device)
        if entry.head is not None:
            entry.head.to(active_device)
        pathway = self.commit_pathway(
            entry.transition,
            complexity_cost=entry.complexity_cost,
            head=entry.head,
        )
        self.archived_pathways[archive_id] = None
        return pathway

    def core_fingerprint(self) -> str:
        """基础稳定态的逐张量 SHA-256；归档不能静默迁移到另一个核心。"""
        metadata = {"kind": "FreeEnergyLM", "state_keys": sorted(self.core.state_dict())}
        state = {
            name: value.detach().cpu()
            for name, value in self.core.state_dict().items()
        }
        return _state_digest(metadata, state)

    @staticmethod
    def _serialize_archived_module(module: nn.Module, role: str) -> dict:
        from fe_llm.energy_lm.models.low_rank_transition import (
            LowRankGenerativeTransition,
            LowRankReadout,
        )

        if role == "transition" and isinstance(module, LowRankGenerativeTransition):
            metadata = {
                "kind": "low_rank_transition",
                "dim": module.dim,
                "rank": module.rank,
                "alpha": module.alpha,
            }
        elif role == "head" and isinstance(module, LowRankReadout):
            metadata = {
                "kind": "low_rank_readout",
                "in_dim": module.in_dim,
                "out_dim": module.out_dim,
                "rank": module.rank,
                "alpha": module.alpha,
            }
        elif role == "head" and isinstance(module, nn.Linear):
            metadata = {
                "kind": "linear_readout",
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bias": module.bias is not None,
            }
        else:
            raise TypeError(
                f"归档仅支持低秩生成性转移及低秩/线性读出，收到 {type(module).__name__}。")
        state = _state_dict_cpu(module)
        return {
            "metadata": metadata,
            "state": state,
            "digest": _state_digest(metadata, state),
        }

    def _deserialize_archived_module(self, record: dict, role: str) -> nn.Module:
        from fe_llm.energy_lm.models.low_rank_transition import (
            LowRankGenerativeTransition,
            LowRankReadout,
        )

        if not isinstance(record, dict):
            raise ValueError("归档模块记录格式错误。")
        metadata = record.get("metadata")
        state = record.get("state")
        expected_digest = record.get("digest")
        if not isinstance(metadata, dict) or not isinstance(state, dict):
            raise ValueError("归档模块缺少 metadata/state。")
        if _state_digest(metadata, state) != expected_digest:
            raise ValueError("归档模块摘要不匹配，文件可能损坏或被篡改。")
        kind = metadata.get("kind")
        if role == "transition" and kind == "low_rank_transition":
            module = LowRankGenerativeTransition(
                self.core.transition,
                dim=int(metadata["dim"]),
                rank=int(metadata["rank"]),
                alpha=float(metadata["alpha"]),
            )
        elif role == "head" and kind == "low_rank_readout":
            module = LowRankReadout(
                self.core.head,
                in_dim=int(metadata["in_dim"]),
                out_dim=int(metadata["out_dim"]),
                rank=int(metadata["rank"]),
                alpha=float(metadata["alpha"]),
            )
        elif role == "head" and kind == "linear_readout":
            module = nn.Linear(
                int(metadata["in_features"]),
                int(metadata["out_features"]),
                bias=bool(metadata["bias"]),
            )
        else:
            raise ValueError(f"不支持的归档模块 kind={kind!r} role={role!r}。")
        module.load_state_dict(state, strict=True)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        return module.cpu()

    def save_archives(self, path: str) -> dict:
        """原子保存所有冷盆地，并返回可审计 manifest。"""
        entries = []
        for archive_id, entry in enumerate(self.archived_pathways):
            if entry is None:
                continue
            transition = self._serialize_archived_module(entry.transition, "transition")
            head = (
                None if entry.head is None
                else self._serialize_archived_module(entry.head, "head"))
            entry_metadata = {
                "archive_id": archive_id,
                "complexity_cost": entry.complexity_cost,
                "original_index": entry.original_index,
                "offload_device": "cpu",
                "transition_digest": transition["digest"],
                "head_digest": None if head is None else head["digest"],
            }
            entries.append({"metadata": entry_metadata,
                            "transition": transition, "head": head})
        core_fingerprint = self.core_fingerprint()
        payload_metadata = {
            "format_version": ARCHIVE_FORMAT_VERSION,
            "core_fingerprint": core_fingerprint,
            "entries": [entry["metadata"] for entry in entries],
        }
        payload_digest = hashlib.sha256(
            json.dumps(payload_metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        payload = {
            **payload_metadata,
            "payload_digest": payload_digest,
            "entries": entries,
        }

        absolute = os.path.abspath(path)
        os.makedirs(os.path.dirname(absolute), exist_ok=True)
        temporary = f"{absolute}.tmp-{os.getpid()}"
        try:
            torch.save(payload, temporary)
            os.replace(temporary, absolute)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        file_digest = hashlib.sha256()
        with open(absolute, "rb") as file:
            for chunk in iter(lambda: file.read(1 << 20), b""):
                file_digest.update(chunk)
        return {
            "format_version": ARCHIVE_FORMAT_VERSION,
            "core_fingerprint": core_fingerprint,
            "archive_count": len(entries),
            "payload_digest": payload_digest,
            "file_sha256": file_digest.hexdigest(),
            "file_size": os.path.getsize(absolute),
        }

    def load_archives(
        self,
        path: str,
        *,
        strict_core: bool = True,
        replace: bool = False,
    ) -> list[int]:
        """安全加载受限张量归档；校验格式、核心指纹、payload 与逐模块摘要。"""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("归档 payload 必须是字典。")
        if payload.get("format_version") != ARCHIVE_FORMAT_VERSION:
            raise ValueError("不支持的归档格式版本。")
        archived_core = payload.get("core_fingerprint")
        if strict_core and archived_core != self.core_fingerprint():
            raise ValueError("归档基础核心指纹不匹配，拒绝恢复。")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError("归档 entries 格式错误。")
        payload_metadata = {
            "format_version": payload["format_version"],
            "core_fingerprint": archived_core,
            "entries": [entry.get("metadata") for entry in entries],
        }
        actual_payload_digest = hashlib.sha256(
            json.dumps(payload_metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if actual_payload_digest != payload.get("payload_digest"):
            raise ValueError("归档 payload 摘要不匹配。")

        loaded = []
        for record in entries:
            if not isinstance(record, dict) or not isinstance(record.get("metadata"), dict):
                raise ValueError("归档 entry 格式错误。")
            metadata = record["metadata"]
            transition_record = record.get("transition")
            head_record = record.get("head")
            if metadata.get("transition_digest") != transition_record.get("digest"):
                raise ValueError("归档 transition 摘要引用不匹配。")
            if metadata.get("head_digest") != (
                    None if head_record is None else head_record.get("digest")):
                raise ValueError("归档 head 摘要引用不匹配。")
            cost = float(metadata["complexity_cost"])
            if cost < 0:
                raise ValueError("归档 complexity_cost 不能为负。")
            transition = self._deserialize_archived_module(transition_record, "transition")
            head = (
                None if head_record is None
                else self._deserialize_archived_module(head_record, "head"))
            entry = ArchivedPathway(
                transition=transition,
                head=head,
                complexity_cost=cost,
                original_index=int(metadata["original_index"]),
                offload_device="cpu",
            )
            if replace and not loaded:
                self.archived_pathways = []
            self.archived_pathways.append(entry)
            loaded.append(len(self.archived_pathways) - 1)
        if replace and not entries:
            self.archived_pathways = []
        return loaded

    @torch.no_grad()
    def pathway_statistics(self, ids: torch.Tensor, start: int = 1) -> dict[str, torch.Tensor]:
        """给出各盆地的最低能路由占比和胜出时相对第二名的能量优势。"""
        scores = self.score_all(ids, start=start)
        choices = scores.argmin(dim=1)
        fractions = torch.bincount(choices, minlength=self.pathway_count).to(scores.dtype)
        fractions = fractions / max(1, ids.size(0))

        advantages = torch.zeros(self.pathway_count, device=scores.device, dtype=scores.dtype)
        if self.pathway_count > 1:
            for pathway in range(self.pathway_count):
                mask = choices == pathway
                if bool(mask.any()):
                    competitors = torch.cat(
                        (scores[:, :pathway], scores[:, pathway + 1:]), dim=1)
                    second_best = competitors.min(dim=1).values
                    advantages[pathway] = (
                        second_best[mask] - scores[mask, pathway]
                    ).clamp_min(0).mean()
        return {
            "route_fraction": fractions,
            "winning_energy_advantage": advantages,
        }

    @torch.no_grad()
    def merge_pathways_if_redundant(
        self,
        keep: int,
        remove: int,
        evidence_ids: torch.Tensor,
        *,
        start: int = 1,
        energy_tolerance: float = 1e-3,
        min_covered_fraction: float = 0.95,
    ) -> tuple[bool, dict[str, float | int]]:
        """若保留盆地能等价解释待删盆地的证据，则合并冗余容量。

        判据只读取因果序列已经产生的 residual-F 与 MDL 代价；不使用技能名或人工
        路由标签。调用者仍应在任务级 held-out 上审计读出能力是否保持。
        """
        if keep == remove:
            raise ValueError("keep 与 remove 必须是不同通路。")
        if not 0 <= keep < self.pathway_count:
            raise IndexError(f"keep={keep} 超出 [0,{self.pathway_count - 1}]。")
        if not 1 <= remove < self.pathway_count:
            raise IndexError(f"remove={remove} 超出 [1,{self.pathway_count - 1}]。")
        if energy_tolerance < 0:
            raise ValueError("energy_tolerance 不能为负。")
        if not 0.0 < min_covered_fraction <= 1.0:
            raise ValueError("min_covered_fraction 必须在 (0,1] 内。")

        scores = self.score_all(evidence_ids, start=start)
        increase = (scores[:, keep] - scores[:, remove]).clamp_min(0)
        covered = float((increase <= energy_tolerance).float().mean().cpu())
        mean_increase = float(increase.mean().cpu())
        merged = covered >= min_covered_fraction
        survivor = keep
        if merged:
            mapping = self.remove_pathway(remove)
            survivor = mapping[keep]
        return merged, {
            "covered_fraction": covered,
            "mean_energy_increase": mean_increase,
            "survivor": survivor,
        }

    @torch.no_grad()
    def retire_pathway_if_inactive(
        self,
        pathway: int,
        recent_ids: torch.Tensor,
        *,
        start: int = 1,
        max_route_fraction: float = 0.01,
    ) -> tuple[bool, dict[str, float]]:
        """在近期证据中几乎从不成为最低能解释时，回收一个新增盆地。"""
        if pathway == 0:
            raise ValueError("基础通路不能被回收。")
        if not 1 <= pathway < self.pathway_count:
            raise IndexError(f"pathway={pathway} 超出 [1,{self.pathway_count - 1}]。")
        if not 0.0 <= max_route_fraction < 1.0:
            raise ValueError("max_route_fraction 必须在 [0,1) 内。")

        stats = self.pathway_statistics(recent_ids, start=start)
        fraction = float(stats["route_fraction"][pathway].cpu())
        advantage = float(stats["winning_energy_advantage"][pathway].cpu())
        retired = fraction <= max_route_fraction
        if retired:
            self.remove_pathway(pathway)
        return retired, {
            "route_fraction": fraction,
            "winning_energy_advantage": advantage,
        }

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
    def residual_scores(
        self,
        ids: torch.Tensor,
        pathway: int,
        start: int = 1,
        *,
        max_relax_steps: int | None = None,
    ) -> torch.Tensor:
        """返回每个样本在指定通路下的归一化稳定后残余自由能。"""
        _, trace = self.forward_pathway(
            ids, pathway, return_trace=True, max_relax_steps=max_relax_steps)
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
    def recalibrate_pathway_cost(
        self,
        stable_ids: torch.Tensor,
        pathway: int,
        *,
        start: int = 1,
        quantile: float = 0.90,
        margin: float = 1e-4,
    ) -> float:
        """拓扑合并/删除后，相对其余稳定解释重新校准已固化盆地的 MDL 代价。"""
        if not 1 <= pathway < self.pathway_count:
            raise IndexError(f"pathway={pathway} 超出 [1,{self.pathway_count - 1}]。")
        if not 0.5 < quantile < 1.0:
            raise ValueError("quantile 应在 (0.5,1.0) 内。")
        other_scores = []
        for other in range(self.pathway_count):
            if other == pathway:
                continue
            raw = self.residual_scores(stable_ids, other, start=start)
            other_scores.append(raw + self.pathway_costs[other].to(raw))
        reference = torch.stack(other_scores, dim=1).min(dim=1).values
        target_raw = self.residual_scores(stable_ids, pathway, start=start)
        false_advantage = reference - target_raw
        cost = torch.quantile(false_advantage, quantile).clamp_min(0) + max(0.0, margin)
        self.pathway_costs[pathway] = cost.to(self.pathway_costs)
        return float(cost.cpu())

    @torch.no_grad()
    def score_all(
        self,
        ids: torch.Tensor,
        start: int = 1,
        *,
        max_relax_steps: int | None = None,
    ) -> torch.Tensor:
        """返回 ``(B,K)``：每个样本在所有生成性通路下的残余自由能。"""
        raw = torch.stack(
            [self.residual_scores(
                ids, pathway, start=start, max_relax_steps=max_relax_steps)
             for pathway in range(self.pathway_count)],
            dim=1,
        )
        return raw + self.pathway_costs[:self.pathway_count].to(raw).unsqueeze(0)

    def _stack_low_rank_parameters(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把共享同一底座、同一秩的低秩通路堆叠；基础通路用零修正表示。"""
        from fe_llm.energy_lm.models.low_rank_transition import LowRankGenerativeTransition

        if not self.grown_pathways:
            raise RuntimeError("批量低秩路由至少需要一个新增通路。")
        if not all(isinstance(pathway, LowRankGenerativeTransition)
                   for pathway in self.grown_pathways):
            raise TypeError("批量低秩路由要求所有新增通路都是 LowRankGenerativeTransition。")
        first = self.grown_pathways[0]
        rank = first.rank
        if any(pathway.rank != rank or pathway.dim != self.core.dim
               or pathway.base_transition is not self.core.transition
               for pathway in self.grown_pathways):
            raise ValueError("所有低秩通路必须共享核心底座、状态维与 rank。")

        down = [torch.zeros_like(first.down.weight)]
        up = [torch.zeros_like(first.up.weight)]
        scale = [torch.zeros_like(first.scale)]
        for pathway in self.grown_pathways:
            down.append(pathway.down.weight)
            up.append(pathway.up.weight)
            scale.append(pathway.scale)
        return torch.stack(down), torch.stack(up), torch.stack(scale)

    @torch.no_grad()
    def indexed_low_rank_scores(
        self,
        ids: torch.Tensor,
        pathways: torch.Tensor,
        *,
        start: int = 1,
        max_relax_steps: int | None = None,
    ) -> torch.Tensor:
        """批量计算逐样本指定低秩通路的 residual-F + MDL。"""
        from fe_llm.energy_lm.models.low_rank_transition import (
            IndexedLowRankGenerativeTransition,
        )

        if pathways.ndim != 1 or pathways.size(0) != ids.size(0):
            raise ValueError("pathways 必须是与 ids batch 对齐的一维索引。")
        if bool(((pathways < 0) | (pathways >= self.pathway_count)).any()):
            raise IndexError("pathways 包含越界通路索引。")
        down, up, scale = self._stack_low_rank_parameters()
        pathways = pathways.to(device=down.device, dtype=torch.long)
        transition = IndexedLowRankGenerativeTransition(
            self.core.transition,
            down[pathways],
            up[pathways],
            scale[pathways],
        )
        _, trace = self.core(
            ids,
            transition_override=transition,
            return_trace=True,
            max_relax_steps=max_relax_steps,
        )
        residual = trace["residual_free_energy_per_dim"]
        start = min(max(0, int(start)), residual.size(1) - 1)
        raw = residual[:, start:].mean(dim=1)
        return raw + self.pathway_costs[pathways].to(raw)

    @torch.no_grad()
    def score_all_low_rank_batched(
        self,
        ids: torch.Tensor,
        start: int = 1,
        *,
        max_relax_steps: int | None = None,
    ) -> torch.Tensor:
        """一次批量前向计算所有共享底座低秩通路的能量。"""
        batch = ids.size(0)
        pathways = torch.arange(self.pathway_count, device=ids.device)
        expanded_pathways = pathways[:, None].expand(-1, batch).reshape(-1)
        expanded_ids = ids.repeat(self.pathway_count, 1)
        scores = self.indexed_low_rank_scores(
            expanded_ids,
            expanded_pathways,
            start=start,
            max_relax_steps=max_relax_steps,
        )
        return scores.view(self.pathway_count, batch).transpose(0, 1)

    @torch.no_grad()
    def route_low_rank_batched(
        self,
        ids: torch.Tensor,
        start: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score_all_low_rank_batched(ids, start=start)
        return scores.argmin(dim=1), scores

    @torch.no_grad()
    def route_cascade(
        self,
        ids: torch.Tensor,
        *,
        start: int = 1,
        screen_relax_steps: int = 1,
        shortlist_size: int = 2,
        batched_low_rank: bool = False,
        screen_prefix_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """用低预算自由能筛选，再只对候选做完整弛豫。

        返回 ``choices, exact_shortlist_scores, shortlist, screen_scores``。最终选择仍来自
        完整 residual-F + MDL，只是未进入 shortlist 的通路以 ``inf`` 标记、不做完整求解。
        廉价阶段不训练额外 router，使用相同生成性动力学和相同显式自由能。
        """
        if not 0 <= screen_relax_steps < self.core.relaxation_steps:
            raise ValueError(
                f"screen_relax_steps 必须在 [0,{self.core.relaxation_steps - 1}] 内。")
        if not 1 <= shortlist_size <= self.pathway_count:
            raise ValueError(f"shortlist_size 必须在 [1,{self.pathway_count}] 内。")
        prefix_length = ids.size(1) if screen_prefix_length is None else int(screen_prefix_length)
        if not 1 <= prefix_length <= ids.size(1):
            raise ValueError(f"screen_prefix_length 必须在 [1,{ids.size(1)}] 内。")

        score_all = (
            self.score_all_low_rank_batched if batched_low_rank else self.score_all)
        screen_ids = ids[:, :prefix_length]
        screen_start = min(start, prefix_length - 1)
        screen_scores = score_all(
            screen_ids, start=screen_start, max_relax_steps=screen_relax_steps)
        shortlist = screen_scores.topk(
            shortlist_size, dim=1, largest=False, sorted=True).indices
        exact_scores = torch.full_like(screen_scores, float("inf"))
        if batched_low_rank:
            expanded_ids = ids[:, None, :].expand(
                -1, shortlist_size, -1).reshape(-1, ids.size(1))
            selected_scores = self.indexed_low_rank_scores(
                expanded_ids, shortlist.reshape(-1), start=start)
            exact_scores.scatter_(1, shortlist, selected_scores.view(ids.size(0), -1))
        else:
            for pathway in range(self.pathway_count):
                selected = (shortlist == pathway).any(dim=1)
                if bool(selected.any()):
                    raw = self.residual_scores(ids[selected], pathway, start=start)
                    exact_scores[selected, pathway] = (
                        raw + self.pathway_costs[pathway].to(raw))
        return exact_scores.argmin(dim=1), exact_scores, shortlist, screen_scores

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
