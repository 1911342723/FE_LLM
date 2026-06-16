"""Internal belief and prediction state containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def zero_vector(dim: int = 128) -> np.ndarray:
    return np.zeros(dim, dtype=np.float32)


def _vector_summary(vec: np.ndarray | None) -> dict[str, Any]:
    if vec is None:
        return {"dim": 0, "norm": 0.0}
    arr = np.asarray(vec, dtype=np.float32)
    return {"dim": int(arr.size), "norm": float(np.linalg.norm(arr))}


@dataclass
class BeliefState:
    """The model's current internal state for a session."""

    intent_vector: np.ndarray
    context_vector: np.ndarray
    confidence: float = 0.0
    assumptions: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    # 多轮对话状态：让 Predictor 能基于"上一轮做了什么"形成对下一观测的真实预期。
    turn_index: int = 0
    last_action: str | None = None
    # 上一轮是否发出了澄清请求（即模型预期下一观测应当补充信息）。
    pending_clarification: bool = False
    # 槽位级记忆：已填槽位（slot_key -> value）与正在等待澄清的槽位键。
    # 把"通用 pending_clarification"升级为"槽位级 belief"的地基（见 经验.md headroom 首证）。
    known_slots: dict[str, str] = field(default_factory=dict)
    pending_slot: str | None = None
    # B2d：活跃任务领域（booking/hotel/reminder…），用于消解"领域未明示的跟进句"——
    # 真实数据(CrossWOZ)验证 belief 的真实 headroom 在状态/领域追踪（见 经验.md B2 系列）。
    active_domain: str | None = None

    @classmethod
    def empty(cls, dim: int = 128) -> "BeliefState":
        vec = zero_vector(dim)
        return cls(intent_vector=vec.copy(), context_vector=vec.copy())

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_vector": _vector_summary(self.intent_vector),
            "context_vector": _vector_summary(self.context_vector),
            "confidence": self.confidence,
            "assumptions": list(self.assumptions),
            "unresolved_questions": list(self.unresolved_questions),
            "turn_index": self.turn_index,
            "last_action": self.last_action,
            "pending_clarification": self.pending_clarification,
            "known_slots": dict(self.known_slots),
            "pending_slot": self.pending_slot,
            "active_domain": self.active_domain,
        }


@dataclass
class PredictionState:
    """Expected latent states before observing the current prompt."""

    expected_intent: np.ndarray
    expected_observation: np.ndarray | None = None
    expected_action_outcome: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_intent": _vector_summary(self.expected_intent),
            "expected_observation": _vector_summary(self.expected_observation),
            "expected_action_outcome": self.expected_action_outcome,
        }


class BeliefStateStore:
    """In-memory v1 state store keyed by session id."""

    def __init__(self, vector_dim: int = 128):
        self.vector_dim = vector_dim
        self._states: dict[str, BeliefState] = {}

    def load(self, session_id: str | None = None) -> BeliefState:
        key = session_id or "__default__"
        if key not in self._states:
            self._states[key] = BeliefState.empty(self.vector_dim)
        return self._states[key]

    def save(self, state: BeliefState, session_id: str | None = None) -> None:
        self._states[session_id or "__default__"] = state

