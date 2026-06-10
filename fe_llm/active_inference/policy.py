"""Action policy generation and selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .observation import Observation
from .perception import ObservationState
from .state import BeliefState
from .surprise import PredictionError, SurpriseScore


class ActionType(Enum):
    ANSWER = "answer"
    ASK_CLARIFICATION = "ask_clarification"
    RETRIEVE = "retrieve"
    REFUSE = "refuse"
    UPDATE_MEMORY = "update_memory"


@dataclass
class CandidateAction:
    action_type: ActionType
    intent_vector: np.ndarray
    rationale: str
    cost: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "intent_vector": {"dim": int(self.intent_vector.size), "norm": float(np.linalg.norm(self.intent_vector))},
            "rationale": self.rationale,
            "cost": self.cost,
        }


class PolicyGenerator:
    """Generates the fixed v1 action set."""

    COSTS = {
        ActionType.ANSWER: 0.10,
        ActionType.ASK_CLARIFICATION: 0.20,
        ActionType.RETRIEVE: 0.25,
        ActionType.REFUSE: 0.25,
        ActionType.UPDATE_MEMORY: 0.30,
    }

    RATIONALES = {
        ActionType.ANSWER: "Respond directly when uncertainty and risk are low.",
        ActionType.ASK_CLARIFICATION: "Ask for more information when the observation is underspecified or inconsistent.",
        ActionType.RETRIEVE: "Retrieve external information when the prompt depends on current or external facts.",
        ActionType.REFUSE: "Refuse when the request has safety risk.",
        ActionType.UPDATE_MEMORY: "Store a memory candidate when the user expresses a stable preference or identity fact.",
    }

    def generate(self, posterior_belief: BeliefState, surprise: SurpriseScore) -> list[CandidateAction]:
        del surprise
        return [
            CandidateAction(
                action_type=action_type,
                intent_vector=posterior_belief.intent_vector.copy(),
                rationale=self.RATIONALES[action_type],
                cost=self.COSTS[action_type],
            )
            for action_type in ActionType
        ]


def build_policy_feature_vector(
    observation_state: ObservationState,
    prediction_error: PredictionError,
    vector_size: int = 128,
) -> np.ndarray:
    """Feature vector used by the optional tiny policy classifier."""

    vec = np.asarray(observation_state.vector, dtype=np.float32)
    if vec.size >= vector_size:
        base = vec[:vector_size]
    else:
        base = np.zeros(vector_size, dtype=np.float32)
        base[: vec.size] = vec
    features = observation_state.features
    external_kind = str(features.get("external_info_kind", "none"))
    scalars = np.array(
        [
            prediction_error.semantic_error,
            prediction_error.intent_error,
            prediction_error.consistency_error,
            prediction_error.uncertainty_error,
            prediction_error.safety_error,
            min(features.get("length", 0), 200) / 200.0,
            float(features.get("has_question", False)),
            float(features.get("is_ambiguous_request", False)),
            float(features.get("needs_external_info", False)),
            float(features.get("external_info_confidence", 0.0)),
            float(external_kind == "weather"),
            float(external_kind == "current_time"),
            float(external_kind == "market_or_news"),
            float(external_kind == "public_schedule"),
            float(external_kind == "lookup"),
            float(features.get("has_memory_request", False)),
            float(features.get("has_safety_risk", False)),
            float(features.get("has_consistency_conflict", False)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, scalars])


class PolicySelector:
    """Selects the lowest expected-free-energy action, optionally calibrated by MLP."""

    # 按 action 的融合权重覆盖：retrieve 在公式层先验偏弱（external 特征未触发时
    # ambiguity=0.55、ev=0.05，而 teacher 标注高 uncertainty 时 ask 的 EFE 可低至 -0.5+，
    # 公式间差距可达 ~1.3），统一权重 0.5 不足以让高置信 MLP 翻盘，导致
    # final retrieve recall 显著低于分类器本身（0.686 vs 0.859）。
    # 权重扫描（1.2/1.6/2.0/2.5/3.0）结果：2.5 时整体 acc 最高（0.960），
    # retrieve recall 0.843 且 precision 仅降 0.011；3.0 开始伤 precision。
    DEFAULT_WEIGHT_OVERRIDES: dict[ActionType, float] = {ActionType.RETRIEVE: 2.5}

    def __init__(
        self,
        classifier_path: str | None = None,
        classifier_weight: float = 0.5,
        classifier_weight_overrides: dict[A++ctionType, float] | None = None,
    ):
        self.classifier_path = classifier_path
        # 该权重按当前 EFE total 尺度校准；过大会让 MLP 覆盖可解释公式的场景保护。
        self.classifier_weight = classifier_weight
        self.classifier_weight_overrides = (
            dict(self.DEFAULT_WEIGHT_OVERRIDES)
            if classifier_weight_overrides is None
            else dict(classifier_weight_overrides)
        )
        self._classifier = None
        self._load_classifier()

    def _weight_for(self, action_type: ActionType) -> float:
        return self.classifier_weight_overrides.get(action_type, self.classifier_weight)

    def select(
        self,
        candidates: list[CandidateAction],
        scores: dict[ActionType, Any],
        observation: Observation | None = None,
        observation_state: ObservationState | None = None,
        prediction_error: PredictionError | None = None,
        fusion_scale: float = 1.0,
    ) -> CandidateAction:
        adjusted: dict[ActionType, float] = {
            action.action_type: float(scores[action.action_type].total) for action in candidates
        }
        # fusion_scale：MLP 是单轮 teacher 数据训练的，不具备多轮上下文。
        # 当多轮证据明确（如澄清请求刚被满足）时由控制器降低融合比例，公式主导。
        probs = self._predict_probs(observation_state, prediction_error) if fusion_scale > 0 else None
        if probs is not None:
            for action_type, prob in probs.items():
                if action_type in adjusted:
                    adjusted[action_type] -= fusion_scale * self._weight_for(action_type) * prob
        selected_type = min(adjusted, key=adjusted.get)
        for action in candidates:
            if action.action_type == selected_type:
                return action
        del observation
        return candidates[0]

    def _load_classifier(self) -> None:
        if not self.classifier_path or not os.path.exists(self.classifier_path):
            return
        try:
            import torch

            ckpt = torch.load(self.classifier_path, map_location="cpu", weights_only=False)
            input_dim = int(ckpt["input_dim"])
            hidden_dim = int(ckpt.get("hidden_dim", 64))
            net = _TinyPolicyNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=len(ActionType))
            state_dict = ckpt["state_dict"]
            try:
                net.load_state_dict(state_dict)
            except RuntimeError:
                stripped = {
                    key.removeprefix("net."): value
                    for key, value in state_dict.items()
                    if key.startswith("net.")
                }
                net.load_state_dict(stripped)
            net.eval()
            self._classifier = net
        except Exception:
            self._classifier = None

    def _predict_probs(
        self,
        observation_state: ObservationState | None,
        prediction_error: PredictionError | None,
    ) -> dict[ActionType, float] | None:
        if self._classifier is None or observation_state is None or prediction_error is None:
            return None
        try:
            import torch

            x = build_policy_feature_vector(observation_state, prediction_error)
            with torch.no_grad():
                logits = self._classifier(torch.tensor(x, dtype=torch.float32).unsqueeze(0))[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return {action_type: float(probs[i]) for i, action_type in enumerate(ActionType)}
        except Exception:
            return None


class _TinyPolicyNet:  # replaced by torch.nn.Module lazily to keep import optional at module load
    def __new__(cls, input_dim: int, hidden_dim: int, output_dim: int):
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
