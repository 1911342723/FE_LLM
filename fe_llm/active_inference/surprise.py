"""Prediction error and surprise estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fe_llm.embedding.base import cosine_distance

from .perception import ObservationState
from .state import PredictionState


@dataclass
class PredictionError:
    semantic_error: float
    intent_error: float
    consistency_error: float
    uncertainty_error: float
    safety_error: float

    def to_dict(self) -> dict[str, float]:
        return {
            "semantic_error": self.semantic_error,
            "intent_error": self.intent_error,
            "consistency_error": self.consistency_error,
            "uncertainty_error": self.uncertainty_error,
            "safety_error": self.safety_error,
        }


@dataclass
class SurpriseScore:
    total: float
    components: PredictionError
    precision_weights: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "components": self.components.to_dict(),
            "precision_weights": dict(self.precision_weights),
        }


def is_clarification_fulfilled(features: dict[str, Any]) -> bool:
    """判断当前观测是否补上了上一轮澄清请求要求的信息。

    口径：不再是模糊请求、且不是寒暄/致谢这类空内容，并带有可用的具体内容长度。
    """

    if features.get("is_ambiguous_request"):
        return False
    if features.get("is_greeting") or features.get("is_thanks"):
        return False
    return int(features.get("length", 0)) >= 6


class PredictionErrorEstimator:
    """Rule-assisted estimator for v1 interpretable prediction errors."""

    def compare(
        self,
        observation_state: ObservationState,
        prediction_state: PredictionState,
    ) -> PredictionError:
        features = observation_state.features
        outcome = prediction_state.expected_action_outcome or {}
        expects_clarification = bool(outcome.get("expects_clarification"))
        fulfilled = expects_clarification and is_clarification_fulfilled(features)

        semantic = self._semantic_error(observation_state, prediction_state)
        intent = 0.65 if features.get("is_ambiguous_request") else 0.15
        consistency = 1.0 if features.get("has_consistency_conflict") else 0.0
        uncertainty = self._uncertainty_error(features)
        safety = 1.0 if features.get("has_safety_risk") else 0.0

        if fulfilled:
            # 上一轮的澄清预期被验证：观测落在模型预测之内，预测误差应当下降。
            # 这是"对外行动改变环境，从而降低未来自由能"的核心度量点。
            intent = min(intent, 0.10)
            if not features.get("needs_external_info") and consistency == 0.0:
                uncertainty = min(uncertainty, 0.30)
            semantic = semantic * 0.6
        elif expects_clarification and features.get("is_ambiguous_request"):
            # 预期被违背：模型预测用户会补充信息，但输入仍然模糊，误差维持高位。
            uncertainty = max(uncertainty, 0.95)

        return PredictionError(
            semantic_error=round(float(semantic), 4),
            intent_error=round(float(intent), 4),
            consistency_error=round(float(consistency), 4),
            uncertainty_error=round(float(uncertainty), 4),
            safety_error=round(float(safety), 4),
        )

    @staticmethod
    def _semantic_error(
        observation_state: ObservationState,
        prediction_state: PredictionState,
    ) -> float:
        expected = prediction_state.expected_intent
        if expected is None or np.linalg.norm(expected) < 1e-8:
            if observation_state.features.get("is_greeting") or observation_state.features.get("is_thanks"):
                return 0.05
            return 0.2
        dim = min(len(observation_state.vector), len(expected))
        dist = cosine_distance(observation_state.vector[:dim], expected[:dim])
        return min(1.0, max(0.0, dist / 2.0))

    @staticmethod
    def _uncertainty_error(features: dict[str, Any]) -> float:
        value = 0.1
        if features.get("is_ambiguous_request"):
            value = max(value, 0.9)
        if features.get("needs_external_info"):
            value = max(value, 0.75)
        if features.get("has_question") and features.get("length", 0) <= 6:
            value = max(value, 0.35)
        if features.get("has_memory_request"):
            value = min(value, 0.2)
        return value


class SurpriseEstimator:
    """Weighted surprise estimator with explicit precision weights."""

    DEFAULT_WEIGHTS = {
        "semantic_error": 0.7,
        "intent_error": 0.8,
        "consistency_error": 1.1,
        "uncertainty_error": 1.0,
        "safety_error": 1.2,
    }

    def __init__(self, precision_weights: dict[str, float] | None = None):
        self.precision_weights = dict(precision_weights or self.DEFAULT_WEIGHTS)

    def score(self, prediction_error: PredictionError) -> SurpriseScore:
        components = prediction_error.to_dict()
        weighted = sum(components[k] * self.precision_weights.get(k, 1.0) for k in components)
        denom = sum(self.precision_weights.get(k, 1.0) for k in components)
        total = weighted / max(denom, 1e-8)
        return SurpriseScore(
            total=round(float(total), 4),
            components=prediction_error,
            precision_weights=dict(self.precision_weights),
        )

