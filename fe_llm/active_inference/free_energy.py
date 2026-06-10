"""Expected free energy scoring for candidate actions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .observation import Observation
from .policy import ActionType, CandidateAction
from .state import BeliefState
from .surprise import SurpriseScore


@dataclass
class ExpectedFreeEnergyScore:
    risk: float
    ambiguity: float
    epistemic_value: float
    action_cost: float
    calibrated_total: float | None = None

    @property
    def total(self) -> float:
        if self.calibrated_total is not None:
            return self.calibrated_total
        return self.risk + self.ambiguity + self.action_cost - self.epistemic_value

    def to_dict(self) -> dict[str, float]:
        return {
            "risk": self.risk,
            "ambiguity": self.ambiguity,
            "epistemic_value": self.epistemic_value,
            "action_cost": self.action_cost,
            "total": round(float(self.total), 4),
        }


class FreeEnergyScorer:
    """Formula-based v1 scorer with transparent score components."""

    def __init__(self, calibration_path: str | None = None):
        self.calibration = FreeEnergyCalibration.load(calibration_path) if calibration_path else None

    def score(
        self,
        candidate_actions: list[CandidateAction],
        posterior_belief: BeliefState,
        surprise: SurpriseScore,
        observation: Observation,
    ) -> dict[ActionType, ExpectedFreeEnergyScore]:
        del posterior_belief
        comps = surprise.components
        features = observation.features
        external = float(features.get("needs_external_info", False))
        memory = float(features.get("has_memory_request", False))
        ambiguity_signal = max(comps.uncertainty_error, comps.intent_error)
        consistency = comps.consistency_error
        safety = comps.safety_error

        out: dict[ActionType, ExpectedFreeEnergyScore] = {}
        for candidate in candidate_actions:
            action = candidate.action_type
            if action == ActionType.ANSWER:
                score = ExpectedFreeEnergyScore(
                    risk=round(2.0 * safety + 0.8 * consistency, 4),
                    ambiguity=round(ambiguity_signal + consistency + 0.8 * external, 4),
                    epistemic_value=0.25 if surprise.total < 0.25 else 0.05,
                    action_cost=candidate.cost,
                )
            elif action == ActionType.ASK_CLARIFICATION:
                score = ExpectedFreeEnergyScore(
                    risk=round(0.8 * safety, 4),
                    ambiguity=0.10,
                    epistemic_value=round(max(0.0, 1.15 * ambiguity_signal + 0.85 * consistency - 0.45 * external), 4),
                    action_cost=candidate.cost,
                )
            elif action == ActionType.RETRIEVE:
                score = ExpectedFreeEnergyScore(
                    risk=round(0.7 * safety, 4),
                    ambiguity=0.15 if external else 0.55,
                    epistemic_value=1.45 if external else 0.05,
                    action_cost=candidate.cost,
                )
            elif action == ActionType.REFUSE:
                score = ExpectedFreeEnergyScore(
                    risk=0.05 if safety else 0.55,
                    ambiguity=0.10,
                    epistemic_value=1.55 if safety else 0.05,
                    action_cost=candidate.cost + (0.0 if safety else 0.45),
                )
            elif action == ActionType.UPDATE_MEMORY:
                score = ExpectedFreeEnergyScore(
                    risk=round(0.5 * safety, 4),
                    ambiguity=0.10,
                    epistemic_value=1.35 if memory else 0.05,
                    action_cost=candidate.cost + (0.0 if memory else 0.65),
                )
            else:
                score = ExpectedFreeEnergyScore(1.0, 1.0, 0.0, candidate.cost)
            out[action] = self._calibrate(score, action)
        return out

    def _calibrate(self, score: ExpectedFreeEnergyScore, action_type: ActionType) -> ExpectedFreeEnergyScore:
        if self.calibration is None:
            return score
        total = self.calibration.total(score, action_type)
        return ExpectedFreeEnergyScore(
            risk=score.risk,
            ambiguity=score.ambiguity,
            epistemic_value=score.epistemic_value,
            action_cost=score.action_cost,
            calibrated_total=round(float(total), 4),
        )


@dataclass(frozen=True)
class FreeEnergyCalibration:
    risk_weight: float = 1.0
    ambiguity_weight: float = 1.0
    epistemic_value_weight: float = 1.0
    action_cost_weight: float = 1.0
    action_bias: dict[str, float] | None = None
    action_weights: dict[str, dict[str, float]] | None = None

    @classmethod
    def load(cls, path: str | None) -> "FreeEnergyCalibration | None":
        if not path or not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            risk_weight=float(data.get("risk_weight", 1.0)),
            ambiguity_weight=float(data.get("ambiguity_weight", 1.0)),
            epistemic_value_weight=float(data.get("epistemic_value_weight", 1.0)),
            action_cost_weight=float(data.get("action_cost_weight", 1.0)),
            action_bias={str(k): float(v) for k, v in data.get("action_bias", {}).items()},
            action_weights={
                str(action): {str(k): float(v) for k, v in weights.items()}
                for action, weights in data.get("action_weights", {}).items()
            },
        )

    def total(self, score: ExpectedFreeEnergyScore, action_type: ActionType) -> float:
        weights = (self.action_weights or {}).get(action_type.value, {})
        risk_weight = weights.get("risk_weight", self.risk_weight)
        ambiguity_weight = weights.get("ambiguity_weight", self.ambiguity_weight)
        epistemic_value_weight = weights.get("epistemic_value_weight", self.epistemic_value_weight)
        action_cost_weight = weights.get("action_cost_weight", self.action_cost_weight)
        bias = (self.action_bias or {}).get(action_type.value, 0.0)
        return (
            risk_weight * score.risk
            + ambiguity_weight * score.ambiguity
            + action_cost_weight * score.action_cost
            - epistemic_value_weight * score.epistemic_value
            + bias
        )
