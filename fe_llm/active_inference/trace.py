"""Trace recording and consistency checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .free_energy import ExpectedFreeEnergyScore
from .observation import Observation
from .perception import ObservationState
from .policy import ActionType, CandidateAction
from .state import BeliefState, PredictionState
from .surprise import PredictionError, SurpriseScore


@dataclass
class InferenceTrace:
    observation: Observation
    observation_state: ObservationState
    prior_belief: BeliefState
    prediction: PredictionState
    prediction_error: PredictionError
    surprise: SurpriseScore
    candidate_actions: list[CandidateAction]
    action_scores: dict[ActionType, ExpectedFreeEnergyScore]
    selected_action: CandidateAction
    posterior_belief: BeliefState

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "observation_state": self.observation_state.to_dict(),
            "prior_belief": self.prior_belief.to_dict(),
            "prediction": self.prediction.to_dict(),
            "prediction_error": self.prediction_error.to_dict(),
            "surprise": self.surprise.to_dict(),
            "candidate_actions": [action.to_dict() for action in self.candidate_actions],
            "action_scores": {key.value: value.to_dict() for key, value in self.action_scores.items()},
            "selected_action": self.selected_action.to_dict(),
            "posterior_belief": self.posterior_belief.to_dict(),
        }


class TraceRecorder:
    """Builds the structured trace for every response."""

    def record(
        self,
        observation: Observation,
        observation_state: ObservationState,
        prior_belief: BeliefState,
        prediction: PredictionState,
        prediction_error: PredictionError,
        surprise: SurpriseScore,
        candidate_actions: list[CandidateAction],
        action_scores: dict[ActionType, ExpectedFreeEnergyScore],
        selected_action: CandidateAction,
        posterior_belief: BeliefState,
    ) -> InferenceTrace:
        return InferenceTrace(
            observation=observation,
            observation_state=observation_state,
            prior_belief=prior_belief,
            prediction=prediction,
            prediction_error=prediction_error,
            surprise=surprise,
            candidate_actions=candidate_actions,
            action_scores=action_scores,
            selected_action=selected_action,
            posterior_belief=posterior_belief,
        )


class TraceConsistencyScorer:
    """Rule-based v1 trace completeness checker."""

    REQUIRED_KEYS = {
        "observation",
        "prediction_error",
        "candidate_actions",
        "action_scores",
        "selected_action",
        "posterior_belief",
    }

    def validate(self, trace: InferenceTrace) -> tuple[bool, list[str]]:
        data = trace.to_dict()
        missing = sorted(key for key in self.REQUIRED_KEYS if key not in data or data[key] in (None, [], {}))
        selected = trace.selected_action.action_type
        if selected not in trace.action_scores:
            missing.append("selected_action_score")
        if not trace.candidate_actions:
            missing.append("candidate_actions_nonempty")
        return len(missing) == 0, missing

