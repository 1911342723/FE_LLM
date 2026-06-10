"""Top-level active inference controller."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .action import ActionRealizer
from .belief_update import BeliefUpdater
from .free_energy import ExpectedFreeEnergyScore, FreeEnergyScorer
from .memory import MemoryCandidate, MemoryManager
from .observation import Observation
from .perception import PerceptionEncoder
from .policy import ActionType, PolicyGenerator, PolicySelector
from .prediction import Predictor
from .state import BeliefStateStore
from .surprise import (
    PredictionError,
    PredictionErrorEstimator,
    SurpriseEstimator,
    SurpriseScore,
    is_clarification_fulfilled,
)
from .trace import InferenceTrace, TraceConsistencyScorer, TraceRecorder


@dataclass
class ModelResponse:
    text: str
    selected_action_type: ActionType
    surprise_score: SurpriseScore
    prediction_error: PredictionError
    action_scores: dict[ActionType, ExpectedFreeEnergyScore]
    trace: InferenceTrace
    memory_candidate: MemoryCandidate | None = None
    recalled_memories: list[MemoryCandidate] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "selected_action_type": self.selected_action_type.value,
            "surprise_score": self.surprise_score.to_dict(),
            "prediction_error": self.prediction_error.to_dict(),
            "action_scores": {key.value: value.to_dict() for key, value in self.action_scores.items()},
            "trace": self.trace.to_dict(),
            "memory_candidate": self.memory_candidate.to_dict() if self.memory_candidate else None,
            "recalled_memories": [item.to_dict() for item in self.recalled_memories or []],
        }


class ActiveInferenceController:
    """Runs the v1 observation -> surprise -> policy -> action -> trace loop."""

    def __init__(
        self,
        use_intent_model: bool = True,
        use_energy_decoder: bool = False,
        policy_classifier_path: str | None = os.path.join("checkpoints", "active_inference", "policy_selector.pt"),
        policy_classifier_weight: float = 0.5,
        free_energy_calibration_path: str | None = os.path.join(
            "checkpoints", "active_inference", "free_energy_weights.json"
        ),
        memory_candidate_path: str | None = os.path.join("data", "active_inference", "memory_candidates.jsonl"),
    ):
        self.perception_encoder = PerceptionEncoder(use_intent_model=use_intent_model)
        self.state_store = BeliefStateStore(vector_dim=self.perception_encoder.vector_dim)
        self.predictor = Predictor()
        self.error_estimator = PredictionErrorEstimator()
        self.surprise_estimator = SurpriseEstimator()
        self.belief_updater = BeliefUpdater()
        self.policy_generator = PolicyGenerator()
        self.free_energy_scorer = FreeEnergyScorer(calibration_path=free_energy_calibration_path)
        self.action_selector = PolicySelector(
            classifier_path=policy_classifier_path,
            classifier_weight=policy_classifier_weight,
        )
        self.action_realizer = ActionRealizer(use_energy_decoder=use_energy_decoder)
        self.trace_recorder = TraceRecorder()
        self.trace_consistency = TraceConsistencyScorer()
        self.memory_manager = MemoryManager(candidate_path=memory_candidate_path)

    def respond(self, text: str, session_id: str | None = None) -> ModelResponse:
        observation = Observation.from_text(text, session_id=session_id)
        observation_state = self.perception_encoder.encode(observation)
        prior_belief = self.state_store.load(session_id)
        prediction = self.predictor.predict(prior_belief)
        prediction_error = self.error_estimator.compare(observation_state, prediction)
        surprise = self.surprise_estimator.score(prediction_error)
        # 多轮闭环：判断上一轮的澄清预期是否被当前观测满足。
        clarification_fulfilled = prior_belief.pending_clarification and is_clarification_fulfilled(
            observation.features
        )
        posterior_belief = self.belief_updater.update(
            prior_belief,
            observation_state,
            prediction_error,
            clarification_fulfilled=clarification_fulfilled,
        )
        # 记忆读回：相关记忆注入信念假设，使历史 update_memory 影响当前行为。
        recalled_memories = self.memory_manager.recall(observation.text, session_id=session_id)
        for memory in recalled_memories:
            note = f"memory: {memory.text}"
            if note not in posterior_belief.assumptions:
                posterior_belief.assumptions.append(note)
        candidates = self.policy_generator.generate(posterior_belief, surprise)
        scores = self.free_energy_scorer.score(candidates, posterior_belief, surprise, observation)
        selected_action = self.action_selector.select(
            candidates,
            scores,
            observation=observation,
            observation_state=observation_state,
            prediction_error=prediction_error,
            # 澄清刚被满足时，单轮 MLP 缺乏多轮上下文，交由可解释公式主导。
            fusion_scale=0.0 if clarification_fulfilled else 1.0,
        )
        # 行动回写：selected action 决定下一轮 Predictor 的预期（如等待澄清）。
        posterior_belief = self.belief_updater.apply_action_feedback(
            posterior_belief, selected_action.action_type.value
        )
        text_output = self.action_realizer.realize(
            selected_action,
            observation,
            posterior_belief,
            surprise,
            recalled_memories=recalled_memories,
        )
        trace = self.trace_recorder.record(
            observation=observation,
            observation_state=observation_state,
            prior_belief=prior_belief,
            prediction=prediction,
            prediction_error=prediction_error,
            surprise=surprise,
            candidate_actions=candidates,
            action_scores=scores,
            selected_action=selected_action,
            posterior_belief=posterior_belief,
        )
        ok, missing = self.trace_consistency.validate(trace)
        if not ok:
            raise RuntimeError(f"Incomplete inference trace: {missing}")
        memory_candidate = self.memory_manager.update_if_needed(trace)
        self.state_store.save(posterior_belief, session_id=session_id)
        return ModelResponse(
            text=text_output,
            selected_action_type=selected_action.action_type,
            surprise_score=surprise,
            prediction_error=prediction_error,
            action_scores=scores,
            trace=trace,
            memory_candidate=memory_candidate,
            recalled_memories=recalled_memories,
        )
