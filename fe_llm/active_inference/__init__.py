"""Active inference prototype for FE-LLM v1."""

from .action import ActionRealizer
from .controller import ActiveInferenceController, ModelResponse
from .free_energy import ExpectedFreeEnergyScore, FreeEnergyScorer
from .observation import Observation
from .policy import ActionType, CandidateAction
from .state import BeliefState, PredictionState
from .surprise import PredictionError, SurpriseScore
from .trace import InferenceTrace

__all__ = [
    "ActionRealizer",
    "ActionType",
    "ActiveInferenceController",
    "BeliefState",
    "CandidateAction",
    "ExpectedFreeEnergyScore",
    "FreeEnergyScorer",
    "InferenceTrace",
    "ModelResponse",
    "Observation",
    "PredictionError",
    "PredictionState",
    "SurpriseScore",
]

