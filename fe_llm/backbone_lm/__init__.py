"""Pretrained-backbone route for FE-LLM mechanism experiments."""

from .backbone import BackboneOutput, PretrainedBackbone
from .energy_head import EnergyHead, IntentAdapter
from .hybrid_decode import hybrid_scores, normalize_candidate_energy, select_hybrid_candidate
from .layer_hook import IntentLayerHook, resolve_module
from .logits_adapter import IntentLogitsAdapter
from .residual_adapter import IntentResidualAdapter
from .types import HybridDecodeStep, IntentState

__all__ = [
    "BackboneOutput",
    "EnergyHead",
    "HybridDecodeStep",
    "IntentAdapter",
    "IntentLayerHook",
    "IntentLogitsAdapter",
    "IntentResidualAdapter",
    "IntentState",
    "PretrainedBackbone",
    "hybrid_scores",
    "normalize_candidate_energy",
    "resolve_module",
    "select_hybrid_candidate",
]
