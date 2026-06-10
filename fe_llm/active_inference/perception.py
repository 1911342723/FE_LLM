"""Perception layer: observation text to latent observation state."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from fe_llm.embedding.hash_embedder import HashEmbedder

from .observation import Observation


@dataclass
class ObservationState:
    """Encoded observation used by prediction-error estimators."""

    vector: np.ndarray
    features: dict[str, Any]
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector": {"dim": int(self.vector.size), "norm": float(np.linalg.norm(self.vector))},
            "features": self.features,
            "text": self.text,
        }


class PerceptionEncoder:
    """Encodes observations with IntentEncoder when available, else hash vectors."""

    def __init__(self, use_intent_model: bool = True, fallback_dim: int = 128):
        self.fallback = HashEmbedder(dimension=fallback_dim)
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._enc_max = 24
        if use_intent_model:
            self._try_load_intent_model()

    @property
    def vector_dim(self) -> int:
        if self._model is not None:
            return int(self._model.intent_dim)
        return int(self.fallback.dimension)

    def encode(self, observation: Observation) -> ObservationState:
        vector = self._encode_with_intent_model(observation.text)
        if vector is None:
            vector = self.fallback.embed_one(observation.text)
        return ObservationState(
            vector=np.asarray(vector, dtype=np.float32),
            features=dict(observation.features),
            text=observation.text,
        )

    def _try_load_intent_model(self) -> None:
        try:
            from fe_llm.config import get_device
            from fe_llm.energy_lm.intent_model import IntentLM
            from fe_llm.energy_lm.intent_train import CKPT_PATH, CKPT_TOK, ENC_MAX
            from fe_llm.energy_lm.tokenizer import CharTokenizer

            if not (os.path.exists(CKPT_PATH) and os.path.exists(CKPT_TOK)):
                return
            self._device = get_device()
            self._enc_max = ENC_MAX
            self._model = IntentLM.load(CKPT_PATH, map_location=self._device).to(self._device).eval()
            self._tokenizer = CharTokenizer.load(CKPT_TOK)
        except Exception:
            self._model = None
            self._tokenizer = None
            self._device = "cpu"

    def _encode_with_intent_model(self, text: str) -> np.ndarray | None:
        if self._model is None or self._tokenizer is None:
            return None
        try:
            import torch

            tok = self._tokenizer
            ids = tok.encode(text)[: self._enc_max - 1] + [tok.sep_id]
            ids = ids + [tok.pad_id] * (self._enc_max - len(ids))
            tensor = torch.tensor([ids], device=self._device, dtype=torch.long)
            with torch.no_grad():
                z = self._model.encoder(tensor)[0].detach().float().cpu().numpy()
            return z
        except Exception:
            return None

