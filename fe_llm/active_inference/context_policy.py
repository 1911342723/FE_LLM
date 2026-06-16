# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/context_policy.py —— 上下文感知动作策略（学习式）
========================================================================
把"当前句 + belief 槽位状态"映射到动作（answer/ask_clarification/retrieve/refuse/
update_memory）。这是控制层的可学习版本：同一句在不同 belief 下可得不同动作，
正是任务型多轮的 headroom 所在。

特征 = 字符袋(utterance) + belief 向量(已知槽位键 multi-hot + 槽位数)。模型 = 小 MLP。
可 save/load，可在真实 controller 之外独立训练/评测（不动既有绿测试）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy import ActionType

ACTIONS = [a.value for a in ActionType]
ACTION_ID = {a: i for i, a in enumerate(ACTIONS)}


class ContextAwarePolicy:
    def __init__(self, vocab: dict[str, int] | None = None, slot_keys: list[str] | None = None,
                 hidden: int = 96, device: str = "cpu"):
        self.vocab = vocab or {}
        self.slot_keys = slot_keys or []
        self.hidden = hidden
        self.device = device
        self.net: nn.Module | None = None

    @property
    def _belief_dim(self) -> int:
        return len(self.slot_keys) + 1

    def _featurize(self, utterance: str, known_slots: dict) -> np.ndarray:
        vec = np.zeros(len(self.vocab) + self._belief_dim, dtype=np.float32)
        for c in utterance:
            idx = self.vocab.get(c)
            if idx is not None:
                vec[idx] += 1.0
        if utterance:
            vec[: len(self.vocab)] /= len(utterance)
        base = len(self.vocab)
        for i, key in enumerate(self.slot_keys):
            if key in known_slots:
                vec[base + i] = 1.0
        vec[base + len(self.slot_keys)] = min(len(known_slots), 4) / 4.0
        return vec

    def fit(self, utterances: list[str], known_slots_list: list[dict], actions: list[str],
            epochs: int = 250, lr: float = 5e-3, seed: int = 42) -> "ContextAwarePolicy":
        if not self.vocab:
            chars = sorted({c for u in utterances for c in u})
            self.vocab = {c: i for i, c in enumerate(chars)}
        if not self.slot_keys:
            self.slot_keys = sorted({k for ks in known_slots_list for k in ks})
        torch.manual_seed(seed)
        X = np.stack([self._featurize(u, ks) for u, ks in zip(utterances, known_slots_list)])
        y = np.array([ACTION_ID[a] for a in actions], dtype=np.int64)
        Xt = torch.tensor(X, device=self.device)
        yt = torch.tensor(y, dtype=torch.long, device=self.device)
        self.net = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden), nn.ReLU(), nn.Linear(self.hidden, len(ACTIONS))
        ).to(self.device)
        counts = np.bincount(y, minlength=len(ACTIONS)).astype(np.float32)
        weight = torch.tensor(counts.sum() / (np.maximum(counts, 1) * len(ACTIONS)), dtype=torch.float32, device=self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(epochs):
            self.net.train()
            opt.zero_grad()
            F.cross_entropy(self.net(Xt), yt, weight=weight).backward()
            opt.step()
        self.net.eval()
        return self

    def predict(self, utterance: str, known_slots: dict) -> str:
        if self.net is None:
            return ActionType.ANSWER.value
        with torch.no_grad():
            x = torch.tensor(self._featurize(utterance, known_slots)[None, :], device=self.device)
            idx = int(self.net(x).argmax(-1).item())
        return ACTIONS[idx]

    def save(self, path: str) -> None:
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"vocab": self.vocab, "slot_keys": self.slot_keys, "hidden": self.hidden,
                    "state_dict": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ContextAwarePolicy":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        obj = cls(vocab=ckpt["vocab"], slot_keys=ckpt["slot_keys"], hidden=ckpt["hidden"], device=device)
        in_dim = len(obj.vocab) + obj._belief_dim
        obj.net = nn.Sequential(
            nn.Linear(in_dim, obj.hidden), nn.ReLU(), nn.Linear(obj.hidden, len(ACTIONS))
        ).to(device)
        obj.net.load_state_dict(ckpt["state_dict"])
        obj.net.eval()
        return obj
