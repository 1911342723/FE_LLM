# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/nlu/slot_intent_nlu.py —— 学习式槽位意图 NLU
==================================================================
把"靠关键词表判断请求意图（订票/订酒店/提醒）"升级为学习式分类器：
特征 = 字符一元 + 二元袋；模型 = 小 MLP。相比固定关键词表，它能泛化到
训练里没出现过的同义/改写表达（keyword 表只命中固定子串）。

意图类别（决定 required_slots）：
    booking  -> ["route"]
    hotel    -> ["city", "date"]
    reminder -> ["time"]
    none     -> []

设计为可选增强：有训练好的 checkpoint 时由感知层调用；否则回退关键词（向后兼容）。
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fe_llm.active_inference.nlu.taxonomy import LEGACY_REQUIRED_SLOTS

# INTENTS 顺序锁定（NLU checkpoint 用 index 编码标签，不可改）。
INTENTS = ["none", "booking", "hotel", "reminder"]
# required_slots 收敛到统一 taxonomy 的单一真相源（值与历史一致，行为不变）。
INTENT_REQUIRED_SLOTS = LEGACY_REQUIRED_SLOTS


def _ngrams(text: str) -> list[str]:
    text = text.strip()
    grams = list(text)
    grams += [text[i : i + 2] for i in range(len(text) - 1)]
    return grams


class SlotIntentNLU:
    """字符 n-gram 袋 + MLP 的意图分类器（脱离关键词表）。"""

    def __init__(self, vocab: dict[str, int] | None = None, hidden: int = 64, device: str = "cpu"):
        self.vocab = vocab or {}
        self.hidden = hidden
        self.device = device
        self.net: nn.Module | None = None

    def _vectorize(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            grams = _ngrams(text)
            for gram in grams:
                idx = self.vocab.get(gram)
                if idx is not None:
                    mat[row, idx] += 1.0
            if grams:
                mat[row] /= len(grams)
        return mat

    def fit(self, texts: list[str], labels: list[str], epochs: int = 300, lr: float = 5e-3, seed: int = 42) -> "SlotIntentNLU":
        if not self.vocab:
            grams = sorted({g for t in texts for g in _ngrams(t)})
            self.vocab = {g: i for i, g in enumerate(grams)}
        torch.manual_seed(seed)
        X = torch.tensor(self._vectorize(texts), device=self.device)
        y = torch.tensor([INTENTS.index(lbl) for lbl in labels], dtype=torch.long, device=self.device)
        self.net = nn.Sequential(
            nn.Linear(len(self.vocab), self.hidden), nn.ReLU(), nn.Linear(self.hidden, len(INTENTS))
        ).to(self.device)
        counts = np.bincount(y.cpu().numpy(), minlength=len(INTENTS)).astype(np.float32)
        weight = torch.tensor(counts.sum() / (np.maximum(counts, 1) * len(INTENTS)), dtype=torch.float32, device=self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(epochs):
            self.net.train()
            opt.zero_grad()
            F.cross_entropy(self.net(X), y, weight=weight).backward()
            opt.step()
        self.net.eval()
        return self

    def predict(self, text: str) -> str:
        if self.net is None:
            return "none"
        with torch.no_grad():
            x = torch.tensor(self._vectorize([text]), device=self.device)
            idx = int(self.net(x).argmax(-1).item())
        return INTENTS[idx]

    def predict_conf(self, text: str) -> tuple[str, float]:
        """返回 (意图, 置信度)。供感知层做置信门控，避免低置信误判触发追问。"""
        if self.net is None:
            return "none", 0.0
        with torch.no_grad():
            x = torch.tensor(self._vectorize([text]), device=self.device)
            probs = torch.softmax(self.net(x), dim=-1)[0]
            idx = int(probs.argmax().item())
        return INTENTS[idx], float(probs[idx].item())

    def required_slots(self, text: str) -> list[str]:
        return list(INTENT_REQUIRED_SLOTS[self.predict(text)])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"vocab": self.vocab, "hidden": self.hidden, "state_dict": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SlotIntentNLU":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        obj = cls(vocab=ckpt["vocab"], hidden=ckpt["hidden"], device=device)
        obj.net = nn.Sequential(
            nn.Linear(len(obj.vocab), obj.hidden), nn.ReLU(), nn.Linear(obj.hidden, len(INTENTS))
        ).to(device)
        obj.net.load_state_dict(ckpt["state_dict"])
        obj.net.eval()
        return obj
