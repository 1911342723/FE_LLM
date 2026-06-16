# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/nlu/slot_value_tagger.py —— 从 0 的字符级槽位值序列标注器
================================================================================
学习式槽位"值"抽取（不止意图）：对每个字符预测 O/CITY/DATE/TIME，再把连续同类合并成 span。
模型：字符嵌入 + 窗口上下文（±W）+ MLP（完全从 0，无 RNN，训练快、确定）。

诚实界定（写入 经验.md）：
    - DATE/TIME 是规则模式（明天/8点/X号），标注器能学到并泛化到新表达；
    - CITY 是开放命名实体，字符级小模型只能记住见过的城市，对未见城市受容量限制
      （这与 gazetteer 同理，是小模型 NER 的固有边界）。

作为独立模块，不强行进感知热路径。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = ["O", "CITY", "DATE", "TIME"]
PAD, UNK = "<pad>", "<unk>"


class SlotValueTagger:
    """字符级窗口标注器：每个字符 → O/CITY/DATE/TIME。"""

    def __init__(self, window: int = 2, emb_dim: int = 16, hidden: int = 64, device: str = "cpu"):
        self.window = window
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.device = device
        self.char2id: dict[str, int] = {}
        self.emb: nn.Embedding | None = None
        self.mlp: nn.Module | None = None

    def _build_vocab(self, texts: list[str]) -> None:
        chars = sorted({c for t in texts for c in t})
        self.char2id = {PAD: 0, UNK: 1}
        for c in chars:
            self.char2id[c] = len(self.char2id)

    def _ids(self, text: str) -> list[int]:
        return [self.char2id.get(c, 1) for c in text]

    def _windowed(self, id_seq: list[int]) -> torch.Tensor:
        """返回 (L, (2W+1)) 的字符 id 窗口矩阵（越界用 PAD=0）。"""
        L = len(id_seq)
        W = self.window
        padded = [0] * W + id_seq + [0] * W
        rows = [padded[i : i + 2 * W + 1] for i in range(L)]
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def _forward_logits(self, id_seq: list[int]) -> torch.Tensor:
        win = self._windowed(id_seq)                 # (L, 2W+1)
        emb = self.emb(win)                          # (L, 2W+1, E)
        feat = emb.reshape(emb.shape[0], -1)         # (L, (2W+1)*E)
        return self.mlp(feat)                        # (L, n_labels)

    def fit(self, texts: list[str], char_labels: list[list[int]], epochs: int = 250, lr: float = 5e-3, seed: int = 42) -> "SlotValueTagger":
        torch.manual_seed(seed)
        self._build_vocab(texts)
        self.emb = nn.Embedding(len(self.char2id), self.emb_dim, padding_idx=0).to(self.device)
        in_dim = (2 * self.window + 1) * self.emb_dim
        self.mlp = nn.Sequential(nn.Linear(in_dim, self.hidden), nn.ReLU(), nn.Linear(self.hidden, len(LABELS))).to(self.device)
        params = list(self.emb.parameters()) + list(self.mlp.parameters())
        flat_labels = [lab for seq in char_labels for lab in seq]
        counts = np.bincount(flat_labels, minlength=len(LABELS)).astype(np.float32)
        weight = torch.tensor(counts.sum() / (np.maximum(counts, 1) * len(LABELS)), dtype=torch.float32, device=self.device)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        seqs = [self._ids(t) for t in texts]
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.zeros((), device=self.device)
            for ids, labs in zip(seqs, char_labels):
                if not ids:
                    continue
                logits = self._forward_logits(ids)
                y = torch.tensor(labs, dtype=torch.long, device=self.device)
                loss = loss + F.cross_entropy(logits, y, weight=weight)
            loss = loss / max(len(seqs), 1)
            loss.backward()
            opt.step()
        self.emb.eval()
        self.mlp.eval()
        return self

    @torch.no_grad()
    def tag(self, text: str) -> list[int]:
        if self.mlp is None or not text:
            return [0] * len(text)
        return self._forward_logits(self._ids(text)).argmax(-1).cpu().tolist()

    def extract_spans(self, text: str) -> list[tuple[str, str]]:
        """返回 [(label, span_text)]，把连续同类（非 O）字符合并成 span。"""
        labels = self.tag(text)
        spans: list[tuple[str, str]] = []
        cur_label, cur_chars = 0, []
        for ch, lab in zip(text, labels):
            if lab != 0 and lab == cur_label:
                cur_chars.append(ch)
            else:
                if cur_label != 0 and cur_chars:
                    spans.append((LABELS[cur_label], "".join(cur_chars)))
                cur_label, cur_chars = lab, [ch] if lab != 0 else []
        if cur_label != 0 and cur_chars:
            spans.append((LABELS[cur_label], "".join(cur_chars)))
        return spans
