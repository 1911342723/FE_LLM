# -*- coding: utf-8 -*-
"""
translation/model.py —— 小型 Transformer seq2seq + 能量递减解码器
================================================================
架构：标准 encoder-decoder Transformer（参数量小，适配单卡 RTX 5060）。

与 FE-LLM 思想的对应：
    - 每一步解码输出词表 logits，能量 E(token) = -logit。
    - 贪心/束搜索 = 每步选能量最低的 token「滚落」，直到吐出 </s>（残余能量耗尽）。
    - 训练用交叉熵 = 最小化 -ln P(token|上下文) = 最小化期望惊奇度。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦位置编码（无参数，给序列注入位置信息）。"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TranslationModel(nn.Module):
    """中英双向翻译 Transformer。"""

    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dim_ff: int = 1024, dropout: float = 0.1,
                 pad_id: int = 0, max_len: int = 128):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.max_len = max_len

        # 中英共享词嵌入（源/目标共用，呼应共享子词词表）
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        # 输出投影到词表：输出即每个 token 的 -能量（logit 越高能量越低）
        self.out = nn.Linear(d_model, vocab_size)
        # 权重绑定：输出层与词嵌入共享，省参数且更稳
        self.out.weight = self.embed.weight

        self._init()

    def _init(self):
        nn.init.normal_(self.embed.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.constant_(self.embed.weight[self.pad_id], 0)

    def _pad_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """生成 padding mask：True 表示该位置是 pad，需被忽略。"""
        return seq == self.pad_id

    def _embed(self, seq: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.pos(self.embed(seq) * math.sqrt(self.d_model)))

    def encode(self, src: torch.Tensor):
        """编码源句，返回 (memory, src_pad_mask)。"""
        src_pad = self._pad_mask(src)
        memory = self.transformer.encoder(
            self._embed(src), src_key_padding_mask=src_pad)
        return memory, src_pad

    def decode_step(self, tgt: torch.Tensor, memory: torch.Tensor,
                    src_pad: torch.Tensor) -> torch.Tensor:
        """
        给定已生成的 tgt 前缀，返回每个位置在词表上的 logits。
        tgt: (B, T)。返回 (B, T, vocab)。
        """
        T = tgt.size(1)
        causal = torch.triu(torch.full((T, T), float("-inf"),
                                       device=tgt.device), diagonal=1)
        tgt_pad = self._pad_mask(tgt)
        dec = self.transformer.decoder(
            self._embed(tgt), memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.out(dec)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """训练前向：teacher forcing。返回 logits (B, T, vocab)。"""
        memory, src_pad = self.encode(src)
        return self.decode_step(tgt_in, memory, src_pad)
