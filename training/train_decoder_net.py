# -*- coding: utf-8 -*-
"""
training/train_decoder_net.py —— 训练能量递减解码网络 DecoderNet
================================================================
运行：python training/train_decoder_net.py

目标：让 DecoderNet 学会预测"加入某候选语义元后到意图的残余能量"，
      最终固化成 checkpoints/decoder_net.pt 供解码器加载，
      使输出路径规划比纯几何更平滑、更具非线性表达力。

这是文档点名"需要真正训练固化成权重"的第二个网络。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fe_llm.config import CHECKPOINT_DIR, get_device
from fe_llm.embedding import get_embedder
from fe_llm.generation.decoder_net import DecoderNet
from fe_llm.generation.tokenizer import build_default_tokenizer
from training.dataset import build_decoder_dataset


def main(epochs: int = 200, lr: float = 1e-3, batch_size: int = 64,
         n_samples: int = 4000):
    device = get_device()
    print(f"[train_decoder] 设备：{device}")

    embedder = get_embedder()
    tokenizer = build_default_tokenizer(embedder)
    print(f"[train_decoder] 语义元数量={len(tokenizer)}")

    print("[train_decoder] 生成解码蒸馏数据 ...")
    intents, currents, cands, residuals = build_decoder_dataset(
        tokenizer, n_samples=n_samples)
    dim = intents.shape[1]
    print(f"[train_decoder] 样本数={len(intents)}  维度={dim}")

    ds = TensorDataset(
        torch.tensor(intents), torch.tensor(currents),
        torch.tensor(cands), torch.tensor(residuals)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = DecoderNet(embed_dim=dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for it, cu, ca, y in loader:
            it, cu, ca, y = it.to(device), cu.to(device), ca.to(device), y.to(device)
            opt.zero_grad()
            pred = model(it, cu, ca)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += loss.item() * len(it)
        if epoch % 40 == 0 or epoch == 1:
            print(f"[train_decoder] epoch {epoch:4d}  loss={total/len(ds):.6f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, "decoder_net.pt")
    model.save(path)
    print(f"[train_decoder] 权重已保存：{path}")

    model.eval()
    with torch.no_grad():
        idx = np.random.choice(len(intents), size=min(3, len(intents)), replace=False)
        for i in idx:
            it = torch.tensor(intents[i]).unsqueeze(0).to(device)
            cu = torch.tensor(currents[i]).unsqueeze(0).to(device)
            ca = torch.tensor(cands[i]).unsqueeze(0).to(device)
            pred = float(model(it, cu, ca).item())
            print(f"  样本{i}: 预测残余={pred:.4f}  标签={residuals[i][0]:.4f}")


if __name__ == "__main__":
    main()
