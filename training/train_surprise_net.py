# -*- coding: utf-8 -*-
"""
training/train_surprise_net.py —— 训练自由能网络 SurpriseNet
=============================================================
运行：python training/train_surprise_net.py

目标：让 SurpriseNet 学会模仿(并平滑化)规则引擎的"误差打分"能力，
      最终固化成 checkpoints/surprise_net.pt 供运行时加载。

这是文档点名"需要真正训练固化成权重"的第一个网络。
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
from fe_llm.free_energy.rule_engine import RuleFreeEnergyEngine
from fe_llm.free_energy.surprise_net import SurpriseNet
from fe_llm.world_model import build_world_model
from training.dataset import build_surprise_dataset


def main(epochs: int = 300, lr: float = 1e-3, batch_size: int = 32):
    device = get_device()
    print(f"[train_surprise] 设备：{device}")

    # 1) 构建世界模型与规则教师
    embedder = get_embedder()
    world = build_world_model(embedder, backend="auto")
    rule = RuleFreeEnergyEngine(world)

    # 2) 蒸馏训练数据
    print("[train_surprise] 生成蒸馏数据 ...")
    signals, expectations, labels = build_surprise_dataset(world, embedder, rule)
    print(f"[train_surprise] 样本数={len(signals)}  维度={signals.shape[1]}")

    ds = TensorDataset(
        torch.tensor(signals), torch.tensor(expectations), torch.tensor(labels)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # 3) 模型 / 优化器 / 损失
    model = SurpriseNet(embed_dim=signals.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()  # 对异常大误差更鲁棒

    # 4) 训练循环
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for s, e, y in loader:
            s, e, y = s.to(device), e.to(device), y.to(device)
            opt.zero_grad()
            pred = model(s, e)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += loss.item() * len(s)
        if epoch % 50 == 0 or epoch == 1:
            print(f"[train_surprise] epoch {epoch:4d}  loss={total/len(ds):.5f}")

    # 5) 固化权重
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, "surprise_net.pt")
    model.save(path)
    print(f"[train_surprise] 权重已保存：{path}")

    # 6) 简单自检：随机抽样对比预测与标签
    model.eval()
    with torch.no_grad():
        idx = np.random.choice(len(signals), size=min(3, len(signals)), replace=False)
        for i in idx:
            s = torch.tensor(signals[i]).unsqueeze(0).to(device)
            e = torch.tensor(expectations[i]).unsqueeze(0).to(device)
            pred = model(s, e).squeeze(0).cpu().numpy()
            print(f"  样本{i}: 预测={np.round(pred,3)}  标签={np.round(labels[i],3)}")


if __name__ == "__main__":
    main()
