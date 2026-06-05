# -*- coding: utf-8 -*-
"""
factory.py —— 模型装配工厂
===========================
便捷地装配一个 FreeEnergyLLM：自动加载已训练的两个权重(若存在)，
否则回退到解析/几何版。让"是否已训练"对调用方透明。
"""

from __future__ import annotations

import os

from .config import CHECKPOINT_DIR, get_device
from .embedding import get_embedder
from .engine import FreeEnergyLLM
from .world_model import build_world_model


def load_model(
    use_trained: bool = True,
    world_backend: str = "auto",
    precision: float = 1.0,
    checkpoint_dir: str = CHECKPOINT_DIR,
) -> FreeEnergyLLM:
    """
    装配 FE-LLM。

    use_trained=True 时，尝试加载：
        checkpoints/surprise_net.pt  → SurpriseNet
        checkpoints/decoder_net.pt   → DecoderNet
    任一缺失则该部分回退到解析/几何版。
    """
    device = get_device()
    embedder = get_embedder()
    world = build_world_model(embedder, backend=world_backend)

    surprise_net = None
    decoder_net = None
    if use_trained:
        surprise_net = _try_load_surprise(checkpoint_dir, device)
        decoder_net = _try_load_decoder(checkpoint_dir, device)

    return FreeEnergyLLM(
        embedder=embedder,
        world=world,
        precision=precision,
        surprise_net=surprise_net,
        decoder_net=decoder_net,
        device=device,
    )


def _try_load_surprise(checkpoint_dir: str, device: str):
    path = os.path.join(checkpoint_dir, "surprise_net.pt")
    if not os.path.exists(path):
        print("[factory] 未找到 surprise_net.pt，自由能引擎使用解析版。")
        return None
    from .free_energy.surprise_net import SurpriseNet

    net = SurpriseNet.load(path, map_location=device).to(device)
    print("[factory] 已加载 SurpriseNet 权重。")
    return net


def _try_load_decoder(checkpoint_dir: str, device: str):
    path = os.path.join(checkpoint_dir, "decoder_net.pt")
    if not os.path.exists(path):
        print("[factory] 未找到 decoder_net.pt，解码器使用几何版。")
        return None
    from .generation.decoder_net import DecoderNet

    net = DecoderNet.load(path, map_location=device).to(device)
    print("[factory] 已加载 DecoderNet 权重。")
    return net
