"""置信度特征与校准。

设计文档 v0.2 §2.2：谷深 −E(z*) 不具备可信度语义
（E'(h,z)=E(h,z)+c(h) 不改变排序/梯度/解，却任意改变 −E(z*)），
故仅使用相对量与几何量作为**待检验特征**，最终由独立验证集的真实成功
标签做后验校准，并与温度缩放基线对比 ECE。
"""
from __future__ import annotations
import numpy as np
import torch


def symmetric_curvature(model, adj, z, n_dirs: int = 4, delta: float = 0.05,
                        generator=None) -> torch.Tensor:
    """对称差分方向曲率： [E(z+δv) − 2E(z) + E(z−δv)] / δ²。

    使用对称差分以避免未收敛时混入一阶梯度项。
    """
    with torch.no_grad():
        e0 = model(adj, z)
        acc = torch.zeros_like(e0)
        for _ in range(n_dirs):
            v = torch.randn(z.shape, generator=generator)
            v = v / v.flatten(1).norm(dim=1).view(-1, 1, 1).clamp(min=1e-8)
            ep = model(adj, z + delta * v)
            em = model(adj, z - delta * v)
            acc = acc + (ep - 2 * e0 + em) / (delta ** 2)
        return acc / n_dirs


def energy_gap(energies: torch.Tensor) -> torch.Tensor:
    """多起点下 E(z*_2) − E(z*_1)。相对量，不受 c(h) 影响，先天合法。"""
    srt, _ = torch.sort(energies, dim=0)
    if srt.shape[0] < 2:
        return torch.zeros_like(srt[0])
    return srt[1] - srt[0]


def consistency(colorings: torch.Tensor) -> torch.Tensor:
    """多起点解码一致性：各起点结果两两相同节点比例的均值。"""
    k = colorings.shape[0]
    if k < 2:
        return torch.zeros(colorings.shape[1])
    acc, cnt = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            acc = acc + (colorings[i] == colorings[j]).float().mean(-1)
            cnt += 1
    return acc / cnt


def expected_calibration_error(probs: np.ndarray, correct: np.ndarray,
                               n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs > lo) & (probs <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(correct[m].mean() - probs[m].mean())
    return float(ece)
