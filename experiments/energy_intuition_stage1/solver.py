"""K 步能量下降求解器：带步长控制、每步下降检查、显式停止条件、全轨迹监控。

注意：通用非凸能量网络不继承凸二次版本的下降保证，故这些机制是必需的。
"""
from __future__ import annotations
import torch

from energy import decode


def descend(model, adj: torch.Tensor, z0: torch.Tensor, steps: int = 5,
            eta: float = 1.0, min_eta: float = 1e-4, tol: float = 1e-6,
            record: bool = False):
    """从 z0 出发做 steps 步带回退的梯度下降。

    每步：试探 z' = z - eta*grad；若 E(z') >= E(z) 则 eta 减半重试（最多 8 次）。
    停止条件：能量变化 < tol 或 eta < min_eta 或步数用尽。
    返回 (z, info)。record=True 时 info['traj'] 记录每步能量与解码。
    """
    z = z0.clone().requires_grad_(True)
    B = z.shape[0]
    eta_v = torch.full((B, 1, 1), eta, device=z.device)
    traj = []

    with torch.enable_grad():
        e = model(adj, z)
        for _ in range(steps):
            grad = torch.autograd.grad(e.sum(), z, create_graph=False)[0]
            accepted = torch.zeros(B, dtype=torch.bool, device=z.device)
            z_new = z.detach().clone()
            e_new = e.detach().clone()
            for _try in range(8):
                cand = z.detach() - eta_v * grad
                e_cand = model(adj, cand).detach()
                better = (e_cand < e.detach()) & (~accepted)
                z_new = torch.where(better.view(-1, 1, 1), cand, z_new)
                e_new = torch.where(better, e_cand, e_new)
                accepted = accepted | better
                if accepted.all():
                    break
                eta_v = torch.where(accepted.view(-1, 1, 1), eta_v, eta_v * 0.5)
            delta = (e.detach() - e_new).abs()
            z = z_new.requires_grad_(True)
            e = model(adj, z)
            if record:
                traj.append({"energy": e_new.detach().cpu(),
                             "coloring": decode(z_new).cpu()})
            if bool((delta < tol).all()) or bool((eta_v < min_eta).all()):
                break

    info = {"traj": traj, "final_energy": e.detach()}
    return z.detach(), info


def multi_start(model, adj, k: int, steps: int, dim=(16, 3), scale: float = 1.0,
                generator=None):
    """多起点下降。主实验固定单起点，此函数仅用于辅助比较，需自行对齐预算。"""
    outs = []
    for _ in range(k):
        z0 = torch.randn(adj.shape[0], *dim, generator=generator) * scale
        z, info = descend(model, adj, z0, steps=steps)
        outs.append((z, info["final_energy"]))
    energies = torch.stack([o[1] for o in outs], dim=0)
    best = energies.argmin(0)
    zs = torch.stack([o[0] for o in outs], dim=0)
    z_best = zs[best, torch.arange(zs.shape[1])]
    return z_best, energies
