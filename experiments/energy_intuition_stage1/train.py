"""训练：展开式优化监督 + 难负样本挖掘 + 辅助排序/平滑项。

要点（对应设计文档 v0.2 §2.3）：
- 展开推理时实际使用的 K 步下降，对第 K 步解码后的任务质量直接反传；
- 把求解器自己找到的「低能但错误」的 z* 加入负样本池；
- 颜色置换等价：标签损失取 3! = 6 种置换中的最小值。
"""
from __future__ import annotations
import itertools
import torch
import torch.nn.functional as F

PERMS = list(itertools.permutations(range(3)))


def perm_invariant_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """对颜色编号置换取最小交叉熵，避免惩罚等价的合法解。"""
    losses = []
    for p in PERMS:
        idx = torch.tensor(p, device=labels.device)
        losses.append(F.cross_entropy(
            logits.reshape(-1, 3), idx[labels].reshape(-1), reduction="none"
        ).view(labels.shape).mean(-1))
    return torch.stack(losses, 0).min(0).values.mean()


def soft_conflict_loss(adj: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """可微的任务质量代理：相邻节点颜色分布的内积之和（越小越好）。

    它与整图合法率方向一致，但主指标评估始终使用独立判定器。
    """
    p = torch.softmax(z, dim=-1)
    sim = torch.bmm(p, p.transpose(1, 2))
    return ((sim * adj).sum((1, 2)) / 2.0).mean()


def unrolled_loss(energy_model, adj, z0, steps: int, eta: float):
    """展开 K 步下降（保留计算图），对终点的任务质量求导。"""
    z = z0
    for _ in range(steps):
        e = energy_model(adj, z).sum()
        g = torch.autograd.grad(e, z, create_graph=True)[0]
        z = z - eta * g
    return soft_conflict_loss(adj, z), z


def rank_margin_loss(energy_model, adj, z_good, z_bad, margin: float = 1.0):
    """辅助项：好方案能量应低于坏方案至少 margin。

    注意：排序损失不固定能量零点，故 E 的绝对值无可信度语义，
    置信度只能使用相对量（见设计文档 v0.2 §2.2）。
    """
    e_good = energy_model(adj, z_good)
    e_bad = energy_model(adj, z_bad)
    return F.relu(margin + e_good - e_bad).mean()


def gradient_penalty(energy_model, adj, z):
    """平滑正则：惩罚 ∂E/∂z 的范数，限制能量面变化率。"""
    z = z.detach().requires_grad_(True)
    e = energy_model(adj, z).sum()
    g = torch.autograd.grad(e, z, create_graph=True)[0]
    return (g.pow(2).sum((1, 2))).mean()
