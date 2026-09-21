"""诊断脚本：①单批次轨迹审计 ②求解器正向检查 ③小样本过拟合。

全部只使用训练/验证数据，不触碰测试集，不冻结配置，不做能力结论。

对应评审指出的四处问题：
1. 排序项中 z_good 是随机噪声 —— 诊断阶段直接关闭排序项；
2. 评估未 model.eval() —— spectral norm 内部状态会漂移，全部评估前显式 eval()；
3. 训练用固定步长展开、评估用带回退求解 —— 本脚本同时记录两种轨迹并对比；
4. 两法监督不同 —— 小样本过拟合中两法使用同一份合法标签与同一任务损失。
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import data as D
import verify as V
from energy import EnergyNet, decode
from baselines import DirectPredictor
from solver import descend
from train import perm_invariant_ce, soft_conflict_loss

LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(str(msg))


# ---------------------------------------------------------------- ②
def solver_forward_check():
    """已知最优点的凸二次能量：E = ||z - z_opt||^2，验证下降与梯度链路。"""
    log("=== ② 求解器正向检查（凸二次能量，已知最优点）===")
    torch.manual_seed(0)
    z_opt = torch.randn(4, 16, 3)

    class Quad(torch.nn.Module):
        def forward(self, adj, z):
            return ((z - z_opt) ** 2).flatten(1).sum(-1)

    m = Quad()
    adj = torch.zeros(4, 16, 16)
    z0 = torch.zeros(4, 16, 3)
    e0 = m(adj, z0)
    for eta in [1.0, 0.5, 0.1]:
        z, info = descend(m, adj, z0, steps=5, eta=eta, record=True)
        e1 = info["final_energy"]
        disp = (z - z0).flatten(1).norm(dim=1)
        log(f"  eta={eta}: E {float(e0.mean()):.4f} -> {float(e1.mean()):.4f}, "
            f"||z5-z0||={float(disp.mean()):.4f}, 记录步数={len(info['traj'])}")
    ok = float(e1.mean()) < float(e0.mean()) * 0.5
    log(f"  结论：求解器链路 {'正常' if ok else '异常（凸问题都降不下去）'}")
    return {"quadratic_降幅正常": bool(ok)}


# ---------------------------------------------------------------- ①
def batch_audit(model, adj, z0, eta, tag):
    """逐步轨迹审计：梯度范数、实际步长、回退次数、能量变化、位移、冲突边。"""
    log(f"=== ① 单批次轨迹审计 [{tag}] ===")
    model.eval()                       # 固定 spectral norm 内部状态
    rows = []
    z = z0.clone().requires_grad_(True)
    eta_v = torch.full((z.shape[0], 1, 1), eta)
    adj_np = adj.numpy().astype(np.int8)
    col0 = decode(z0).numpy()
    for step in range(5):
        e = model(adj, z)
        g = torch.autograd.grad(e.sum(), z)[0]
        gnorm = float(g.flatten(1).norm(dim=1).mean())
        backoffs = 0
        cur_eta = eta_v.clone()
        accepted = torch.zeros(z.shape[0], dtype=torch.bool)
        z_new = z.detach().clone()
        e_new = e.detach().clone()
        for _ in range(8):
            cand = z.detach() - cur_eta * g
            with torch.no_grad():
                e_cand = model(adj, cand)
            better = (e_cand < e.detach()) & (~accepted)
            z_new = torch.where(better.view(-1, 1, 1), cand, z_new)
            e_new = torch.where(better, e_cand, e_new)
            accepted |= better
            if accepted.all():
                break
            cur_eta = torch.where(accepted.view(-1, 1, 1), cur_eta, cur_eta * 0.5)
            backoffs += 1
        eta_v = cur_eta
        col = decode(z_new).numpy()
        rows.append({
            "step": step + 1,
            "grad_norm": gnorm,
            "eta": float(eta_v.mean()),
            "backoffs": backoffs,
            "energy": float(e_new.mean()),
            "delta_energy": float((e.detach() - e_new).mean()),
            "disp_from_z0": float((z_new - z0).flatten(1).norm(dim=1).mean()),
            "changed_node_frac": float(np.mean(col != col0)),
            "violations": float(np.mean([V.violation_count(a, c)
                                         for a, c in zip(adj_np, col)])),
        })
        log("  " + json.dumps(rows[-1], ensure_ascii=False))
        z = z_new.requires_grad_(True)
    return rows


def task_grad_audit(model, adj, z0, eta):
    """任务损失单独对能量网络各层的梯度范数。

    总梯度非零可能只是辅助损失在更新，必须单独看任务损失的回传。
    """
    log("=== ① 任务损失 -> 能量网络 的梯度范数（仅任务损失，无辅助项）===")
    model.train()
    z = z0.clone().requires_grad_(True)
    for _ in range(5):
        e = model(adj, z).sum()
        g = torch.autograd.grad(e, z, create_graph=True)[0]
        z = z - eta * g
    loss = soft_conflict_loss(adj, z)
    model.zero_grad()
    loss.backward()
    out = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = float(p.grad.norm())
            if gn > 0 or "readout" in name or "pair" in name:
                out[name] = gn
    total = float(np.sqrt(sum(v ** 2 for v in out.values())))
    nonzero = sum(1 for v in out.values() if v > 1e-12)
    log(f"  任务损失={float(loss):.6f}  总梯度范数={total:.6e}  "
        f"非零参数组={nonzero}/{len(out)}")
    for k in list(out)[:8]:
        log(f"    {k}: {out[k]:.3e}")
    return {"task_loss": float(loss), "total_grad_norm": total,
            "nonzero_param_groups": nonzero, "n_param_groups": len(out)}


# ---------------------------------------------------------------- ③
def overfit_small(n_graphs, steps=400, eta=1.0, seed=11):
    """小样本过拟合：关闭排序项与梯度惩罚，两法使用同一份合法标签与同一任务损失。"""
    log(f"=== ③ 小样本过拟合 n_graphs={n_graphs} ===")
    splits = D.make_splits(seed, n_train=n_graphs, n_val=1, n_test=1, n_ood=1)
    train = splits["train"]
    adj_np = [a for a, _ in train]
    labels_np = np.stack([g for _, g in train])
    adj = torch.tensor(np.stack(adj_np), dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)

    # 同一监督：两法都用真实合法着色的置换不变交叉熵
    torch.manual_seed(seed)
    e_model = EnergyNet(spectral=False)     # ④ 单因素：先关 spectral norm
    d_model = DirectPredictor()
    opt_e = torch.optim.Adam(e_model.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(d_model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(adj.shape[0], 16, 3, generator=g)

    hist = []
    for it in range(steps):
        # 能量法：展开 5 步，对终点 logits 施加与基线相同的监督
        e_model.train()
        z = z0.clone().requires_grad_(True)
        for _ in range(5):
            e = e_model(adj, z).sum()
            gz = torch.autograd.grad(e, z, create_graph=True)[0]
            z = z - eta * gz
        loss_e = perm_invariant_ce(z, labels)       # 同一任务损失
        opt_e.zero_grad(); loss_e.backward(); opt_e.step()

        d_model.train()
        loss_d = perm_invariant_ce(d_model(adj), labels)
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        if (it + 1) % max(1, steps // 8) == 0:
            e_model.eval(); d_model.eval()
            zt, _ = descend(e_model, adj, z0, steps=5, eta=eta)
            col_e = decode(zt).numpy()
            with torch.no_grad():
                col_d = d_model(adj).argmax(-1).numpy()
            rec = {
                "iter": it + 1,
                "loss_energy": float(loss_e.detach()),
                "loss_direct": float(loss_d.detach()),
                "succ_energy": float(np.mean([V.is_valid(a, c)
                                              for a, c in zip(adj_np, col_e)])),
                "succ_direct": float(np.mean([V.is_valid(a, c)
                                              for a, c in zip(adj_np, col_d)])),
                "viol_energy": float(np.mean([V.violation_count(a, c)
                                              for a, c in zip(adj_np, col_e)])),
                "viol_direct": float(np.mean([V.violation_count(a, c)
                                              for a, c in zip(adj_np, col_d)])),
            }
            hist.append(rec)
            log("  " + json.dumps(rec, ensure_ascii=False))
    return hist


# ---------------------------------------------------------------- 训练/评估轨迹差异
def traj_mismatch(model, adj, z0, eta):
    """③ 相关：训练用固定步长展开 vs 评估用带回退求解，两条轨迹差多少。"""
    log("=== 训练(固定步长) vs 评估(带回退) 轨迹差异 ===")
    model.eval()
    z = z0.clone().requires_grad_(True)
    for _ in range(5):
        e = model(adj, z).sum()
        gz = torch.autograd.grad(e, z, create_graph=False)[0]
        z = (z - eta * gz).detach().requires_grad_(True)
    z_fixed = z.detach()
    z_bt, _ = descend(model, adj, z0, steps=5, eta=eta)
    diff = float((z_fixed - z_bt).flatten(1).norm(dim=1).mean())
    same = float((decode(z_fixed) == decode(z_bt)).float().mean())
    log(f"  ||z_fixed - z_backtrack|| = {diff:.6f}, 解码相同节点比例 = {same:.4f}")
    return {"traj_l2_diff": diff, "decode_agreement": same}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="diagnosis.json")
    ap.add_argument("--eta", type=float, default=1.0)
    args = ap.parse_args()
    t0 = time.time()
    res = {"note": "诊断，仅训练/验证数据；不冻结配置，不碰测试集，不做能力结论"}

    res["solver_forward_check"] = solver_forward_check()

    splits = D.make_splits(11, n_train=16, n_val=1, n_test=1, n_ood=1)
    adj = torch.tensor(np.stack([a for a, _ in splits["train"]]), dtype=torch.float32)
    g = torch.Generator().manual_seed(11)
    z0 = torch.randn(adj.shape[0], 16, 3, generator=g)

    torch.manual_seed(11)
    m_sn = EnergyNet(spectral=True)
    res["audit_untrained_spectral"] = batch_audit(m_sn, adj, z0, args.eta, "未训练 + spectral_norm")
    res["task_grad_spectral"] = task_grad_audit(m_sn, adj, z0, args.eta)
    res["traj_mismatch_spectral"] = traj_mismatch(m_sn, adj, z0, args.eta)

    torch.manual_seed(11)
    m_no = EnergyNet(spectral=False)
    res["audit_untrained_nospectral"] = batch_audit(m_no, adj, z0, args.eta, "未训练 + 无 spectral_norm")
    res["task_grad_nospectral"] = task_grad_audit(m_no, adj, z0, args.eta)

    res["overfit_1"] = overfit_small(1, steps=200)
    res["overfit_16"] = overfit_small(16, steps=400)

    res["elapsed_s"] = time.time() - t0
    res["log"] = LOG
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    log(f"诊断完成，耗时 {res['elapsed_s']:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
