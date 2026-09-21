"""预算标定：只使用验证集，确认两法是否脱离地板效应（0% 合法率）。

本脚本**不触碰测试集**，不做任何统计裁决，不写入 frozen_config.json。
目的：在正式冻结前确认 Gate-1 的 5pp 门槛有意义（基线非 0）。
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import data as D
import verify as V
from energy import decode
from solver import descend
from run_gate1 import train_energy, train_direct, eval_success, K_STEPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--density", type=float, default=0.35)
    ap.add_argument("--out", default="calibration.json")
    args = ap.parse_args()

    splits = D.make_splits(args.seed, args.n_train, args.n_val,
                           n_test=1, n_ood=1, density=args.density)
    t0 = time.time()
    e_model = train_energy(splits["train"], args.seed, args.epochs, eta=1.0)
    t_e = time.time() - t0
    t0 = time.time()
    d_model = train_direct(splits["train"], args.seed, args.epochs)
    t_d = time.time() - t0

    def evaluate(subset, tag, n_max=500):
        """同时评估训练集与验证集。

        训练集全 0 -> 连拟合都做不到（优化/容量问题）；
        训练集高而验证集低 -> 泛化问题。两者排查方向不同。
        """
        sub = subset[:n_max]
        adj_np = [a for a, _ in sub]
        adj = torch.tensor(np.stack(adj_np), dtype=torch.float32)
        g = torch.Generator().manual_seed(args.seed)
        z0 = torch.randn(adj.shape[0], 16, 3, generator=g)
        z_star, _ = descend(e_model, adj, z0, steps=K_STEPS, eta=1.0)
        col_e = decode(z_star).numpy()
        with torch.no_grad():
            col_d = d_model(adj).argmax(-1).numpy()
        return {
            f"{tag}_success_energy": eval_success(adj_np, col_e),
            f"{tag}_success_direct": eval_success(adj_np, col_d),
            f"diag_{tag}_edge_rate_energy": float(np.mean(
                [V.edge_satisfy_rate(a, c) for a, c in zip(adj_np, col_e)])),
            f"diag_{tag}_edge_rate_direct": float(np.mean(
                [V.edge_satisfy_rate(a, c) for a, c in zip(adj_np, col_d)])),
            f"diag_{tag}_violations_energy": float(np.mean(
                [V.violation_count(a, c) for a, c in zip(adj_np, col_e)])),
            f"diag_{tag}_violations_direct": float(np.mean(
                [V.violation_count(a, c) for a, c in zip(adj_np, col_d)])),
        }

    res = {
        "note": "仅训练集/验证集，预算标定，不构成能力结论，不做统计裁决",
        "config": vars(args),
        "train_time_energy_s": t_e, "train_time_direct_s": t_d,
    }
    res.update(evaluate(splits["train"], "train"))
    res.update(evaluate(splits["val"], "val"))
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
