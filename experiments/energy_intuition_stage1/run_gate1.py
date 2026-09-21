"""Gate-1 主实验：5 种子配对比较 + 配对 t 检验 + 种子级 bootstrap。

预注册协议见 docs/STAGE1_PREREGISTRATION.md，本脚本严格执行：
- 种子固定 [11, 23, 37, 51, 71]，全部报告，不删不补；
- 主比较 = 五步能量下降 vs 直接前向预测；单起点、五步，不按测试结果挑起点；
- bootstrap 10000 次，重采样种子固定 20260920，双侧 95% percentile CI；
- 通过条件三项同时满足：均值提升 ≥5pp、p<0.05、CI 下界 >0；
- 测试集在配置冻结前不可加载（哈希门禁）。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import data as D
import verify as V
from energy import EnergyNet, decode
from baselines import DirectPredictor, sample_best_of_n
from solver import descend
from train import unrolled_loss, rank_margin_loss, gradient_penalty, perm_invariant_ce

SEEDS = [11, 23, 37, 51, 71]          # 预注册，硬编码
BOOTSTRAP_SEED = 20260920             # 预注册
N_BOOTSTRAP = 10000
ALPHA = 0.05
MIN_GAIN_PP = 5.0                     # 最低实际收益门槛（百分点）
K_STEPS = 5
ROOT = Path(__file__).parent
FROZEN = ROOT / "frozen_config.json"
SMOKE_CONFIG = ROOT / "smoke_config.json"
PROTOCOL = ROOT.parent.parent / "docs" / "STAGE1_PREREGISTRATION.md"


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def freeze(cfg: dict) -> None:
    """显式冻结，只允许执行一次；已存在则拒绝覆盖。"""
    if FROZEN.exists():
        existing = json.loads(FROZEN.read_text(encoding="utf-8"))
        raise RuntimeError(
            "配置已于 %s 冻结，哈希 %s，拒绝覆盖。"
            "如确需变更，请在 docs/STAGE1_PREREGISTRATION.md 追加协议修订记录，"
            "并手动重命名旧文件留痕。"
            % (existing.get("frozen_at"), existing.get("hash")))
    proto_hash = (hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
                  if PROTOCOL.exists() else None)
    FROZEN.write_text(json.dumps({
        "hash": config_hash(cfg),
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol_sha256": proto_hash,
        "config": cfg,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("配置已冻结 -> %s" % FROZEN)


def require_frozen(cfg: dict, smoke: bool = False) -> None:
    """测试集门禁：配置未冻结或与冻结配置不符则拒绝加载测试集。

    冒烟模式走独立的 smoke 分区，绝不触碰、也绝不写入正式冻结文件。
    """
    if smoke:
        SMOKE_CONFIG.write_text(json.dumps(
            {"hash": config_hash(cfg), "config": cfg, "note": "冒烟配置，非预注册冻结"},
            ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if not FROZEN.exists():
        raise RuntimeError(
            "测试集被锁定：请先冻结配置（写入 frozen_config.json）。"
            "超参与 checkpoint 选择只允许使用验证集。")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen.get("hash") != config_hash(cfg):
        raise RuntimeError("当前配置与冻结配置不一致，测试集拒绝开启。")


def eval_success(adjs, colorings) -> float:
    """主指标：整图完全合法比例。"""
    ok = [V.is_valid(a, c) for a, c in zip(adjs, colorings)]
    return float(np.mean(ok))


def train_energy(train_set, seed, epochs, eta, device="cpu", log=print):
    torch.manual_seed(seed + 1000)
    model = EnergyNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    adjs = torch.tensor(np.stack([a for a, _ in train_set]), dtype=torch.float32)
    neg_pool = []
    for ep in range(epochs):
        perm = torch.randperm(len(adjs))
        for i in range(0, len(adjs), 64):
            adj = adjs[perm[i:i + 64]].to(device)
            z0 = torch.randn(adj.shape[0], 16, 3, device=device, requires_grad=True)
            loss_task, z_star = unrolled_loss(model, adj, z0, K_STEPS, eta)
            loss = loss_task + 0.1 * gradient_penalty(model, adj, z_star.detach())
            # 难负样本挖掘：求解器自己找到的「低能但错误」的解
            with torch.no_grad():
                col = decode(z_star).cpu().numpy()
                bad = [k for k in range(len(col))
                       if not V.is_valid(adj[k].cpu().numpy().astype(np.int8), col[k])]
            if bad:
                z_bad = z_star.detach()[bad]
                z_good = torch.randn_like(z_bad) * 0.1
                loss = loss + 0.1 * rank_margin_loss(model, adj[bad], z_good, z_bad)
            opt.zero_grad()
            loss.backward()
            opt.step()
        log(f"[energy seed={seed}] epoch {ep+1}/{epochs} "
            f"loss={float(loss.detach()):.4f}")
    return model


def train_direct(train_set, seed, epochs, device="cpu", log=print):
    torch.manual_seed(seed + 2000)
    model = DirectPredictor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    adjs = torch.tensor(np.stack([a for a, _ in train_set]), dtype=torch.float32)
    labels = torch.tensor(np.stack([g for _, g in train_set]), dtype=torch.long)
    for ep in range(epochs):
        perm = torch.randperm(len(adjs))
        for i in range(0, len(adjs), 64):
            idx = perm[i:i + 64]
            adj, lab = adjs[idx].to(device), labels[idx].to(device)
            logits = model(adj)
            loss = perm_invariant_ce(logits, lab)   # 颜色置换等价
            opt.zero_grad()
            loss.backward()
            opt.step()
        log(f"[direct seed={seed}] epoch {ep+1}/{epochs} "
            f"loss={float(loss.detach()):.4f}")
    return model


def run_seed(seed, cfg, smoke=False, log=print):
    require_frozen(cfg, smoke=smoke)   # 门禁前置：未冻结立即拒绝，不浪费训练时间
    splits = D.make_splits(seed, cfg["n_train"], cfg["n_val"], cfg["n_test"],
                           cfg["density"], cfg["ood_density"], cfg["n_ood"])
    t0 = time.time()
    e_model = train_energy(splits["train"], seed, cfg["epochs"], cfg["eta"], log=log)
    t_energy_train = time.time() - t0
    t0 = time.time()
    d_model = train_direct(splits["train"], seed, cfg["epochs"], log=log)
    t_direct_train = time.time() - t0

    test = splits["test"]
    adj_np = [a for a, _ in test]
    adj = torch.tensor(np.stack(adj_np), dtype=torch.float32)

    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(adj.shape[0], 16, 3, generator=g)   # 单起点，固定
    t0 = time.time()
    z_star, _ = descend(e_model, adj, z0, steps=K_STEPS, eta=cfg["eta"])
    t_energy_inf = time.time() - t0
    acc_energy = eval_success(adj_np, decode(z_star).numpy())

    t0 = time.time()
    with torch.no_grad():
        acc_direct = eval_success(adj_np, d_model(adj).argmax(-1).numpy())
    t_direct_inf = time.time() - t0

    # 辅助比较（不得替换主比较）
    acc_zero = eval_success(adj_np, decode(z0).numpy())
    g2 = torch.Generator().manual_seed(seed + 7)
    acc_sample = eval_success(
        adj_np, sample_best_of_n(e_model, adj, n=K_STEPS + 1, generator=g2).numpy())

    return {
        "seed": seed,
        "acc_energy": acc_energy, "acc_direct": acc_direct,
        "delta": acc_energy - acc_direct,
        "aux_zero_step": acc_zero, "aux_sample_best_of_n": acc_sample,
        "time_train_energy_s": t_energy_train, "time_train_direct_s": t_direct_train,
        "time_infer_energy_s": t_energy_inf, "time_infer_direct_s": t_direct_inf,
    }


def paired_stats(deltas: list[float]) -> dict:
    from scipy import stats
    d = np.array(deltas, dtype=float)
    t_res = stats.ttest_rel(d + 0, np.zeros_like(d))     # 配对 t 检验（等价单样本）
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.array([rng.choice(d, size=len(d), replace=True).mean()
                     for _ in range(N_BOOTSTRAP)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    mean_pp = float(d.mean() * 100)
    passed = (mean_pp >= MIN_GAIN_PP) and (float(t_res.pvalue) < ALPHA) and (lo * 100 > 0)
    return {
        "mean_gain_pp": mean_pp,
        "p_value": float(t_res.pvalue),
        "bootstrap_ci95_pp": [float(lo * 100), float(hi * 100)],
        "passed": bool(passed),
        "verdict": "PASS" if passed else "证据不足（不表述为已证伪，也不表述为趋势向好）",
        "note": "配对 t 检验与配对 bootstrap 源自同一批数据，不构成两份独立证据。",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="小规模冒烟测试；不写入正式冻结文件，不做统计裁决")
    ap.add_argument("--freeze", action="store_true",
                    help="显式冻结正式配置（只能执行一次），不运行实验")
    ap.add_argument("--out", default="results_gate1.json")
    args = ap.parse_args()

    cfg = {"n_train": 5000, "n_val": 1000, "n_test": 1000, "n_ood": 1000,
           "density": 0.35, "ood_density": 0.5, "epochs": 10, "eta": 1.0,
           "k_steps": K_STEPS, "seeds": SEEDS}
    if args.smoke:
        cfg.update({"n_train": 300, "n_val": 100, "n_test": 100, "n_ood": 100,
                    "epochs": 1, "seeds": [11]})

    if args.freeze:
        if args.smoke:
            raise SystemExit("--freeze 不能与 --smoke 同用。")
        freeze(cfg)
        return

    rows = [run_seed(s, cfg, smoke=args.smoke) for s in cfg["seeds"]]
    result = {"config": cfg, "config_hash": config_hash(cfg),
              "smoke": bool(args.smoke), "per_seed": rows}
    if not args.smoke and FROZEN.exists():
        result["frozen"] = json.loads(FROZEN.read_text(encoding="utf-8"))
    if len(rows) >= 2:
        result["gate1"] = paired_stats([r["delta"] for r in rows])
    else:
        result["gate1"] = {"verdict": "冒烟测试，不做统计裁决"}
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(json.dumps(result["gate1"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
