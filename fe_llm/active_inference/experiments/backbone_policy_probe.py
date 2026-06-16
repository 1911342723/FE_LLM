"""N2 step1 离线对照 probe：底座句向量做 policy 特征。

研究问题（见 `docs/FE-LLM预训练底座N2执行方案.md` 第 2-3 节）：
把 action 策略分类器的「句向量来源」，从 8.9M 自建 `IntentEncoder` 换成
**冻结 Qwen2.5-0.5B 的 mean-pooled hidden state**，5 类动作分类的
balanced accuracy 会不会更好？

公平对照（唯一变量 = 句向量来源）：
- 同一份 `policy_teacher.jsonl`；
- 同一 `split_samples(seed, val_ratio)`，划分完全一致；
- 同样的 18 个标量特征（teacher `surprise_components` + 规则特征，与编码器无关，
  两臂逐元素相同）；
- 同样的 `TinyPolicyNet` 结构与训练超参；
- 只换 base 句向量：intent(128) vs backbone mean-pool(hidden_size)。

为预防「backbone 维度更高=参数更多才赢」的质疑，额外加一条**维度匹配对照**：
把 backbone mean-pool 向量用 PCA 降到与 intent 相同维度(128)后再训同一分类器。

判定（N2 文档第 3 节）：
- 通过(PASS)：backbone bal_acc 明显高于 intent（delta ≥ +0.02）；
- 部分(PARTIAL)：持平（|delta| < 0.02）；
- 失败(FAIL)：明显低于（delta ≤ -0.02）。

默认 dry-run（只打印计划）；加 `--run` 真正训练对照。

用法：
    python -m fe_llm.active_inference.experiments.backbone_policy_probe --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from fe_llm.active_inference.training.train_policy import (
    DEFAULT_DATA,
    TinyPolicyNet,
    build_dataset,
    class_weights,
    evaluate,
    load_samples,
    split_samples,
)
from fe_llm.config import get_device

# build_policy_feature_vector 把 base 句向量定长到 128，其后接 18 个标量特征。
INTENT_BASE_DIM = 128
SCALAR_DIM = 18

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_REPORT_JSON = os.path.join("docs", "reports", "backbone_policy_probe.json")
DEFAULT_REPORT_MD = os.path.join("docs", "reports", "backbone_policy_probe.md")
DEFAULT_CACHE_DIR = os.path.join("checkpoints", "backbone_lm", "policy_probe_cache")

# 判定阈值：balanced accuracy 的「明显」差异定为 2 个百分点。
MARGIN = 0.02


def backbone_encode_pools(
    prompts: list[str],
    *,
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
    cache_path: str | None = None,
) -> dict[str, np.ndarray]:
    """一次前向同时取两种句向量：mean-pool 与 last-token（按 attention_mask 掩码）。

    last-token 对因果 LM 更合理（末 token attends 整条序列），与 mean-pool 一起给
    底座最强机会再与自建 IntentEncoder 对照。用 max(index where mask==1) 定位最后一个
    真实 token，对左/右 padding 都鲁棒。
    """

    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached["mean"].shape[0] == len(prompts):
            print(f"[probe] 命中缓存 {cache_path} mean={cached['mean'].shape}", flush=True)
            return {"mean": cached["mean"], "last": cached["last"]}
        print(f"[probe] 缓存样本数不符（{cached['mean'].shape[0]}≠{len(prompts)}），重算", flush=True)

    import torch

    from fe_llm.backbone_lm import PretrainedBackbone

    backbone = PretrainedBackbone.from_pretrained(model_name, freeze=True).to(device)
    backbone.eval()
    mean_vecs: list[np.ndarray] = []
    last_vecs: list[np.ndarray] = []
    total = len(prompts)
    for start in range(0, total, batch_size):
        batch = prompts[start : start + batch_size]
        enc = backbone.encode_texts(batch, max_length=max_length, device=device)
        attn = enc["attention_mask"]
        out = backbone(enc["input_ids"], attn)
        hidden = out.hidden_states  # [B, T, H]
        mask = attn.unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        mean_vecs.append((summed / counts).float().cpu().numpy())
        # 最后一个真实 token 的下标 = argmax(mask * 位置序号)，对左右 padding 都成立。
        positions = torch.arange(attn.shape[1], device=attn.device).unsqueeze(0)
        last_idx = (attn * positions).argmax(dim=1)  # [B]
        last_hidden = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx]
        last_vecs.append(last_hidden.float().cpu().numpy())
        done = min(start + batch_size, total)
        if done % (batch_size * 10) == 0 or done == total:
            print(f"[probe] backbone 编码 {done}/{total}", flush=True)

    pools = {
        "mean": np.vstack(mean_vecs).astype(np.float32),
        "last": np.vstack(last_vecs).astype(np.float32),
    }
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, mean=pools["mean"], last=pools["last"])
        print(f"[probe] 缓存写入 {cache_path} mean={pools['mean'].shape}", flush=True)
    return pools


def pca_project(train: np.ndarray, val: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """在 train 上拟合标准化 + PCA，把 train/val 投影到 dim 维（维度匹配对照用）。"""

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    tr = (train - mean) / std
    va = (val - mean) / std
    # 经济 SVD 取前 dim 个主成分。
    _, _, vt = np.linalg.svd(tr, full_matrices=False)
    components = vt[:dim]  # [dim, H]
    return (tr @ components.T).astype(np.float32), (va @ components.T).astype(np.float32)


def train_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden: int,
    epochs: int,
    seed: int,
    batch: int,
    lr: float,
    device: str,
    use_class_weights: bool,
) -> dict:
    """与 train_policy.train 同口径：选 balanced_accuracy 最优的 epoch。"""

    import torch
    import torch.nn.functional as F

    model = TinyPolicyNet(input_dim=int(x_train.shape[1]), hidden_dim=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    cw = class_weights(y_train).to(device) if use_class_weights else None

    best = {"accuracy": 0.0, "balanced_accuracy": -1.0, "per_class_recall": {}}
    for epoch in range(1, epochs + 1):
        order = np.random.default_rng(seed + epoch).permutation(len(x_train))
        model.train()
        for s in range(0, len(order), batch):
            idx = order[s : s + batch]
            xb = torch.tensor(x_train[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb, weight=cw)
            loss.backward()
            opt.step()
        metrics = evaluate(model, x_val, y_val, device)
        if metrics["balanced_accuracy"] > best["balanced_accuracy"]:
            best = metrics
    return best


def verdict(delta: float) -> str:
    if delta >= MARGIN:
        return "PASS"
    if delta <= -MARGIN:
        return "FAIL"
    return "PARTIAL"


def write_reports(payload: dict, json_path: str, md_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    arms = payload["arms"]
    lines = [
        "# N2 step1 · 底座句向量做 policy 特征（离线对照 probe）",
        "",
        f"更新日期：{payload['date']}",
        "",
        "> 唯一变量 = 句向量来源（自建 IntentEncoder vs 冻结 Qwen2.5-0.5B，mean-pool 与 last-token 两种池化）。",
        "> 同 split / 同 18 标量 / 同 TinyPolicyNet 超参；判定指标 = balanced accuracy。",
        "",
        "## 配置",
        "",
        f"- 数据：`{payload['data']}`（train {payload['n_train']} / val {payload['n_val']}）",
        f"- 底座：`{payload['model_name']}`（hidden_size={payload['backbone_hidden']}）",
        f"- 超参：hidden={payload['hidden']}, epochs={payload['epochs']}, "
        f"seed={payload['seed']}, batch={payload['batch']}, lr={payload['lr']}, "
        f"class_weights={payload['class_weights']}",
        f"- 判定边界：|delta| ≥ {MARGIN} 记「明显」",
        "",
        "## 结果",
        "",
        "| 臂 | 句向量来源 | 输入维度 | val accuracy | val balanced_acc |",
        "|---|---|---:|---:|---:|",
    ]
    for arm in arms:
        lines.append(
            f"| {arm['name']} | {arm['source']} | {arm['input_dim']} | "
            f"{arm['accuracy']:.4f} | {arm['balanced_accuracy']:.4f} |"
        )
    lines += ["", "## 判定", ""]
    hl = payload["headline"]
    lines.append(
        f"- **头条（{hl['name']}）**：best_backbone {hl['backbone_bal']:.4f} − intent "
        f"{hl['intent_bal']:.4f} = delta **{hl['delta']:+.4f}** → **{hl['verdict']}**"
    )
    for cmp in payload["comparisons"]:
        lines.append(
            f"- {cmp['name']}：backbone {cmp['backbone_bal']:.4f} − intent "
            f"{cmp['intent_bal']:.4f} = delta **{cmp['delta']:+.4f}** → {cmp['verdict']}"
        )
    lines += ["", "## 逐类召回（balanced_acc 分解）", ""]
    for arm in arms:
        recs = ", ".join(f"{k}={v:.3f}" for k, v in arm["per_class_recall"].items())
        lines.append(f"- {arm['name']}：{recs}")
    lines += ["", "## 结论", "", payload["conclusion"], ""]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    samples = load_samples(args.data)
    if len(samples) < 10:
        raise RuntimeError(f"样本太少：{len(samples)} 条，来自 {args.data}")
    train_samples, val_samples = split_samples(samples, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[probe] device={device} train={len(train_samples)} val={len(val_samples)}", flush=True)

    # intent 臂：完全复用 train_policy.build_dataset（base=intent 句向量 + 18 标量）。
    x_train_intent, y_train = build_dataset(train_samples, use_intent_model=True)
    x_val_intent, y_val = build_dataset(val_samples, use_intent_model=True)

    # 18 标量与编码器无关，从 intent 特征里切出来，给 backbone 臂逐元素复用。
    scalars_train = x_train_intent[:, INTENT_BASE_DIM:]
    scalars_val = x_val_intent[:, INTENT_BASE_DIM:]
    assert scalars_train.shape[1] == SCALAR_DIM, scalars_train.shape

    # backbone 句向量：一次前向取 mean-pool 与 last-token 两种池化。
    pools_train = backbone_encode_pools(
        [s["prompt"] for s in train_samples],
        model_name=args.model_name,
        batch_size=args.bb_batch,
        max_length=args.max_length,
        device=device,
        cache_path=os.path.join(args.cache_dir, f"bb_train_s{args.seed}_n{len(train_samples)}.npz"),
    )
    pools_val = backbone_encode_pools(
        [s["prompt"] for s in val_samples],
        model_name=args.model_name,
        batch_size=args.bb_batch,
        max_length=args.max_length,
        device=device,
        cache_path=os.path.join(args.cache_dir, f"bb_val_s{args.seed}_n{len(val_samples)}.npz"),
    )
    backbone_hidden = int(pools_train["mean"].shape[1])

    common = dict(
        hidden=args.hidden,
        epochs=args.epochs,
        seed=args.seed,
        batch=args.batch,
        lr=args.lr,
        device=device,
        use_class_weights=not args.no_class_weights,
    )

    print("[probe] 训练 intent 臂 ...", flush=True)
    m_intent = train_probe(x_train_intent, y_train, x_val_intent, y_val, **common)
    intent_bal = m_intent["balanced_accuracy"]
    print(f"[probe] intent bal_acc={intent_bal:.4f}", flush=True)

    arms = [
        {
            "name": "intent",
            "source": "IntentEncoder 8.9M (128d)",
            "input_dim": int(x_train_intent.shape[1]),
            **{k: m_intent[k] for k in ("accuracy", "balanced_accuracy", "per_class_recall")},
        }
    ]
    comparisons: list[dict] = []
    best_backbone_bal = -1.0

    # 对 mean / last 两种池化，各跑「全维」与「PCA 维度匹配」两条对照。
    for pool in ("mean", "last"):
        vec_train = pools_train[pool]
        vec_val = pools_val[pool]
        x_tr = np.concatenate([vec_train, scalars_train], axis=1).astype(np.float32)
        x_va = np.concatenate([vec_val, scalars_val], axis=1).astype(np.float32)
        print(f"[probe] 训练 backbone({pool},full) 臂 ...", flush=True)
        m_full = train_probe(x_tr, y_train, x_va, y_val, **common)
        print(f"[probe] backbone({pool},full) bal_acc={m_full['balanced_accuracy']:.4f}", flush=True)
        best_backbone_bal = max(best_backbone_bal, m_full["balanced_accuracy"])
        arms.append(
            {
                "name": f"backbone_{pool}_full",
                "source": f"Qwen2.5-0.5B {pool}-pool ({backbone_hidden}d)",
                "input_dim": int(x_tr.shape[1]),
                **{k: m_full[k] for k in ("accuracy", "balanced_accuracy", "per_class_recall")},
            }
        )
        d_full = m_full["balanced_accuracy"] - intent_bal
        comparisons.append(
            {
                "name": f"backbone_{pool}_full vs intent",
                "intent_bal": intent_bal,
                "backbone_bal": m_full["balanced_accuracy"],
                "delta": d_full,
                "verdict": verdict(d_full),
            }
        )
        if args.pca_dim > 0:
            print(f"[probe] PCA 维度匹配对照 backbone({pool})→{args.pca_dim}d ...", flush=True)
            pca_train, pca_val = pca_project(vec_train, vec_val, args.pca_dim)
            x_tr_p = np.concatenate([pca_train, scalars_train], axis=1).astype(np.float32)
            x_va_p = np.concatenate([pca_val, scalars_val], axis=1).astype(np.float32)
            m_pca = train_probe(x_tr_p, y_train, x_va_p, y_val, **common)
            print(f"[probe] backbone({pool},pca{args.pca_dim}) bal_acc={m_pca['balanced_accuracy']:.4f}", flush=True)
            arms.append(
                {
                    "name": f"backbone_{pool}_pca{args.pca_dim}",
                    "source": f"Qwen2.5-0.5B {pool}-pool→PCA{args.pca_dim}",
                    "input_dim": int(x_tr_p.shape[1]),
                    **{k: m_pca[k] for k in ("accuracy", "balanced_accuracy", "per_class_recall")},
                }
            )
            d_pca = m_pca["balanced_accuracy"] - intent_bal
            comparisons.append(
                {
                    "name": f"backbone_{pool}_pca{args.pca_dim} vs intent",
                    "intent_bal": intent_bal,
                    "backbone_bal": m_pca["balanced_accuracy"],
                    "delta": d_pca,
                    "verdict": verdict(d_pca),
                }
            )

    # 头条判定：给 backbone 最强机会（两种池化全维里最好的一条）对照 intent。
    headline = {
        "name": "best_backbone(full) vs intent",
        "intent_bal": intent_bal,
        "backbone_bal": best_backbone_bal,
        "delta": best_backbone_bal - intent_bal,
        "verdict": verdict(best_backbone_bal - intent_bal),
    }
    conclusion = _conclude(headline, comparisons)

    import datetime

    payload = {
        "date": datetime.date.today().isoformat(),
        "data": args.data,
        "model_name": args.model_name,
        "backbone_hidden": backbone_hidden,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "hidden": args.hidden,
        "epochs": args.epochs,
        "seed": args.seed,
        "batch": args.batch,
        "lr": args.lr,
        "class_weights": not args.no_class_weights,
        "margin": MARGIN,
        "arms": arms,
        "headline": headline,
        "comparisons": comparisons,
        "conclusion": conclusion,
    }
    write_reports(payload, args.report_json, args.report_md)
    print("\n[probe] === 结论 ===", flush=True)
    print(conclusion, flush=True)
    print(f"[probe] 报告：{args.report_json} / {args.report_md}", flush=True)
    return payload


def _conclude(headline: dict, comparisons: list[dict]) -> str:
    v = headline["verdict"]
    delta = headline["delta"]
    head = {
        "PASS": f"在最有利的池化方式下，backbone 句向量明显优于自建 IntentEncoder（best delta {delta:+.4f}）：底座句向量对 action 分类有额外价值，可进入 N2 step2 接入 PerceptionEncoder。",
        "PARTIAL": f"即便取最有利的池化方式，backbone 句向量与自建 IntentEncoder 仍只持平（best delta {delta:+.4f}）：冻结 0.5B 底座的通用句向量在 action 分类上无额外优势——自建 8.9M IntentEncoder 已吃满该任务，符合「瓶颈在任务不在表示规模」。N2 step1 不构成接入理由。",
        "FAIL": f"即便取最有利的池化方式，backbone 句向量仍明显劣于自建 IntentEncoder（best delta {delta:+.4f}）：底座通用句向量不如专训小模型 intent 向量，记录阴性，回到机制层。",
    }[v]
    detail = "；".join(f"{c['name']} {c['delta']:+.4f}({c['verdict']})" for c in comparisons)
    return head + " 各对照：" + detail + "。"


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bb-batch", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--pca-dim", type=int, default=128, help="维度匹配对照维度；0=关闭")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    ap.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    ap.add_argument("--run", action="store_true", help="真正训练对照；缺省只打印计划（dry-run）")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if not args.run:
        print(
            "[probe] dry-run：N2 step1 底座 vs IntentEncoder 句向量对照。\n"
            f"  数据={args.data}\n  底座={args.model_name}\n"
            f"  超参 hidden={args.hidden} epochs={args.epochs} seed={args.seed}\n"
            f"  维度匹配对照 pca_dim={args.pca_dim}\n"
            "  加 --run 真正执行。"
        )
        return
    run(args)


if __name__ == "__main__":
    main()
