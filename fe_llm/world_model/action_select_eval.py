# -*- coding: utf-8 -*-
"""
fe_llm/world_model/action_select_eval.py —— 分层在「动作选择」标尺上的单变量消融
==============================================================================
M1 在生成标尺（decode_loss）上判分层不利，但生成可能是错的标尺（同翻译教训）。
本脚本把分层放到 FE-LLM 真正有价值且有标签的场——动作选择——并做最干净的消融：

    同一个 HierarchicalPredictiveEncoder，唯一变量是 relax_steps：
        relax_steps=0  → z_global = masked mean（等价单向量池化，无分层弛豫）= 对照
        relax_steps=K  → z_global = 弛豫后的 gist（有分层）             = 实验

两条都用 action 标签（policy_teacher）端到端监督训练，同初始化/同数据/同预算，
比 val balanced accuracy。若分层 K>0 显著优于 0，则分层在对的标尺上有价值，保留；
否则分层未证明价值，按 v2 设计第 8 节放弃强加分层。

默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.world_model.action_select_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fe_llm.active_inference.policy import ActionType
from fe_llm.active_inference.training.train_policy import DEFAULT_DATA as POLICY_DATA
from fe_llm.active_inference.training.train_policy import load_samples as load_policy_samples
from fe_llm.config import get_device
from fe_llm.energy_lm.models.tokenizer import CharTokenizer
from fe_llm.world_model.hierarchical_encoder import HierarchicalPredictiveEncoder

REPORT_JSON = os.path.join("docs", "reports", "v2_m1_action_ablation.json")
REPORT_MD = os.path.join("docs", "reports", "v2_m1_action_ablation.md")

ENC_MAX = 24


def encode_prompt_ids(tok: CharTokenizer, text: str) -> list[int]:
    ids = tok.encode(text)[: ENC_MAX - 1] + [tok.sep_id]
    return (ids + [tok.pad_id] * ENC_MAX)[:ENC_MAX]


def balanced_accuracy(pred: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    recalls = []
    for c in range(n_classes):
        mask = y == c
        if mask.any():
            recalls.append(float((pred[mask] == y[mask]).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def stratified_split(y: np.ndarray, n_classes: int, seed: int, val_frac=0.2):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * (1 - val_frac))) if len(idx) > 1 else len(idx)
        tr.extend(idx[:cut].tolist())
        va.extend(idx[cut:].tolist())
    rng.shuffle(tr)
    rng.shuffle(va)
    return np.array(tr, dtype=np.int64), np.array(va, dtype=np.int64)


def train_variant(relax_steps: int, ids: np.ndarray, y: np.ndarray, pad_id: int,
                  vocab_size: int, n_actions: int, args, device, tag: str) -> float:
    torch.manual_seed(args.seed)
    encoder = HierarchicalPredictiveEncoder(
        vocab_size=vocab_size, max_len=ENC_MAX, dim=args.dim, n_heads=args.n_heads,
        intent_dim=args.intent_dim, depth=args.depth, relax_steps=relax_steps, alpha=args.alpha,
    )
    head = nn.Linear(args.intent_dim, n_actions)
    model = nn.ModuleDict({"enc": encoder, "head": head}).to(device)

    tr, va = stratified_split(y, n_actions, args.seed)
    ids_t = torch.tensor(ids, dtype=torch.long, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    counts = np.bincount(y[tr], minlength=n_actions).astype(np.float32)
    w = torch.tensor(counts.sum() / (np.maximum(counts, 1) * n_actions), dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        order = tr.copy()
        rng.shuffle(order)
        for i in range(0, len(order), args.batch):
            ch = order[i : i + args.batch]
            b_ids = ids_t[ch]
            mask = (b_ids != pad_id).float()
            state = encoder(b_ids, attention_mask=mask)
            logits = head(state.z_global)
            loss = F.cross_entropy(logits, y_t[ch], weight=w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            mask = (ids_t[va] != pad_id).float()
            pred = head(encoder(ids_t[va], attention_mask=mask).z_global).argmax(-1).cpu().numpy()
        bal = balanced_accuracy(pred, y[va], n_actions)
        best = max(best, bal)
        if ep % 5 == 0 or ep == args.epochs:
            print(f"[{tag}] ep {ep:3d} val_bal_acc={bal:.4f} (best {best:.4f})", flush=True)
    return best


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hierarchy ablation on action selection (relax_steps=0 vs K).")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--n-policy", type=int, default=0, help="0=全部")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--intent-dim", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--relax-k", type=int, default=5, help="实验组弛豫步数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[action-ablation] dry-run：未训练。")
    print(f"[action-ablation] 对照 relax_steps=0 vs 实验 relax_steps={args.relax_k}，唯一变量=分层弛豫。")
    print(f"[action-ablation] epochs/batch/dim/intent_dim = {args.epochs}/{args.batch}/{args.dim}/{args.intent_dim}")
    print("[action-ablation] 真正运行请显式追加 --run。")


def run(args: argparse.Namespace) -> dict:
    device = get_device() if args.device == "auto" else args.device
    samples = [s for s in load_policy_samples(POLICY_DATA)]
    if args.n_policy > 0:
        samples = samples[: args.n_policy]
    action_idx = {a.value: i for i, a in enumerate(ActionType)}
    prompts = [s["prompt"] for s in samples]
    y = np.array([action_idx[s["action_type"]] for s in samples], dtype=np.int64)
    chars = sorted({c for p in prompts for c in p})
    tok = CharTokenizer(chars)
    ids = np.array([encode_prompt_ids(tok, p) for p in prompts], dtype=np.int64)
    n_actions = len(ActionType)
    dist = {a.value: int((y == i).sum()) for a, i in zip(ActionType, range(n_actions))}
    print(f"[action-ablation] device={device} 样本={len(samples)} 字表={tok.vocab_size} 类别分布={dist}")

    control = train_variant(0, ids, y, tok.pad_id, tok.vocab_size, n_actions, args, device, tag="relax0")
    hier = train_variant(args.relax_k, ids, y, tok.pad_id, tok.vocab_size, n_actions, args, device, tag=f"relax{args.relax_k}")

    delta = hier - control
    verdict = "PASS: 分层提升动作选择" if delta > 0.01 else "FAIL: 分层未提升动作选择"
    result = {
        "n_samples": len(samples),
        "vocab_size": tok.vocab_size,
        "class_dist": dist,
        "epochs": args.epochs,
        "relax_k": args.relax_k,
        "control_relax0_best_bal_acc": round(control, 4),
        "hier_relaxK_best_bal_acc": round(hier, 4),
        "delta": round(delta, 4),
        "verdict": verdict,
        "note": "唯一变量=分层弛豫(relax_steps)；同初始化/数据/预算。动作选择是 FE-LLM 真正有价值的标尺。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# FE-LLM V2-M1 重定向：分层在「动作选择」标尺上的消融",
        "",
        f"- 判定：**{verdict}**",
        f"- 样本：{result['n_samples']}，字表：{result['vocab_size']}，epochs：{result['epochs']}",
        f"- 对照 relax_steps=0 best balanced acc：{result['control_relax0_best_bal_acc']}",
        f"- 实验 relax_steps={args.relax_k} best balanced acc：{result['hier_relaxK_best_bal_acc']}",
        f"- delta（实验-对照）：{result['delta']}",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.run:
        print_dry_run(args)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
