# -*- coding: utf-8 -*-
"""
fe_llm/world_model/m1_judgments.py —— V2-M1 判定二/三：分层是否换来价值
=====================================================================
判定一已 FAIL（分层抬高 decode_loss）。本脚本检验分层是否换来"更干净的抽象意图 +
分层 surprise 可定位"，以决定这个代价值不值（v2 设计第 8 节）。

小预算训练一个 v2 HierarchicalIntentLM（dialogues），然后：

判定二a（意图分工）：z_global 与 z_local(masked mean) 各做 action 分类（policy_teacher
    标签），比 balanced accuracy。预期 z_global > z_local（顶层承载抽象意图）。
判定二b（局部分工）：z_local 与 z_global(broadcast) 各预测 prompt 局部 token，比准确率。
    预期 z_local > z_global（底层承载局部词形）。
判定三（分层 surprise 可定位）：对 prompt 做"词序打乱"（局部/表面扰动，token 多集不变）
    与"语义拼接"（前半 A + 后半 B，gist 不连贯）两种破坏，比顶层自由能。
    预期 语义拼接 > 词序打乱（顶层误差对语义更敏感，而非对表面顺序）。

默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.world_model.m1_judgments --run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from argparse import Namespace
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
from fe_llm.energy_lm.data.corpus import load_dialogues
from fe_llm.energy_lm.models.tokenizer import CharTokenizer
from fe_llm.energy_lm.training.intent_train import make_dataset
from fe_llm.world_model import HierarchicalIntentLM
from fe_llm.world_model.m1_train_eval import (
    DEC_MAX,
    DIM,
    ENC_DEPTH,
    ENC_MAX,
    INTENT_DIM,
    NHEADS,
    DEC_DEPTH,
    train_model,
)

REPORT_JSON = os.path.join("docs", "reports", "v2_m1_judgments.json")
REPORT_MD = os.path.join("docs", "reports", "v2_m1_judgments.md")


def encode_prompt_ids(tok: CharTokenizer, text: str) -> list[int]:
    ids = tok.encode(text)[: ENC_MAX - 1] + [tok.sep_id]
    return (ids + [tok.pad_id] * ENC_MAX)[:ENC_MAX]


@torch.no_grad()
def encode_global_local(encoder, tok, texts, device, batch=128):
    """返回 (z_global:(N,d), z_local_pooled:(N,d))。"""
    zg, zl = [], []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        ids = torch.tensor([encode_prompt_ids(tok, t) for t in chunk], device=device)
        mask = (ids != tok.pad_id).float()
        state = encoder(ids, attention_mask=mask)
        m = mask.unsqueeze(-1)
        pooled = (state.z_local * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        zg.append(state.z_global.cpu())
        zl.append(pooled.cpu())
    return torch.cat(zg), torch.cat(zl)


def _balanced_accuracy(pred: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    recalls = []
    for c in range(n_classes):
        mask = y == c
        if mask.any():
            recalls.append(float((pred[mask] == y[mask]).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def linear_probe(X: torch.Tensor, y: np.ndarray, n_classes: int, device, seed=0,
                 epochs=300, lr=1e-2, metric="balanced") -> float:
    """冻结特征上的线性探针，返回 val 指标（balanced accuracy 或 accuracy）。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    split = int(len(y) * 0.8)
    tr, va = idx[:split], idx[split:]
    Xtr = X[tr].to(device)
    ytr = torch.tensor(y[tr], dtype=torch.long, device=device)
    Xva = X[va].to(device)
    yva = y[va]
    clf = nn.Linear(X.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    counts = np.bincount(y[tr], minlength=n_classes).astype(np.float32)
    w = torch.tensor(counts.sum() / (np.maximum(counts, 1) * n_classes), dtype=torch.float32, device=device)
    for _ in range(epochs):
        clf.train()
        opt.zero_grad()
        loss = F.cross_entropy(clf(Xtr), ytr, weight=w)
        loss.backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        pred = clf(Xva).argmax(-1).cpu().numpy()
    if metric == "balanced":
        return _balanced_accuracy(pred, yva, n_classes)
    return float((pred == yva).mean())


@torch.no_grad()
def local_token_probe_features(encoder, tok, texts, device, batch=128, max_pos=200000):
    """收集 (z_local[i], token_id) 与 (z_global, token_id) 用于局部 token 预测对照。"""
    zl_feat, zg_feat, toks = [], [], []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        ids = torch.tensor([encode_prompt_ids(tok, t) for t in chunk], device=device)
        mask = (ids != tok.pad_id)
        state = encoder(ids, attention_mask=mask.float())
        zloc = state.z_local                                  # (B,L,d)
        zglob = state.z_global.unsqueeze(1).expand(-1, zloc.shape[1], -1)
        valid = mask & (ids != tok.sep_id)
        zl_feat.append(zloc[valid].cpu())
        zg_feat.append(zglob[valid].cpu())
        toks.append(ids[valid].cpu())
        if sum(t.shape[0] for t in toks) >= max_pos:
            break
    return torch.cat(zl_feat), torch.cat(zg_feat), torch.cat(toks).numpy()


@torch.no_grad()
def mean_free_energy(encoder, tok, texts, device, batch=128) -> float:
    total, n = 0.0, 0
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        ids = torch.tensor([encode_prompt_ids(tok, t) for t in chunk], device=device)
        mask = (ids != tok.pad_id).float()
        state = encoder(ids, attention_mask=mask)
        total += float(state.free_energy) * len(chunk)
        n += len(chunk)
    return total / max(n, 1)


def shuffle_chars(text: str, rng: random.Random) -> str:
    chars = list(text)
    rng.shuffle(chars)
    return "".join(chars)


def mash(text_a: str, text_b: str) -> str:
    half_a = text_a[: max(1, len(text_a) // 2)]
    half_b = text_b[len(text_b) // 2 :]
    return half_a + half_b


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V2-M1 judgments 2/3: does hierarchy buy a cleaner intent + locatable surprise.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--n-dialogue", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-policy", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[m1-judge] dry-run：未训练。")
    print(f"[m1-judge] n_dialogue/epochs/n_policy = {args.n_dialogue}/{args.epochs}/{args.n_policy}")
    print("[m1-judge] 判定二a意图分工 / 判定二b局部分工 / 判定三分层surprise定位。")
    print("[m1-judge] 真正运行请显式追加 --run。")


def run(args: argparse.Namespace) -> dict:
    device = get_device() if args.device == "auto" else args.device
    dias = load_dialogues()
    if args.n_dialogue > 0:
        dias = dias[: args.n_dialogue]
    chars = sorted({c for p, r in dias for c in p + r})
    tok = CharTokenizer(chars)
    data = make_dataset(tok, dias, ENC_MAX, DEC_MAX)
    print(f"[m1-judge] device={device} 对话={len(data[0])} 字表={tok.vocab_size}")

    torch.manual_seed(args.seed)
    v2 = HierarchicalIntentLM(
        vocab_size=tok.vocab_size, enc_max=ENC_MAX, dec_max=DEC_MAX, dim=DIM,
        enc_depth=ENC_DEPTH, dec_depth=DEC_DEPTH, n_heads=NHEADS, intent_dim=INTENT_DIM,
    ).to(device)
    train_model(v2, data, tok, Namespace(epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed), device, tag="v2")
    encoder = v2.encoder
    encoder.eval()

    # ---- 判定二a：意图分工（z_global vs z_local → action）----
    policy_samples = [s for s in load_policy_samples(POLICY_DATA)][: args.n_policy]
    action_idx = {a.value: i for i, a in enumerate(ActionType)}
    prompts = [s["prompt"] for s in policy_samples]
    y_action = np.array([action_idx[s["action_type"]] for s in policy_samples], dtype=np.int64)
    zg, zl = encode_global_local(encoder, tok, prompts, device)
    zg_bal = linear_probe(zg, y_action, len(ActionType), device, seed=args.seed, metric="balanced")
    zl_bal = linear_probe(zl, y_action, len(ActionType), device, seed=args.seed, metric="balanced")

    # ---- 判定二b：局部分工（z_local vs z_global → 局部 token）----
    zl_feat, zg_feat, toks = local_token_probe_features(encoder, tok, [p for p, _ in dias][:1500], device)
    zl_tok_acc = linear_probe(zl_feat, toks, tok.vocab_size, device, seed=args.seed, epochs=120, metric="accuracy")
    zg_tok_acc = linear_probe(zg_feat, toks, tok.vocab_size, device, seed=args.seed, epochs=120, metric="accuracy")

    # ---- 判定三：分层 surprise 定位（语义拼接 vs 词序打乱 的顶层自由能）----
    rng = random.Random(args.seed)
    base_prompts = [p for p, _ in dias][:1000]
    shuffled = [shuffle_chars(p, rng) for p in base_prompts]
    mashed = [mash(base_prompts[i], base_prompts[(i + 1) % len(base_prompts)]) for i in range(len(base_prompts))]
    f_normal = mean_free_energy(encoder, tok, base_prompts, device)
    f_shuffle = mean_free_energy(encoder, tok, shuffled, device)
    f_mash = mean_free_energy(encoder, tok, mashed, device)

    j2a = zg_bal > zl_bal
    j2b = zl_tok_acc > zg_tok_acc
    j3 = f_mash > f_shuffle
    earns = j2a and j2b and j3
    result = {
        "n_dialogue": len(data[0]),
        "n_policy": len(policy_samples),
        "judgment2a_intent": {
            "z_global_action_bal_acc": round(zg_bal, 4),
            "z_local_action_bal_acc": round(zl_bal, 4),
            "pass": bool(j2a),
        },
        "judgment2b_local": {
            "z_local_token_acc": round(zl_tok_acc, 4),
            "z_global_token_acc": round(zg_tok_acc, 4),
            "pass": bool(j2b),
        },
        "judgment3_surprise": {
            "free_energy_normal": round(f_normal, 4),
            "free_energy_shuffle": round(f_shuffle, 4),
            "free_energy_mash": round(f_mash, 4),
            "pass_mash_gt_shuffle": bool(j3),
        },
        "hierarchy_earns_its_keep": bool(earns),
        "note": "判定一(decode_loss)已 FAIL；本组若全过，说明分层用 decode_loss 代价换来了更干净的意图+可定位 surprise，代价值得。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# FE-LLM V2-M1 判定二/三：分层是否值这个 decode_loss 代价",
        "",
        f"- 综合：**{'分层值得（二/三过）' if earns else '分层未充分证明价值'}**",
        "",
        "## 判定二a 意图分工（z_global vs z_local → action）",
        f"- z_global balanced acc：{result['judgment2a_intent']['z_global_action_bal_acc']}",
        f"- z_local  balanced acc：{result['judgment2a_intent']['z_local_action_bal_acc']}",
        f"- 通过（z_global 更会分意图）：{result['judgment2a_intent']['pass']}",
        "",
        "## 判定二b 局部分工（z_local vs z_global → 局部 token）",
        f"- z_local token acc：{result['judgment2b_local']['z_local_token_acc']}",
        f"- z_global token acc：{result['judgment2b_local']['z_global_token_acc']}",
        f"- 通过（z_local 更会分局部 token）：{result['judgment2b_local']['pass']}",
        "",
        "## 判定三 分层 surprise 定位（顶层自由能）",
        f"- 正常：{result['judgment3_surprise']['free_energy_normal']}",
        f"- 词序打乱（表面）：{result['judgment3_surprise']['free_energy_shuffle']}",
        f"- 语义拼接（gist 破坏）：{result['judgment3_surprise']['free_energy_mash']}",
        f"- 通过（语义破坏 > 词序打乱）：{result['judgment3_surprise']['pass_mash_gt_shuffle']}",
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
