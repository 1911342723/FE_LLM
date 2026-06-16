# -*- coding: utf-8 -*-
"""
fe_llm/world_model/m1_train_eval.py —— V2-M1 判定一：同预算对照（v1 单向量 vs v2 分层）
=====================================================================================
对应 docs/FE-LLM从0自建v2架构设计.md 第 8 节判定一。

公平对照：同一对话语料、同一损失配方（decode CE + 2·InfoNCE + 0.1·approach）、
同一超参与训练预算，唯一变量是架构：
    v1 = IntentLM（单全局意图向量 + EnergyDecoder）
    v2 = HierarchicalIntentLM（z_global + z_local 分层预测编码 + SlotEnergyDecoder）

判定一通过标准：v2 best decode_loss ≤ v1 best decode_loss × 1.02（分层不破坏生成）。

说明（诚实）：这是"架构级"对照而非单变量消融——v2 的解码器也多了 slot cross-attention。
报告会标注这一点。默认 dry-run，不自动重训；真正训练需 --run。

运行：python -m fe_llm.world_model.m1_train_eval --run --n 3000 --epochs 12
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

from fe_llm.config import get_device
from fe_llm.energy_lm.data.corpus import load_dialogues
from fe_llm.energy_lm.models.intent_model import IntentLM
from fe_llm.energy_lm.models.tokenizer import CharTokenizer
from fe_llm.energy_lm.training.intent_train import make_dataset
from fe_llm.world_model import HierarchicalIntentLM

REPORT_JSON = os.path.join("docs", "reports", "v2_m1_decode_compare.json")
REPORT_MD = os.path.join("docs", "reports", "v2_m1_decode_compare.md")

ENC_MAX = 24
DEC_MAX = 24
DIM = 256
ENC_DEPTH = 4
DEC_DEPTH = 6
NHEADS = 8
INTENT_DIM = 128
LAMBDA_APPROACH = 0.1


def _forward(model, p_ids, ri):
    """统一前向：返回 logits, h_intent, z_pred, z_target（v1/v2 各自取全局意图）。"""

    if isinstance(model, HierarchicalIntentLM):
        state = model.encoder(p_ids)
        with torch.no_grad():
            z_target = model.encoder(ri).z_global
        logits, h_intent = model.decoder(ri, state.z_global, state.z_local)
        return logits, h_intent, state.z_global, z_target
    z_pred = model.encoder(p_ids)
    with torch.no_grad():
        z_target = model.encoder(ri)
    logits, h_intent = model.decoder(ri, z_pred)
    return logits, h_intent, z_pred, z_target


def train_model(model, data, tok, args, device, tag: str) -> float:
    prompts, resp_in, resp_tgt = data
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = random.Random(args.seed)
    best = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        idx = list(range(len(prompts)))
        rng.shuffle(idx)
        tot_l, nt = 0.0, 0
        t0 = time.time()
        for i in range(0, len(idx), args.batch):
            ch = idx[i : i + args.batch]
            p_ids = torch.tensor(prompts[ch], device=device, dtype=torch.long)
            ri = torch.tensor(resp_in[ch], device=device, dtype=torch.long)
            rt = torch.tensor(resp_tgt[ch], device=device, dtype=torch.long)

            logits, h_intent, z_pred, z_target = _forward(model, p_ids, ri)
            mask = rt != tok.pad_id
            l_decode = F.cross_entropy(logits[mask], rt[mask])
            sim = F.cosine_similarity(z_pred.unsqueeze(1), z_target.unsqueeze(0), dim=-1)
            labels = torch.arange(sim.size(0), device=device)
            l_intent = F.cross_entropy(sim / 0.07, labels)
            dist = torch.norm(h_intent - z_pred.detach().unsqueeze(1), dim=-1)
            violations = F.relu(dist[:, 1:] - dist[:, :-1])
            l_approach = violations[mask[:, 1:]].mean() if mask[:, 1:].any() else torch.tensor(0.0, device=device)
            loss = l_decode + 2.0 * l_intent + LAMBDA_APPROACH * l_approach

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_l += float(l_decode.detach()) * int(mask.sum())
            nt += int(mask.sum().item())
        sched.step()
        avg_l = tot_l / max(1, nt)
        best = min(best, avg_l)
        print(f"[{tag}] ep {ep:3d} decode_loss={avg_l:.4f} 用时={time.time() - t0:.0f}s", flush=True)
    return best


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V2-M1 judgment 1: v1 single-vector vs v2 hierarchical decode_loss.")
    parser.add_argument("--run", action="store_true", help="真正训练对照；默认只 dry-run。")
    parser.add_argument("--n", type=int, default=3000, help="取前 n 条对话，0=全部")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--relax-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[v2-m1] dry-run：未训练。")
    print(f"[v2-m1] n/epochs/batch = {args.n}/{args.epochs}/{args.batch}")
    print("[v2-m1] 对照：v1 IntentLM(单向量) vs v2 HierarchicalIntentLM(分层)，同预算同损失。")
    print("[v2-m1] 真正训练请显式追加 --run。")


def run(args: argparse.Namespace) -> dict:
    device = get_device() if args.device == "auto" else args.device
    dias = load_dialogues()
    if args.n > 0:
        dias = dias[: args.n]
    # 直接从对话字符建字表（不用 build_tokenizer：其相对 import 在包重构后已失效，
    # 指向不存在的 models.corpus；改用 translation_train 同款构造方式）。
    chars = sorted({c for p, r in dias for c in p + r})
    tok = CharTokenizer(chars)
    data = make_dataset(tok, dias, ENC_MAX, DEC_MAX)
    print(f"[v2-m1] device={device} 对话={len(data[0])} 字表={tok.vocab_size}")

    torch.manual_seed(args.seed)
    v1 = IntentLM(
        vocab_size=tok.vocab_size, enc_max=ENC_MAX, dec_max=DEC_MAX, dim=DIM,
        enc_depth=ENC_DEPTH, dec_depth=DEC_DEPTH, n_heads=NHEADS, intent_dim=INTENT_DIM,
    ).to(device)
    v1_params = sum(p.numel() for p in v1.parameters()) / 1e6
    v1_loss = train_model(v1, data, tok, args, device, tag="v1")

    torch.manual_seed(args.seed)
    v2 = HierarchicalIntentLM(
        vocab_size=tok.vocab_size, enc_max=ENC_MAX, dec_max=DEC_MAX, dim=DIM,
        enc_depth=ENC_DEPTH, dec_depth=DEC_DEPTH, n_heads=NHEADS, intent_dim=INTENT_DIM,
        relax_steps=args.relax_steps,
    ).to(device)
    v2_params = sum(p.numel() for p in v2.parameters()) / 1e6
    v2_loss = train_model(v2, data, tok, args, device, tag="v2")

    verdict = "PASS: v2 不劣于 v1" if v2_loss <= v1_loss * 1.02 else "FAIL: v2 明显劣于 v1"
    result = {
        "dialogues": len(data[0]),
        "vocab_size": tok.vocab_size,
        "epochs": args.epochs,
        "batch": args.batch,
        "relax_steps": args.relax_steps,
        "v1_params_m": round(v1_params, 2),
        "v2_params_m": round(v2_params, 2),
        "v1_decode_loss": round(v1_loss, 4),
        "v2_decode_loss": round(v2_loss, 4),
        "ratio_v2_over_v1": round(v2_loss / max(v1_loss, 1e-9), 4),
        "verdict": verdict,
        "note": "架构级对照（v2 解码器多 slot cross-attention），非单变量消融。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# FE-LLM V2-M1 判定一：v1 单向量 vs v2 分层（同预算 decode_loss）",
        "",
        f"- 判定：**{verdict}**",
        f"- 对话条数：{result['dialogues']}，字表：{result['vocab_size']}，epochs：{result['epochs']}",
        f"- 参数量：v1 {result['v1_params_m']}M / v2 {result['v2_params_m']}M",
        f"- v1 best decode_loss：{result['v1_decode_loss']}",
        f"- v2 best decode_loss：{result['v2_decode_loss']}",
        f"- v2/v1 比值：{result['ratio_v2_over_v1']}",
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
