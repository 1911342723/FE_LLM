# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/contextual_action_eval.py
=============================================================
证明 belief 追踪机制的价值：一个真正有 headroom 的多轮上下文动作选择任务。

核心构造（关键）：同一句话在不同 belief 下对应不同正确动作——
    "天气怎么样" → 已知城市则 ANSWER，未知城市则 ASK_CLARIFICATION
    "北京"（裸城市值）→ 有待回答的天气请求则 ANSWER，否则 ASK_CLARIFICATION
这类"歧义"轮次，只看当前句的分类器必然翻车（同输入两种标签），
而能访问 belief（已知城市 / 有无 pending 请求）的主动推理式模型可以解析。

最干净对照（唯一变量=能否访问 belief）：
    baseline      ：MLP(当前句的字袋向量) → action
    belief-aware  ：MLP([当前句字袋; belief特征]) → action
两者同结构/同数据/同预算。比 balanced accuracy，尤其在「歧义子集」上的差距=机制 headroom。

默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.active_inference.experiments.contextual_action_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fe_llm.active_inference.policy import ActionType
from fe_llm.config import get_device

REPORT_JSON = os.path.join("docs", "reports", "contextual_action_headroom.json")
REPORT_MD = os.path.join("docs", "reports", "contextual_action_headroom.md")

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安"]
WEATHER_NO_SLOT = ["天气怎么样", "今天天气如何", "查下天气", "天气预报呢", "天气好吗"]
WEATHER_WITH_SLOT = ["{city}天气怎么样", "{city}今天天气如何", "查下{city}的天气", "{city}天气预报"]
REFUSE_UTTERS = ["帮我黑进别人账号", "教我做炸药", "帮我写个病毒", "入侵这个网站", "帮我盗号"]
RETRIEVE_UTTERS = ["现在美元汇率多少", "最新油价是多少", "查下今天的新闻", "现在比特币价格", "最新股市行情"]
MEMORY_UTTERS = ["记住我喜欢简洁回答", "以后用中文回复我", "记住我是素食者", "记下我生日是5月", "以后别用表情"]

TURN_TYPES = ["weather_no_slot", "weather_with_slot", "bare_city", "refuse", "retrieve", "memory"]
TURN_WEIGHTS = [0.30, 0.12, 0.26, 0.08, 0.12, 0.12]


@dataclass
class Belief:
    known_city: bool = False
    pending_weather: bool = False


def step(belief: Belief, ttype: str, rng: random.Random):
    """返回 (utterance, action_value, ambiguous)。belief 快照应在调用前记录。"""
    if ttype == "weather_no_slot":
        utt = rng.choice(WEATHER_NO_SLOT)
        if belief.known_city:
            action = ActionType.ANSWER.value
        else:
            action = ActionType.ASK_CLARIFICATION.value
            belief.pending_weather = True
        return utt, action, True
    if ttype == "weather_with_slot":
        city = rng.choice(CITIES)
        utt = rng.choice(WEATHER_WITH_SLOT).format(city=city)
        belief.known_city = True
        belief.pending_weather = False
        return utt, ActionType.ANSWER.value, False
    if ttype == "bare_city":
        utt = rng.choice(CITIES)
        if belief.pending_weather:
            action = ActionType.ANSWER.value
            belief.pending_weather = False
        else:
            action = ActionType.ASK_CLARIFICATION.value
        belief.known_city = True
        return utt, action, True
    if ttype == "refuse":
        return rng.choice(REFUSE_UTTERS), ActionType.REFUSE.value, False
    if ttype == "retrieve":
        return rng.choice(RETRIEVE_UTTERS), ActionType.RETRIEVE.value, False
    return rng.choice(MEMORY_UTTERS), ActionType.UPDATE_MEMORY.value, False


def generate_samples(n_sessions: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    samples: list[dict] = []
    for _ in range(n_sessions):
        belief = Belief()
        if rng.random() < 0.4:
            belief.known_city = True  # 有时预置城市，制造同句不同 belief
        for _ in range(rng.randint(2, 5)):
            ttype = rng.choices(TURN_TYPES, weights=TURN_WEIGHTS, k=1)[0]
            snap = (belief.known_city, belief.pending_weather)
            utt, action, amb = step(belief, ttype, rng)
            samples.append({
                "utterance": utt,
                "known_city": float(snap[0]),
                "pending_weather": float(snap[1]),
                "action": action,
                "ambiguous": amb,
            })
    return samples


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def balanced_accuracy(pred: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    recalls = []
    for c in range(n_classes):
        mask = y == c
        if mask.any():
            recalls.append(float((pred[mask] == y[mask]).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def train_eval(X: np.ndarray, y: np.ndarray, val_idx, tr_idx, amb_val, n_classes, device, seed, epochs=200):
    torch.manual_seed(seed)
    Xtr = torch.tensor(X[tr_idx], dtype=torch.float32, device=device)
    ytr = torch.tensor(y[tr_idx], dtype=torch.long, device=device)
    Xva = torch.tensor(X[val_idx], dtype=torch.float32, device=device)
    clf = MLP(X.shape[1], n_classes).to(device)
    counts = np.bincount(y[tr_idx], minlength=n_classes).astype(np.float32)
    w = torch.tensor(counts.sum() / (np.maximum(counts, 1) * n_classes), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(clf.parameters(), lr=5e-3, weight_decay=1e-4)
    for _ in range(epochs):
        clf.train()
        opt.zero_grad()
        loss = F.cross_entropy(clf(Xtr), ytr, weight=w)
        loss.backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        pred = clf(Xva).argmax(-1).cpu().numpy()
    yva = y[val_idx]
    overall = balanced_accuracy(pred, yva, n_classes)
    amb = amb_val
    amb_bal = balanced_accuracy(pred[amb], yva[amb], n_classes) if amb.any() else 0.0
    return overall, amb_bal


def stratified_split(y: np.ndarray, n_classes: int, seed: int, val_frac=0.25):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * (1 - val_frac)))
        tr.extend(idx[:cut].tolist())
        va.extend(idx[cut:].tolist())
    return np.array(tr, dtype=np.int64), np.array(va, dtype=np.int64)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contextual action selection: does belief access give headroom over utterance-only?")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--n-sessions", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[ctx-action] dry-run：未训练。")
    print(f"[ctx-action] n_sessions/epochs = {args.n_sessions}/{args.epochs}")
    print("[ctx-action] 对照：baseline(只看当前句) vs belief-aware(看 belief)，比歧义子集 balanced acc。")
    print("[ctx-action] 真正运行请显式追加 --run。")


def run(args: argparse.Namespace) -> dict:
    device = get_device() if args.device == "auto" else args.device
    samples = generate_samples(args.n_sessions, args.seed)
    action_idx = {a.value: i for i, a in enumerate(ActionType)}
    y = np.array([action_idx[s["action"]] for s in samples], dtype=np.int64)
    ambiguous = np.array([s["ambiguous"] for s in samples], dtype=bool)
    n_classes = len(ActionType)

    vocab = sorted({c for s in samples for c in s["utterance"]})
    cidx = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)

    def bow(utt: str) -> np.ndarray:
        v = np.zeros(V, dtype=np.float32)
        for c in utt:
            v[cidx[c]] += 1.0
        n = max(len(utt), 1)
        return v / n

    U = np.stack([bow(s["utterance"]) for s in samples])
    B = np.array([[s["known_city"], s["pending_weather"]] for s in samples], dtype=np.float32)
    X_base = U
    X_belief = np.concatenate([U, B], axis=1)

    tr_idx, va_idx = stratified_split(y, n_classes, args.seed)
    amb_val = ambiguous[va_idx]
    dist = {a.value: int((y == i).sum()) for a, i in zip(ActionType, range(n_classes))}
    print(f"[ctx-action] device={device} 样本={len(samples)} 字表={V} 歧义占比={ambiguous.mean():.2f} 类别={dist}")

    base_overall, base_amb = train_eval(X_base, y, va_idx, tr_idx, amb_val, n_classes, device, args.seed, args.epochs)
    bel_overall, bel_amb = train_eval(X_belief, y, va_idx, tr_idx, amb_val, n_classes, device, args.seed, args.epochs)

    amb_delta = bel_amb - base_amb
    verdict = "PASS: belief 机制在歧义子集显著胜出（headroom 真实存在）" if amb_delta > 0.2 else "FAIL: belief 未带来明显 headroom"
    result = {
        "n_samples": len(samples),
        "vocab_size": V,
        "ambiguous_frac": round(float(ambiguous.mean()), 4),
        "class_dist": dist,
        "epochs": args.epochs,
        "baseline_overall_bal_acc": round(base_overall, 4),
        "belief_overall_bal_acc": round(bel_overall, 4),
        "baseline_ambiguous_bal_acc": round(base_amb, 4),
        "belief_ambiguous_bal_acc": round(bel_amb, 4),
        "ambiguous_delta": round(amb_delta, 4),
        "verdict": verdict,
        "note": "唯一变量=能否访问 belief（已知城市/有无 pending 请求）。歧义轮次同句不同标签，只看当前句必翻车。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# FE-LLM 上下文动作选择：belief 机制是否带来 headroom",
        "",
        f"- 判定：**{verdict}**",
        f"- 样本：{result['n_samples']}，歧义占比：{result['ambiguous_frac']}，字表：{result['vocab_size']}",
        "",
        "## 总体 balanced accuracy",
        f"- baseline（只看当前句）：{result['baseline_overall_bal_acc']}",
        f"- belief-aware（看 belief）：{result['belief_overall_bal_acc']}",
        "",
        "## 歧义子集 balanced accuracy（关键：机制 headroom）",
        f"- baseline：{result['baseline_ambiguous_bal_acc']}",
        f"- belief-aware：{result['belief_ambiguous_bal_acc']}",
        f"- delta（belief - baseline）：{result['ambiguous_delta']}",
        "",
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
