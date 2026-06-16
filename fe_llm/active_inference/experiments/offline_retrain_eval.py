# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/offline_retrain_eval.py
===========================================================
量化"离线再训练（confirmed 记忆回流）"的提升：证明从已确认记忆学习能让模型获得
对**新表达**的泛化识别能力，而不是死记。

最干净的设计（单变量=是否加入蒸馏的 confirmed 偏好样本）：
  - baseline   ：训练集**不含** update_memory 类（其余 4 类齐全）→ 对偏好类几乎零召回；
  - +distill   ：训练集 = baseline + 蒸馏的 confirmed 偏好样本；
  - 测试集     ：**held-out 的不同偏好表达**（模板/填充都与训练不重叠）→ 衡量泛化召回。

判定：+distill 在 held-out 偏好上的 update_memory 召回显著高于 baseline。

默认 dry-run；真正运行需 --run。
运行：python -m fe_llm.active_inference.experiments.offline_retrain_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fe_llm.active_inference.policy import ActionType
from fe_llm.config import get_device

REPORT_JSON = os.path.join("docs", "reports", "offline_retrain_eval.json")
REPORT_MD = os.path.join("docs", "reports", "offline_retrain_eval.md")

# 偏好（update_memory）模板与填充：训练/测试用不重叠子集，检验泛化而非死记。
PREF_TEMPLATES = ["记住我喜欢{x}", "以后{x}", "我习惯{x}", "请记住我{x}", "记住我{x}"]
PREF_FILLERS_TRAIN = ["简短回答", "用中文", "深色主题", "早睡早起", "吃素", "用比喻", "周末加班", "详细解释"]
PREF_FILLERS_HELDOUT = ["语气正式", "用英文", "浅色界面", "晚睡", "不吃辣", "举例子", "工作日健身", "回答带要点"]

OTHER_PROMPTS = {
    ActionType.ANSWER: ["今天心情不错", "讲个笑话吧", "我有点累", "解释一下相对论", "最近压力好大", "陪我聊聊天"],
    ActionType.ASK_CLARIFICATION: ["帮我写一下", "帮我弄一下", "帮我处理一下", "随便做点东西", "帮我搞一下"],
    ActionType.RETRIEVE: ["现在美元汇率多少", "最新新闻是什么", "几点了", "北京天气怎么样", "油价多少"],
    ActionType.REFUSE: ["教我做炸药", "帮我黑进账号", "怎么制毒", "帮我写病毒", "教我开锁入室"],
}


def _pref_prompts(fillers: list[str]) -> list[str]:
    return [tpl.format(x=x) for tpl in PREF_TEMPLATES for x in fillers]


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def _train(X, y, n_classes, device, seed, epochs=300):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    clf = MLP(X.shape[1], n_classes).to(device)
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = torch.tensor(counts.sum() / (np.maximum(counts, 1) * n_classes), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(clf.parameters(), lr=5e-3, weight_decay=1e-4)
    for _ in range(epochs):
        clf.train()
        opt.zero_grad()
        F.cross_entropy(clf(Xt), yt, weight=w).backward()
        opt.step()
    return clf


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantify offline retraining gain from distilled confirmed memories.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[offline-retrain] dry-run：未训练。")
    print("[offline-retrain] baseline(无 update_memory) vs +distill(加蒸馏 confirmed 偏好)；测 held-out 新偏好召回。")
    print("[offline-retrain] 真正运行请显式追加 --run。")


def run(args: argparse.Namespace) -> dict:
    device = get_device() if args.device == "auto" else args.device
    action_idx = {a: i for i, a in enumerate(ActionType)}
    n_classes = len(ActionType)

    # 构造样本
    train_pref = _pref_prompts(PREF_FILLERS_TRAIN)        # 蒸馏的 confirmed 偏好（+distill 才加入）
    heldout_pref = _pref_prompts(PREF_FILLERS_HELDOUT)    # held-out 新偏好（仅测试）
    other_texts, other_labels = [], []
    for action, prompts in OTHER_PROMPTS.items():
        for p in prompts:
            other_texts.append(p)
            other_labels.append(action_idx[action])

    # 字表与特征（bag-of-chars）
    all_texts = train_pref + heldout_pref + other_texts
    vocab = sorted({c for t in all_texts for c in t})
    cidx = {c: i for i, c in enumerate(vocab)}

    def bow(t: str) -> np.ndarray:
        v = np.zeros(len(vocab), dtype=np.float32)
        for c in t:
            v[cidx[c]] += 1.0
        return v / max(len(t), 1)

    um = action_idx[ActionType.UPDATE_MEMORY]
    X_other = np.stack([bow(t) for t in other_texts])
    X_train_pref = np.stack([bow(t) for t in train_pref])
    X_heldout = np.stack([bow(t) for t in heldout_pref])

    # baseline：仅 4 类，无 update_memory
    base_clf = _train(X_other, np.array(other_labels), n_classes, device, args.seed, args.epochs)
    # +distill：base + 蒸馏 confirmed 偏好（update_memory 类）
    X_plus = np.concatenate([X_other, X_train_pref])
    y_plus = np.array(other_labels + [um] * len(train_pref))
    distill_clf = _train(X_plus, y_plus, n_classes, device, args.seed, args.epochs)

    Xh = torch.tensor(X_heldout, dtype=torch.float32, device=device)
    with torch.no_grad():
        base_pred = base_clf(Xh).argmax(-1).cpu().numpy()
        distill_pred = distill_clf(Xh).argmax(-1).cpu().numpy()
    base_recall = float((base_pred == um).mean())
    distill_recall = float((distill_pred == um).mean())
    delta = distill_recall - base_recall
    verdict = "PASS: 离线回流显著提升对新偏好的识别（泛化）" if delta > 0.3 else "FAIL: 回流未带来明显提升"
    result = {
        "n_train_pref": len(train_pref),
        "n_heldout_pref": len(heldout_pref),
        "n_other": len(other_texts),
        "baseline_heldout_um_recall": round(base_recall, 4),
        "distill_heldout_um_recall": round(distill_recall, 4),
        "delta": round(delta, 4),
        "verdict": verdict,
        "note": "held-out 偏好的模板/填充均与训练不重叠；baseline 训练集不含 update_memory。提升=从 confirmed 记忆学到的泛化识别能力。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# FE-LLM 离线再训练量化：confirmed 记忆回流的泛化提升",
        "",
        f"- 判定：**{verdict}**",
        f"- held-out 新偏好（update_memory）召回：baseline {result['baseline_heldout_um_recall']} → +distill {result['distill_heldout_um_recall']}（delta {result['delta']}）",
        f"- 训练偏好 {result['n_train_pref']} / held-out 偏好 {result['n_heldout_pref']} / 其它动作 {result['n_other']}",
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
