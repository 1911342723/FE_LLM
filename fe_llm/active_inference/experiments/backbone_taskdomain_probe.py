"""N2 step1b：在【有 headroom】的 9 领域任务 NLU 上对照句向量表示。

step1（`backbone_policy_probe`，action 5 分类，已饱和 ~0.957）显示 backbone 句向量
对已饱和任务无增益。本实验把同一「只换表示」的对照搬到一个**有 headroom 的域内任务**
——9 领域任务识别（`经验.md`/`task_nlu_eval` 记录 balanced acc ~0.857，因 flight/train
等相邻领域措辞重叠而未饱和）——检验核心命题：**更强表示是否只在有 headroom 的任务上
才显价值**。

唯一变量 = utterance 的向量表示；分类器结构 / 训练超参 / session 切分全部相同：
- charbow ：字符词袋（`train_task_nlu` 现基线）
- intent  ：自建 8.9M IntentEncoder 句向量
- backbone_mean / backbone_last：冻结 Qwen2.5-0.5B mean-pool / last-token（±PCA 维度匹配）

判定（balanced accuracy，margin=0.02）：
- backbone 明显 > intent → 在 headroom 任务上底座表示有价值（为底座找到用武之地）；
- backbone ≈ intent 且都 > charbow → 更强连续表示有用，但自建已够；
- 全部 ≈ → 即便有 headroom，表示也非瓶颈。

默认 dry-run；`--run` 真跑。
用法：python -m fe_llm.active_inference.experiments.backbone_taskdomain_probe --run
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from fe_llm.active_inference.experiments.backbone_policy_probe import (
    MARGIN,
    backbone_encode_pools,
    pca_project,
    verdict,
)
from fe_llm.active_inference.observation import Observation
from fe_llm.active_inference.perception import PerceptionEncoder
from fe_llm.config import get_device

DEFAULT_CORPUS = os.path.join("data", "dialogue", "teacher_task_oriented.jsonl")
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_REPORT_JSON = os.path.join("docs", "reports", "backbone_taskdomain_probe.json")
DEFAULT_REPORT_MD = os.path.join("docs", "reports", "backbone_taskdomain_probe.md")
DEFAULT_CACHE_DIR = os.path.join("checkpoints", "backbone_lm", "taskdomain_probe_cache")


def balanced_acc(pred: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    recalls = {}
    for c in range(n_classes):
        m = y == c
        if m.any():
            recalls[c] = float((pred[m] == y[m]).mean())
    bal = float(np.mean(list(recalls.values()))) if recalls else 0.0
    return {"balanced_accuracy": bal, "accuracy": float((pred == y).mean()), "per_class": recalls}


def intent_vectors(utterances: list[str], device: str) -> np.ndarray:
    """用自建 IntentEncoder（PerceptionEncoder）把每条 utterance 编成句向量。"""

    encoder = PerceptionEncoder(use_intent_model=True)
    vecs = []
    kinds = set()
    for i, text in enumerate(utterances):
        state = encoder.encode(Observation.from_text(text))
        vecs.append(np.asarray(state.vector, dtype=np.float32))
        kinds.add(state.encoder_kind)
        if (i + 1) % 2000 == 0:
            print(f"[probe] intent 编码 {i + 1}/{len(utterances)}", flush=True)
    if kinds != {"intent_model"}:
        print(f"[probe][警告] intent 编码出现非 intent_model 来源：{kinds}", flush=True)
    return np.vstack(vecs).astype(np.float32)


def char_bow(utterances: list[str], vocab: list[str]) -> np.ndarray:
    cidx = {c: i for i, c in enumerate(vocab)}
    out = np.zeros((len(utterances), len(vocab)), dtype=np.float32)
    for r, text in enumerate(utterances):
        for c in text:
            j = cidx.get(c)
            if j is not None:
                out[r, j] += 1.0
        out[r] /= max(len(text), 1)
    return out


def train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_classes: int,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
    device: str,
) -> dict:
    """与 train_task_nlu 同口径：全批 GD、class 加权、留出 balanced acc。"""

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    clf = nn.Sequential(nn.Linear(x_train.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, n_classes)).to(device)
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    w = torch.tensor(counts.sum() / (np.maximum(counts, 1) * n_classes), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)
    for _ in range(epochs):
        clf.train()
        opt.zero_grad()
        F.cross_entropy(clf(xt), yt, weight=w).backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        pred = clf(torch.tensor(x_val, dtype=torch.float32, device=device)).argmax(-1).cpu().numpy()
    return balanced_acc(pred, y_val, n_classes)


def load_split(corpus: str, seed: int, val_frac: float):
    rows = []
    with open(corpus, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    task_rows = [r for r in rows if r["action"] in ("ask_clarification", "answer")]
    domains = sorted({r["domain"] for r in task_rows})
    did = {d: i for i, d in enumerate(domains)}
    sessions = defaultdict(list)
    for r in task_rows:
        sessions[r["session_id"]].append(r)
    sids = sorted(sessions)
    rng = np.random.default_rng(seed)
    rng.shuffle(sids)
    cut = int(len(sids) * (1 - val_frac))
    train_rows = [r for s in sids[:cut] for r in sessions[s]]
    val_rows = [r for s in sids[cut:] for r in sessions[s]]
    vocab = sorted({c for r in task_rows for c in r["utterance"]})
    return task_rows, domains, did, train_rows, val_rows, vocab


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    task_rows, domains, did, train_rows, val_rows, vocab = load_split(args.corpus, args.seed, args.val_frac)
    n_classes = len(domains)
    y_train = np.array([did[r["domain"]] for r in train_rows], dtype=np.int64)
    y_val = np.array([did[r["domain"]] for r in val_rows], dtype=np.int64)
    utt_train = [r["utterance"] for r in train_rows]
    utt_val = [r["utterance"] for r in val_rows]
    print(
        f"[probe] device={device} domains={n_classes} train={len(train_rows)} val={len(val_rows)}",
        flush=True,
    )

    common = dict(n_classes=n_classes, hidden=args.hidden, epochs=args.epochs, lr=args.lr, seed=args.seed, device=device)

    # 1) char-BoW 基线（复现 train_task_nlu）。
    x_tr_bow = char_bow(utt_train, vocab)
    x_va_bow = char_bow(utt_val, vocab)
    print("[probe] 训练 charbow 臂 ...", flush=True)
    m_bow = train_classifier(x_tr_bow, y_train, x_va_bow, y_val, **common)
    print(f"[probe] charbow bal_acc={m_bow['balanced_accuracy']:.4f}", flush=True)

    # 2) 自建 IntentEncoder 句向量。
    print("[probe] intent 编码 ...", flush=True)
    iv_train = intent_vectors(utt_train, device)
    iv_val = intent_vectors(utt_val, device)
    print("[probe] 训练 intent 臂 ...", flush=True)
    m_intent = train_classifier(iv_train, y_train, iv_val, y_val, **common)
    print(f"[probe] intent bal_acc={m_intent['balanced_accuracy']:.4f}", flush=True)

    # 3) 冻结 Qwen2.5-0.5B：mean-pool 与 last-token（各带 PCA 维度匹配）。
    pools_train = backbone_encode_pools(
        utt_train,
        model_name=args.model_name,
        batch_size=args.bb_batch,
        max_length=args.max_length,
        device=device,
        cache_path=os.path.join(args.cache_dir, f"bb_train_s{args.seed}_n{len(utt_train)}.npz"),
    )
    pools_val = backbone_encode_pools(
        utt_val,
        model_name=args.model_name,
        batch_size=args.bb_batch,
        max_length=args.max_length,
        device=device,
        cache_path=os.path.join(args.cache_dir, f"bb_val_s{args.seed}_n{len(utt_val)}.npz"),
    )
    backbone_hidden = int(pools_train["mean"].shape[1])

    arms = [
        {"name": "charbow", "source": f"char-BoW ({len(vocab)}d)", "input_dim": len(vocab), **m_bow},
        {"name": "intent", "source": "IntentEncoder 8.9M", "input_dim": int(iv_train.shape[1]), **m_intent},
    ]
    intent_bal = m_intent["balanced_accuracy"]
    bow_bal = m_bow["balanced_accuracy"]
    best_backbone_bal = -1.0

    for pool in ("mean", "last"):
        vt, vv = pools_train[pool], pools_val[pool]
        print(f"[probe] 训练 backbone({pool},full) 臂 ...", flush=True)
        m_full = train_classifier(vt, y_train, vv, y_val, **common)
        print(f"[probe] backbone({pool},full) bal_acc={m_full['balanced_accuracy']:.4f}", flush=True)
        best_backbone_bal = max(best_backbone_bal, m_full["balanced_accuracy"])
        arms.append(
            {
                "name": f"backbone_{pool}_full",
                "source": f"Qwen2.5-0.5B {pool}-pool ({backbone_hidden}d)",
                "input_dim": int(vt.shape[1]),
                **m_full,
            }
        )
        if args.pca_dim > 0:
            pt, pv = pca_project(vt, vv, args.pca_dim)
            print(f"[probe] 训练 backbone({pool},pca{args.pca_dim}) 臂 ...", flush=True)
            m_pca = train_classifier(pt, y_train, pv, y_val, **common)
            print(f"[probe] backbone({pool},pca{args.pca_dim}) bal_acc={m_pca['balanced_accuracy']:.4f}", flush=True)
            arms.append(
                {
                    "name": f"backbone_{pool}_pca{args.pca_dim}",
                    "source": f"Qwen2.5-0.5B {pool}-pool→PCA{args.pca_dim}",
                    "input_dim": int(pt.shape[1]),
                    **m_pca,
                }
            )

    comparisons = [
        {"name": "intent vs charbow", "a": intent_bal, "b": bow_bal, "delta": intent_bal - bow_bal, "verdict": verdict(intent_bal - bow_bal)},
        {"name": "best_backbone(full) vs charbow", "a": best_backbone_bal, "b": bow_bal, "delta": best_backbone_bal - bow_bal, "verdict": verdict(best_backbone_bal - bow_bal)},
        {"name": "best_backbone(full) vs intent", "a": best_backbone_bal, "b": intent_bal, "delta": best_backbone_bal - intent_bal, "verdict": verdict(best_backbone_bal - intent_bal)},
    ]
    headline = comparisons[2]
    conclusion = _conclude(headline, comparisons, bow_bal, intent_bal, best_backbone_bal)

    payload = {
        "date": datetime.date.today().isoformat(),
        "task": "9-domain task NLU (headroom)",
        "corpus": args.corpus,
        "model_name": args.model_name,
        "backbone_hidden": backbone_hidden,
        "n_classes": n_classes,
        "domains": domains,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "hidden": args.hidden,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "margin": MARGIN,
        "arms": arms,
        "comparisons": comparisons,
        "conclusion": conclusion,
    }
    write_reports(payload, args.report_json, args.report_md)
    print("\n[probe] === 结论 ===", flush=True)
    print(conclusion, flush=True)
    print(f"[probe] 报告：{args.report_json} / {args.report_md}", flush=True)
    return payload


def _conclude(headline: dict, comparisons: list[dict], bow: float, intent: float, backbone: float) -> str:
    bb_vs_intent = headline["delta"]
    bb_vs_bow = comparisons[1]["delta"]
    intent_vs_bow = comparisons[0]["delta"]
    if bb_vs_intent >= MARGIN:
        head = (
            f"在【有 headroom】的 9 领域任务 NLU 上，backbone 句向量明显优于自建 IntentEncoder"
            f"（delta {bb_vs_intent:+.4f}）：与 step1 的饱和任务相反——**更强表示只在有 headroom 的任务上才显价值**，"
            "为底座找到了用武之地（可考虑把任务领域理解接 backbone 表示）。"
        )
    elif backbone - intent <= -MARGIN:
        head = (
            f"即便在有 headroom 的任务上，backbone 句向量仍明显劣于自建 IntentEncoder（delta {bb_vs_intent:+.4f}）："
            "底座通用表示不适配本任务，自建专训编码器更好。"
        )
    else:
        head = (
            f"在有 headroom 的 9 领域任务上，backbone 与自建 IntentEncoder 仍持平（delta {bb_vs_intent:+.4f}）："
            "表示不是该任务的瓶颈（领域措辞重叠需更多上下文/数据，而非更强句向量）。"
        )
    head += (
        f" 参照：charbow={bow:.4f}、intent={intent:.4f}、best_backbone={backbone:.4f}；"
        f"intent−charbow {intent_vs_bow:+.4f}（{comparisons[0]['verdict']}）、"
        f"backbone−charbow {bb_vs_bow:+.4f}（{comparisons[1]['verdict']}）。"
    )
    return head


def write_reports(payload: dict, json_path: str, md_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    lines = [
        "# N2 step1b · 有 headroom 任务上的句向量表示对照（9 领域任务 NLU）",
        "",
        f"更新日期：{payload['date']}",
        "",
        "> 唯一变量 = utterance 表示（charbow / 自建 IntentEncoder / 冻结 Qwen2.5-0.5B mean·last）。",
        "> 同分类器结构 / 同训练超参 / 同 session 切分；判定指标 = balanced accuracy。",
        "> 对照动机：step1 在已饱和的 action 分类上 backbone 无增益；本实验换到有 headroom 的任务验证表示价值。",
        "",
        "## 配置",
        "",
        f"- 语料：`{payload['corpus']}`（{payload['n_classes']} 领域：{', '.join(payload['domains'])}）",
        f"- 切分：按 session，train {payload['n_train']} / val {payload['n_val']}（seed={payload['seed']}）",
        f"- 底座：`{payload['model_name']}`（hidden={payload['backbone_hidden']}）",
        f"- 分类器：Linear→ReLU→Linear(hidden={payload['hidden']})，epochs={payload['epochs']}, lr={payload['lr']}, class_weights=on",
        f"- 判定边界：|delta| ≥ {payload['margin']}",
        "",
        "## 结果",
        "",
        "| 臂 | 表示来源 | 输入维度 | val accuracy | val balanced_acc |",
        "|---|---|---:|---:|---:|",
    ]
    for arm in payload["arms"]:
        lines.append(
            f"| {arm['name']} | {arm['source']} | {arm['input_dim']} | "
            f"{arm['accuracy']:.4f} | {arm['balanced_accuracy']:.4f} |"
        )
    lines += ["", "## 判定", ""]
    for cmp in payload["comparisons"]:
        lines.append(f"- **{cmp['name']}**：delta **{cmp['delta']:+.4f}** → **{cmp['verdict']}**")
    lines += ["", "## 结论", "", payload["conclusion"], ""]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bb-batch", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--pca-dim", type=int, default=128, help="维度匹配对照维度；0=关闭")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    ap.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    ap.add_argument("--run", action="store_true")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if not args.run:
        print(
            "[probe] dry-run：N2 step1b 有 headroom 任务(9领域NLU)上 charbow/intent/backbone 表示对照。\n"
            f"  语料={args.corpus}\n  底座={args.model_name}\n"
            f"  超参 hidden={args.hidden} epochs={args.epochs} seed={args.seed} pca_dim={args.pca_dim}\n"
            "  加 --run 真正执行。"
        )
        return
    run(args)


if __name__ == "__main__":
    main()
