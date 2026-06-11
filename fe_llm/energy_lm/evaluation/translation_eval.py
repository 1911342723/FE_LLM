# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/translation_eval.py —— 翻译版 IntentLM 泛化评测
================================================================
在完全未见的 opus-100 验证集上检验三件事：
1. 泛化：未见中文句能否重建出可读英文（字符级小模型口径）；
2. 能量机制：逐字残余能量是否朝意图收敛；
3. 决策分歧：hybrid 选字与纯 logit argmax 的差异是否存在。

指标（字符级小模型的诚实口径，不与 SOTA 翻译比）：
    - word-F1：生成英文与参考英文按词集合的 F1；
    - char-1gram F1；
    - exact match；
    - 能量整体下降比例。

运行：python fe_llm/energy_lm/translation_eval.py [--limit 200]
输出：docs/reports/translation_eval.{md,json}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from fe_llm.config import get_device
from fe_llm.energy_lm.models.intent_model import IntentLM
from fe_llm.energy_lm.models.tokenizer import CharTokenizer
from fe_llm.energy_lm.training.translation_train import CKPT_PATH, CKPT_TOK, DEC_MAX, ENC_MAX

REPORT_JSON = os.path.join("docs", "reports", "translation_eval.json")
REPORT_MD = os.path.join("docs", "reports", "translation_eval.md")

# 额外的手写探针：训练集之外、口语化的句子，直观感受泛化。
PROBES = ["你好", "我爱你", "今天天气很好", "我有点累", "谢谢你的帮助", "我们明天见"]


class Translator:
    """加载翻译版 IntentLM，提供 hybrid/logit 两种解码。"""

    def __init__(self, device: str | None = None):
        self.device = device or get_device()
        self.model = IntentLM.load(CKPT_PATH, map_location=self.device).to(self.device).eval()
        self.tok = CharTokenizer.load(CKPT_TOK)

    @torch.no_grad()
    def translate(self, zh: str, decode_mode: str = "hybrid", top_k: int = 8,
                  alpha: float = 1.0, max_new: int = DEC_MAX - 2):
        tok = self.tok
        p_ids = tok.encode(zh)[: ENC_MAX - 1] + [tok.sep_id]
        p_ids = p_ids + [tok.pad_id] * (ENC_MAX - len(p_ids))
        p_tensor = torch.tensor([p_ids], device=self.device)
        z_intent = self.model.encoder(p_tensor)

        gen_ids = [tok.bos_id]
        energies: list[float] = []
        disagreement = 0
        steps = 0
        for _ in range(max_new):
            dec_input = gen_ids + [tok.pad_id] * (DEC_MAX - len(gen_ids))
            dec_tensor = torch.tensor([dec_input[:DEC_MAX]], device=self.device)
            logits, h_intent = self.model.decoder(dec_tensor, z_intent)
            pos = len(gen_ids) - 1
            logit_row = logits[0, pos].clone()
            for sp in (tok.mask_id, tok.bos_id, tok.sep_id, tok.pad_id, tok.unk_id):
                logit_row[sp] = -1e9
            energies.append(float(torch.norm(h_intent[0, pos] - z_intent[0])))

            tid_prob = int(logit_row.argmax())
            if decode_mode == "hybrid" and len(gen_ids) < DEC_MAX:
                tid = self._hybrid_choice(gen_ids, z_intent, logit_row, top_k, alpha)
            else:
                tid = tid_prob
            steps += 1
            if tid != tid_prob:
                disagreement += 1
            if tid == tok.eos_id:
                break
            gen_ids.append(tid)
        text = "".join(tok.id_to_tok[t] for t in gen_ids[1:])
        return text, {
            "energies": energies,
            "disagreement_steps": disagreement,
            "total_steps": steps,
        }

    @torch.no_grad()
    def _hybrid_choice(self, gen_ids, z_intent, logit_row, top_k: int, alpha: float) -> int:
        tok = self.tok
        topk = torch.topk(logit_row, k=min(top_k, logit_row.numel()))
        cand_ids = topk.indices.tolist()
        pos = len(gen_ids)
        batch = []
        for cand in cand_ids:
            seq = gen_ids + [cand]
            batch.append((seq + [tok.pad_id] * (DEC_MAX - len(seq)))[:DEC_MAX])
        batch_tensor = torch.tensor(batch, device=self.device)
        z_batch = z_intent.expand(len(cand_ids), -1)
        _, h_intent = self.model.decoder(batch_tensor, z_batch)
        dists = torch.norm(h_intent[:, pos] - z_batch, dim=-1)
        log_probs = torch.log_softmax(logit_row, dim=-1)[topk.indices]
        dist_norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)
        return int(cand_ids[int((log_probs - alpha * dist_norm).argmax())])


def word_f1(pred: str, ref: str) -> float:
    p, r = set(pred.split()), set(ref.split())
    if not p or not r:
        return 0.0
    overlap = len(p & r)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def char_f1(pred: str, ref: str) -> float:
    p, r = set(pred.replace(" ", "")), set(ref.replace(" ", ""))
    if not p or not r:
        return 0.0
    overlap = len(p & r)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="评测的验证对数量")
    args = ap.parse_args()

    val_path = os.path.join("data", "translation", "opus100_val.jsonl")
    pairs = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pairs.append((item["zh"], item["en"]))
    pairs = pairs[: args.limit]

    translator = Translator()
    rows = []
    for zh, ref in pairs:
        pred, info = translator.translate(zh)
        energies = info["energies"]
        rows.append(
            {
                "zh": zh,
                "ref": ref,
                "pred": pred,
                "word_f1": round(word_f1(pred, ref), 4),
                "char_f1": round(char_f1(pred, ref), 4),
                "exact": pred.strip() == ref.strip(),
                "energy_descent": bool(energies and energies[-1] < energies[0]),
                "disagreement_rate": round(info["disagreement_steps"] / max(info["total_steps"], 1), 4),
            }
        )

    probe_rows = []
    for zh in PROBES:
        pred, _ = translator.translate(zh)
        probe_rows.append({"zh": zh, "pred": pred})

    n = len(rows)
    summary = {
        "val_pairs": n,
        "mean_word_f1": round(sum(r["word_f1"] for r in rows) / max(n, 1), 4),
        "mean_char_f1": round(sum(r["char_f1"] for r in rows) / max(n, 1), 4),
        "exact_match": sum(1 for r in rows if r["exact"]),
        "word_f1_ge_0.5": sum(1 for r in rows if r["word_f1"] >= 0.5),
        "word_f1_eq_0": sum(1 for r in rows if r["word_f1"] == 0.0),
        "energy_descent_rate": round(sum(1 for r in rows if r["energy_descent"]) / max(n, 1), 4),
        "mean_disagreement_rate": round(sum(r["disagreement_rate"] for r in rows) / max(n, 1), 4),
    }

    lines = [
        "# FE-LLM 翻译泛化评测（opus-100 zh→en，IntentLM 架构）",
        "",
        "在完全未见的验证集上检验：意图弛豫 + 能量递减解码能否做跨语言重建。",
        "字符级 8M 小模型口径，目标是验证架构泛化性，不与 SOTA 翻译对比。",
        "",
        f"- 未见验证对：{summary['val_pairs']}",
        f"- 平均 word-F1：{summary['mean_word_f1']:.3f}（≥0.5 的占 {summary['word_f1_ge_0.5']}/{n}，=0 的占 {summary['word_f1_eq_0']}/{n}）",
        f"- 平均 char-F1：{summary['mean_char_f1']:.3f}",
        f"- exact match：{summary['exact_match']}/{n}",
        f"- 能量整体下降比例：{summary['energy_descent_rate']:.0%}",
        f"- hybrid 选字分歧率：{summary['mean_disagreement_rate']:.1%}",
        "",
        "## 手写探针（训练集外）",
        "",
        "| 中文 | 生成英文 |",
        "|---|---|",
    ]
    for row in probe_rows:
        lines.append(f"| {row['zh']} | {row['pred']} |")
    lines.extend(["", "## 验证集样例（前 20 条）", "", "| 中文 | 参考 | 生成 | word-F1 |", "|---|---|---|---:|"])
    for row in rows[:20]:
        lines.append(f"| {row['zh']} | {row['ref']} | {row['pred']} | {row['word_f1']:.2f} |")
    report = "\n".join(lines) + "\n"
    print(report)

    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows, "probes": probe_rows}, f, ensure_ascii=False, indent=2)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
