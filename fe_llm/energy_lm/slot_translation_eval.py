# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/slot_translation_eval.py —— M2 判定评测
========================================================
与 translation_eval.py 同一验证集、同一指标口径，唯一变量是槽位化意图。

判定（草案第 5 节 M2）：
    未见句 mean word-F1 相对单向量版（0.073）显著提升（目标 ≥0.3）
    → 瓶颈在表达结构，架构成立，值得上规模；
    否则 → 转预训练底座路线。

运行：python fe_llm/energy_lm/slot_translation_eval.py [--limit 200]
输出：docs/reports/slot_translation_eval.{md,json}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from fe_llm.config import get_device
from fe_llm.energy_lm.slot_intent_model import SlotIntentLM
from fe_llm.energy_lm.slot_translation_train import CKPT_PATH, CKPT_TOK
from fe_llm.energy_lm.tokenizer import CharTokenizer
from fe_llm.energy_lm.translation_train import DEC_MAX, ENC_MAX
from fe_llm.energy_lm.translation_eval import PROBES, char_f1, word_f1

REPORT_JSON = os.path.join("docs", "reports", "slot_translation_eval.json")
REPORT_MD = os.path.join("docs", "reports", "slot_translation_eval.md")

# 单向量版基线（docs/reports/translation_eval.json），用于 M2 对比。
BASELINE_WORD_F1 = 0.0735


class SlotTranslator:
    def __init__(self, device: str | None = None):
        self.device = device or get_device()
        self.model = SlotIntentLM.load(CKPT_PATH, map_location=self.device).to(self.device).eval()
        self.tok = CharTokenizer.load(CKPT_TOK)

    @torch.no_grad()
    def translate(self, zh: str, max_new: int = DEC_MAX - 2):
        tok = self.tok
        p_ids = tok.encode(zh)[: ENC_MAX - 1] + [tok.sep_id]
        p_ids = p_ids + [tok.pad_id] * (ENC_MAX - len(p_ids))
        p_tensor = torch.tensor([p_ids], device=self.device)
        z_global, slots, salience = self.model.encoder(p_tensor)

        gen_ids = [tok.bos_id]
        energies: list[float] = []
        coverage_trace: list[float] = []
        for _ in range(max_new):
            dec_input = gen_ids + [tok.pad_id] * (DEC_MAX - len(gen_ids))
            dec_tensor = torch.tensor([dec_input[:DEC_MAX]], device=self.device)
            logits, h_intent = self.model.decoder(dec_tensor, z_global, slots)
            pos = len(gen_ids) - 1
            logit_row = logits[0, pos].clone()
            for sp in (tok.mask_id, tok.bos_id, tok.sep_id, tok.pad_id, tok.unk_id):
                logit_row[sp] = -1e9
            # 复合能量记录：全局残余 + salience 加权槽位覆盖（可溯源量）。
            h_seq = h_intent[0, : pos + 1]                            # (i+1, d)
            global_e = float(torch.norm(h_intent[0, pos] - z_global[0]))
            cov = float(
                (salience[0] * torch.cdist(slots[0], h_seq).min(dim=-1).values).sum()
            )
            energies.append(global_e)
            coverage_trace.append(cov)

            tid = int(logit_row.argmax())
            if tid == tok.eos_id:
                break
            gen_ids.append(tid)
        text = "".join(tok.id_to_tok[t] for t in gen_ids[1:])
        return text, {"energies": energies, "coverage": coverage_trace}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    val_path = os.path.join("data", "translation", "opus100_val.jsonl")
    pairs = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pairs.append((item["zh"], item["en"]))
    pairs = pairs[: args.limit]

    translator = SlotTranslator()
    rows = []
    for zh, ref in pairs:
        pred, info = translator.translate(zh)
        energies = info["energies"]
        coverage = info["coverage"]
        rows.append({
            "zh": zh,
            "ref": ref,
            "pred": pred,
            "word_f1": round(word_f1(pred, ref), 4),
            "char_f1": round(char_f1(pred, ref), 4),
            "exact": pred.strip() == ref.strip(),
            "energy_descent": bool(energies and energies[-1] < energies[0]),
            "coverage_descent": bool(coverage and coverage[-1] < coverage[0]),
        })

    probe_rows = []
    for zh in PROBES:
        pred, _ = translator.translate(zh)
        probe_rows.append({"zh": zh, "pred": pred})

    n = len(rows)
    unique_outputs = len({r["pred"] for r in rows})
    mean_wf1 = round(sum(r["word_f1"] for r in rows) / max(n, 1), 4)
    summary = {
        "val_pairs": n,
        "mean_word_f1": mean_wf1,
        "baseline_word_f1": BASELINE_WORD_F1,
        "improvement_ratio": round(mean_wf1 / BASELINE_WORD_F1, 2) if BASELINE_WORD_F1 else None,
        "mean_char_f1": round(sum(r["char_f1"] for r in rows) / max(n, 1), 4),
        "exact_match": sum(1 for r in rows if r["exact"]),
        "word_f1_ge_0.5": sum(1 for r in rows if r["word_f1"] >= 0.5),
        "word_f1_eq_0": sum(1 for r in rows if r["word_f1"] == 0.0),
        "unique_outputs": unique_outputs,
        "energy_descent_rate": round(sum(1 for r in rows if r["energy_descent"]) / max(n, 1), 4),
        "coverage_descent_rate": round(sum(1 for r in rows if r["coverage_descent"]) / max(n, 1), 4),
        "m2_verdict": "PASS (>=0.3)" if mean_wf1 >= 0.3 else (
            "PARTIAL (significant improvement)" if mean_wf1 >= 2 * BASELINE_WORD_F1 else "FAIL"
        ),
    }

    lines = [
        "# FE-LLM M2 判定评测：槽位化意图 vs 单向量意图（opus-100 zh→en）",
        "",
        "同数据、同规模、同训练预算，唯一变量是意图表示结构。",
        "",
        f"- 未见验证对：{n}",
        f"- **mean word-F1：{summary['mean_word_f1']:.3f}**（单向量基线 {BASELINE_WORD_F1}，{summary['improvement_ratio']}x）",
        f"- 输出多样性：{unique_outputs}/{n} 种（单向量版 67/200）",
        f"- word-F1≥0.5：{summary['word_f1_ge_0.5']}/{n}；=0：{summary['word_f1_eq_0']}/{n}",
        f"- 全局能量下降率：{summary['energy_descent_rate']:.0%}；覆盖能量下降率：{summary['coverage_descent_rate']:.0%}",
        f"- **M2 判定：{summary['m2_verdict']}**",
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
