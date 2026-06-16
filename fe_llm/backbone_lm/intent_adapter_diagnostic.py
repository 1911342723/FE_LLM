"""Diagnostics for whether structured intent is separable enough."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from fe_llm.backbone_lm import IntentAdapter, PretrainedBackbone
from fe_llm.backbone_lm.slot_translation_p1_train import DEFAULT_BACKBONE
from fe_llm.backbone_lm.slot_translation_p1_train import load_pairs as load_translation_pairs
from fe_llm.backbone_lm.slot_translation_p1_predict import DEFAULT_VAL_PATH
from fe_llm.config import get_device

DEFAULT_CKPT_PATH = os.path.join("checkpoints", "backbone_lm", "slot_translation_p15_logits_128_seed42.pt")
REPORT_JSON = os.path.join("docs", "reports", "intent_adapter_diagnostic.json")
REPORT_MD = os.path.join("docs", "reports", "intent_adapter_diagnostic.md")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose FE-LLM IntentAdapter separability.")
    parser.add_argument("--run", action="store_true", help="真正加载模型并生成诊断；默认 dry-run。")
    parser.add_argument("--model-name", default=DEFAULT_BACKBONE)
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--report-json", default=REPORT_JSON)
    parser.add_argument("--report-md", default=REPORT_MD)
    parser.add_argument("--device", default="auto")
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[intent-diag] dry-run：未加载模型，未生成诊断。")
    print(f"[intent-diag] model_name = {args.model_name}")
    print(f"[intent-diag] ckpt_path = {args.ckpt_path}")
    print(f"[intent-diag] val_path = {args.val_path}")
    print("[intent-diag] 真正诊断请显式追加 --run。")


def load_intent_adapter(args: argparse.Namespace):
    device = get_device() if args.device == "auto" else args.device
    backbone = PretrainedBackbone.from_pretrained(args.model_name, freeze=True).to(device)
    hidden_size = int(getattr(backbone.model.config, "hidden_size"))
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    adapter = IntentAdapter(
        hidden_size=hidden_size,
        intent_dim=int(cfg.get("intent_dim", 128)),
        n_slots=int(cfg.get("n_slots", 8)),
        n_heads=int(cfg.get("n_heads", 4)),
    ).to(device)
    adapter.load_state_dict(ckpt["intent_adapter"])
    adapter.eval()
    return device, backbone, adapter


@torch.no_grad()
def encode_texts(backbone: PretrainedBackbone, adapter: IntentAdapter, texts: list[str], device):
    encoded = backbone.encode_texts(texts, device=device)
    output = backbone(encoded["input_ids"], encoded.get("attention_mask"))
    return adapter(output.hidden_states, encoded.get("attention_mask"))


def summarize_intents(global_intent: torch.Tensor, slot_salience: torch.Tensor) -> dict[str, float | int]:
    z = F.normalize(global_intent, dim=-1)
    sim = z @ z.T
    n = sim.shape[0]
    offdiag = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
    nearest = sim.masked_fill(torch.eye(n, dtype=torch.bool, device=sim.device), -1e9).argmax(dim=-1)
    entropy = (-(slot_salience.clamp_min(1e-8).log() * slot_salience).sum(dim=-1)).mean()
    return {
        "n": int(n),
        "mean_offdiag_cosine": round(float(offdiag.mean()), 4) if offdiag.numel() else 0.0,
        "max_offdiag_cosine": round(float(offdiag.max()), 4) if offdiag.numel() else 0.0,
        "min_offdiag_cosine": round(float(offdiag.min()), 4) if offdiag.numel() else 0.0,
        "nearest_unique_count": int(len(set(int(x) for x in nearest.detach().cpu()))),
        "slot_salience_entropy": round(float(entropy), 4),
    }


def render_markdown(summary: dict[str, float | int], examples: list[dict[str, str | int]]) -> str:
    lines = [
        "# FE-LLM IntentAdapter 表达诊断",
        "",
        "## 汇总",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## 最近邻样例", "", "| idx | nearest_idx | text | nearest_text |", "|---:|---:|---|---|"])
    for item in examples[:20]:
        lines.append(f"| {item['idx']} | {item['nearest_idx']} | {item['text']} | {item['nearest_text']} |")
    return "\n".join(lines) + "\n"


def diagnose(args: argparse.Namespace) -> dict:
    device, backbone, adapter = load_intent_adapter(args)
    pairs = load_translation_pairs(args.val_path, limit=args.limit)
    texts = [pair.zh for pair in pairs]
    state = encode_texts(backbone, adapter, texts, device)
    summary = summarize_intents(state.global_intent, state.slot_salience)
    z = F.normalize(state.global_intent, dim=-1)
    sim = z @ z.T
    nearest = sim.masked_fill(torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device), -1e9).argmax(dim=-1)
    examples = [
        {
            "idx": idx,
            "nearest_idx": int(nearest[idx]),
            "text": texts[idx],
            "nearest_text": texts[int(nearest[idx])],
        }
        for idx in range(len(texts))
    ]
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "examples": examples}, f, ensure_ascii=False, indent=2)
    report = render_markdown(summary, examples)
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    return {"summary": summary, "examples": examples}


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.run:
        print_dry_run(args)
        return 0
    diagnose(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
