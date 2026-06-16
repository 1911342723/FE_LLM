"""Generate A/B/C prediction JSONL for FE-LLM P1 evaluation.

默认 dry-run，不下载模型、不生成预测。真实生成需显式传入 --run。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from fe_llm.backbone_lm import EnergyHead, IntentAdapter, IntentState, PretrainedBackbone, select_hybrid_candidate
from fe_llm.backbone_lm.slot_translation_p1_train import DEFAULT_BACKBONE, DEFAULT_CKPT_NAME, TranslationPair
from fe_llm.backbone_lm.slot_translation_p1_train import load_pairs as load_translation_pairs
from fe_llm.config import get_device

DEFAULT_VAL_PATH = os.path.join("data", "translation", "opus100_val.jsonl")
DEFAULT_CKPT_PATH = os.path.join("checkpoints", "backbone_lm", DEFAULT_CKPT_NAME)
DEFAULT_PRED_PATH = os.path.join("docs", "reports", "slot_translation_p1_predictions.jsonl")


def build_translate_prompt(zh: str) -> str:
    return f"Chinese: {zh.strip()}\nEnglish:"


def clean_generation(text: str) -> str:
    text = text.strip()
    if "\n" in text:
        text = text.splitlines()[0].strip()
    return text


def make_random_intent(state: IntentState, seed: int = 0) -> IntentState:
    generator = torch.Generator(device=state.global_intent.device)
    generator.manual_seed(seed)
    return IntentState(
        global_intent=torch.randn(
            state.global_intent.shape,
            generator=generator,
            device=state.global_intent.device,
            dtype=state.global_intent.dtype,
        ),
        intent_slots=torch.randn(
            state.intent_slots.shape,
            generator=generator,
            device=state.intent_slots.device,
            dtype=state.intent_slots.dtype,
        ),
        slot_salience=torch.full_like(state.slot_salience, 1.0 / state.slot_salience.shape[1]),
    )


def prediction_record(
    group: str,
    pair: TranslationPair,
    pred: str,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = info or {}
    record = {
        "group": group,
        "zh": pair.zh,
        "ref": pair.en,
        "pred": clean_generation(pred),
    }
    for key in (
        "residual_start",
        "residual_end",
        "coverage_start",
        "coverage_end",
        "disagreement_rate",
    ):
        if key in info and info[key] is not None:
            record[key] = info[key]
    return record


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FE-LLM P1 A/B/C prediction JSONL.")
    parser.add_argument("--run", action="store_true", help="真正加载模型并生成预测；默认只 dry-run。")
    parser.add_argument("--model-name", default=DEFAULT_BACKBONE)
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--pred-path", default=DEFAULT_PRED_PATH)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[p1-predict] dry-run：未下载模型，未生成预测。")
    print(f"[p1-predict] model_name = {args.model_name}")
    print(f"[p1-predict] ckpt_path = {args.ckpt_path}")
    print(f"[p1-predict] val_path = {args.val_path}")
    print(f"[p1-predict] pred_path = {args.pred_path}")
    print("[p1-predict] 真正生成请显式追加 --run。")


def load_components(args: argparse.Namespace):
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
    energy_head = EnergyHead(hidden_size=hidden_size, intent_dim=int(cfg.get("intent_dim", 128))).to(device)
    adapter.load_state_dict(ckpt["adapter"])
    energy_head.load_state_dict(ckpt["energy_head"])
    adapter.eval()
    energy_head.eval()
    return device, backbone, adapter, energy_head


@torch.no_grad()
def encode_intent(
    backbone: PretrainedBackbone,
    adapter: IntentAdapter,
    text: str,
    max_length: int = 256,
    device: str | torch.device = "cpu",
) -> IntentState:
    encoded = backbone.encode_texts([text], max_length=max_length, device=device)
    output = backbone(encoded["input_ids"], encoded.get("attention_mask"))
    return adapter(output.hidden_states, encoded.get("attention_mask"))


def _energy_info(
    energy_head: EnergyHead,
    hidden: torch.Tensor,
    intent: IntentState,
) -> dict[str, float | None]:
    if hidden.shape[1] == 0:
        return {"residual_start": None, "residual_end": None, "coverage_start": None, "coverage_end": None}
    energies = energy_head(hidden, intent)
    residual = energies["residual_energy"][0]
    coverage = energies["coverage_energy"][0]
    return {
        "residual_start": round(float(residual[0]), 4),
        "residual_end": round(float(residual[-1]), 4),
        "coverage_start": round(float(coverage[0]), 4),
        "coverage_end": round(float(coverage[-1]), 4),
    }


@torch.no_grad()
def generate_one(
    backbone: PretrainedBackbone,
    adapter: IntentAdapter,
    energy_head: EnergyHead,
    pair: TranslationPair,
    group: str,
    args: argparse.Namespace,
    device: str | torch.device,
) -> tuple[str, dict[str, Any]]:
    prompt = build_translate_prompt(pair.zh)
    encoded_prompt = backbone.encode_texts([prompt], device=device)
    prompt_len = int(encoded_prompt["input_ids"].shape[1])
    generated = encoded_prompt["input_ids"]
    intent = encode_intent(backbone, adapter, pair.zh, device=device)
    if group == "C":
        intent = make_random_intent(intent, seed=args.seed)

    disagreement = 0
    steps = 0
    for _ in range(args.max_new):
        output = backbone(generated)
        logits = output.logits[0, -1]
        log_probs = torch.log_softmax(logits, dim=-1)
        prob_token = int(log_probs.argmax())
        token_id = prob_token

        if group in {"B", "C"}:
            topk = torch.topk(log_probs, k=min(args.top_k, log_probs.numel()))
            candidate_ids = topk.indices
            candidate_batches = [
                torch.cat([generated[0], candidate_id.reshape(1)], dim=0)
                for candidate_id in candidate_ids
            ]
            candidate_input = torch.stack(candidate_batches, dim=0)
            candidate_out = backbone(candidate_input)
            candidate_hidden = candidate_out.hidden_states[:, prompt_len:, :]
            candidate_intent = IntentState(
                global_intent=intent.global_intent.expand(candidate_hidden.shape[0], -1),
                intent_slots=intent.intent_slots.expand(candidate_hidden.shape[0], -1, -1),
                slot_salience=intent.slot_salience.expand(candidate_hidden.shape[0], -1),
            )
            candidate_energy = energy_head(candidate_hidden, candidate_intent)["total_energy"].mean(dim=1)
            step = select_hybrid_candidate(candidate_ids, topk.values, candidate_energy, alpha=args.alpha)
            token_id = step.token_id
            if token_id != prob_token:
                disagreement += 1

        steps += 1
        token_tensor = torch.tensor([[token_id]], device=generated.device, dtype=generated.dtype)
        generated = torch.cat([generated, token_tensor], dim=1)
        if token_id == getattr(backbone.tokenizer, "eos_token_id", None):
            break

    new_ids = generated[0, prompt_len:]
    pred = backbone.tokenizer.decode(new_ids, skip_special_tokens=True)
    final_out = backbone(generated)
    gen_hidden = final_out.hidden_states[:, prompt_len:, :]
    info = _energy_info(energy_head, gen_hidden, intent)
    info["disagreement_rate"] = round(disagreement / max(steps, 1), 4)
    return clean_generation(pred), info


def write_predictions(args: argparse.Namespace) -> int:
    device, backbone, adapter, energy_head = load_components(args)
    pairs = load_translation_pairs(args.val_path, limit=args.limit)
    os.makedirs(os.path.dirname(args.pred_path), exist_ok=True)
    count = 0
    with open(args.pred_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            for group in ("A", "B", "C"):
                pred, info = generate_one(backbone, adapter, energy_head, pair, group, args, device)
                f.write(json.dumps(prediction_record(group, pair, pred, info), ensure_ascii=False) + "\n")
                count += 1
    print(f"[p1-predict] wrote {count} rows -> {args.pred_path}")
    return count


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.run:
        print_dry_run(args)
        return 0
    write_predictions(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
