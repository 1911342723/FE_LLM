"""Generate A/B/C prediction JSONL with intent residual injected by layer hooks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from fe_llm.backbone_lm import IntentAdapter, IntentLayerHook, IntentResidualAdapter, PretrainedBackbone
from fe_llm.backbone_lm.slot_translation_p1_train import DEFAULT_BACKBONE, TranslationPair
from fe_llm.backbone_lm.slot_translation_p1_train import load_pairs as load_translation_pairs
from fe_llm.backbone_lm.slot_translation_p1_predict import (
    DEFAULT_PRED_PATH,
    DEFAULT_VAL_PATH,
    build_translate_prompt,
    clean_generation,
    make_random_intent,
    prediction_record,
)
from fe_llm.backbone_lm.slot_translation_p2_train import DEFAULT_CKPT_NAME
from fe_llm.config import get_device

DEFAULT_CKPT_PATH = os.path.join("checkpoints", "backbone_lm", DEFAULT_CKPT_NAME)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FE-LLM P2c layer-hook prediction JSONL.")
    parser.add_argument("--run", action="store_true", help="真正加载模型并生成预测；默认只 dry-run。")
    parser.add_argument("--model-name", default=DEFAULT_BACKBONE)
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH)
    parser.add_argument("--intent-ckpt-path", default="")
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--pred-path", default=DEFAULT_PRED_PATH)
    parser.add_argument("--layer-path", default="model.layers.0")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--control-mode", choices=["random", "mismatch"], default="mismatch")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[p2c-predict] dry-run：未下载模型，未生成预测。")
    print(f"[p2c-predict] model_name = {args.model_name}")
    print(f"[p2c-predict] ckpt_path = {args.ckpt_path}")
    print(f"[p2c-predict] layer_path = {args.layer_path}")
    print(f"[p2c-predict] pred_path = {args.pred_path}")
    print("[p2c-predict] 真正生成请显式追加 --run。")


def load_components(args: argparse.Namespace):
    device = get_device() if args.device == "auto" else args.device
    backbone = PretrainedBackbone.from_pretrained(args.model_name, freeze=True).to(device)
    hidden_size = int(getattr(backbone.model.config, "hidden_size"))
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    intent_adapter = IntentAdapter(
        hidden_size=hidden_size,
        intent_dim=int(cfg.get("intent_dim", 128)),
        n_slots=int(cfg.get("n_slots", 8)),
        n_heads=int(cfg.get("n_heads", 4)),
    ).to(device)
    residual_adapter = IntentResidualAdapter(
        hidden_size=hidden_size,
        intent_dim=int(cfg.get("intent_dim", 128)),
        adapter_dim=int(cfg.get("adapter_dim", 128)),
    ).to(device)
    if args.intent_ckpt_path:
        intent_ckpt = torch.load(args.intent_ckpt_path, map_location=device, weights_only=False)
        intent_adapter.load_state_dict(intent_ckpt["intent_adapter"])
    else:
        intent_adapter.load_state_dict(ckpt["intent_adapter"])
    residual_adapter.load_state_dict(ckpt["residual_adapter"])
    intent_adapter.eval()
    residual_adapter.eval()
    return device, backbone, intent_adapter, residual_adapter


@torch.no_grad()
def encode_intent(backbone: PretrainedBackbone, intent_adapter: IntentAdapter, text: str, device):
    encoded = backbone.encode_texts([text], device=device)
    output = backbone(encoded["input_ids"], encoded.get("attention_mask"))
    return intent_adapter(output.hidden_states, encoded.get("attention_mask"))


@torch.no_grad()
def forward_logits(backbone: PretrainedBackbone, input_ids: torch.Tensor):
    output = backbone.model(input_ids=input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
    return output.logits[0, -1]


@torch.no_grad()
def generate_one(
    backbone: PretrainedBackbone,
    intent_adapter: IntentAdapter,
    residual_adapter: IntentResidualAdapter,
    pair: TranslationPair,
    group: str,
    args: argparse.Namespace,
    device,
    intent_override=None,
) -> tuple[str, dict[str, Any]]:
    prompt = build_translate_prompt(pair.zh)
    encoded_prompt = backbone.encode_texts([prompt], device=device)
    prompt_len = int(encoded_prompt["input_ids"].shape[1])
    generated = encoded_prompt["input_ids"]
    intent = encode_intent(backbone, intent_adapter, pair.zh, device=device)
    if group == "C":
        if args.control_mode == "mismatch" and intent_override is not None:
            intent = intent_override
        else:
            intent = make_random_intent(intent, seed=args.seed)

    disagreement = 0
    steps = 0
    for _ in range(args.max_new):
        base_logits = forward_logits(backbone, generated)
        prob_token = int(base_logits.argmax())
        token_id = prob_token
        if group in {"B", "C"}:
            with IntentLayerHook(backbone.model, [args.layer_path], residual_adapter, intent, gamma=args.gamma):
                hooked_logits = forward_logits(backbone, generated)
            token_id = int(hooked_logits.argmax())
            if token_id != prob_token:
                disagreement += 1
        steps += 1
        generated = torch.cat(
            [generated, torch.tensor([[token_id]], device=generated.device, dtype=generated.dtype)],
            dim=1,
        )
        if token_id == getattr(backbone.tokenizer, "eos_token_id", None):
            break

    new_ids = generated[0, prompt_len:]
    pred = clean_generation(backbone.tokenizer.decode(new_ids, skip_special_tokens=True))
    return pred, {"disagreement_rate": round(disagreement / max(steps, 1), 4)}


def write_predictions(args: argparse.Namespace) -> int:
    device, backbone, intent_adapter, residual_adapter = load_components(args)
    pairs = load_translation_pairs(args.val_path, limit=args.limit)
    mismatch_intents = []
    if args.control_mode == "mismatch":
        mismatch_intents = [
            encode_intent(backbone, intent_adapter, pairs[(idx + 1) % len(pairs)].zh, device=device)
            for idx in range(len(pairs))
        ]
    os.makedirs(os.path.dirname(args.pred_path), exist_ok=True)
    count = 0
    with open(args.pred_path, "w", encoding="utf-8") as f:
        for idx, pair in enumerate(pairs):
            for group in ("A", "B", "C"):
                pred, info = generate_one(
                    backbone,
                    intent_adapter,
                    residual_adapter,
                    pair,
                    group,
                    args,
                    device,
                    intent_override=mismatch_intents[idx] if mismatch_intents else None,
                )
                f.write(json.dumps(prediction_record(group, pair, pred, info), ensure_ascii=False) + "\n")
                count += 1
    print(f"[p2c-predict] wrote {count} rows -> {args.pred_path}")
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
