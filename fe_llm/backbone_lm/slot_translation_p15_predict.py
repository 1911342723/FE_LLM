"""Generate A/B/C prediction JSONL for FE-LLM P1.5 logits-adapter experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from fe_llm.backbone_lm import IntentAdapter, IntentLogitsAdapter, IntentState, PretrainedBackbone
from fe_llm.backbone_lm.slot_translation_p1_train import DEFAULT_BACKBONE, TranslationPair
from fe_llm.backbone_lm.slot_translation_p1_train import load_pairs as load_translation_pairs
from fe_llm.backbone_lm.slot_translation_p1_train import unique_candidate_ids
from fe_llm.backbone_lm.slot_translation_p1_predict import (
    DEFAULT_PRED_PATH,
    DEFAULT_VAL_PATH,
    build_translate_prompt,
    clean_generation,
    make_random_intent,
    prediction_record,
)
from fe_llm.backbone_lm.slot_translation_p15_train import DEFAULT_CKPT_NAME
from fe_llm.config import get_device

DEFAULT_CKPT_PATH = os.path.join("checkpoints", "backbone_lm", DEFAULT_CKPT_NAME)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate FE-LLM P1.5 A/B/C prediction JSONL.")
    parser.add_argument("--run", action="store_true", help="真正加载模型并生成预测；默认只 dry-run。")
    parser.add_argument("--model-name", default=DEFAULT_BACKBONE)
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--pred-path", default=DEFAULT_PRED_PATH)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[p15-predict] dry-run：未下载模型，未生成预测。")
    print(f"[p15-predict] model_name = {args.model_name}")
    print(f"[p15-predict] ckpt_path = {args.ckpt_path}")
    print(f"[p15-predict] pred_path = {args.pred_path}")
    print("[p15-predict] 真正生成请显式追加 --run。")


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
    logits_adapter = IntentLogitsAdapter(
        hidden_size=hidden_size,
        intent_dim=int(cfg.get("intent_dim", 128)),
        adapter_dim=int(cfg.get("adapter_dim", 128)),
    ).to(device)
    intent_adapter.load_state_dict(ckpt["intent_adapter"])
    logits_adapter.load_state_dict(ckpt["logits_adapter"])
    intent_adapter.eval()
    logits_adapter.eval()
    return device, backbone, intent_adapter, logits_adapter


@torch.no_grad()
def encode_intent(
    backbone: PretrainedBackbone,
    intent_adapter: IntentAdapter,
    text: str,
    device: str | torch.device,
) -> IntentState:
    encoded = backbone.encode_texts([text], device=device)
    output = backbone(encoded["input_ids"], encoded.get("attention_mask"))
    return intent_adapter(output.hidden_states, encoded.get("attention_mask"))


@torch.no_grad()
def generate_one(
    backbone: PretrainedBackbone,
    intent_adapter: IntentAdapter,
    logits_adapter: IntentLogitsAdapter,
    pair: TranslationPair,
    group: str,
    args: argparse.Namespace,
    device: str | torch.device,
) -> tuple[str, dict[str, Any]]:
    prompt = build_translate_prompt(pair.zh)
    encoded_prompt = backbone.encode_texts([prompt], device=device)
    prompt_len = int(encoded_prompt["input_ids"].shape[1])
    generated = encoded_prompt["input_ids"]
    intent = encode_intent(backbone, intent_adapter, pair.zh, device=device)
    if group == "C":
        intent = make_random_intent(intent, seed=args.seed)

    disagreement = 0
    steps = 0
    for _ in range(args.max_new):
        output = backbone(generated)
        log_probs = torch.log_softmax(output.logits[0, -1], dim=-1)
        prob_token = int(log_probs.argmax())
        token_id = prob_token
        if group in {"B", "C"}:
            topk = torch.topk(log_probs, k=min(args.top_k, log_probs.numel()))
            candidate_ids = unique_candidate_ids(prob_token, [int(x) for x in topk.indices])
            candidate_tensors = [
                torch.cat([generated[0], torch.tensor([candidate_id], device=device, dtype=generated.dtype)], dim=0)
                for candidate_id in candidate_ids
            ]
            candidate_input = torch.stack(candidate_tensors, dim=0)
            candidate_out = backbone(candidate_input)
            candidate_hidden = candidate_out.hidden_states[:, -1, :].unsqueeze(0)
            bias = logits_adapter(candidate_hidden, intent)[0]
            candidate_log_probs = log_probs[torch.tensor(candidate_ids, device=device)]
            scores = candidate_log_probs + args.beta * bias
            token_id = int(candidate_ids[int(scores.argmax())])
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
    device, backbone, intent_adapter, logits_adapter = load_components(args)
    pairs = load_translation_pairs(args.val_path, limit=args.limit)
    os.makedirs(os.path.dirname(args.pred_path), exist_ok=True)
    count = 0
    with open(args.pred_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            for group in ("A", "B", "C"):
                pred, info = generate_one(backbone, intent_adapter, logits_adapter, pair, group, args, device)
                f.write(json.dumps(prediction_record(group, pair, pred, info), ensure_ascii=False) + "\n")
                count += 1
    print(f"[p15-predict] wrote {count} rows -> {args.pred_path}")
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
