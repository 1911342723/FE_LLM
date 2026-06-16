"""P1.5 training: frozen backbone + intent-conditioned logits adapter.

默认 dry-run，不下载模型、不开始训练。真实训练需显式传入 --run。
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from fe_llm.backbone_lm import IntentAdapter, IntentLogitsAdapter, IntentState, PretrainedBackbone
from fe_llm.backbone_lm.slot_translation_p1_train import (
    DEFAULT_BACKBONE,
    DEFAULT_TRAIN_PATH,
    batch_iter,
    load_pairs,
    set_seed,
    translate_prompt,
    unique_candidate_ids,
)
from fe_llm.config import get_device

DEFAULT_OUTPUT_DIR = os.path.join("checkpoints", "backbone_lm")
DEFAULT_CKPT_NAME = "slot_translation_p15_logits.pt"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FE-LLM P1.5 intent-conditioned logits adapter.")
    parser.add_argument("--run", action="store_true", help="真正开始训练；默认只 dry-run。")
    parser.add_argument("--model-name", default=DEFAULT_BACKBONE)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ckpt-name", default=DEFAULT_CKPT_NAME)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--intent-dim", type=int, default=128)
    parser.add_argument("--n-slots", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--adapter-dim", type=int, default=128)
    parser.add_argument("--candidate-top-k", type=int, default=4)
    parser.add_argument("--candidate-steps", type=int, default=1)
    parser.add_argument("--candidate-examples", type=int, default=4)
    parser.add_argument("--intent-contrast-weight", type=float, default=0.5)
    parser.add_argument("--intent-contrast-margin", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[p15-train] dry-run：未下载模型，未开始训练。")
    print(f"[p15-train] backbone = {args.model_name}")
    print(f"[p15-train] train_path = {args.train_path}")
    print(f"[p15-train] output_dir = {args.output_dir}")
    print(f"[p15-train] batch/limit/epochs = {args.batch}/{args.limit}/{args.epochs}")
    print("[p15-train] 真正训练请显式追加 --run。")


@torch.no_grad()
def _token_ids(tokenizer, text: str, device: str | torch.device) -> torch.Tensor:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    return encoded["input_ids"][0].to(device)


def _single_intent(state: IntentState, index: int) -> IntentState:
    return IntentState(
        global_intent=state.global_intent[index : index + 1],
        intent_slots=state.intent_slots[index : index + 1],
        slot_salience=state.slot_salience[index : index + 1],
    )


def logits_adapter_loss(
    backbone: PretrainedBackbone,
    logits_adapter: IntentLogitsAdapter,
    intent_state: IntentState,
    zh_texts: list[str],
    en_texts: list[str],
    args: argparse.Namespace,
    device: str | torch.device,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    tokenizer = backbone.tokenizer
    n_examples = min(len(zh_texts), args.candidate_examples)
    for sample_idx in range(n_examples):
        prompt_ids = _token_ids(tokenizer, translate_prompt(zh_texts[sample_idx]), device)
        target_ids = _token_ids(tokenizer, en_texts[sample_idx], device)
        if target_ids.numel() == 0:
            continue
        true_intent = _single_intent(intent_state, sample_idx)
        wrong_intent = _single_intent(intent_state, (sample_idx + 1) % int(intent_state.global_intent.shape[0]))
        n_steps = min(args.candidate_steps, int(target_ids.numel()))
        for pos in range(n_steps):
            prefix = torch.cat([prompt_ids, target_ids[:pos]], dim=0)
            prefix_out = backbone(prefix.unsqueeze(0))
            log_probs = torch.log_softmax(prefix_out.logits[0, -1], dim=-1)
            topk = torch.topk(log_probs, k=min(args.candidate_top_k, log_probs.numel()))
            candidate_ids = unique_candidate_ids(int(target_ids[pos]), [int(x) for x in topk.indices])
            candidate_tensors = [
                torch.cat([prefix, torch.tensor([candidate_id], device=device, dtype=prefix.dtype)], dim=0)
                for candidate_id in candidate_ids
            ]
            candidate_input = torch.stack(candidate_tensors, dim=0)
            candidate_out = backbone(candidate_input)
            candidate_hidden = candidate_out.hidden_states[:, -1, :].unsqueeze(0)
            expanded_intent = IntentState(
                global_intent=true_intent.global_intent,
                intent_slots=true_intent.intent_slots,
                slot_salience=true_intent.slot_salience,
            )
            bias = logits_adapter(candidate_hidden, expanded_intent)
            target = torch.zeros(1, device=device, dtype=torch.long)
            rank_loss = F.cross_entropy(bias, target)

            gold_hidden = candidate_hidden[:, 0, :]
            true_bias = logits_adapter(gold_hidden, true_intent)
            wrong_bias = logits_adapter(gold_hidden, wrong_intent)
            contrast = F.relu(wrong_bias - true_bias + args.intent_contrast_margin).mean()
            losses.append(rank_loss + args.intent_contrast_weight * contrast)
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def train(args: argparse.Namespace) -> str:
    set_seed(args.seed)
    pairs = load_pairs(args.train_path, limit=args.limit)
    if len(pairs) < 2:
        raise ValueError("P1.5 training requires at least two translation pairs")

    device = get_device() if args.device == "auto" else args.device
    print(f"[p15-train] device = {device}")
    backbone = PretrainedBackbone.from_pretrained(args.model_name, freeze=True).to(device)
    hidden_size = int(getattr(backbone.model.config, "hidden_size"))
    intent_adapter = IntentAdapter(
        hidden_size=hidden_size,
        intent_dim=args.intent_dim,
        n_slots=args.n_slots,
        n_heads=args.n_heads,
    ).to(device)
    logits_adapter = IntentLogitsAdapter(
        hidden_size=hidden_size,
        intent_dim=args.intent_dim,
        adapter_dim=args.adapter_dim,
    ).to(device)
    opt = torch.optim.AdamW(
        list(intent_adapter.parameters()) + list(logits_adapter.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_batches = 0
        for batch in batch_iter(pairs, args.batch, rng):
            if len(batch) < 2:
                continue
            zh_texts = [item.zh for item in batch]
            en_texts = [item.en for item in batch]
            src = backbone.encode_texts(zh_texts, device=device)
            src_out = backbone(src["input_ids"], src.get("attention_mask"))
            intent = intent_adapter(src_out.hidden_states, src.get("attention_mask"))
            loss = logits_adapter_loss(backbone, logits_adapter, intent, zh_texts, en_texts, args, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(intent_adapter.parameters()) + list(logits_adapter.parameters()),
                1.0,
            )
            opt.step()
            total_loss += float(loss.detach())
            total_batches += 1
        print(f"[p15-train] epoch {epoch} logits_loss={total_loss / max(total_batches, 1):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, args.ckpt_name)
    torch.save(
        {
            "intent_adapter": intent_adapter.state_dict(),
            "logits_adapter": logits_adapter.state_dict(),
            "config": vars(args),
        },
        ckpt_path,
    )
    print(f"[p15-train] saved adapter/logits -> {ckpt_path}")
    return ckpt_path


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.run:
        print_dry_run(args)
        return 0
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
