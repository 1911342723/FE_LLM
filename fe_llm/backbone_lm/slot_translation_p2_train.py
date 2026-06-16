"""P2 training: intent-conditioned residual adapter over frozen backbone hidden states."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from fe_llm.backbone_lm import IntentAdapter, IntentResidualAdapter, IntentState, PretrainedBackbone
from fe_llm.backbone_lm.slot_translation_p1_train import (
    DEFAULT_BACKBONE,
    DEFAULT_TRAIN_PATH,
    batch_iter,
    load_pairs,
    set_seed,
    translate_prompt,
)
from fe_llm.config import get_device

DEFAULT_OUTPUT_DIR = os.path.join("checkpoints", "backbone_lm")
DEFAULT_CKPT_NAME = "slot_translation_p2_residual.pt"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FE-LLM P2 intent residual adapter.")
    parser.add_argument("--run", action="store_true", help="真正开始训练；默认只 dry-run。")
    parser.add_argument("--model-name", default=DEFAULT_BACKBONE)
    parser.add_argument("--intent-ckpt-path", default="")
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
    parser.add_argument("--candidate-steps", type=int, default=1)
    parser.add_argument("--candidate-examples", type=int, default=4)
    parser.add_argument("--intent-contrast-weight", type=float, default=0.5)
    parser.add_argument("--intent-contrast-margin", type=float, default=0.2)
    parser.add_argument("--negative-mode", choices=["next", "hard"], default="next")
    parser.add_argument("--delta-norm-weight", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[p2-train] dry-run：未下载模型，未开始训练。")
    print(f"[p2-train] backbone = {args.model_name}")
    print(f"[p2-train] train_path = {args.train_path}")
    print(f"[p2-train] output_dir = {args.output_dir}")
    print(f"[p2-train] ckpt_name = {args.ckpt_name}")
    print("[p2-train] 真正训练请显式追加 --run。")


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


def hard_negative_indices(intent_state: IntentState) -> list[int]:
    """Pick the most similar non-self global intent in the batch."""

    z = F.normalize(intent_state.global_intent.detach(), dim=-1)
    sim = z @ z.T
    sim.fill_diagonal_(-1e9)
    return [int(i) for i in sim.argmax(dim=-1)]


def p2_loss(
    backbone: PretrainedBackbone,
    residual_adapter: IntentResidualAdapter,
    intent_state: IntentState,
    zh_texts: list[str],
    en_texts: list[str],
    args: argparse.Namespace,
    device: str | torch.device,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    tokenizer = backbone.tokenizer
    lm_head = getattr(backbone.model, "lm_head")
    n_examples = min(len(zh_texts), args.candidate_examples)
    hard_negatives = hard_negative_indices(intent_state) if args.negative_mode == "hard" else []
    for sample_idx in range(n_examples):
        prompt_ids = _token_ids(tokenizer, translate_prompt(zh_texts[sample_idx]), device)
        target_ids = _token_ids(tokenizer, en_texts[sample_idx], device)
        if target_ids.numel() == 0:
            continue
        true_intent = _single_intent(intent_state, sample_idx)
        wrong_idx = hard_negatives[sample_idx] if hard_negatives else (sample_idx + 1) % int(intent_state.global_intent.shape[0])
        wrong_intent = _single_intent(intent_state, wrong_idx)
        n_steps = min(args.candidate_steps, int(target_ids.numel()))
        for pos in range(n_steps):
            prefix = torch.cat([prompt_ids, target_ids[:pos]], dim=0)
            out = backbone(prefix.unsqueeze(0))
            hidden = out.hidden_states[:, -1, :]
            adapted, delta = residual_adapter(hidden, true_intent, gamma=args.gamma)
            logits = lm_head(adapted)
            gold = target_ids[pos].reshape(1)
            ce = F.cross_entropy(logits, gold)

            wrong_adapted, _ = residual_adapter(hidden, wrong_intent, gamma=args.gamma)
            true_gold_logit = logits[:, gold.item()]
            wrong_gold_logit = lm_head(wrong_adapted)[:, gold.item()]
            contrast = F.relu(wrong_gold_logit - true_gold_logit + args.intent_contrast_margin).mean()
            norm = delta.pow(2).mean()
            losses.append(ce + args.intent_contrast_weight * contrast + args.delta_norm_weight * norm)
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def train(args: argparse.Namespace) -> str:
    set_seed(args.seed)
    pairs = load_pairs(args.train_path, limit=args.limit)
    if len(pairs) < 2:
        raise ValueError("P2 training requires at least two translation pairs")

    device = get_device() if args.device == "auto" else args.device
    print(f"[p2-train] device = {device}")
    backbone = PretrainedBackbone.from_pretrained(args.model_name, freeze=True).to(device)
    hidden_size = int(getattr(backbone.model.config, "hidden_size"))
    intent_adapter = IntentAdapter(hidden_size, args.intent_dim, args.n_slots, args.n_heads).to(device)
    if args.intent_ckpt_path:
        ckpt = torch.load(args.intent_ckpt_path, map_location=device, weights_only=False)
        intent_adapter.load_state_dict(ckpt["intent_adapter"])
        intent_adapter.eval()
        for param in intent_adapter.parameters():
            param.requires_grad_(False)
    residual_adapter = IntentResidualAdapter(hidden_size, args.intent_dim, args.adapter_dim).to(device)
    opt = torch.optim.AdamW(
        [param for param in list(intent_adapter.parameters()) + list(residual_adapter.parameters()) if param.requires_grad],
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
            loss = p2_loss(backbone, residual_adapter, intent, zh_texts, en_texts, args, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(intent_adapter.parameters()) + list(residual_adapter.parameters()),
                1.0,
            )
            opt.step()
            total_loss += float(loss.detach())
            total_batches += 1
        print(f"[p2-train] epoch {epoch} residual_loss={total_loss / max(total_batches, 1):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, args.ckpt_name)
    torch.save(
        {
            "intent_adapter": intent_adapter.state_dict(),
            "residual_adapter": residual_adapter.state_dict(),
            "config": vars(args),
        },
        ckpt_path,
    )
    print(f"[p2-train] saved adapter/residual -> {ckpt_path}")
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
