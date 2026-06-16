"""Train IntentAdapter separability before deeper generation injection."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from fe_llm.backbone_lm import IntentAdapter, PretrainedBackbone
from fe_llm.backbone_lm.slot_translation_p1_train import (
    DEFAULT_BACKBONE,
    DEFAULT_TRAIN_PATH,
    batch_iter,
    load_pairs,
    set_seed,
)
from fe_llm.config import get_device

DEFAULT_OUTPUT_DIR = os.path.join("checkpoints", "backbone_lm")
DEFAULT_CKPT_NAME = "intent_adapter_separable.pt"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train IntentAdapter with contrastive and salience losses.")
    parser.add_argument("--run", action="store_true", help="真正开始训练；默认只 dry-run。")
    parser.add_argument("--model-name", default=DEFAULT_BACKBONE)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ckpt-name", default=DEFAULT_CKPT_NAME)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--intent-dim", type=int, default=128)
    parser.add_argument("--n-slots", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--slot-div-weight", type=float, default=0.05)
    parser.add_argument("--source-spread-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[intent-train] dry-run：未下载模型，未开始训练。")
    print(f"[intent-train] backbone = {args.model_name}")
    print(f"[intent-train] train_path = {args.train_path}")
    print(f"[intent-train] output_dir = {args.output_dir}")
    print(f"[intent-train] ckpt_name = {args.ckpt_name}")
    print("[intent-train] 真正训练请显式追加 --run。")


def contrastive_loss(z_src: torch.Tensor, z_tgt: torch.Tensor, temperature: float) -> torch.Tensor:
    src = F.normalize(z_src, dim=-1)
    tgt = F.normalize(z_tgt, dim=-1)
    logits = src @ tgt.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def slot_diversity_loss(slots: torch.Tensor) -> torch.Tensor:
    norm = F.normalize(slots, dim=-1)
    gram = norm @ norm.transpose(1, 2)
    eye = torch.eye(gram.shape[-1], device=gram.device).unsqueeze(0)
    return ((gram - eye) ** 2).mean()


def salience_entropy(salience: torch.Tensor) -> torch.Tensor:
    return (-(salience.clamp_min(1e-8).log() * salience).sum(dim=-1)).mean()


def global_spread_loss(z: torch.Tensor) -> torch.Tensor:
    """Penalize high off-diagonal cosine among source intents."""

    if z.shape[0] < 2:
        return torch.tensor(0.0, device=z.device)
    norm = F.normalize(z, dim=-1)
    sim = norm @ norm.T
    offdiag = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
    return F.relu(offdiag).mean()


def train(args: argparse.Namespace) -> str:
    set_seed(args.seed)
    pairs = load_pairs(args.train_path, limit=args.limit)
    if len(pairs) < 2:
        raise ValueError("IntentAdapter training requires at least two translation pairs")

    device = get_device() if args.device == "auto" else args.device
    print(f"[intent-train] device = {device}")
    backbone = PretrainedBackbone.from_pretrained(args.model_name, freeze=True).to(device)
    hidden_size = int(getattr(backbone.model.config, "hidden_size"))
    adapter = IntentAdapter(hidden_size, args.intent_dim, args.n_slots, args.n_heads).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)

    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        steps = 0
        for batch in batch_iter(pairs, args.batch, rng):
            zh_texts = [item.zh for item in batch]
            en_texts = [item.en for item in batch]
            src = backbone.encode_texts(zh_texts, device=device)
            tgt = backbone.encode_texts(en_texts, device=device)
            src_out = backbone(src["input_ids"], src.get("attention_mask"))
            tgt_out = backbone(tgt["input_ids"], tgt.get("attention_mask"))
            src_state = adapter(src_out.hidden_states, src.get("attention_mask"))
            tgt_state = adapter(tgt_out.hidden_states, tgt.get("attention_mask"))

            l_contrast = contrastive_loss(src_state.global_intent, tgt_state.global_intent, args.temperature)
            l_entropy = 0.5 * (salience_entropy(src_state.slot_salience) + salience_entropy(tgt_state.slot_salience))
            l_div = 0.5 * (slot_diversity_loss(src_state.intent_slots) + slot_diversity_loss(tgt_state.intent_slots))
            l_spread = global_spread_loss(src_state.global_intent)
            loss = (
                l_contrast
                + args.entropy_weight * l_entropy
                + args.slot_div_weight * l_div
                + args.source_spread_weight * l_spread
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            steps += 1
        print(f"[intent-train] epoch {epoch} intent_loss={total / max(steps, 1):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, args.ckpt_name)
    torch.save({"intent_adapter": adapter.state_dict(), "config": vars(args)}, ckpt_path)
    print(f"[intent-train] saved intent adapter -> {ckpt_path}")
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
