"""P1 training scaffold: frozen backbone + intent adapter + energy head.

默认只 dry-run 打印配置，不下载模型、不开始训练。真实训练需显式传入 --run。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from fe_llm.backbone_lm import EnergyHead, IntentAdapter, IntentState, PretrainedBackbone
from fe_llm.config import get_device

DEFAULT_BACKBONE = "Qwen/Qwen2.5-0.5B"
DEFAULT_TRAIN_PATH = os.path.join("data", "translation", "opus100_train.jsonl")
DEFAULT_OUTPUT_DIR = os.path.join("checkpoints", "backbone_lm")
DEFAULT_CKPT_NAME = "slot_translation_p1_head.pt"


@dataclass(frozen=True)
class TranslationPair:
    zh: str
    en: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pairs(path: str, limit: int = 0) -> list[TranslationPair]:
    pairs: list[TranslationPair] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                pairs.append(TranslationPair(zh=item["zh"], en=item["en"]))
            except (json.JSONDecodeError, KeyError):
                continue
            if limit > 0 and len(pairs) >= limit:
                break
    return pairs


def batch_iter(
    pairs: list[TranslationPair],
    batch_size: int,
    rng: random.Random,
) -> list[list[TranslationPair]]:
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    return [[pairs[j] for j in idx[i : i + batch_size]] for i in range(0, len(idx), batch_size)]


def roll_intent_state(state: IntentState) -> IntentState:
    """Batch 内错配负样本：第 i 条英文配第 i-1 条中文 intent。"""

    state.validate()
    if state.global_intent.shape[0] < 2:
        raise ValueError("negative pairing requires batch size >= 2")
    return IntentState(
        global_intent=state.global_intent.roll(shifts=1, dims=0),
        intent_slots=state.intent_slots.roll(shifts=1, dims=0),
        slot_salience=state.slot_salience.roll(shifts=1, dims=0),
    )


def sequence_energy(
    energy_head: EnergyHead,
    target_hidden: torch.Tensor,
    intent_state: IntentState,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return masked mean total energy per sample."""

    energies = energy_head(target_hidden, intent_state)["total_energy"]
    mask = attention_mask.to(dtype=energies.dtype, device=energies.device)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (energies * mask).sum(dim=1) / denom


def translate_prompt(zh: str) -> str:
    return f"Chinese: {zh.strip()}\nEnglish:"


def unique_candidate_ids(gold_id: int, negative_ids: list[int]) -> list[int]:
    """Keep gold first, then top-k negatives without duplicates."""

    out = [gold_id]
    for item in negative_ids:
        if item != gold_id and item not in out:
            out.append(item)
    return out


def _single_intent_state(state: IntentState, index: int) -> IntentState:
    return IntentState(
        global_intent=state.global_intent[index : index + 1],
        intent_slots=state.intent_slots[index : index + 1],
        slot_salience=state.slot_salience[index : index + 1],
    )


@torch.no_grad()
def _tokenize_without_special(tokenizer, text: str, device: str | torch.device) -> torch.Tensor:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    return encoded["input_ids"][0].to(device)


def candidate_ranking_loss(
    backbone: PretrainedBackbone,
    energy_head: EnergyHead,
    intent_state: IntentState,
    zh_texts: list[str],
    en_texts: list[str],
    args: argparse.Namespace,
    device: str | torch.device,
) -> torch.Tensor:
    """Decode-time ranking: gold next token energy should beat top-k negatives.

    该损失直接模拟 hybrid 解码使用的候选集合，弥补"只看整段 hidden 能量"
    与"逐 token 选字"之间的训练/推理错位。
    """

    if (
        args.candidate_steps <= 0
        or (args.candidate_loss_weight <= 0 and args.intent_contrast_weight <= 0)
    ):
        return torch.tensor(0.0, device=device)
    losses: list[torch.Tensor] = []
    n_examples = min(len(zh_texts), args.candidate_examples)
    tokenizer = backbone.tokenizer
    for sample_idx in range(n_examples):
        prompt_ids = _tokenize_without_special(tokenizer, translate_prompt(zh_texts[sample_idx]), device)
        target_ids = _tokenize_without_special(tokenizer, en_texts[sample_idx], device)
        if target_ids.numel() == 0:
            continue
        sample_intent = _single_intent_state(intent_state, sample_idx)
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
            candidate_hidden = candidate_out.hidden_states[:, prompt_ids.shape[0] :, :]
            candidate_intent = IntentState(
                global_intent=sample_intent.global_intent.expand(candidate_hidden.shape[0], -1),
                intent_slots=sample_intent.intent_slots.expand(candidate_hidden.shape[0], -1, -1),
                slot_salience=sample_intent.slot_salience.expand(candidate_hidden.shape[0], -1),
            )
            candidate_energy = energy_head(candidate_hidden, candidate_intent)["total_energy"].mean(dim=1)
            gold_energy = candidate_energy[0]
            negative_energy = candidate_energy[1:]
            step_losses: list[torch.Tensor] = []
            if args.candidate_loss_weight > 0 and negative_energy.numel() > 0:
                step_losses.append(
                    args.candidate_loss_weight
                    * F.relu(gold_energy - negative_energy + args.candidate_margin).mean()
                )
            if args.intent_contrast_weight > 0 and intent_state.global_intent.shape[0] > 1:
                wrong_idx = (sample_idx + 1) % int(intent_state.global_intent.shape[0])
                wrong_intent = _single_intent_state(intent_state, wrong_idx)
                wrong_energy = energy_head(candidate_hidden[:1], wrong_intent)["total_energy"].mean()
                step_losses.append(
                    args.intent_contrast_weight
                    * F.relu(gold_energy - wrong_energy + args.intent_contrast_margin)
                )
            if step_losses:
                losses.append(torch.stack(step_losses).sum())
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FE-LLM P1 adapter/head on zh->en pairs.")
    parser.add_argument("--run", action="store_true", help="真正开始训练；默认只 dry-run。")
    parser.add_argument("--check-env", action="store_true", help="只检查数据、依赖和设备，不下载模型。")
    parser.add_argument("--model-name", default=DEFAULT_BACKBONE)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--candidate-loss-weight", type=float, default=0.0)
    parser.add_argument("--candidate-margin", type=float, default=0.2)
    parser.add_argument("--candidate-steps", type=int, default=0)
    parser.add_argument("--candidate-examples", type=int, default=1)
    parser.add_argument("--candidate-top-k", type=int, default=4)
    parser.add_argument("--intent-contrast-weight", type=float, default=0.0)
    parser.add_argument("--intent-contrast-margin", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--intent-dim", type=int, default=128)
    parser.add_argument("--n-slots", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser


def print_dry_run(args: argparse.Namespace) -> None:
    print("[p1-train] dry-run：未下载模型，未开始训练。")
    print(f"[p1-train] backbone = {args.model_name}")
    print(f"[p1-train] train_path = {args.train_path}")
    print(f"[p1-train] output_dir = {args.output_dir}")
    print(f"[p1-train] batch/limit/epochs = {args.batch}/{args.limit}/{args.epochs}")
    print("[p1-train] 真正训练请显式追加 --run。")


def check_environment(args: argparse.Namespace) -> dict[str, object]:
    train_path_exists = os.path.exists(args.train_path)
    sample_pairs = len(load_pairs(args.train_path, limit=2)) if train_path_exists else 0
    transformers_available = importlib.util.find_spec("transformers") is not None
    device = get_device() if args.device == "auto" else args.device
    return {
        "model_name": args.model_name,
        "train_path": args.train_path,
        "train_path_exists": train_path_exists,
        "sample_pairs": sample_pairs,
        "has_min_pairs": sample_pairs >= 2,
        "transformers_available": transformers_available,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "ready_for_run": bool(train_path_exists and sample_pairs >= 2 and transformers_available),
    }


def print_env_check(result: dict[str, object]) -> None:
    print("[p1-train] env-check：未下载模型，未开始训练。")
    for key, value in result.items():
        print(f"[p1-train] {key} = {value}")


def train(args: argparse.Namespace) -> str:
    set_seed(args.seed)
    pairs = load_pairs(args.train_path, limit=args.limit)
    if len(pairs) < 2:
        raise ValueError("P1 training requires at least two translation pairs for negative pairing")

    device = get_device() if args.device == "auto" else args.device
    print(f"[p1-train] device = {device}")
    backbone = PretrainedBackbone.from_pretrained(args.model_name, freeze=True).to(device)

    hidden_size = int(getattr(backbone.model.config, "hidden_size"))
    adapter = IntentAdapter(
        hidden_size=hidden_size,
        intent_dim=args.intent_dim,
        n_slots=args.n_slots,
        n_heads=args.n_heads,
    ).to(device)
    energy_head = EnergyHead(hidden_size=hidden_size, intent_dim=args.intent_dim).to(device)
    opt = torch.optim.AdamW(
        list(adapter.parameters()) + list(energy_head.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    rng = random.Random(42)
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        energy_head.train()
        total_loss = 0.0
        total_batches = 0
        for batch in batch_iter(pairs, args.batch, rng):
            if len(batch) < 2:
                continue
            zh_texts = [item.zh for item in batch]
            en_texts = [item.en for item in batch]
            src = backbone.encode_texts(zh_texts, max_length=args.max_source_length, device=device)
            tgt = backbone.encode_texts(en_texts, max_length=args.max_target_length, device=device)

            src_out = backbone(src["input_ids"], src.get("attention_mask"))
            tgt_out = backbone(tgt["input_ids"], tgt.get("attention_mask"))
            intent = adapter(src_out.hidden_states, src.get("attention_mask"))
            neg_intent = roll_intent_state(intent)

            target_mask = tgt.get("attention_mask")
            if target_mask is None:
                target_mask = torch.ones(tgt_out.hidden_states.shape[:2], device=device)
            pos_energy = sequence_energy(energy_head, tgt_out.hidden_states, intent, target_mask)
            neg_energy = sequence_energy(energy_head, tgt_out.hidden_states, neg_intent, target_mask)

            # 排序目标：正确 zh->en 配对的能量，应低于 batch 内错配配对。
            pair_loss = F.relu(pos_energy - neg_energy + args.margin).mean()
            cand_loss = candidate_ranking_loss(
                backbone,
                energy_head,
                intent,
                zh_texts,
                en_texts,
                args,
                device,
            )
            loss = pair_loss + cand_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(adapter.parameters()) + list(energy_head.parameters()), 1.0)
            opt.step()

            total_loss += float(loss.detach())
            total_batches += 1
        avg_loss = total_loss / max(total_batches, 1)
        print(f"[p1-train] epoch {epoch} rank_loss={avg_loss:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, DEFAULT_CKPT_NAME)
    torch.save(
        {
            "adapter": adapter.state_dict(),
            "energy_head": energy_head.state_dict(),
            "config": vars(args),
        },
        ckpt_path,
    )
    print(f"[p1-train] saved adapter/head -> {ckpt_path}")
    return ckpt_path


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.check_env:
        print_env_check(check_environment(args))
        return 0
    if not args.run:
        print_dry_run(args)
        return 0
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
