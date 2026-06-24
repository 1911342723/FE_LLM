# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/training/chat_train.py —— 训练一个小聊天模型（从 0 可溯源架构）
================================================================================
用**自有的因果 PER 机制**（SeqEnergyNet，预测-误差弛豫 + 因果掩码，**非 Transformer 底座**）
从 0 训一个字符级中文聊天模型。已被 `lm_scaling_eval` 证明：该架构 held-out 困惑度随规模单调下降、
与 Transformer 同档——所以这是"能随规模变好"的正经训练入口，而不是玩具。

数据：data/dialogue/*.jsonl（{"prompt","response"} 格式）。默认 dialogues.jsonl（干净短对话），
可 --extra 叠加 dialogues_lccc_highentropy.jsonl（真实口语对话）扩充。

序列布局（自回归 teacher-forcing，链式分解精确无独立假设）：
    [prompt] [SEP] [BOS] r1 r2 … rn [EOS]
    只在回应区(BOS..rn)计损失：位置 i 预测下一字 → 学"给定上文与已生成前缀，推进一个字"。

口径：训练集 next-char loss/acc + **held-out 验证困惑度**（按回应区）；存最佳(val)。
训完可 --sample 看样例回复、--chat 进入交互对话。

运行：
    训练：  python -m fe_llm.energy_lm.training.chat_train --epochs 40 --dim 256 --depth 5 --extra
    采样：  python -m fe_llm.energy_lm.training.chat_train --sample
    对话：  python -m fe_llm.energy_lm.training.chat_train --chat
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn.functional as F

from fe_llm.config import get_device
from fe_llm.energy_lm.models.seq_net import SeqEnergyNet
from fe_llm.energy_lm.models.tokenizer import CharTokenizer

CKPT_DIR = os.path.join("checkpoints", "energy_lm")
CKPT_NET = os.path.join(CKPT_DIR, "chat_model.pt")
CKPT_TOK = os.path.join(CKPT_DIR, "chat_tokenizer.json")

DATA_MAIN = os.path.join("data", "dialogue", "dialogues.jsonl")
DATA_EXTRA = os.path.join("data", "dialogue", "dialogues_lccc_highentropy.jsonl")

PROBES = ["你好", "在吗", "你是谁", "今天天气怎么样", "谢谢你", "我有点难过", "晚安"]


# ----------------------------------------------------------------------------- 数据
def load_pairs(paths: list[str], max_pairs: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[chat] 跳过不存在的数据：{path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                    p, r = o["prompt"].strip(), o["response"].strip()
                except (ValueError, KeyError):
                    continue
                if p and r:
                    out.append((p, r))
    if max_pairs > 0:
        out = out[:max_pairs]
    return out


def build_example(tok: CharTokenizer, prompt: str, response: str, max_len: int):
    """拼 [p][SEP][BOS] r... [EOS]；返回 ids 与回应区(BOS..rn)监督标记。位置 i 监督目标=ids[i+1]。"""
    c = tok.encode(prompt)
    r = tok.encode(response)
    seq = c + [tok.sep_id, tok.bos_id] + r + [tok.eos_id]
    if len(seq) > max_len:
        return None  # 超长直接丢，保证回应区完整
    bos_pos = len(c) + 1
    sup = [False] * len(seq)
    for i in range(bos_pos, len(seq) - 1):
        sup[i] = True
    seq = seq + [tok.pad_id] * (max_len - len(seq))
    sup = sup + [False] * (max_len - len(sup))
    return seq, sup


def make_dataset(tok, pairs, max_len):
    seqs, sups = [], []
    for p, r in pairs:
        ex = build_example(tok, p, r, max_len)
        if ex is None:
            continue
        seqs.append(ex[0]); sups.append(ex[1])
    return np.array(seqs, dtype=np.int64), np.array(sups, dtype=bool)


# ----------------------------------------------------------------------------- 评估/生成
@torch.no_grad()
def eval_val(net, seqs, sups, tok, device, batch):
    net.eval()
    tot, ntok = 0.0, 0
    for s in range(0, len(seqs), batch):
        seq = torch.tensor(seqs[s:s + batch], device=device)
        sup = torch.tensor(sups[s:s + batch], device=device)
        logits = -net(seq)
        pl = logits[:, :-1, :]; tg = seq[:, 1:]; m = sup[:, :-1]
        if m.any():
            l = F.cross_entropy(pl[m], tg[m], reduction="sum")
            tot += float(l); ntok += int(m.sum())
    nats = tot / max(1, ntok)
    return nats, float(np.exp(nats))


@torch.no_grad()
def generate(net, tok, prompt, max_len, device, temperature=0.8, top_k=20):
    net.eval()
    ids = tok.encode(prompt)[: max_len - 3] + [tok.sep_id, tok.bos_id]
    start = len(ids)
    for _ in range(max_len - start):
        pad = ids + [tok.pad_id] * (max_len - len(ids))
        seq = torch.tensor([pad[:max_len]], device=device)
        logits = -net(seq)[0, len(ids) - 1]
        for sp in (tok.pad_id, tok.bos_id, tok.sep_id, tok.mask_id, tok.unk_id):
            logits[sp] = -1e9
        if temperature <= 1e-5:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k > 0:
                v, idx = torch.topk(logits, min(top_k, logits.numel()))
                probs = torch.softmax(v, dim=-1)
                nxt = int(idx[int(torch.multinomial(probs, 1))])
            else:
                nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
        if nxt == tok.eos_id:
            break
        ids.append(nxt)
    return "".join(tok.id_to_tok[i] for i in ids[start:])


# ----------------------------------------------------------------------------- 训练
def train(args):
    device = get_device()
    print(f"[chat] 设备：{device}")
    paths = [p.strip() for p in args.data.split(",") if p.strip()] + ([DATA_EXTRA] if args.extra else [])
    pairs = load_pairs(paths, args.max_pairs)
    if not pairs:
        print("[chat] 没有可用数据，退出。"); return
    chars = sorted({ch for p, r in pairs for ch in (p + r)})
    tok = CharTokenizer(chars)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in perm]
    n_val = max(1, int(len(pairs) * args.val_frac))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    tr_seq, tr_sup = make_dataset(tok, train_pairs, args.max_len)
    va_seq, va_sup = make_dataset(tok, val_pairs, args.max_len)
    print(f"[chat] 训练 {len(tr_seq)} / 验证 {len(va_seq)} 对（过滤超长后），字表 {tok.vocab_size}，定长 {args.max_len}")

    net = SeqEnergyNet(tok.vocab_size, args.max_len, dim=args.dim, depth=args.depth,
                       n_heads=args.heads, use_synapse=not args.no_synapse).to(device)
    syn_tag = "阉割(无突触#2)" if args.no_synapse else "完整原型(含可学突触#2)"
    print(f"[chat] 参数量 {sum(p.numel() for p in net.parameters())/1e6:.2f}M (dim={args.dim} depth={args.depth}) · {syn_tag}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    g = torch.Generator(device="cpu"); g.manual_seed(args.seed)
    best_val = 1e9
    for ep in range(1, args.epochs + 1):
        net.train()
        idx = torch.randperm(len(tr_seq), generator=g).tolist()
        tot, nc, nt = 0.0, 0, 0
        t0 = time.time()
        for s in range(0, len(idx), args.batch):
            ch = idx[s:s + args.batch]
            seq = torch.tensor(tr_seq[ch], device=device)
            sup = torch.tensor(tr_sup[ch], device=device)
            logits = -net(seq)
            pl = logits[:, :-1, :]; tg = seq[:, 1:]; m = sup[:, :-1]
            sl = pl[m]; t = tg[m]
            loss = F.cross_entropy(sl, t)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * t.numel()
            nc += int((sl.argmax(-1) == t).sum()); nt += int(t.numel())
        sched.step()
        tr_loss, tr_acc = tot / max(1, nt), nc / max(1, nt)
        val_nats, val_ppl = eval_val(net, va_seq, va_sup, tok, device, args.batch)
        flag = ""
        if val_nats < best_val:
            best_val = val_nats
            os.makedirs(CKPT_DIR, exist_ok=True)
            net.save(CKPT_NET); tok.save(CKPT_TOK)
            flag = " *best"
        print(f"[chat] ep {ep:3d} | train_loss={tr_loss:.4f} acc={tr_acc:.1%} | "
              f"val_loss={val_nats:.4f} val_ppl={val_ppl:.2f}{flag} | {time.time()-t0:.0f}s", flush=True)

    print(f"\n[chat] 完成，最佳 val_loss={best_val:.4f} (ppl={np.exp(best_val):.2f}) → {CKPT_NET}")
    if os.path.exists(CKPT_NET):
        net = SeqEnergyNet.load(CKPT_NET, map_location=device).to(device)  # 用最佳 checkpoint，而非过拟合的末轮
    print("[chat] 样例回复（最佳 checkpoint）：")
    for p in PROBES:
        print(f"  你: {p}\n  AI: {generate(net, tok, p, args.max_len, device, args.temperature, args.top_k)}")


def load_for_infer(device):
    if not os.path.exists(CKPT_NET):
        print(f"[chat] 找不到模型 {CKPT_NET}，请先训练。"); return None, None
    net = SeqEnergyNet.load(CKPT_NET, map_location=device).to(device)
    tok = CharTokenizer.load(CKPT_TOK)
    return net, tok


def sample(args):
    device = get_device()
    net, tok = load_for_infer(device)
    if net is None:
        return
    for p in PROBES:
        print(f"你: {p}\nAI: {generate(net, tok, p, net.max_len, device, args.temperature, args.top_k)}\n")


def chat(args):
    device = get_device()
    net, tok = load_for_infer(device)
    if net is None:
        return
    print("[chat] 进入对话（输入 exit/quit 退出）")
    while True:
        try:
            p = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if p.lower() in ("exit", "quit", "再见"):
            break
        if p:
            print("AI:", generate(net, tok, p, net.max_len, device, args.temperature, args.top_k))


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="从 0 可溯源架构训练小聊天模型（SeqEnergyNet/因果 PER）。")
    ap.add_argument("--data", default=DATA_MAIN)
    ap.add_argument("--extra", action="store_true", help="叠加 LCCC 真实口语对话扩充")
    ap.add_argument("--max-pairs", type=int, default=0, help="取前 n 对，0=全部")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-synapse", action="store_true", help="消融：关掉 PER 辨识点 #2 可学突触基底（退化为因果注意力）")
    ap.add_argument("--sample", action="store_true", help="加载已训模型打印样例回复")
    ap.add_argument("--chat", action="store_true", help="加载已训模型进入交互对话")
    return ap


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = build_arg_parser().parse_args(argv)
    if args.sample:
        sample(args)
    elif args.chat:
        chat(args)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
