# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/training/code_train.py —— 从 0 训练字符级代码模型（显式自由能 / PER / Transformer）
=======================================================================================================
默认使用 **FreeEnergyLM**：显式定义自由能，以共享动力学从不稳定状态弛豫到稳定状态；
同时保留旧 **SeqEnergyNet**（`--arch per`）与标准 Transformer（`--arch transformer`）作为历史基线。
三者共用字符语料、tokenizer、训练与评估路径，便于后续做诚实对照。

为什么字符级：代码字符集小（ASCII 为主，词表一两百），局部结构强，字级无需 BPE 即可建模，
且与本项目既有的字符级架构一致。指标用 **bits/char (bpc)**——跨 tokenizer/arch 可比。

口径（标准自回归 next-char LM）：
    语料 = data/code/python_corpus.txt（prepare_code.py 产出的连续字符流）
    定长窗口切块；位置 i 预测下一字符；损失=交叉熵；指标=held-out bpc 与 ppl。
    旧 SeqEnergyNet 输出 -logits；FreeEnergyLM/Transformer 直接输出 logits；下游统一换算。

公平对照要点：两 arch 共用同一 corpus + 同 seed → **同一组验证块**；同 tokenizer（同 corpus 同 min_freq
确定性一致）；同 ctx / batch / lr / 调度 / bf16；用 --max-steps 对齐"看到的 token 数"。

运行：
    准备数据： python -m fe_llm.energy_lm.data.prepare_code --target-mb 200
    新核心自检：python -m fe_llm.energy_lm.training.code_train --smoke --arch free_energy
    冒烟自检： python -m fe_llm.energy_lm.training.code_train --smoke --arch transformer
    PER 训练： python -m fe_llm.energy_lm.training.code_train --arch per --hours 4.5 --dim 768 --depth 12
    TF 对照：  python -m fe_llm.energy_lm.training.code_train --arch transformer --dim 768 --depth 7 --max-steps 78000
    采样测试： python -m fe_llm.energy_lm.training.code_train --sample --arch per --prompt "def quicksort(arr):"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn.functional as F

from fe_llm.config import get_device
from fe_llm.energy_lm.models.free_energy_lm import FreeEnergyLM
from fe_llm.energy_lm.models.seq_net import SeqEnergyNet
from fe_llm.energy_lm.models.transformer_lm import CharTransformerLM
from fe_llm.energy_lm.models.tokenizer import CharTokenizer

CKPT_DIR = os.path.join("checkpoints", "energy_lm")
CKPT_TOK = os.path.join(CKPT_DIR, "code_tokenizer.json")
# PER 默认路径（向后兼容 code_web 等导入）
CKPT_NET = os.path.join(CKPT_DIR, "code_model.pt")
CKPT_LAST = os.path.join(CKPT_DIR, "code_model_last.pt")
CKPT_META = os.path.join(CKPT_DIR, "code_model_meta.json")

CORPUS = os.path.join("data", "code", "python_corpus.txt")

PROBES = [
    "def quicksort(arr):\n",
    "import numpy as np\n\n",
    "class Stack:\n",
    "for i in range(10):\n    ",
    "def fibonacci(n):\n    ",
    "if __name__ == \"__main__\":\n",
]


# ----------------------------------------------------------------------------- arch / 路径
def ckpt_paths(arch: str):
    """按 arch 区分 checkpoint 路径；所有架构共用 tokenizer。"""
    suffix = {"per": "", "transformer": "_tf", "free_energy": "_fe"}[arch]
    return (os.path.join(CKPT_DIR, f"code_model{suffix}.pt"),
            os.path.join(CKPT_DIR, f"code_model{suffix}_last.pt"),
            CKPT_TOK,
            os.path.join(CKPT_DIR, f"code_model{suffix}_meta.json"))


def build_model(args, vocab: int):
    """按 arch 造模型，返回 (net, 描述标签)。"""
    if args.arch == "transformer":
        net = CharTransformerLM(vocab, args.ctx, dim=args.dim, depth=args.depth,
                                n_heads=args.heads, ffn_mult=args.ffn_mult, dropout=args.dropout)
        tag = f"标准 Transformer (ffn×{args.ffn_mult})"
    elif args.arch == "free_energy":
        net = FreeEnergyLM(vocab, args.ctx, dim=args.dim,
                           relaxation_steps=args.relax_steps,
                           tolerance=args.relax_tolerance)
        tag = f"显式自由能动力学（共享弛豫×≤{args.relax_steps}）"
    else:
        net = SeqEnergyNet(vocab, args.ctx, dim=args.dim, depth=args.depth, n_heads=args.heads,
                           dropout=args.dropout, use_synapse=not args.no_synapse)
        tag = "阉割(无#2)" if args.no_synapse else "完整 PER 原型(含可学突触#2)"
    return net, tag


def _logits(net, seq):
    """统一换算：旧 SeqEnergyNet 返回 -logits，其余模型直接返回 logits。"""
    out = net(seq)
    return -out if getattr(net, "returns_energy", True) else out


def load_any(path: str, map_location: str = "cpu"):
    """按 checkpoint 里的 arch 字段加载正确的模型类（缺省视为 per）。"""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if ck.get("arch") == "transformer":
        return CharTransformerLM.load(path, map_location)
    if ck.get("arch") == "free_energy":
        return FreeEnergyLM.load(path, map_location)
    return SeqEnergyNet.load(path, map_location)


# ----------------------------------------------------------------------------- 数据
def build_tokenizer(text: str, min_freq: int) -> CharTokenizer:
    cnt = Counter(text)
    chars = sorted(c for c, n in cnt.items() if n >= min_freq)
    return CharTokenizer(chars)


def encode_corpus(text: str, tok: CharTokenizer) -> np.ndarray:
    cps = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    max_cp = int(cps.max()) if len(cps) else 0
    table = np.full(max_cp + 1, tok.unk_id, dtype=np.int64)
    for t, i in tok.tok_to_id.items():
        if len(t) == 1:
            o = ord(t)
            if o <= max_cp:
                table[o] = i
    return table[cps]


def make_chunks(ids: np.ndarray, L: int) -> np.ndarray:
    n = (len(ids) // L) * L
    return ids[:n].reshape(-1, L)


# ----------------------------------------------------------------------------- 评估/生成
@torch.no_grad()
def eval_bpc(net, chunks, device, batch, amp_dtype, max_batches=200) -> tuple[float, float]:
    net.eval()
    tot, ntok = 0.0, 0
    vocab = net.vocab_size
    for bi, s in enumerate(range(0, len(chunks), batch)):
        if bi >= max_batches:
            break
        seq = torch.as_tensor(chunks[s:s + batch], device=device, dtype=torch.long)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = _logits(net, seq)
            l = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab).float(),
                                seq[:, 1:].reshape(-1), reduction="sum")
        tot += float(l); ntok += seq[:, 1:].numel()
    nats = tot / max(1, ntok)
    return nats / math.log(2), math.exp(min(20.0, nats))


@torch.no_grad()
def generate(net, tok, prompt, ctx, device, max_new=160, temperature=0.7, top_k=20,
             top_p=0.0, repetition_penalty=1.0, amp_dtype=None) -> str:
    """自回归续写。支持 top-k / top-p(nucleus) / 重复惩罚。"""
    net.eval()
    ids = tok.encode(prompt)
    n_prompt = len(ids)
    if not ids:
        ids = [tok.bos_id]; n_prompt = 0
    specials = (tok.pad_id, tok.bos_id, tok.sep_id, tok.mask_id, tok.unk_id)
    for _ in range(max_new):
        window = ids[-ctx:]
        seq = torch.as_tensor([window], device=device, dtype=torch.long)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = _logits(net, seq)[0, len(window) - 1].float()
        for sp in specials:
            logits[sp] = -1e9
        if repetition_penalty and repetition_penalty != 1.0:
            for tid in set(ids[-ctx:]):
                if logits[tid] > 0:
                    logits[tid] /= repetition_penalty
                else:
                    logits[tid] *= repetition_penalty
        if temperature <= 1e-5:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if 0.0 < top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                probs = torch.softmax(sl, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                keep = cum <= top_p
                keep[0] = True
                sl = sl[keep]; si = si[keep]
                p = torch.softmax(sl, dim=-1)
                nxt = int(si[int(torch.multinomial(p, 1))])
            elif top_k > 0:
                v, idx = torch.topk(logits, min(top_k, logits.numel()))
                p = torch.softmax(v, dim=-1)
                nxt = int(idx[int(torch.multinomial(p, 1))])
            else:
                nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
        if nxt == tok.eos_id:
            break
        ids.append(nxt)
    return "".join(tok.id_to_tok[i] for i in ids[n_prompt:])


def syntactic_valid(code: str) -> bool:
    """用 ast.parse 判断是否为语法合法的 Python（容错：补不全的尾部按合法看待前缀）。"""
    import ast
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


# ----------------------------------------------------------------------------- 学习率
def lr_at(step: int, warmup: int, total: int, peak: float, floor_frac: float = 0.02) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if total <= warmup:
        return peak
    prog = min(1.0, (step - warmup) / (total - warmup))
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return peak * (floor_frac + (1 - floor_frac) * cos)


# ----------------------------------------------------------------------------- 训练
def train(args) -> None:
    device = args.device.strip() or get_device()
    amp_dtype = torch.bfloat16 if (args.amp and device.startswith("cuda")) else None
    net_path, last_path, tok_path, meta_path = ckpt_paths(args.arch)
    print(f"[code] arch={args.arch} 设备={device} amp={'bf16' if amp_dtype else 'off'}", flush=True)

    if not os.path.exists(args.corpus):
        print(f"[code] 找不到语料 {args.corpus}，请先：python -m fe_llm.energy_lm.data.prepare_code")
        return
    with open(args.corpus, "r", encoding="utf-8") as f:
        text = f.read()
    if args.max_train_mb > 0:
        text = text[: int(args.max_train_mb * 1_000_000)]
    print(f"[code] 语料字符数={len(text)/1e6:.1f}M", flush=True)

    tok = build_tokenizer(text, args.min_char_freq)
    print(f"[code] 字表（含特殊符）={tok.vocab_size}", flush=True)
    ids = encode_corpus(text, tok)
    del text
    chunks = make_chunks(ids, args.ctx)
    del ids
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(chunks))
    chunks = chunks[perm]
    n_val = max(1, int(len(chunks) * args.val_frac))
    val_chunks = chunks[:n_val]
    train_chunks = chunks[n_val:]
    print(f"[code] 块大小={args.ctx}  训练块={len(train_chunks)}  验证块={len(val_chunks)}（同 seed→两 arch 同验证集）", flush=True)

    net, tag = build_model(args, tok.vocab_size)
    net = net.to(device)
    n_params = sum(p.numel() for p in net.parameters())
    if isinstance(net, FreeEnergyLM):
        shape = f"dim={net.dim} relax_steps≤{net.relaxation_steps} tol={net.tolerance:g}"
    else:
        shape = f"dim={net.dim} depth={net.depth} heads={net.n_heads}"
    print(f"[code] 参数量={n_params/1e6:.2f}M  {shape} · {tag}", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    start_step = 0
    if args.resume and os.path.exists(last_path):
        try:
            ck = torch.load(last_path, map_location=device, weights_only=False)
            net.load_state_dict(ck["state_dict"])
            start_step = int(ck.get("step", 0))
            print(f"[code] 续训：从 {last_path} 恢复，step={start_step}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[code] 续训失败（{e}），从头开始", flush=True)

    os.makedirs(CKPT_DIR, exist_ok=True)
    tok.save(tok_path)

    budget = args.hours * 3600.0
    t0 = time.time()
    last_eval = t0
    step = start_step
    best_bpc = float("inf")
    train_idx = np.arange(len(train_chunks))
    if args.max_steps > 0:
        est_total = args.max_steps
    else:
        est_total = max(args.warmup + 1, int(budget / 0.2))
    bs = args.batch
    accum = max(1, args.accum)
    measured = False
    t_measure = None
    step_measure = None
    paths = (net_path, last_path, tok_path, meta_path)

    cap = f"max_steps={args.max_steps}" if args.max_steps > 0 else f"预算 {args.hours:.1f}h"
    print(f"[code] 开始训练：{cap}，bs={bs}×accum{accum}，eval 每 {args.eval_min:.0f} 分钟", flush=True)
    net.train()
    running = 0.0
    running_fe = 0.0
    nrun = 0
    stop = False
    while not stop:
        np.random.default_rng(args.seed + step).shuffle(train_idx)
        opt.zero_grad(set_to_none=True)
        for bstart in range(0, len(train_idx) - bs, bs):
            elapsed = time.time() - t0
            if elapsed >= budget or (args.max_steps > 0 and step >= args.max_steps):
                stop = True
                break
            sel = train_idx[bstart:bstart + bs]
            seq = torch.as_tensor(train_chunks[sel], device=device, dtype=torch.long)
            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = _logits(net, seq)
                task_loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, tok.vocab_size),
                                            seq[:, 1:].reshape(-1))
                fe_loss = getattr(net, "last_free_energy_loss", None)
                if isinstance(net, FreeEnergyLM) and fe_loss is not None:
                    loss = task_loss + args.free_energy_weight * fe_loss
                else:
                    loss = task_loss
            (loss / accum).backward()
            running += float(task_loss.detach())
            if isinstance(net, FreeEnergyLM) and fe_loss is not None:
                running_fe += float(fe_loss.detach())
            nrun += 1
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, args.warmup, est_total, args.lr)
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1

            if args.max_steps == 0:
                # warmup 后用实测步速校正总步数（让 cosine 落到预算末端）
                if not measured and step - start_step >= args.warmup + 50:
                    t_measure = time.time(); step_measure = step
                    measured = True
                elif measured and step - step_measure == 200:
                    per_step = (time.time() - t_measure) / 200.0
                    est_total = step + int((budget - (time.time() - t0)) / max(1e-6, per_step))
                    print(f"[code] 实测步速 {per_step*1000:.0f}ms/step → 估算总步数≈{est_total}", flush=True)

            if step % args.log_every == 0:
                denom = max(1, nrun)
                avg = running / denom
                running = 0.0
                fe_note = ""
                if isinstance(net, FreeEnergyLM):
                    fe_avg = running_fe / denom
                    fe_note = f" | residual_F/dim {fe_avg:.4f}"
                    running_fe = 0.0
                nrun = 0
                gpu = (torch.cuda.max_memory_allocated() / 1e9) if device.startswith("cuda") else 0.0
                print(f"[code] step {step:>7} | loss {avg:.4f} bpc {avg/math.log(2):.3f} | "
                      f"lr {opt.param_groups[0]['lr']:.2e}{fe_note} | {elapsed/60:.1f}min | peakGPU {gpu:.2f}G", flush=True)

            if time.time() - last_eval >= args.eval_min * 60:
                _do_eval_and_ckpt(net, tok, val_chunks, device, bs, amp_dtype, args, step, t0,
                                  best_bpc_ref := [best_bpc], paths)
                best_bpc = best_bpc_ref[0]
                last_eval = time.time()
                net.train()

    print(f"[code] 停止（{(time.time()-t0)/3600:.2f}h, step {step}），收尾评估…", flush=True)
    best_bpc_ref = [best_bpc]
    _do_eval_and_ckpt(net, tok, val_chunks, device, bs, amp_dtype, args, step, t0, best_bpc_ref, paths, final=True)
    print(f"[code] 完成。arch={args.arch} best_bpc={best_bpc_ref[0]:.4f} → {net_path}", flush=True)


def _do_eval_and_ckpt(net, tok, val_chunks, device, bs, amp_dtype, args, step, t0, best_ref, paths, final=False):
    net_path, last_path, tok_path, meta_path = paths
    bpc, ppl = eval_bpc(net, val_chunks, device, bs, amp_dtype)
    elapsed = time.time() - t0
    flag = ""
    _save(net, step, last_path)
    if bpc < best_ref[0]:
        best_ref[0] = bpc
        _save(net, step, net_path)
        flag = " *best"
    # 采样 + 语法合法率（客观看生成结构质量）
    probes = PROBES if final else PROBES[:3]
    samples = []
    n_valid = 0
    for p in probes:
        out = generate(net, tok, p, net.max_len, device, max_new=120,
                       temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                       repetition_penalty=args.rep_penalty, amp_dtype=amp_dtype)
        full = p + out
        ok = syntactic_valid(full)
        n_valid += int(ok)
        samples.append({"prompt": p, "text": full, "ast_ok": ok})
    valid_rate = n_valid / max(1, len(probes))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"arch": args.arch, "step": step, "val_bpc": round(bpc, 4), "val_ppl": round(ppl, 2),
                   "best_bpc": round(best_ref[0], 4), "elapsed_h": round(elapsed / 3600, 3),
                   "dim": net.dim, "depth": net.depth, "n_heads": net.n_heads,
                   "ctx": net.max_len, "vocab": net.vocab_size,
                   "params_M": round(sum(p.numel() for p in net.parameters()) / 1e6, 2),
                   "ffn_mult": getattr(net, "ffn_mult", None),
                   "use_synapse": getattr(net, "use_synapse", None),
                   "ast_valid_rate": round(valid_rate, 3)}, f, ensure_ascii=False, indent=2)
    print(f"[code] === eval @ step {step} ({elapsed/60:.1f}min) | val_bpc {bpc:.4f} ppl {ppl:.2f} | "
          f"ast合法率 {valid_rate:.0%}{flag} ===", flush=True)
    for s in samples[:3]:
        shown = s["text"].replace("\n", "\\n")
        print(f"[code]   [{'ok' if s['ast_ok'] else '×'}][{s['prompt'].strip()[:20]!r}] → {shown[:150]}", flush=True)


def _save(net, step, path):
    if isinstance(net, FreeEnergyLM):
        torch.save(net.checkpoint(step), path)
    elif getattr(net, "returns_energy", True):  # SeqEnergyNet / PER
        torch.save({"arch": "per", "vocab_size": net.vocab_size, "max_len": net.max_len, "dim": net.dim,
                    "depth": net.depth, "n_heads": net.n_heads, "use_synapse": net.use_synapse,
                    "step": step, "state_dict": net.state_dict()}, path)
    else:  # Transformer
        torch.save({"arch": "transformer", "vocab_size": net.vocab_size, "max_len": net.max_len,
                    "dim": net.dim, "depth": net.depth, "n_heads": net.n_heads,
                    "ffn_mult": net.ffn_mult, "step": step, "state_dict": net.state_dict()}, path)


# ----------------------------------------------------------------------------- 采样/冒烟
def sample(args) -> None:
    device = args.device.strip() or get_device()
    amp_dtype = torch.bfloat16 if (args.amp and device.startswith("cuda")) else None
    net_path, last_path, tok_path, _ = ckpt_paths(args.arch)
    path = net_path if os.path.exists(net_path) else last_path
    if not (os.path.exists(path) and os.path.exists(tok_path)):
        print(f"[code] 找不到 {args.arch} 模型，请先训练。")
        return
    net = load_any(path, device).to(device)
    tok = CharTokenizer.load(tok_path)
    prompts = [args.prompt] if args.prompt else PROBES
    for p in prompts:
        out = generate(net, tok, p, net.max_len, device, max_new=args.max_new,
                       temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                       repetition_penalty=args.rep_penalty, amp_dtype=amp_dtype)
        print("=" * 60)
        print(p + out)
        print(f"  [ast 合法: {syntactic_valid(p + out)}]")


def smoke(args) -> None:
    """小配置快速自检：跑通 CUDA + 前后向 + 采样 + 报告峰值显存，不产出正式权重。"""
    device = args.device.strip() or get_device()
    amp_dtype = torch.bfloat16 if (args.amp and device.startswith("cuda")) else None
    print(f"[smoke] arch={args.arch} 设备={device} amp={'bf16' if amp_dtype else 'off'}", flush=True)
    if not os.path.exists(args.corpus):
        text = ("def f(x):\n    return x*x\n\n" * 4000)
        print("[smoke] 无语料，用合成片段自检", flush=True)
    else:
        with open(args.corpus, "r", encoding="utf-8") as f:
            text = f.read(2_000_000)
    tok = build_tokenizer(text, 1)
    ids = encode_corpus(text, tok)
    chunks = make_chunks(ids, args.ctx)
    net, tag = build_model(args, tok.vocab_size)
    net = net.to(device)
    n_params = sum(p.numel() for p in net.parameters())
    if isinstance(net, FreeEnergyLM):
        shape = f"dim={net.dim} relax_steps≤{net.relaxation_steps} tol={net.tolerance:g}"
    else:
        shape = f"dim={net.dim} depth={net.depth} heads={net.n_heads}"
    print(f"[smoke] {tag} 参数={n_params/1e6:.2f}M {shape} ctx={args.ctx} bs={args.batch} vocab={tok.vocab_size}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    net.train()
    t0 = time.time()
    nstep = 30
    loss = None
    for i in range(nstep):
        sel = np.random.randint(0, len(chunks), size=args.batch)
        seq = torch.as_tensor(chunks[sel], device=device, dtype=torch.long)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = _logits(net, seq)
            task_loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, tok.vocab_size), seq[:, 1:].reshape(-1))
            fe_loss = getattr(net, "last_free_energy_loss", None)
            loss = (task_loss + args.free_energy_weight * fe_loss
                    if isinstance(net, FreeEnergyLM) and fe_loss is not None else task_loss)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    dt = (time.time() - t0) / nstep
    gpu = (torch.cuda.max_memory_allocated() / 1e9) if device.startswith("cuda") else 0.0
    fe_note = (f" | residual_F/dim {float(net.last_free_energy_loss.detach()):.3f}"
               if isinstance(net, FreeEnergyLM) and net.last_free_energy_loss is not None else "")
    print(f"[smoke] {nstep} 步 OK | {dt*1000:.0f}ms/step | 峰值显存 {gpu:.2f}G | "
          f"末步 task_loss {float(task_loss.detach()):.3f}{fe_note}", flush=True)
    out = generate(net, tok, "def ", args.ctx, device, max_new=40, amp_dtype=amp_dtype)
    print(f"[smoke] 采样 'def ' → {('def ' + out)!r}", flush=True)
    if dt > 0:
        print(f"[smoke] 估算 4.5h ≈ {int(4.5*3600/dt)} 步（仅供选规模参考）", flush=True)


# ----------------------------------------------------------------------------- CLI
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="从 0 训练字符级代码模型（显式自由能 / PER / Transformer）。")
    ap.add_argument("--arch", choices=["free_energy", "per", "transformer"], default="free_energy",
                    help="模型架构；默认使用显式自由能动力学")
    ap.add_argument("--ffn-mult", type=int, default=4, help="Transformer 前馈倍数（其他架构忽略）")
    ap.add_argument("--relax-steps", type=int, default=8,
                    help="显式自由能模型的最大弛豫步数（不是网络层数）")
    ap.add_argument("--relax-tolerance", type=float, default=1e-4,
                    help="显式自由能模型逐位置自适应停止阈值")
    ap.add_argument("--free-energy-weight", type=float, default=2.0,
                    help="显式自由能模型外循环的残余自由能权重（机制裁决默认 2.0）")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--hours", type=float, default=4.5, help="训练时间预算（小时）")
    ap.add_argument("--max-steps", type=int, default=0, help=">0 时按步数停止（对齐两 arch 看到的 token）")
    ap.add_argument("--device", default="", help="留空=自动；可填 cpu / cuda")
    ap.add_argument("--dim", type=int, default=640)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=256, help="上下文窗口（字符）")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--accum", type=int, default=2, help="梯度累积步数（等效 batch=batch×accum）")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--min-char-freq", type=int, default=10, help="字符入表的最低频次（稀有→UNK）")
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--max-train-mb", type=float, default=0, help=">0 时只用前 N MB 语料（调试）")
    ap.add_argument("--eval-min", type=float, default=12.0, help="每隔多少分钟 eval+采样+存档")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.0, help=">0 时启用 nucleus 采样（覆盖 top-k）")
    ap.add_argument("--rep-penalty", type=float, default=1.0, help="重复惩罚（>1 抑制重复）")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", dest="amp", action="store_true", default=True, help="bf16 混合精度（默认开）")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--no-synapse", action="store_true", help="消融：关掉可学突触 #2（仅 per）")
    ap.add_argument("--resume", action="store_true", help="从 *_last.pt 续训")
    ap.add_argument("--sample", action="store_true", help="加载已训模型采样")
    ap.add_argument("--prompt", default="", help="采样起手 prompt")
    ap.add_argument("--smoke", action="store_true", help="小配置自检（验证显存/跑通）")
    return ap


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = build_arg_parser().parse_args(argv)
    if args.smoke:
        smoke(args)
    elif args.sample:
        sample(args)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
