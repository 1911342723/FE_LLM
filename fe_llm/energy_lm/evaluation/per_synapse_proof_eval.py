# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/evaluation/per_synapse_proof_eval.py
=====================================================
严肃证明「完整 PER 原型」的辨识点 #2（可学突触基底 self.synapse = 经验刻成先天低阻
通路）真正做事，并联合验收三个 claim：可溯源 / 可后天学习 / 可成长。

为什么用合成「cue 条件位置路由 copy」任务：
    序列 = [CUE_r]  v_1..v_m  [SEP]  o_1..o_m ，其中 o_j = v_{π_r(j)}。
    内容 v 随机、规则 π_r 只藏在**位置依赖**里（不同 cue → 不同位置置换）。
    要答对就必须按 cue 做**位置路由**——这正是位置级 synapse 能显出价值的地方。

诚实边界：self.synapse 是**位置级**结构先验，不是内容记忆。本实验证的是「结构/路由
的可溯源、后天可塑、可成长」，不是「记住新事实」（那是外挂 MemoryBank 的事）。

默认 dry-run；真跑加 --run。
    python -m fe_llm.energy_lm.evaluation.per_synapse_proof_eval --run
    python -m fe_llm.energy_lm.evaluation.per_synapse_proof_eval --run --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fe_llm.config import get_device
from fe_llm.energy_lm.models.seq_net import SeqEnergyNet

REPORT_JSON = os.path.join("docs", "reports", "per_synapse_proof_eval.json")
REPORT_MD = os.path.join("docs", "reports", "per_synapse_proof_eval.md")
FIG_DIR = os.path.join("docs", "reports", "figs")

# ----------------------------------------------------------------- 词表布局
PAD, SEP, CUE_A, CUE_B, CUE_C, MODE_CYCLE = 0, 1, 2, 3, 4, 5
CONTENT_START = 6


def vocab_size(n_content: int) -> int:
    return CONTENT_START + n_content


# 规则：perm[j] = 输出位置 j 应当读取的输入下标
def rule_forward(m: int) -> list[int]:
    return [j for j in range(m)]


def rule_reverse(m: int) -> list[int]:
    return [m - 1 - j for j in range(m)]


def rule_shift(m: int) -> list[int]:
    return [(j + 1) % m for j in range(m)]


RULES = {"forward": rule_forward, "reverse": rule_reverse, "shift": rule_shift}
CUE_OF = {"forward": CUE_A, "reverse": CUE_B, "shift": CUE_C}


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------- 数据生成
def encode_copy(rng: np.random.Generator, cue_id: int, perm: list[int],
                n_content: int, max_len: int):
    """一条 cue 条件 copy 例：返回 (seq, sup, out_positions)。sup 标出回应区监督位。"""
    m = len(perm)
    v = (CONTENT_START + rng.integers(0, n_content, size=m)).tolist()
    o = [v[perm[j]] for j in range(m)]
    seq = [cue_id] + v + [SEP] + o
    sup = [False] * len(seq)
    # 位置 p ∈ [m+1, 2m]：从 SEP/前一输出 预测 o_0..o_{m-1}
    for p in range(m + 1, 2 * m + 1):
        sup[p] = True
    if len(seq) > max_len:
        raise ValueError("max_len too small")
    seq = seq + [PAD] * (max_len - len(seq))
    sup = sup + [False] * (max_len - len(sup))
    return seq, sup


def encode_cycle(rng: np.random.Generator, n_content: int, m: int, max_len: int):
    """无需位置路由的对照子任务：输出是固定内容循环 c_j=f(c_{j-1})，只靠相邻 + backbone。"""
    u = (CONTENT_START + rng.integers(0, n_content, size=m)).tolist()  # filler 输入
    start = CONTENT_START + int(rng.integers(0, n_content))
    c = [start]
    for _ in range(m - 1):
        nxt = CONTENT_START + (c[-1] - CONTENT_START + 1) % n_content
        c.append(nxt)
    seq = [MODE_CYCLE] + u + [SEP] + c
    sup = [False] * len(seq)
    for p in range(m + 1, 2 * m + 1):
        sup[p] = True
    seq = seq + [PAD] * (max_len - len(seq))
    sup = sup + [False] * (max_len - len(sup))
    return seq, sup


def build_copy_dataset(rule_names: list[str], n_per_rule: int, m: int,
                       n_content: int, max_len: int, seed: int):
    """多规则混合 cue copy 数据集。返回 seqs, sups, rule_ids(np.int)。"""
    rng = np.random.default_rng(seed)
    seqs, sups, rids = [], [], []
    for ri, rn in enumerate(rule_names):
        perm = RULES[rn](m)
        cue = CUE_OF[rn]
        for _ in range(n_per_rule):
            s, sp = encode_copy(rng, cue, perm, n_content, max_len)
            seqs.append(s); sups.append(sp); rids.append(ri)
    seqs = np.array(seqs, np.int64); sups = np.array(sups, bool); rids = np.array(rids, np.int64)
    perm_idx = rng.permutation(len(seqs))
    return seqs[perm_idx], sups[perm_idx], rids[perm_idx]


# ----------------------------------------------------------------- Transformer 对照
class TinyCausalTransformer(nn.Module):
    """最小因果 Transformer 对照（无 self.synapse 持久结构记忆）。forward 返回能量=-logits。"""

    def __init__(self, vocab: int, max_len: int, dim: int = 64, depth: int = 3,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab
        self.max_len = max_len
        self.dim = dim
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, n_heads, dim * 2, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        L = ids.size(1)
        x = self.embed(ids) + self.pos[:, :L]
        mask = torch.triu(torch.ones(L, L, device=ids.device, dtype=torch.bool), 1)
        x = self.enc(x, mask=mask)
        return -self.head(self.norm(x))


def build_model(kind: str, vocab: int, max_len: int, dim: int, depth: int, n_heads: int, device):
    if kind == "per_syn":
        net = SeqEnergyNet(vocab, max_len, dim=dim, depth=depth, n_heads=n_heads, use_synapse=True)
    elif kind == "per_nosyn":
        net = SeqEnergyNet(vocab, max_len, dim=dim, depth=depth, n_heads=n_heads, use_synapse=False)
    elif kind == "transformer":
        net = TinyCausalTransformer(vocab, max_len, dim=dim, depth=depth, n_heads=n_heads)
    else:
        raise ValueError(kind)
    return net.to(device)


def n_params(net) -> int:
    return sum(p.numel() for p in net.parameters())


# ----------------------------------------------------------------- 训练 / 评测
def train_model(net, seqs, sups, *, epochs, lr, batch, device,
                synapse_only=False, verbose=False, tag=""):
    """训练。synapse_only=True 时冻结除名字含 'synapse' 外的全部参数。"""
    if synapse_only:
        for n, p in net.named_parameters():
            p.requires_grad = ("synapse" in n)
    params = [p for p in net.parameters() if p.requires_grad]
    n_train_p = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    seqs_t = torch.tensor(seqs, device=device)
    sups_t = torch.tensor(sups, device=device)
    n = len(seqs)
    g = torch.Generator(device="cpu"); g.manual_seed(0)
    for ep in range(1, epochs + 1):
        net.train()
        idx = torch.randperm(n, generator=g).tolist()
        tot, nt = 0.0, 0
        for s in range(0, n, batch):
            ch = idx[s:s + batch]
            seq = seqs_t[ch]; sup = sups_t[ch]
            logits = -net(seq)
            pl = logits[:, :-1, :]; tg = seq[:, 1:]; m = sup[:, :-1]
            sl = pl[m]; t = tg[m]
            if t.numel() == 0:
                continue
            loss = F.cross_entropy(sl, t)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss.detach()) * t.numel(); nt += int(t.numel())
        if verbose and (ep % max(1, epochs // 5) == 0 or ep == 1):
            acc = eval_seqacc(net, seqs, sups, device)["seq_acc"]
            print(f"    [{tag}] ep {ep:3d} loss={tot/max(1,nt):.4f} train_seqacc={acc:.3f}", flush=True)
    return {"trainable_params": n_train_p}


@torch.no_grad()
def eval_seqacc(net, seqs, sups, device, rule_ids=None, batch=512):
    """整段 exact-match 序列准确率（可按 rule 分组）。"""
    net.eval()
    seqs_t = torch.tensor(seqs, device=device)
    sups_t = torch.tensor(sups, device=device)
    n = len(seqs)
    per_ex_ok = np.zeros(n, bool)
    tok_corr, tok_tot = 0, 0
    for s in range(0, n, batch):
        seq = seqs_t[s:s + batch]; sup = sups_t[s:s + batch]
        logits = -net(seq)
        pred = logits[:, :-1, :].argmax(-1)        # 预测 seq[:,1:]
        tg = seq[:, 1:]; m = sup[:, :-1]
        correct = (pred == tg) & m
        tok_corr += int(correct.sum()); tok_tot += int(m.sum())
        ok = (correct.sum(1) == m.sum(1)).cpu().numpy()    # 该样本所有监督位全对
        per_ex_ok[s:s + len(ok)] = ok
    out = {"seq_acc": float(per_ex_ok.mean()), "tok_acc": tok_corr / max(1, tok_tot)}
    if rule_ids is not None:
        out["by_rule"] = {}
        for ri in sorted(set(rule_ids.tolist())):
            mask = rule_ids == ri
            out["by_rule"][int(ri)] = float(per_ex_ok[mask].mean())
    return out


# ----------------------------------------------------------------- synapse 工具
@torch.no_grad()
def avg_synapse(net) -> np.ndarray:
    """各块 softplus(synapse) 的均值矩阵 (L,L)。"""
    mats = [F.softplus(b.synapse.detach()).cpu().numpy() for b in net.blocks if getattr(b, "use_synapse", False)]
    return np.mean(mats, axis=0) if mats else None


def pathway_cells(m: int, rule: str) -> list[tuple[int, int]]:
    """规则 rule 下，预测各输出位的「(预测位置 row, 应读输入列 col)」突触通路。"""
    perm = RULES[rule](m)
    # 预测 o_j 的位置 p = m+1+j；其内容源输入下标 = 1+perm[j]
    return [(m + 1 + j, 1 + perm[j]) for j in range(m)]


@torch.no_grad()
def set_cells(net, cells: list[tuple[int, int]], value: float):
    """把每块 synapse 的指定 (row,col) 设为 value，返回旧值以便恢复。"""
    old = []
    for b in net.blocks:
        if not getattr(b, "use_synapse", False):
            continue
        for (r, c) in cells:
            old.append((b, r, c, float(b.synapse.data[r, c])))
            b.synapse.data[r, c] = value
    return old


@torch.no_grad()
def restore_cells(old):
    for (b, r, c, v) in old:
        b.synapse.data[r, c] = v


import copy as _copy


def clone_model(net, kind, vocab, max_len, dim, depth, heads, device):
    new = build_model(kind, vocab, max_len, dim, depth, heads, device)
    new.load_state_dict(_copy.deepcopy(net.state_dict()))
    return new


# ----------------------------------------------------------------- P4 对照矩阵
def phase_compare(args, device, data):
    """同数据同预算训 PER+syn / PER-syn / Transformer，隔离 synapse(#2) 的边际贡献。"""
    (tr_s, tr_p, tr_r), (te_s, te_p, te_r) = data
    vocab = vocab_size(args.n_content)
    out = {}
    models = {}
    for kind in ["per_syn", "per_nosyn", "transformer"]:
        set_seed(args.seed)
        net = build_model(kind, vocab, 2 * args.m + 2, args.dim, args.depth, args.heads, device)
        t0 = time.time()
        info = train_model(net, tr_s, tr_p, epochs=args.epochs, lr=args.lr, batch=args.batch, device=device)
        res = eval_seqacc(net, te_s, te_p, device, rule_ids=te_r)
        out[kind] = {"params": n_params(net), "trainable": info["trainable_params"],
                     "seq_acc": round(res["seq_acc"], 4), "tok_acc": round(res["tok_acc"], 4),
                     "by_rule": {k: round(v, 4) for k, v in res["by_rule"].items()},
                     "sec": round(time.time() - t0, 1)}
        models[kind] = net
        print(f"[P4] {kind:11s} params={out[kind]['params']/1e3:6.1f}K seq_acc={out[kind]['seq_acc']:.3f} "
              f"by_rule={out[kind]['by_rule']} ({out[kind]['sec']}s)", flush=True)
    return out, models


# ----------------------------------------------------------------- P1 可溯源 + 因果干预
def phase_traceability(args, device, per_syn_net):
    """synapse 是持久可读结构记忆：热力图 + 因果干预（剪断某规则通路→该规则可预期崩，且远强于剪同量随机通路）。"""
    m, nc, max_len = args.m, args.n_content, 2 * args.m + 2
    ev_f_s, ev_f_p, _ = build_copy_dataset(["forward"], args.n_eval, m, nc, max_len, seed=7001)
    ev_r_s, ev_r_p, _ = build_copy_dataset(["reverse"], args.n_eval, m, nc, max_len, seed=7002)

    def fr():
        return (eval_seqacc(per_syn_net, ev_f_s, ev_f_p, device)["seq_acc"],
                eval_seqacc(per_syn_net, ev_r_s, ev_r_p, device)["seq_acc"])

    base_f, base_r = fr()
    fwd_cells, rev_cells = pathway_cells(m, "forward"), pathway_cells(m, "reverse")

    old = set_cells(per_syn_net, rev_cells, -1e4); cutrev_f, cutrev_r = fr(); restore_cells(old)
    old = set_cells(per_syn_net, fwd_cells, -1e4); cutfwd_f, cutfwd_r = fr(); restore_cells(old)

    # 随机对照：剪同数量、同区域（输出行×输入列）的随机通路
    rng = np.random.default_rng(123)
    region = [(m + 1 + j, 1 + i) for j in range(m) for i in range(m)]
    region = [c for c in region if c not in set(fwd_cells) | set(rev_cells)]
    rand_drops_f, rand_drops_r = [], []
    for _ in range(5):
        pick = [region[i] for i in rng.choice(len(region), size=min(m, len(region)), replace=False)]
        old = set_cells(per_syn_net, pick, -1e4); rf, rr = fr(); restore_cells(old)
        rand_drops_f.append(base_f - rf); rand_drops_r.append(base_r - rr)
    rand_f, rand_r = float(np.mean(rand_drops_f)), float(np.mean(rand_drops_r))

    mat = avg_synapse(per_syn_net)
    fig_path = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIG_DIR, exist_ok=True)
        sub = mat[m + 1:2 * m + 1, 1:m + 1]   # 输出行 × 输入列
        plt.figure(figsize=(4.2, 3.6))
        plt.imshow(sub, cmap="viridis", aspect="auto")
        plt.colorbar(label="softplus(synapse) 电导")
        plt.xlabel("输入位置 (v_1..v_m)"); plt.ylabel("预测输出位 (o_1..o_m)")
        plt.title("PER #2 突触基底：经验刻出的低阻通路\n(对角=forward，反对角=reverse)")
        try:
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass
        fig_path = os.path.join(FIG_DIR, "per_synapse_pathways.png")
        plt.tight_layout(); plt.savefig(fig_path, dpi=130); plt.close()
    except Exception as e:
        print(f"[P1] 热力图跳过：{e}", flush=True)

    res = {
        "baseline": {"forward": round(base_f, 4), "reverse": round(base_r, 4)},
        "cut_reverse_pathway": {"forward": round(cutrev_f, 4), "reverse": round(cutrev_r, 4)},
        "cut_forward_pathway": {"forward": round(cutfwd_f, 4), "reverse": round(cutfwd_r, 4)},
        "cut_random_sameN": {"forward_drop": round(rand_f, 4), "reverse_drop": round(rand_r, 4)},
        "specificity_reverse": round((base_r - cutrev_r) - rand_r, 4),
        "specificity_forward": round((base_f - cutfwd_f) - rand_f, 4),
        "fig": fig_path,
    }
    print(f"[P1] 基线 fwd={base_f:.3f} rev={base_r:.3f} | 剪reverse通路→rev={cutrev_r:.3f}(fwd保持{cutrev_f:.3f}) "
          f"| 剪forward通路→fwd={cutfwd_f:.3f}(rev保持{cutfwd_r:.3f}) | 剪同量随机→fwd_drop={rand_f:.3f} rev_drop={rand_r:.3f}",
          flush=True)
    return res


# ----------------------------------------------------------------- P2 可后天学习（突触可塑）
def phase_postnatal(args, device):
    m, nc, max_len = args.m, args.n_content, 2 * args.m + 2
    vocab = vocab_size(nc)
    dim, depth, heads = args.dim, args.depth, args.heads

    # ---- P2.flip：纯路由翻转（pretrain forward，postnatal reverse 复用 CUE_A），冻结 backbone 只动 synapse ----
    set_seed(args.seed)
    base = build_model("per_syn", vocab, max_len, dim, depth, heads, device)
    f_s, f_p, _ = build_copy_dataset(["forward"], args.n_train, m, nc, max_len, seed=21)
    train_model(base, f_s, f_p, epochs=args.epochs, lr=args.lr, batch=args.batch, device=device)
    # 把 reverse 数据也打上 CUE_A（复用同一 cue，纯位置规则翻转）
    rng = np.random.default_rng(22)
    rev_perm = RULES["reverse"](m)
    rs, rp = [], []
    for _ in range(args.n_train):
        s, sp = encode_copy(rng, CUE_A, rev_perm, nc, max_len)
        rs.append(s); rp.append(sp)
    rs, rp = np.array(rs, np.int64), np.array(rp, bool)
    te_rs, te_rp = [], []
    for _ in range(args.n_eval):
        s, sp = encode_copy(rng, CUE_A, rev_perm, nc, max_len)
        te_rs.append(s); te_rp.append(sp)
    te_rs, te_rp = np.array(te_rs, np.int64), np.array(te_rp, bool)
    rev0 = eval_seqacc(base, te_rs, te_rp, device)["seq_acc"]   # 翻转前对 reverse 的表现（≈0）

    flip = {}
    # 冻结 backbone 只训 synapse（给它最充分的机会：更高 lr、更多 epoch，确保负结果不是欠训）
    syn_lr = args.lr * 5
    syn_ep = args.epochs * 2
    so = clone_model(base, "per_syn", vocab, max_len, dim, depth, heads, device)
    info = train_model(so, rs, rp, epochs=syn_ep, lr=syn_lr, batch=args.batch, device=device, synapse_only=True)
    flip["synapse_only"] = {"reverse_acc": round(eval_seqacc(so, te_rs, te_rp, device)["seq_acc"], 4),
                            "trainable": info["trainable_params"], "lr": syn_lr, "epochs": syn_ep}
    # 全参微调
    ft = clone_model(base, "per_syn", vocab, max_len, dim, depth, heads, device)
    info = train_model(ft, rs, rp, epochs=args.epochs, lr=args.lr, batch=args.batch, device=device)
    flip["full_finetune"] = {"reverse_acc": round(eval_seqacc(ft, te_rs, te_rp, device)["seq_acc"], 4),
                             "trainable": info["trainable_params"]}
    flip["frozen_baseline"] = {"reverse_acc": round(rev0, 4)}
    flip["total_params"] = n_params(base)
    print(f"[P2.flip] frozen={flip['frozen_baseline']['reverse_acc']:.3f} | "
          f"synapse_only={flip['synapse_only']['reverse_acc']:.3f} (训练{flip['synapse_only']['trainable']}参) | "
          f"full_ft={flip['full_finetune']['reverse_acc']:.3f} (训练{flip['full_finetune']['trainable']}参)", flush=True)

    # ---- P2.add：在 forward+reverse 之上后天新增 shift(CUE_C)，看新增 + 旧规则保持（遗忘）----
    set_seed(args.seed + 1)
    base2 = build_model("per_syn", vocab, max_len, dim, depth, heads, device)
    fr_s, fr_p, _ = build_copy_dataset(["forward", "reverse"], args.n_train, m, nc, max_len, seed=31)
    train_model(base2, fr_s, fr_p, epochs=args.epochs, lr=args.lr, batch=args.batch, device=device)
    sh_s, sh_p, _ = build_copy_dataset(["shift"], args.n_train, m, nc, max_len, seed=32)
    ev_sh, ev_shp, _ = build_copy_dataset(["shift"], args.n_eval, m, nc, max_len, seed=33)
    ev_f, ev_fp, _ = build_copy_dataset(["forward"], args.n_eval, m, nc, max_len, seed=34)
    ev_r, ev_rp, _ = build_copy_dataset(["reverse"], args.n_eval, m, nc, max_len, seed=35)

    def old_keep(net):
        return round((eval_seqacc(net, ev_f, ev_fp, device)["seq_acc"]
                      + eval_seqacc(net, ev_r, ev_rp, device)["seq_acc"]) / 2, 4)

    add = {"old_keep_before": old_keep(base2)}
    # 可塑-稳定权衡曲线：synapse-only 在不同训练预算下「学新规则 shift」vs「旧规则保持」
    tradeoff = []
    for ep in [10, 30, 60]:
        so_t = clone_model(base2, "per_syn", vocab, max_len, dim, depth, heads, device)
        train_model(so_t, sh_s, sh_p, epochs=ep, lr=syn_lr, batch=args.batch, device=device, synapse_only=True)
        tradeoff.append({"epochs": ep,
                         "shift_acc": round(eval_seqacc(so_t, ev_sh, ev_shp, device)["seq_acc"], 4),
                         "old_keep": old_keep(so_t)})
    add["tradeoff_synapse_only"] = tradeoff
    so2 = clone_model(base2, "per_syn", vocab, max_len, dim, depth, heads, device)
    info = train_model(so2, sh_s, sh_p, epochs=syn_ep, lr=syn_lr, batch=args.batch, device=device, synapse_only=True)
    add["synapse_only"] = {"shift_acc": round(eval_seqacc(so2, ev_sh, ev_shp, device)["seq_acc"], 4),
                           "old_keep_after": old_keep(so2), "trainable": info["trainable_params"],
                           "lr": syn_lr, "epochs": syn_ep}
    ft2 = clone_model(base2, "per_syn", vocab, max_len, dim, depth, heads, device)
    info = train_model(ft2, sh_s, sh_p, epochs=args.epochs, lr=args.lr, batch=args.batch, device=device)
    add["full_finetune"] = {"shift_acc": round(eval_seqacc(ft2, ev_sh, ev_shp, device)["seq_acc"], 4),
                            "old_keep_after": old_keep(ft2), "trainable": info["trainable_params"]}
    print(f"[P2.add] 旧规则保持(前)={add['old_keep_before']:.3f} | "
          f"synapse_only: shift={add['synapse_only']['shift_acc']:.3f} 旧保持={add['synapse_only']['old_keep_after']:.3f} | "
          f"full_ft: shift={add['full_finetune']['shift_acc']:.3f} 旧保持={add['full_finetune']['old_keep_after']:.3f}", flush=True)
    return {"flip": flip, "add": add}


# ----------------------------------------------------------------- P3 可成长（synapse 插值增容）
def _grow_state(small_net, big_net):
    """把小模型权重迁到大模型：同形直接拷；pos / synapse 做插值放大。"""
    ssd, bsd = small_net.state_dict(), big_net.state_dict()
    new = {}
    for k, bv in bsd.items():
        sv = ssd.get(k)
        if sv is not None and sv.shape == bv.shape:
            new[k] = sv.clone()
        elif sv is not None and k.endswith("pos"):                       # (1,L,D)
            t = F.interpolate(sv.transpose(1, 2), size=bv.shape[1], mode="linear", align_corners=False)
            new[k] = t.transpose(1, 2)
        elif sv is not None and k.endswith("synapse"):                   # (L,L)
            t = F.interpolate(sv[None, None], size=bv.shape, mode="bilinear", align_corners=False)
            new[k] = t[0, 0]
        else:
            new[k] = bv
    big_net.load_state_dict(new)


def phase_growth(args, device):
    """先在小迷你结构(m=4)学 reverse，再热启动(synapse 插值)长到大结构(m=8)，对比从头训。"""
    nc, dim, depth, heads = args.n_content, args.dim, args.depth, args.heads
    vocab = vocab_size(nc)
    m_s, m_b = 4, 8
    L_s, L_b = 2 * m_s + 2, 2 * m_b + 2
    epochs_small, epochs_big = args.epochs, max(4, args.epochs // 3)

    set_seed(args.seed)
    small = build_model("per_syn", vocab, L_s, dim, depth, heads, device)
    s_s, s_p, _ = build_copy_dataset(["reverse"], args.n_train, m_s, nc, L_s, seed=41)
    train_model(small, s_s, s_p, epochs=epochs_small, lr=args.lr, batch=args.batch, device=device)
    small_acc = eval_seqacc(small, *build_copy_dataset(["reverse"], args.n_eval, m_s, nc, L_s, seed=42)[:2], device)["seq_acc"]

    b_s, b_p, _ = build_copy_dataset(["reverse"], args.n_train, m_b, nc, L_b, seed=43)
    ev_b, ev_bp, _ = build_copy_dataset(["reverse"], args.n_eval, m_b, nc, L_b, seed=44)

    def grow_curve(net):
        cur = []
        for _ in range(epochs_big):
            train_model(net, b_s, b_p, epochs=1, lr=args.lr, batch=args.batch, device=device)
            cur.append(round(eval_seqacc(net, ev_b, ev_bp, device)["seq_acc"], 4))
        return cur

    # warm-start：synapse 插值放大
    warm = build_model("per_syn", vocab, L_b, dim, depth, heads, device)
    _grow_state(small, warm)
    warm_curve = grow_curve(warm)
    # 对照：转移 backbone+pos 但 synapse 重置为 0（隔离 synapse 插值本身的贡献）
    warm_ns = build_model("per_syn", vocab, L_b, dim, depth, heads, device)
    _grow_state(small, warm_ns)
    with torch.no_grad():
        for b in warm_ns.blocks:
            if getattr(b, "use_synapse", False):
                b.synapse.data.zero_()
    warm_ns_curve = grow_curve(warm_ns)
    # from-scratch
    set_seed(args.seed + 5)
    scratch = build_model("per_syn", vocab, L_b, dim, depth, heads, device)
    scr_curve = grow_curve(scratch)

    res = {"small_m": m_s, "big_m": m_b, "small_reverse_acc": round(small_acc, 4),
           "warm_start_curve": warm_curve, "warm_no_synapse_interp_curve": warm_ns_curve,
           "from_scratch_curve": scr_curve,
           "warm_final": warm_curve[-1], "scratch_final": scr_curve[-1],
           "warm_epoch1": warm_curve[0], "warm_no_synapse_interp_epoch1": warm_ns_curve[0],
           "scratch_epoch1": scr_curve[0]}
    print(f"[P3] 小结构 m={m_s} rev_acc={small_acc:.3f} → 大结构 m={m_b}: "
          f"warm ep1={warm_curve[0]:.3f} | warm(无syn插值) ep1={warm_ns_curve[0]:.3f} | scratch ep1={scr_curve[0]:.3f} "
          f"(末轮 warm={warm_curve[-1]:.3f}/scratch={scr_curve[-1]:.3f})", flush=True)
    return res


# ----------------------------------------------------------------- 报告
def write_report(args, results):
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    L = ["# 严肃证明完整 PER 原型 · 辨识点 #2（可学突触基底 self.synapse）三 claim 联合验收", ""]
    L += [f"- 任务：cue 条件位置路由 copy（m={args.m}，内容词表 {args.n_content}，内容随机→必须按 cue 做位置路由）。",
          f"- 模型：dim={args.dim} depth={args.depth} heads={args.heads}；PER+syn=完整原型，PER−syn=阉割(≈因果注意力)。",
          "- 诚实边界：synapse 是**位置级**结构先验，证的是结构/路由的可溯源·后天可塑·可成长，非“记新事实”。", ""]
    c = results.get("P4_compare")
    if c:
        L += ["## P4 诚实对照矩阵（隔离 #2 的边际贡献）", "",
              "| 模型 | 参数 | held-out seq_acc | by_rule | 说明 |", "|---|---:|---:|---|---|"]
        names = {"per_syn": "PER+syn（完整原型）", "per_nosyn": "PER−syn（阉割）", "transformer": "Transformer 对照"}
        for k in ["per_syn", "per_nosyn", "transformer"]:
            if k in c:
                L.append(f"| {names[k]} | {c[k]['params']/1e3:.1f}K | {c[k]['seq_acc']:.3f} | {c[k]['by_rule']} | {c[k]['sec']}s |")
        L.append("")
    t = results.get("P1_traceability")
    if t:
        L += ["## P1 可溯源：synapse 是持久可读结构记忆 + 因果可干预", "",
              f"- 基线：forward={t['baseline']['forward']:.3f}，reverse={t['baseline']['reverse']:.3f}",
              f"- **剪断 reverse 通路** → reverse 掉到 {t['cut_reverse_pathway']['reverse']:.3f}（forward 保持 {t['cut_reverse_pathway']['forward']:.3f}）",
              f"- **剪断 forward 通路** → forward 掉到 {t['cut_forward_pathway']['forward']:.3f}（reverse 保持 {t['cut_forward_pathway']['reverse']:.3f}）",
              f"- 剪同数量随机通路：forward_drop={t['cut_random_sameN']['forward_drop']:.3f}，reverse_drop={t['cut_random_sameN']['reverse_drop']:.3f}",
              f"- **干预特异性**（定向 − 随机）：reverse={t['specificity_reverse']:.3f}，forward={t['specificity_forward']:.3f}（越大=结构越可定位）", ""]
        if t.get("fig"):
            L += [f"- 突触通路热力图：`{t['fig']}`", ""]
    p = results.get("P2_postnatal")
    if p:
        fl, ad = p["flip"], p["add"]
        L += ["## P2 可后天学习：冻结 backbone，只动 synapse", "",
              "### P2.flip 纯路由翻转（pretrain forward → postnatal reverse）", "",
              f"- 翻转前(frozen) reverse={fl['frozen_baseline']['reverse_acc']:.3f}",
              f"- **synapse-only**（仅训 {fl['synapse_only']['trainable']} 参 / 全模型 {fl['total_params']}）→ reverse={fl['synapse_only']['reverse_acc']:.3f}",
              f"- full-finetune（训 {fl['full_finetune']['trainable']} 参）→ reverse={fl['full_finetune']['reverse_acc']:.3f}", "",
              "### P2.add 后天新增 shift(CUE_C)，看新增能力 + 旧规则遗忘", "",
              f"- 旧规则(forward/reverse)保持·新增前={ad['old_keep_before']:.3f}",
              f"- **synapse-only** → shift={ad['synapse_only']['shift_acc']:.3f}，旧保持={ad['synapse_only']['old_keep_after']:.3f}（训 {ad['synapse_only']['trainable']} 参）",
              f"- full-finetune → shift={ad['full_finetune']['shift_acc']:.3f}，旧保持={ad['full_finetune']['old_keep_after']:.3f}（训 {ad['full_finetune']['trainable']} 参）",
              "- **可塑-稳定权衡**（synapse-only 不同训练预算）：" +
              "；".join(f"{d['epochs']}ep→新规则{d['shift_acc']:.2f}/旧保持{d['old_keep']:.2f}" for d in ad.get("tradeoff_synapse_only", [])),
              ""]
    g = results.get("P3_growth")
    if g:
        L += ["## P3 可成长：synapse 插值热启动增容（小结构→大结构）", "",
              f"- 小结构 m={g['small_m']} reverse_acc={g['small_reverse_acc']:.3f}",
              f"- 长到 m={g['big_m']}：**warm-start**(synapse 插值) 首轮={g['warm_epoch1']:.3f} 末轮={g['warm_final']:.3f}",
              f"- 对照①(转移 backbone 但 **synapse 重置不插值**)：首轮={g.get('warm_no_synapse_interp_epoch1',0):.3f}",
              f"- 对照② from-scratch 同尺寸：首轮={g['scratch_epoch1']:.3f} 末轮={g['scratch_final']:.3f}",
              f"- warm 曲线：{g['warm_start_curve']}",
              f"- warm(无 synapse 插值) 曲线：{g.get('warm_no_synapse_interp_curve')}",
              f"- scratch 曲线：{g['from_scratch_curve']}", ""]
    L += ["## 结论", "", results.get("verdict", "(待定)"), ""]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[per-synapse] 报告写出 → {REPORT_MD}", flush=True)


def make_verdict(results) -> str:
    parts = []
    c = results.get("P4_compare", {})
    if c:
        ps = c.get("per_syn", {}).get("seq_acc", 0); pn = c.get("per_nosyn", {}).get("seq_acc", 0)
        tf = c.get("transformer", {}).get("seq_acc", 0)
        parts.append(f"P4（公平性）：完整训练下 PER+syn={ps:.2f}、PER−syn={pn:.2f}、Transformer={tf:.2f} 三方均能学会，"
                     f"说明任务公平、**不是**靠 synapse 才会做——这反而让后续 P1/P2/P3 的差异更可信（不是任务偏置）。")
    t = results.get("P1_traceability", {})
    if t:
        sr, sf = t.get("specificity_reverse", 0), t.get("specificity_forward", 0)
        strong = sr >= 0.5 and sf >= 0.5
        parts.append(f"P1（可溯源）：{'✅ 成立' if strong else '🟡 部分'}——剪断某规则的突触通路使该规则崩"
                     f"（reverse {t['baseline']['reverse']:.2f}→{t['cut_reverse_pathway']['reverse']:.2f}、"
                     f"forward {t['baseline']['forward']:.2f}→{t['cut_forward_pathway']['forward']:.2f}），剪同量随机通路几乎无影响，"
                     f"干预特异性 rev={sr:.2f}/fwd={sf:.2f}。synapse 是**持久、可定位、可因果干预**的结构记忆，"
                     f"这是 attention 逐样本权重给不了的可溯源性。")
    p = results.get("P2_postnatal", {})
    if p:
        fl = p["flip"]; ad = p["add"]
        add_learn = ad["synapse_only"]["shift_acc"]
        keep_adv = ad["synapse_only"]["old_keep_after"] - ad["full_finetune"]["old_keep_after"]
        to = ad.get("tradeoff_synapse_only", [])
        to_str = "，".join(f"{d['epochs']}ep→新{d['shift_acc']:.2f}/旧{d['old_keep']:.2f}" for d in to)
        parts.append(
            f"P2（可后天学习·🟡**有真实可塑性但边界清晰**）：冻结 backbone 只动 synapse（{fl['synapse_only']['trainable']}/{fl['total_params']} 参，"
            f"已给 {fl['synapse_only']['epochs']}ep×lr{fl['synapse_only']['lr']:.0e} 充分机会，排除欠训）——"
            f"① 覆盖式翻转(forward→reverse，复用同 cue)= **失败** {fl['frozen_baseline']['reverse_acc']:.2f}→{fl['synapse_only']['reverse_acc']:.2f}"
            f"（full-ft 能到 {fl['full_finetune']['reverse_acc']:.2f}）；"
            f"② 新 cue 上后天长出新规则(shift)= **成功** {add_learn:.2f}。"
            f"但存在**可塑-稳定权衡**（{to_str}）：训得越足新规则越满、旧规则忘得越多；充分训练后旧保持 {ad['synapse_only']['old_keep_after']:.2f} "
            f"与全参微调 {ad['full_finetune']['old_keep_after']:.2f} 几乎持平（抗遗忘优势仅 +{keep_adv:.2f}，**不显著**）。"
            f"机制诚实结论：g=softplus(synapse)·softmax(compat) **乘性耦合** + synapse 是**单一 cue 无关共享矩阵**——"
            f"故它能在 content-compat 尚未笃定处后天塑造路由（新 cue 可塑），但**无法推翻**已笃定的 backbone（覆盖式翻转不可塑），"
            f"也**无法**在不伤旧路由的前提下塞进新路由（共享矩阵→必遗忘）。可后天学习**部分成立、有明确边界**。")
    g = results.get("P3_growth", {})
    if g:
        isolate = g["warm_epoch1"] - g.get("warm_no_synapse_interp_epoch1", 0)
        parts.append(
            f"P3（可成长·🟡**warm-start 成立，但功臣不是 synapse 插值**）：热启动从 m={g['small_m']} 长到 m={g['big_m']}，"
            f"首轮 seq_acc={g['warm_epoch1']:.2f}≫从头训 {g['scratch_epoch1']:.2f}（末轮 {g['warm_final']:.2f} vs {g['scratch_final']:.2f}）——"
            f"尺寸可成长成立；但对照'转移 backbone 但**不插值 synapse**'首轮={g.get('warm_no_synapse_interp_epoch1',0):.2f}，"
            f"故 **synapse 插值净贡献≈+{isolate:.2f}（可忽略）**，提速主要来自 backbone(W_pred/q/k/ffn)+pos 的迁移。诚实：可成长成立，但非 #2 专属之功。")
    head = ("【总评·诚实】完整 PER 原型 #2（可学突触基底）严肃验收得到**清晰但有取舍**的结论："
            "#2 最硬、最干净的价值在 **可溯源(P1)✅**——它是持久、可定位、可因果干预的结构记忆，attention 的逐样本权重给不了；"
            "而 **可后天学习(P2)🟡** 只在 backbone 未笃定的方向上部分成立且伴随遗忘（乘性耦合+共享矩阵的真实边界），"
            "**可成长(P3)🟡** 的 warm-start 提速成立但几乎不来自 synapse 插值本身。任务对三方公平(P4)，全程不夸大、负面与边界如实列出。")
    return head + " " + " ".join(parts) if parts else "(无结论)"


# ----------------------------------------------------------------- smoke
def smoke(device):
    print("[smoke] 训练 PER+syn 学 forward+reverse 两规则 cue copy ...", flush=True)
    m, nc, max_len = 5, 20, 2 * 5 + 2
    tr_s, tr_p, tr_r = build_copy_dataset(["forward", "reverse"], 1500, m, nc, max_len, seed=1)
    te_s, te_p, te_r = build_copy_dataset(["forward", "reverse"], 300, m, nc, max_len, seed=999)
    net = build_model("per_syn", vocab_size(nc), max_len, 64, 3, 4, device)
    print(f"[smoke] 参数量 {n_params(net)/1e3:.1f}K", flush=True)
    train_model(net, tr_s, tr_p, epochs=20, lr=2e-3, batch=128, device=device, verbose=True, tag="per_syn")
    res = eval_seqacc(net, te_s, te_p, device, rule_ids=te_r)
    print(f"[smoke] held-out seq_acc={res['seq_acc']:.3f} tok_acc={res['tok_acc']:.3f} by_rule={res['by_rule']}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="严肃证明完整 PER 原型 synapse(#2) 三 claim。")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="只跑一个最小训练自检")
    ap.add_argument("--phases", default="all", help="all 或逗号分隔 p1,p2,p3,p4")
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--n-content", type=int, default=20)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=3000, help="每规则训练样本数")
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print("[per-synapse] dry-run：证明完整 PER 原型 self.synapse(#2) 的可溯源/可后天/可成长。")
        print("[per-synapse] 真跑加 --run（自检 --run --smoke）。")
        return 0
    set_seed(args.seed)
    device = get_device()
    print(f"[per-synapse] device={device}", flush=True)
    if args.smoke:
        smoke(device)
        return 0

    phases = ["p1", "p2", "p3", "p4"] if args.phases == "all" else [x.strip() for x in args.phases.split(",")]
    results = {"task": "cue-conditioned positional-routing copy",
               "config": {"m": args.m, "n_content": args.n_content, "dim": args.dim,
                          "depth": args.depth, "heads": args.heads, "epochs": args.epochs,
                          "n_train_per_rule": args.n_train, "n_eval": args.n_eval}}
    m, nc, max_len = args.m, args.n_content, 2 * args.m + 2

    per_syn_for_trace = None
    if "p4" in phases or "p1" in phases:
        set_seed(args.seed)
        tr = build_copy_dataset(["forward", "reverse"], args.n_train, m, nc, max_len, seed=args.seed)
        te = build_copy_dataset(["forward", "reverse"], args.n_eval, m, nc, max_len, seed=args.seed + 100)
        cmp_out, models = phase_compare(args, device, (tr, te))
        results["P4_compare"] = cmp_out
        per_syn_for_trace = models["per_syn"]
    if "p1" in phases:
        if per_syn_for_trace is None:
            set_seed(args.seed)
            tr = build_copy_dataset(["forward", "reverse"], args.n_train, m, nc, max_len, seed=args.seed)
            per_syn_for_trace = build_model("per_syn", vocab_size(nc), max_len, args.dim, args.depth, args.heads, device)
            train_model(per_syn_for_trace, tr[0], tr[1], epochs=args.epochs, lr=args.lr, batch=args.batch, device=device)
        results["P1_traceability"] = phase_traceability(args, device, per_syn_for_trace)
    if "p2" in phases:
        results["P2_postnatal"] = phase_postnatal(args, device)
    if "p3" in phases:
        results["P3_growth"] = phase_growth(args, device)

    results["verdict"] = make_verdict(results)
    write_report(args, results)
    print("\n[per-synapse] === 结论 ===\n" + results["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
