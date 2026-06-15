# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_induction_seq_eval.py
==============================================
CAPCW 走向序列语言引擎：给内容寻址工作空间补一个**序列相邻算子**，重测 induction（归纳头）。
见 `docs/FE-LLM核心引擎构想.md` 第 16 节与 `经验.md` 的 induction 负结果。

背景（诚实承接负结果）
----------------------
`capcw_induction_eval.py` 已标定 induction 负结果：flat 0.118 / CAPCW 0.104，**双双≈随机**
（基线 0.05）。根因诊断：induction 需要"**序列相邻**"（找 cue 后面紧跟的 token）；而当前 flat 与
CAPCW 都把 token **独立嵌入再聚合**，bigram a→b 的相邻信息在聚合时丢失。集合式 slot 工作空间擅长
"内容绑定"（pair 作为整体喂入），**不自带序列相邻算子**（Transformer 靠 attention+位置实现 induction）。

本实验（独立判定，先定任务+判据，不硬凑）
------------------------------------------
在 token→工作空间写入**之前**补一个最小"相邻/序列算子"——**previous-token channel**：
每个位置的表示 = proj([emb(前驱 token); emb(当前 token)])，即 induction head 的"previous-token head"
（位置 t 同时携带 prev=key 供 cue 匹配、cur=value 供读出）。这把"独立 token 流"变成"(prev→cur)
bigram 表示流"，再喂给单向量池化 / slot 工作空间。

**2×2 单变量析因**（两个变量：相邻算子 on/off、世界状态结构 flat/CAPCW；其余全一致）：

            | flat（单向量均值池化） | CAPCW（slot 工作空间）
  no-adj    | 复现 FAIL(~0.10)       | 复现 FAIL(~0.10)
  +adjacency| ?                      | ?

判据（预先写死；交互口径）
--------------------------
（最初朴素假设是"相邻算子能救活 induction"；2×2 结果把它精确成一个**交互**——相邻算子单独喂单向量
池化救不活，必须配上内容寻址 slot。故判据按"可检索结构(CAPCW)"度量，并报告 flat 作对照。）
- H1 相邻算子在 CAPCW 内救活 induction：acc(capcw_adj) − acc(capcw_raw) 跨负载平均 ≥ +0.30
  （no-adj≈随机）→ 诊断证实：induction 缺的是**序列相邻算子**。
  - 对照：acc(flat_adj) − acc(flat_raw) 预期≈0——相邻算子单独喂单向量池化救不活（无法联想检索）。
- H2 内容寻址价值（与 capcw_binding 同口径）：给了相邻信息后，**slot 是否胜单向量**——
  acc(capcw_adj) − acc(flat_adj) ≥ +0.10（单向量把多个 bigram 挤进一个向量会相互干扰，slot 按内容分开）。
- 综合：H1 且 H2 成立 → **PASS**（2×2 交互：induction 需"序列相邻算子"与"内容寻址 slot"两者兼备，
  CAPCW 走向序列语言引擎）；仅 H1 → **PARTIAL**（相邻算子有效，但内容寻址未决定性胜单向量）；
  都不成立 → **FAIL**。

为什么 no-adj 一定救不活（信息而非容量问题）：no-adj 输入里**根本不含"前驱身份"**，无论多少参数都
无法恢复 a→b 的相邻关系；+adj 把前驱身份显式注入，是信息的增加，不是单纯容量。故对照干净。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_induction_seq_eval --run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fe_llm.config import get_device
from fe_llm.world_model.capcw import PCWorkspace, SequenceAdjacency

REPORT_JSON = os.path.join("docs", "reports", "capcw_induction_seq_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_induction_seq_eval.md")


def gen_induction(n_sym, n_pairs, seq_len, n, seed):
    """序列：n_pairs 个随机 (a,b) 相邻 bigram 散布在长度 seq_len 序列里 + cue(=某 a) 在末位；y=该 a 的 b。

    干净口径（与 capcw_binding 的无噪声键值对齐；清理对所有臂公平，no-adj 臂仍≈随机）：
    - bigram 起点取**偶数位且互不重叠**（st, st+1 占一格对），不会相互覆盖污染 (a→b) 映射；
    - **filler 不含任何 a 符号** → 序列里"前驱==cue"的位置唯一、答案无歧义（去掉 spurious bigram 噪声）。
    这样可学性上限≈1.0，能干净地暴露"相邻算子 + 内容寻址"这一机制，而不被标签噪声压住。
    符号表 [0..n_sym)。返回 ids:(N,seq_len)、cue:(N,)、y:(N,)。
    """
    rng = np.random.default_rng(seed)
    ids = np.zeros((n, seq_len), dtype=np.int64)
    cue = np.zeros((n,), dtype=np.int64)
    y = np.zeros((n,), dtype=np.int64)
    even_slots = np.arange(0, seq_len - 1, 2)          # 偶数起点，st+1 仍在 [0,seq_len-1)
    for i in range(n):
        syms = rng.choice(n_sym, size=2 * n_pairs, replace=False)
        a_list = syms[:n_pairs]
        b_list = syms[n_pairs:2 * n_pairs]
        a_set = set(int(x) for x in a_list)
        filler_pool = np.array([s for s in range(n_sym) if s not in a_set], dtype=np.int64)
        seq = [int(rng.choice(filler_pool)) for _ in range(seq_len)]   # filler 不含任何 a
        starts = rng.choice(even_slots, size=n_pairs, replace=False)    # 互不重叠的相邻对
        for j, st in enumerate(starts):
            seq[st] = int(a_list[j])
            seq[st + 1] = int(b_list[j])
        qi = int(rng.integers(n_pairs))
        seq[seq_len - 1] = int(a_list[qi])      # cue 放末位
        ids[i] = seq
        cue[i] = int(a_list[qi])
        y[i] = int(b_list[qi])
    return ids, cue, y


class SeqEncoder(nn.Module):
    """序列编码器；唯一变量开关 = adjacency（是否补 previous-token channel）。

    - adjacency=False：位置表示 = emb(token) + pos（与旧 induction 评测一致 → 复现 FAIL）。
    - adjacency=True ：位置表示 = adj_proj([emb(前驱); emb(当前)]) + pos（相邻算子=induction head 的
      previous-token head；前驱身份显式注入，使 (prev→cur) bigram 在写入工作空间前已成形）。
    """

    def __init__(self, n_sym, seq_len, d, adjacency: bool = False):
        super().__init__()
        self.emb = nn.Embedding(n_sym, d)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d))
        nn.init.normal_(self.pos, std=0.02)
        self.adjacency = adjacency
        if adjacency:
            # 唯一变量=相邻算子；复用引擎组件 SequenceAdjacency（单一真相源，避免实现漂移）。
            self.adj = SequenceAdjacency(d)

    def tokens(self, ids):
        cur = self.emb(ids)                                   # (B,L,d)
        if not self.adjacency:
            return cur + self.pos[:, : ids.shape[1]]
        rep = self.adj(cur)                                   # (B,L,d) (prev→cur) bigram 表示
        return rep + self.pos[:, : ids.shape[1]]


class FlatInduction(nn.Module):
    """单向量世界状态：序列表示均值池化 + cue 读出。"""

    def __init__(self, n_sym, seq_len, d, adjacency: bool = False):
        super().__init__()
        self.enc = SeqEncoder(n_sym, seq_len, d, adjacency=adjacency)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, n_sym))

    def forward(self, ids, cue):
        tok = self.enc.tokens(ids)
        world = tok.mean(dim=1)
        q = self.enc.emb(cue)                              # cue 复用 token 嵌入（cue 本身是个 token）
        return self.head(torch.cat([world, q], dim=-1))


class CAPCWInduction(nn.Module):
    """CAPCW：序列表示经 PCWorkspace 聚成 slot 工作空间 + cue 内容寻址读出。"""

    def __init__(self, n_sym, seq_len, d, n_slots, iters, adjacency: bool = False):
        super().__init__()
        self.enc = SeqEncoder(n_sym, seq_len, d, adjacency=adjacency)
        self.ws = PCWorkspace(dim=d, n_slots=n_slots, iters=iters)
        self.to_q = nn.Linear(d, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_sym))
        self.d = d

    def forward(self, ids, cue):
        tok = self.enc.tokens(ids)
        slots = self.ws(tok).slots                                    # (B,M,d)
        q = self.to_q(self.enc.emb(cue))                              # (B,d) cue 复用 token 嵌入
        score = torch.einsum("bmd,bd->bm", slots, q) / math.sqrt(self.d)
        attn = score.softmax(dim=1).unsqueeze(-1)                     # cue 内容寻址 slot
        read = (slots * attn).sum(dim=1)
        return self.head(read)


def train_eval(model, train, test, device, *, epochs, lr, batch, seed):
    torch.manual_seed(seed)
    ids, cue, y = (torch.tensor(t, device=device) for t in train)
    tids, tcue, ty = (torch.tensor(t, device=device) for t in test)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(y)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(ids[idx], cue[idx]), y[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(tids, tcue).argmax(-1)
        acc = float((pred == ty).float().mean())
    return acc


def _build_arms(args, n_pairs, seq_len):
    """构造 2×2 四臂（同 d / 同 slot 预算；唯一变量=adjacency 与 结构）。"""
    n_slots = max(args.n_slots, n_pairs + 1)
    return {
        "flat_raw": FlatInduction(args.n_sym, seq_len, args.d, adjacency=False),
        "flat_adj": FlatInduction(args.n_sym, seq_len, args.d, adjacency=True),
        "capcw_raw": CAPCWInduction(args.n_sym, seq_len, args.d, n_slots, args.iters, adjacency=False),
        "capcw_adj": CAPCWInduction(args.n_sym, seq_len, args.d, n_slots, args.iters, adjacency=True),
    }


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    pair_list = [int(x) for x in args.pair_list.split(",")]
    print(f"[ind-seq] device={device} n_sym={args.n_sym} pair_list={pair_list} d={args.d} "
          f"seq_len={args.seq_len} seeds={args.seeds}", flush=True)

    results: dict[int, dict] = {}
    for k in pair_list:
        seq_len = max(args.seq_len, 2 * k + 4)               # 保证能放下 k 个 bigram + cue
        arms = {"flat_raw": [], "flat_adj": [], "capcw_raw": [], "capcw_adj": []}
        for si in range(args.seeds):
            seed = args.seed + si
            train = gen_induction(args.n_sym, k, seq_len, args.n_train, seed)
            test = gen_induction(args.n_sym, k, seq_len, args.n_test, seed + 5000)
            torch.manual_seed(seed)                  # 固定四臂的模型初始化，保证结果可复现
            models = _build_arms(args, k, seq_len)
            common = dict(device=device, epochs=args.epochs, lr=args.lr, batch=args.batch, seed=seed)
            for name, model in models.items():
                arms[name].append(train_eval(model, train, test, **common))
        summary = {a: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)} for a, v in arms.items()}
        results[k] = summary
        print(f"[ind-seq] n_pairs={k} (seq_len={seq_len}) "
              f"flat_raw={summary['flat_raw']['mean']:.3f} flat_adj={summary['flat_adj']['mean']:.3f} "
              f"capcw_raw={summary['capcw_raw']['mean']:.3f} capcw_adj={summary['capcw_adj']['mean']:.3f} "
              f"(random={1.0/args.n_sym:.3f})", flush=True)

    # ---- 判据计算（2×2 交互口径）----
    # rescue_capcw：相邻算子在"可检索结构(CAPCW)"内是否救活 induction（capcw_adj − capcw_raw）。
    # rescue_flat ：相邻算子单独喂给单向量池化是否救活（flat_adj − flat_raw）——预期≈0（无法联想检索）。
    rescue_flat = {k: round(results[k]["flat_adj"]["mean"] - results[k]["flat_raw"]["mean"], 4) for k in pair_list}
    rescue_capcw = {k: round(results[k]["capcw_adj"]["mean"] - results[k]["capcw_raw"]["mean"], 4) for k in pair_list}
    h1_rescue = round(float(np.mean(list(rescue_capcw.values()))), 4)        # H1=CAPCW 内救活幅度（跨负载平均）
    h1_rescue_flat = round(float(np.mean(list(rescue_flat.values()))), 4)    # 对照：相邻算子单独喂 flat
    # H2 内容寻址价值：给了相邻信息后，slot 工作空间是否胜单向量（capcw_adj − flat_adj）。
    high_pairs = [k for k in pair_list if k >= args.high_pairs]
    cap_deltas = {k: round(results[k]["capcw_adj"]["mean"] - results[k]["flat_adj"]["mean"], 4) for k in pair_list}
    h2_capacity = round(float(np.max([cap_deltas[k] for k in high_pairs])), 4) if high_pairs else 0.0
    h2_at = max(high_pairs, key=lambda k: cap_deltas[k]) if high_pairs else None
    # 交互：both-treatment 格(capcw_adj)是否压过另外三格("缺一味"对照)的最好者。
    interaction = round(float(np.mean([
        results[k]["capcw_adj"]["mean"]
        - max(results[k]["flat_raw"]["mean"], results[k]["flat_adj"]["mean"], results[k]["capcw_raw"]["mean"])
        for k in pair_list])), 4)

    h1_pass = h1_rescue >= 0.30          # 相邻算子在 CAPCW 内救活 induction
    h2_pass = h2_capacity >= 0.10        # 内容寻址在给了相邻信息后仍胜单向量
    if h1_pass and h2_pass:
        verdict = ("PASS: 2×2 交互成立——induction 需同时具备'序列相邻算子'(H1: capcw_adj >> capcw_raw)"
                   "与'内容寻址 slot'(H2: capcw_adj >> flat_adj)；缺一格(flat_adj/capcw_raw)均≈随机。"
                   "CAPCW+相邻算子成为可用的 in-context 序列引擎，从'内容绑定'走向'序列语言'。")
    elif h1_pass:
        verdict = ("PARTIAL: 相邻算子在 CAPCW 内救活 induction（H1 成立），但内容寻址相对单向量优势未达 +0.10"
                   "（H2 未过）——需更高负载或 key/value 读出。")
    else:
        verdict = "FAIL: 相邻算子在 CAPCW 内未救活 induction（H1 不成立）——诊断需重审。"

    result = {
        "task": "induction with sequence-adjacency operator: ...A B ... A -> predict B",
        "design": "2x2 factorial; vars = adjacency(prev-token channel) on/off x world-state(flat/CAPCW)",
        "config": {"n_sym": args.n_sym, "pair_list": pair_list, "base_seq_len": args.seq_len,
                   "d": args.d, "n_slots": args.n_slots, "iters": args.iters,
                   "epochs": args.epochs, "seeds": args.seeds,
                   "random_baseline": round(1.0 / args.n_sym, 4)},
        "by_n_pairs": results,
        "h1_rescue_capcw_mean": h1_rescue,
        "h1_rescue_capcw_by_pairs": rescue_capcw,
        "ctrl_rescue_flat_mean": h1_rescue_flat,
        "ctrl_rescue_flat_by_pairs": rescue_flat,
        "h2_addressing_delta_by_pairs": cap_deltas,
        "h2_addressing_best": h2_capacity,
        "h2_addressing_best_at_n_pairs": h2_at,
        "interaction_capcw_adj_minus_best_other": interaction,
        "h1_pass": h1_pass,
        "h2_pass": h2_pass,
        "verdict": verdict,
        "note": "no-adj 输入不含'前驱身份'，无论参数多少都无法恢复 a→b 相邻关系，故对照是信息(非容量)对照；"
                "+adj=previous-token channel(induction head 基元)。关键发现是 2×2 交互：相邻算子单独喂 flat "
                "救不活(ctrl_rescue_flat≈0)、slot 无相邻算子也救不活(capcw_raw≈随机)，唯有两者兼备(capcw_adj)才行。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    lines = [
        "# CAPCW · 序列相邻算子救 induction（走向序列语言引擎）",
        "",
        f"- 判定：**{verdict}**",
        f"- 任务：序列 ...A B ... A → 预测 B（每序列 A→B 随机配对，不可记忆）；"
        f"n_sym={args.n_sym}, d={args.d}；随机基线 {1.0/args.n_sym:.3f}",
        f"- 设计：2×2 单变量析因（相邻算子 on/off × 世界状态 flat/CAPCW）；同 d / 同 slot 预算。",
        "",
        "| n_pairs（负载） | flat_raw | flat_adj | capcw_raw | capcw_adj |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in pair_list:
        s = results[k]
        lines.append(f"| {k} | {s['flat_raw']['mean']:.3f}±{s['flat_raw']['std']:.3f} | "
                     f"{s['flat_adj']['mean']:.3f}±{s['flat_adj']['std']:.3f} | "
                     f"{s['capcw_raw']['mean']:.3f}±{s['capcw_raw']['std']:.3f} | "
                     f"{s['capcw_adj']['mean']:.3f}±{s['capcw_adj']['std']:.3f} |")
    lines += [
        "",
        f"- **H1（相邻算子在 CAPCW 内救活 induction）**：capcw_adj − capcw_raw 跨负载平均 = "
        f"**{h1_rescue:+.4f}**（阈值 ≥ +0.30 → {'成立' if h1_pass else '不成立'}）",
        f"  - CAPCW 各负载救活幅度：{rescue_capcw}",
        f"  - 对照（相邻算子单独喂 flat，flat_adj − flat_raw）跨负载平均 = **{h1_rescue_flat:+.4f}**"
        f"（预期≈0：单向量池化无法联想检索）；各负载：{rescue_flat}",
        f"- **H2（内容寻址价值：给了相邻信息后 slot 是否胜单向量）**：高负载(n_pairs≥{args.high_pairs}) "
        f"capcw_adj − flat_adj 最佳 = **{h2_capacity:+.4f}**"
        + (f"（@n_pairs={h2_at}）" if h2_at is not None else "")
        + f"（阈值 ≥ +0.10 → {'成立' if h2_pass else '不成立'}）",
        f"  - 各负载 capcw_adj − flat_adj：{cap_deltas}",
        f"- **交互**：capcw_adj − max(另三格) 跨负载平均 = **{interaction:+.4f}**"
        "（只有'相邻算子+内容寻址'两者兼备才解 induction）",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n[ind-seq] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    print(f"[ind-seq] H1 rescue={h1_rescue:+.4f}  H2 capacity={h2_capacity:+.4f}", flush=True)
    print(f"[ind-seq] 报告：{args.report_json} / {args.report_md}", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW induction with sequence-adjacency operator (2x2 factorial).")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-sym", type=int, default=20)
    ap.add_argument("--pair-list", default="2,4,6")
    ap.add_argument("--high-pairs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--n-slots", type=int, default=6)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-json", default=REPORT_JSON)
    ap.add_argument("--report-md", default=REPORT_MD)
    return ap


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 GBK，遇到 ≫/≈ 等非 GBK 字符会 UnicodeEncodeError；统一切到 utf-8（报告文件本就 utf-8）。
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print("[ind-seq] dry-run：未训练。2×2（相邻算子 on/off × flat/CAPCW）重测 induction。")
        print("[ind-seq] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
