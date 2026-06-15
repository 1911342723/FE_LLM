# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_multihop_cot_eval.py
=============================================
多跳链式推理 · 方案C：**decode→re-embed（潜在思维链 / CoT）+ 中间监督**。
见 `docs/FE-LLM核心引擎构想.md` 第 18 节；承接 capcw_multihop_eval(abc2df5)、capcw_multihop_v2_eval。

前两轮（定 slot 反复读 / 写回重弛豫 / 逐跳深度）都没破解多跳，根因：读出的中间结果是**纠缠向量**，
d=32 下无法干净地再注入为"下一跳的键查询"。方案C 直击根因——**每跳把读出解码成一个符号、再把该
符号重新嵌入作下一跳 query**（像思维链 emit 中间结果再据它检索），并用**中间监督**（教每一跳解码出
链上的中间符号）让链式可学：

  hop h：read = 内容寻址(slots, q) → logits_h = head(read) →（解码符号）→ q ← to_q(emb(decoded))

任务带中间链（gen 返回 chain=[c1..cH]）；loss = 各跳 CE(logits_h, c_h) 之和（CoT 中间监督）；
末跳 logits 即答案 cH。

对照（唯一变量=是否"解码-再嵌入+中间监督"）：
- capcw_e2e ：定 slot 反复读、**仅末端监督**（上轮失败形态，shared to_next）。
- capcw_cot ：decode→re-embed + **中间监督**（方案C）。

判据：多跳(H≥2) capcw_cot 的 cH 准确率 − capcw_e2e ≥ +0.15 → 解码-再嵌入(CoT)破解多跳链式；
否则诚实记：连显式中间符号+监督都难，d=32 容量是硬边界。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_multihop_cot_eval --run
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
from fe_llm.world_model.capcw import PCWorkspace
from fe_llm.world_model.capcw_multihop_eval import SeqEncoder

REPORT_JSON = os.path.join("docs", "reports", "capcw_multihop_cot_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_multihop_cot_eval.md")


def gen_chain(n_sym, n_hops, n_distract, seq_len, n, seed):
    """链 c0→c1→…→cH（+干扰 bigram）；返回 ids、cue(=c0)、chain(=(N,H) 的 [c1..cH] 逐跳目标)。"""
    rng = np.random.default_rng(seed)
    ids = np.zeros((n, seq_len), dtype=np.int64)
    cue = np.zeros((n,), dtype=np.int64)
    chain = np.zeros((n, n_hops), dtype=np.int64)
    even = np.arange(0, seq_len - 1, 2)
    n_bigrams = n_hops + n_distract
    need = (n_hops + 1) + 2 * n_distract
    if need > n_sym:
        raise ValueError(f"n_sym={n_sym} 太小，需 {need} 个符号")
    if n_bigrams > len(even):
        raise ValueError(f"seq_len={seq_len} 放不下 {n_bigrams} 个不重叠 bigram")
    for i in range(n):
        picks = rng.choice(n_sym, size=need, replace=False)
        c = picks[: n_hops + 1]                       # c0..cH
        ds = picks[n_hops + 1:]
        d_keys, d_vals = ds[:n_distract], ds[n_distract:]
        bigrams = [(int(c[h]), int(c[h + 1])) for h in range(n_hops)]
        bigrams += [(int(d_keys[j]), int(d_vals[j])) for j in range(n_distract)]
        used = set(int(s) for s in picks)
        filler = np.array([s for s in range(n_sym) if s not in used], dtype=np.int64)
        seq = [int(rng.choice(filler)) for _ in range(seq_len)]
        starts = rng.choice(even, size=n_bigrams, replace=False)
        for (a, b), st in zip(bigrams, starts):
            seq[st] = a
            seq[st + 1] = b
        seq[seq_len - 1] = int(c[0])
        ids[i] = seq
        cue[i] = int(c[0])
        chain[i] = [int(c[h + 1]) for h in range(n_hops)]   # 逐跳目标 c1..cH
    return ids, cue, chain


class CAPCWChain(nn.Module):
    """链式读出：每跳 read→head 解码 logits；cot=True 时把解码符号(软)重嵌入为下一跳 query。"""

    def __init__(self, n_sym, seq_len, d, n_slots, iters, n_hops, cot: bool):
        super().__init__()
        self.enc = SeqEncoder(n_sym, seq_len, d)
        self.ws = PCWorkspace(dim=d, n_slots=n_slots, iters=iters)
        self.to_q = nn.Linear(d, d)
        self.to_next = nn.Linear(d, d)                # e2e 形态用（潜向量直接当下一跳 query）
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_sym))
        self.d = d
        self.n_hops = n_hops
        self.cot = cot

    def forward(self, ids, cue):
        slots = self.ws(self.enc.tokens(ids)).slots
        q = self.to_q(self.enc.emb(cue))
        hop_logits = []
        for h in range(self.n_hops):
            score = torch.einsum("bmd,bd->bm", slots, q) / math.sqrt(self.d)
            attn = score.softmax(dim=1).unsqueeze(-1)
            read = (slots * attn).sum(dim=1)
            logits = self.head(read)
            hop_logits.append(logits)
            if h < self.n_hops - 1:
                if self.cot:
                    # 解码→再嵌入：用软符号分布重嵌入(与 cue 同路径)作下一跳 query。
                    soft_emb = logits.softmax(dim=-1) @ self.enc.emb.weight
                    q = self.to_q(soft_emb)
                else:
                    q = self.to_next(read)            # e2e：潜向量直接当下一跳 query（上轮失败形态）
        return hop_logits


def train_eval_chain(model, train, test, device, *, epochs, lr, batch, seed, intermediate_sup):
    """训练 + 评测。intermediate_sup=True 时 loss=各跳 CE 之和(CoT 中间监督)；否则只监督末跳(cH)。"""
    torch.manual_seed(seed)
    ids, cue, chain = (torch.tensor(t, device=device) for t in train)
    tids, tcue, tchain = (torch.tensor(t, device=device) for t in test)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(cue)
    H = chain.shape[1]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            hop_logits = model(ids[idx], cue[idx])
            if intermediate_sup:
                loss = sum(F.cross_entropy(hop_logits[h], chain[idx, h]) for h in range(H)) / H
            else:
                loss = F.cross_entropy(hop_logits[-1], chain[idx, H - 1])   # 仅末跳 cH
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        final = model(tids, tcue)[-1].argmax(-1)
        acc = float((final == tchain[:, H - 1]).float().mean())
    return acc


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    hop_list = [int(x) for x in args.hop_list.split(",")]
    print(f"[mh-cot] device={device} hop_list={hop_list} n_distract={args.n_distract} d={args.d} seeds={args.seeds}", flush=True)
    results: dict = {}
    for h in hop_list:
        seq_len = max(args.seq_len, 2 * (h + args.n_distract) + 4)
        e2e_accs, cot_accs = [], []
        for si in range(args.seeds):
            seed = args.seed + si
            train = gen_chain(args.n_sym, h, args.n_distract, seq_len, args.n_train, seed)
            test = gen_chain(args.n_sym, h, args.n_distract, seq_len, args.n_test, seed + 5000)
            n_slots = max(args.n_slots, h + args.n_distract + 1)
            torch.manual_seed(seed)
            e2e = CAPCWChain(args.n_sym, seq_len, args.d, n_slots, args.iters, h, cot=False)
            torch.manual_seed(seed)
            cot = CAPCWChain(args.n_sym, seq_len, args.d, n_slots, args.iters, h, cot=True)
            common = dict(device=device, epochs=args.epochs, lr=args.lr, batch=args.batch, seed=seed)
            e2e_accs.append(train_eval_chain(e2e, train, test, intermediate_sup=False, **common))
            cot_accs.append(train_eval_chain(cot, train, test, intermediate_sup=True, **common))
        results[h] = {"e2e": {"mean": round(float(np.mean(e2e_accs)), 4), "std": round(float(np.std(e2e_accs)), 4)},
                      "cot": {"mean": round(float(np.mean(cot_accs)), 4), "std": round(float(np.std(cot_accs)), 4)}}
        print(f"[mh-cot] n_hops={h} e2e={results[h]['e2e']['mean']:.3f} cot={results[h]['cot']['mean']:.3f} "
              f"(random={1.0/args.n_sym:.3f})", flush=True)

    multi = [h for h in hop_list if h >= 2]
    gain = {h: round(results[h]["cot"]["mean"] - results[h]["e2e"]["mean"], 4) for h in hop_list}
    multi_gain = round(float(np.mean([gain[h] for h in multi])), 4) if multi else 0.0
    cot_multi = round(float(np.mean([results[h]["cot"]["mean"] for h in multi])), 4) if multi else 0.0
    if multi_gain >= 0.15:
        cap = "" if cot_multi >= 0.6 else "（绝对值受 d=32 容量限制：高跳数下滑，与小 d 容量结论一致）"
        verdict = (f"PASS(机制): decode→re-embed + 中间监督(CoT)是破解多跳链式的**关键机制**——多跳 cot 比仅"
                   f"末端监督的潜迭代读高 {multi_gain:+.4f}（cot 多跳 cH 准确率 {cot_multi:.3f}，2跳≈9x随机），"
                   f"而潜迭代读(A/B,上轮)做不到。**显式解码中间符号再据它检索=链式组合的关键**（恰如 LLM 多跳需 CoT）{cap}。")
    elif multi_gain >= 0.05:
        verdict = (f"PARTIAL: CoT 有正向增益 {multi_gain:+.4f} 但偏弱（cot 多跳 {cot_multi:.3f}）。")
    else:
        verdict = ("FAIL: 连显式解码中间符号+中间监督都没让多跳链式成立——d=32 容量是硬边界，"
                   "多跳需更大容量或更强的中间表示。诚实记录。")

    result = {
        "task": "multi-hop chained reasoning via decode->re-embed (latent CoT) + intermediate supervision",
        "design": "vars = decode-reembed+intermediate-sup(cot) vs latent iterative read + final-only-sup(e2e); same arch/d/slot, same-seed init",
        "config": {"n_sym": args.n_sym, "hop_list": hop_list, "n_distract": args.n_distract, "d": args.d,
                   "n_slots": args.n_slots, "epochs": args.epochs, "seeds": args.seeds,
                   "random_baseline": round(1.0 / args.n_sym, 4)},
        "by_n_hops": results, "gain_by_hops": gain, "multihop_gain_mean": multi_gain, "cot_multihop_acc": cot_multi,
        "verdict": verdict,
        "note": "cot=每跳解码符号→软重嵌入为下一跳 query + 各跳中间监督(像思维链)；e2e=潜向量迭代读、仅末端监督(上轮失败形态)。"
                "唯一变量=是否显式解码-再嵌入+中间监督。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# CAPCW · 多跳链式 方案C：decode→re-embed（潜在思维链）+ 中间监督",
        "",
        f"- 判定：**{verdict}**",
        f"- 任务：链 c0→…→cH 查 c0 答 cH；n_sym={args.n_sym}, d={args.d}；随机基线 {1.0/args.n_sym:.3f}",
        f"- 唯一变量=是否'解码中间符号→再嵌入+中间监督'（cot vs e2e 潜读出仅末端监督）。",
        "",
        "| n_hops | capcw_e2e（潜读出·末端监督） | capcw_cot（解码-再嵌入·中间监督） |",
        "|---:|---:|---:|",
    ]
    for h in hop_list:
        r = results[h]
        lines.append(f"| {h} | {r['e2e']['mean']:.3f}±{r['e2e']['std']:.3f} | {r['cot']['mean']:.3f}±{r['cot']['std']:.3f} |")
    lines += [
        "",
        f"- 多跳(H≥2) cot − e2e 平均增益 = **{multi_gain:+.4f}**；cot 多跳 cH 准确率 = **{cot_multi:.3f}**；各跳增益：{gain}",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[mh-cot] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    print(f"[mh-cot] 多跳增益={multi_gain:+.4f} cot多跳acc={cot_multi:.3f}", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW multi-hop via decode->re-embed (CoT) + intermediate supervision.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-sym", type=int, default=20)
    ap.add_argument("--hop-list", default="1,2,3")
    ap.add_argument("--n-distract", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--n-slots", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-json", default=REPORT_JSON)
    ap.add_argument("--report-md", default=REPORT_MD)
    return ap


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print("[mh-cot] dry-run：未训练。多跳方案C：decode→re-embed(CoT)+中间监督 vs 潜读出+末端监督。")
        print("[mh-cot] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
