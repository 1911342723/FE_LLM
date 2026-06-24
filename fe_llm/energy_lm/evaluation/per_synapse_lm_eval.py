# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/evaluation/per_synapse_lm_eval.py
==================================================
Carrier #2（真实应用）：把**完整 PER 原型**（含辨识点 #2 可学突触基底 self.synapse）
接到**真实中文字符级聊天 LM** 上，做诚实消融：完整原型 vs 阉割版（去 #2，≈因果注意力），
同数据/同预算比 **held-out 困惑度**；并展示真实模型上的可溯源（synapse 热力图 + 逐字能量）。

诚实口径：这里看的是 #2 在**真实语言建模**上的边际贡献——可能小（合成 P4 已示三方都能学），
如实报告，不夸大。

默认 dry-run；真跑加 --run。
    python -m fe_llm.energy_lm.evaluation.per_synapse_lm_eval --run
"""

from __future__ import annotations

import argparse
import copy
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
from fe_llm.energy_lm.training.chat_train import (DATA_MAIN, DATA_EXTRA, PROBES,
                                                  eval_val, generate, load_pairs, make_dataset)

REPORT_JSON = os.path.join("docs", "reports", "per_synapse_lm_eval.json")
REPORT_MD = os.path.join("docs", "reports", "per_synapse_lm_eval.md")
FIG_DIR = os.path.join("docs", "reports", "figs")


def train_lm(net, tr_seq, tr_sup, va_seq, va_sup, tok, device, *, epochs, batch, lr, seed, tag):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    best_val, best_state = 1e9, None
    for ep in range(1, epochs + 1):
        net.train()
        idx = torch.randperm(len(tr_seq), generator=g).tolist()
        tot, nt = 0.0, 0
        t0 = time.time()
        for s in range(0, len(idx), batch):
            ch = idx[s:s + batch]
            seq = torch.tensor(tr_seq[ch], device=device)
            sup = torch.tensor(tr_sup[ch], device=device)
            logits = -net(seq)
            pl = logits[:, :-1, :]; tg = seq[:, 1:]; m = sup[:, :-1]
            sl = pl[m]; t = tg[m]
            loss = F.cross_entropy(sl, t)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * t.numel(); nt += int(t.numel())
        sched.step()
        val_nats, val_ppl = eval_val(net, va_seq, va_sup, tok, device, batch)
        if val_nats < best_val:
            best_val = val_nats; best_state = copy.deepcopy(net.state_dict())
        if ep % max(1, epochs // 5) == 0 or ep == 1:
            print(f"    [{tag}] ep {ep:3d} train_loss={tot/max(1,nt):.4f} val_ppl={val_ppl:.2f} "
                  f"(best {np.exp(best_val):.2f}) {time.time()-t0:.0f}s", flush=True)
    if best_state is not None:
        net.load_state_dict(best_state)
    return float(best_val), float(np.exp(best_val))


@torch.no_grad()
def energy_trace(net, tok, prompt, device, max_steps=24):
    """逐字生成并记录所选字的能量（残余能量可溯源）。返回 (生成文本, [(字,能量)...])。"""
    net.eval()
    ids = tok.encode(prompt)[: net.max_len - 3] + [tok.sep_id, tok.bos_id]
    start = len(ids); trace = []
    for _ in range(min(max_steps, net.max_len - start)):
        pad = ids + [tok.pad_id] * (net.max_len - len(ids))
        seq = torch.tensor([pad[:net.max_len]], device=device)
        energy = net(seq)[0, len(ids) - 1]                 # 能量=-logits
        for sp in (tok.pad_id, tok.bos_id, tok.sep_id, tok.mask_id, tok.unk_id):
            energy[sp] = 1e9
        nxt = int(energy.argmin())
        if nxt == tok.eos_id:
            break
        trace.append((tok.id_to_tok[nxt], round(float(energy[nxt]), 3)))
        ids.append(nxt)
    return "".join(t for t, _ in trace), trace


def save_synapse_fig(net, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mats = [F.softplus(b.synapse.detach()).cpu().numpy() for b in net.blocks if getattr(b, "use_synapse", False)]
        if not mats:
            return None
        mat = np.mean(mats, axis=0)
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.figure(figsize=(4.6, 4.0))
        plt.imshow(mat, cmap="viridis", aspect="auto")
        plt.colorbar(label="softplus(synapse) 电导")
        plt.xlabel("源位置 i"); plt.ylabel("目标位置 j")
        plt.title("真实聊天 LM · PER #2 突触基底(因果下三角)\n经验刻出的持久位置依赖")
        os.makedirs(FIG_DIR, exist_ok=True)
        plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
        return path
    except Exception as e:
        print(f"[lm] 热力图跳过：{e}", flush=True)
        return None


def _compare_one_seed(args, pairs, seed, device, want_nets=False):
    """给定 seed：同一 train/val 切分上训完整 vs 阉割，返回 (ppl_full, ppl_abl, tok, nets)。"""
    chars = sorted({ch for p, r in pairs for ch in (p + r)})
    tok = CharTokenizer(chars)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    pr = [pairs[i] for i in perm]
    n_val = max(1, int(len(pr) * 0.05))
    va_pairs, tr_pairs = pr[:n_val], pr[n_val:]
    tr_seq, tr_sup = make_dataset(tok, tr_pairs, args.max_len)
    va_seq, va_sup = make_dataset(tok, va_pairs, args.max_len)
    ppl = {}; nets = {}
    for use_syn, kind in [(True, "per_syn"), (False, "per_nosyn")]:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        net = SeqEnergyNet(tok.vocab_size, args.max_len, dim=args.dim, depth=args.depth,
                           n_heads=args.heads, use_synapse=use_syn).to(device)
        _, best_ppl = train_lm(net, tr_seq, tr_sup, va_seq, va_sup, tok, device,
                               epochs=args.epochs, batch=args.batch, lr=args.lr, seed=seed, tag=f"{kind}@s{seed}")
        ppl[kind] = best_ppl
        nets[kind] = net if want_nets else None
        net_params = sum(p.numel() for p in net.parameters())
    return ppl["per_syn"], ppl["per_nosyn"], tok, nets, net_params


def run(args):
    device = get_device()
    print(f"[lm] device={device}", flush=True)
    paths = [args.data] + ([DATA_EXTRA] if args.extra else [])
    pairs = load_pairs(paths, args.max_pairs)
    if not pairs:
        print("[lm] 无数据，退出。"); return 1
    print(f"[lm] 共 {len(pairs)} 对，多种子={args.n_seeds}，定长 {args.max_len}", flush=True)

    seeds = [args.seed + i for i in range(max(1, args.n_seeds))]
    per_seed = []
    tok = nets = None
    full_params = 0
    for si, sd in enumerate(seeds):
        pf, pa, tok_i, nets_i, npar = _compare_one_seed(args, pairs, sd, device, want_nets=(si == 0))
        per_seed.append({"seed": sd, "per_syn": round(pf, 3), "per_nosyn": round(pa, 3),
                         "delta_abl_minus_full": round(pa - pf, 3)})
        print(f"[lm] seed {sd}: per_syn={pf:.3f} per_nosyn={pa:.3f} Δ={pa-pf:+.3f}", flush=True)
        if si == 0:
            tok, nets, full_params = tok_i, nets_i, npar
    deltas = np.array([d["delta_abl_minus_full"] for d in per_seed])
    full_ppls = np.array([d["per_syn"] for d in per_seed])
    abl_ppls = np.array([d["per_nosyn"] for d in per_seed])
    out = {"per_syn": {"params": full_params, "val_ppl": round(float(full_ppls.mean()), 3),
                       "val_ppl_std": round(float(full_ppls.std()), 3)},
           "per_nosyn": {"params": full_params, "val_ppl": round(float(abl_ppls.mean()), 3),
                         "val_ppl_std": round(float(abl_ppls.std()), 3)},
           "per_seed": per_seed,
           "delta_mean": round(float(deltas.mean()), 3), "delta_std": round(float(deltas.std()), 3),
           "wins_full": int((deltas > 0).sum()), "n_seeds": len(seeds)}
    print(f"[lm] 跨 {len(seeds)} 种子：Δ(阉割−完整)均值={out['delta_mean']:+.3f}±{out['delta_std']:.3f}，"
          f"完整更优 {out['wins_full']}/{len(seeds)} 次", flush=True)

    # 可溯源：真实模型 synapse 热力图 + 逐字能量 trace + 样例生成
    fig = save_synapse_fig(nets["per_syn"], os.path.join(FIG_DIR, "per_synapse_lm_pathways.png"))
    samples = []
    for p in PROBES[:5]:
        gen_s, _ = energy_trace(nets["per_syn"], tok, p, device)
        gen_n, _ = energy_trace(nets["per_nosyn"], tok, p, device)
        samples.append({"prompt": p, "per_syn": gen_s, "per_nosyn": gen_n})
    demo_prompt = PROBES[0]
    _, etrace = energy_trace(nets["per_syn"], tok, demo_prompt, device, max_steps=12)

    ppl_full, ppl_abl = out["per_syn"]["val_ppl"], out["per_nosyn"]["val_ppl"]
    delta, dstd, wins, ns = out["delta_mean"], out["delta_std"], out["wins_full"], out["n_seeds"]
    rel = round(100.0 * delta / ppl_abl, 2) if ppl_abl else 0.0
    robust = (delta > 0) and (delta > dstd) and (wins == ns)        # 方向稳健：均值>0、大于波动、全胜
    if robust:
        verdict = (f"✅ 真实聊天语言建模上，**完整原型(含 #2)held-out 困惑度更低且方向稳健**："
                   f"均值 {ppl_full:.2f} vs 阉割 {ppl_abl:.2f}（Δ={delta:+.2f}±{dstd:.2f}，相对 {rel:.1f}%，"
                   f"{wins}/{ns} 种子完整更优）——#2 可学突触基底在真实语言上带来**小而一致**的正收益，且结构(热力图)+逐字能量轨迹可溯源。")
    elif delta > 0:
        verdict = (f"🟡 真实语言上完整原型均值更优但**未稳健过噪声**：{ppl_full:.2f} vs {ppl_abl:.2f}"
                   f"（Δ={delta:+.2f}±{dstd:.2f}，{wins}/{ns} 胜）——#2 收益若有也很小、被种子波动淹没；"
                   f"其确证价值在**可溯源**（synapse 持久结构 + 逐字能量可读），而非降 ppl。")
    else:
        verdict = (f"🟡 诚实负/持平：完整原型 {ppl_full:.2f} 未优于阉割 {ppl_abl:.2f}（Δ={delta:+.2f}±{dstd:.2f}）——"
                   f"#2 在该规模真实语言建模上不靠降 ppl 取胜；价值在可溯源/结构干预（见合成 P1）。")

    results = {
        "task": "real Chinese char-level chat LM: full PER prototype (#2) vs ablated (no synapse)",
        "config": {"data": os.path.basename(args.data), "n_pairs": len(pairs), "max_len": args.max_len,
                   "dim": args.dim, "depth": args.depth, "heads": args.heads, "epochs": args.epochs, "n_seeds": ns},
        "ablation": out, "ppl_delta_abl_minus_full": delta, "ppl_delta_std": dstd, "ppl_delta_rel_pct": rel,
        "samples": samples, "energy_trace_demo": {"prompt": demo_prompt, "trace": etrace},
        "fig": fig, "verdict": verdict,
    }
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    _write_md(results)
    print("\n[lm] === 结论 ===\n" + verdict, flush=True)
    return 0


def _write_md(r):
    o = r["ablation"]
    L = ["# Carrier #2 · 真实聊天 LM：完整 PER 原型(#2) vs 阉割版 诚实消融", "",
         f"- 数据：{r['config']['data']}（{r['config']['n_pairs']} 对），字表级中文；"
         f"dim={r['config']['dim']} depth={r['config']['depth']} max_len={r['config']['max_len']} "
         f"epochs={r['config']['epochs']}，**{r['config']['n_seeds']} 个随机种子**。",
         f"- 口径：同数据/同预算，held-out（5%）回应区困惑度；多种子取均值±std 防单次噪声。", "",
         "## 消融：held-out 困惑度（多种子均值±std）", "",
         "| 模型 | 参数 | held-out ppl(均值±std) |", "|---|---:|---:|",
         f"| 完整原型 PER+syn(#2) | {o['per_syn']['params']/1e6:.2f}M | **{o['per_syn']['val_ppl']:.2f} ± {o['per_syn']['val_ppl_std']:.2f}** |",
         f"| 阉割 PER−syn(无#2) | {o['per_nosyn']['params']/1e6:.2f}M | {o['per_nosyn']['val_ppl']:.2f} ± {o['per_nosyn']['val_ppl_std']:.2f} |",
         "",
         f"- Δppl(阉割−完整) = **{r['ppl_delta_abl_minus_full']:+.2f} ± {r['ppl_delta_std']:.2f}**"
         f"（相对 {r['ppl_delta_rel_pct']:+.1f}%）；完整更优 {o['wins_full']}/{o['n_seeds']} 种子",
         "",
         "| seed | 完整 ppl | 阉割 ppl | Δ |", "|---|---:|---:|---:|"]
    for d in o["per_seed"]:
        L.append(f"| {d['seed']} | {d['per_syn']:.2f} | {d['per_nosyn']:.2f} | {d['delta_abl_minus_full']:+.2f} |")
    L.append("")
    if r.get("fig"):
        L += ["## 可溯源：真实模型 synapse 持久结构", "", f"- 突触热力图：`{r['fig']}`", ""]
    et = r["energy_trace_demo"]
    L += ["## 可溯源：逐字能量轨迹（残余能量可读）", "",
          f"- prompt「{et['prompt']}」逐字(字→能量)：" + " ".join(f"{c}({e})" for c, e in et["trace"]), ""]
    L += ["## 样例生成（完整 vs 阉割）", "", "| prompt | 完整原型 | 阉割版 |", "|---|---|---|"]
    for s in r["samples"]:
        L.append(f"| {s['prompt']} | {s['per_syn']} | {s['per_nosyn']} |")
    L += ["", "## 结论", "", r["verdict"], ""]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[lm] 报告写出 → {REPORT_MD}", flush=True)


def build_arg_parser():
    ap = argparse.ArgumentParser(description="真实聊天 LM 上 完整 PER 原型(#2) vs 阉割 诚实消融。")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--data", default=DATA_MAIN)
    ap.add_argument("--extra", action="store_true", help="叠加 LCCC 真实口语对话")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-seeds", type=int, default=3, help="多种子聚合,防单次噪声(方向稳健性)")
    return ap


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print("[lm] dry-run：真实聊天 LM 上完整 PER 原型(#2) vs 阉割消融。真跑加 --run。")
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
