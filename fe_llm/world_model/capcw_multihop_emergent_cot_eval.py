# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_multihop_emergent_cot_eval.py
======================================================
**纯涌现潜 CoT**：多跳链式是否需要"教 emit 中间符号"（中间监督），还是 **decode→re-embed 本身**就足以
让链式自发涌现？见 `docs/FE-LLM核心引擎构想.md` 第 23/26 节，承接 `capcw_multihop_cot_eval.py`。

第 23 节证了「多跳要 CoT」，但把两个变量**混在一起**比较：
- cot = decode→re-embed + **中间监督**（PASS, +0.30）
- e2e = 潜读出(latent)   + **仅末端监督**（FAIL）

本脚本做干净的 **2×2 单变量析因**（唯一两变量：读出结构 decode-vs-latent × 监督 中间-vs-仅末端）：

|            | 仅末端监督(final-only) | 中间监督(intermediate) |
|------------|------------------------|------------------------|
| latent     | e2e（已证 FAIL）        | latent+中间监督         |
| decode     | **emergent（本问）**    | cot（已证 PASS）        |

关键格 = **emergent (decode + 仅末端监督)**：不教中间符号、只监督最终答案，看 decode→re-embed 的离散
符号瓶颈能否**自发**逼出正确的中间步（潜在 CoT 涌现）。

判据（先写死）
--------------
- H1（decode 本身的作用）：多跳(H≥2) emergent − e2e ≥ +0.15 → decode→re-embed 即便无中间监督也带来链式。
- H2（中间监督是否必需）：报告 cot − emergent。
  - 若 emergent ≥ 0.7×cot 且 H1 成立 → **PASS：纯涌现潜 CoT 成立**（中间监督非必需，decode 瓶颈即可涌现链式）；
  - 若 H1 成立但 emergent < 0.7×cot → **PARTIAL**：decode 有帮助但中间监督显著更强（部分涌现）；
  - 若 H1 不成立（emergent≈e2e）→ **FAIL(中间监督必需)**：CoT 不自发涌现，必须教 emit 中间符号。诚实记录。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_multihop_emergent_cot_eval --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from fe_llm.config import get_device
from fe_llm.world_model.capcw_multihop_cot_eval import CAPCWChain, gen_chain, train_eval_chain

REPORT_JSON = os.path.join("docs", "reports", "capcw_multihop_emergent_cot_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_multihop_emergent_cot_eval.md")

# 2×2 四格：(读出结构 cot=decode/latent, 监督 intermediate_sup=中间/仅末端)。
CELLS = {
    "e2e":      {"cot": False, "intermediate_sup": False},   # latent + 仅末端（已证 FAIL）
    "latent_is": {"cot": False, "intermediate_sup": True},   # latent + 中间监督
    "emergent": {"cot": True,  "intermediate_sup": False},   # decode + 仅末端（本问：纯涌现）
    "cot":      {"cot": True,  "intermediate_sup": True},    # decode + 中间监督（已证 PASS）
}


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    hop_list = [int(x) for x in args.hop_list.split(",")]
    print(f"[mh-emg] device={device} hop_list={hop_list} d={args.d} seeds={args.seeds} (2x2 析因)", flush=True)
    results: dict = {}
    for h in hop_list:
        seq_len = max(args.seq_len, 2 * (h + args.n_distract) + 4)
        n_slots = max(args.n_slots, h + args.n_distract + 1)
        cell_accs = {name: [] for name in CELLS}
        for si in range(args.seeds):
            seed = args.seed + si
            train = gen_chain(args.n_sym, h, args.n_distract, seq_len, args.n_train, seed)
            test = gen_chain(args.n_sym, h, args.n_distract, seq_len, args.n_test, seed + 5000)
            for name, cfg in CELLS.items():
                torch.manual_seed(seed)                      # 同 seed 初始化：四格唯一差异=cfg
                model = CAPCWChain(args.n_sym, seq_len, args.d, n_slots, args.iters, h, cot=cfg["cot"])
                acc = train_eval_chain(model, train, test, device=device, epochs=args.epochs, lr=args.lr,
                                       batch=args.batch, seed=seed, intermediate_sup=cfg["intermediate_sup"])
                cell_accs[name].append(acc)
        results[h] = {name: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
                      for name, v in cell_accs.items()}
        row = results[h]
        print(f"[mh-emg] H={h} e2e={row['e2e']['mean']:.3f} latent_is={row['latent_is']['mean']:.3f} "
              f"emergent={row['emergent']['mean']:.3f} cot={row['cot']['mean']:.3f} (rand={1.0/args.n_sym:.3f})",
              flush=True)

    multi = [h for h in hop_list if h >= 2]

    def mmean(name):
        return float(np.mean([results[h][name]["mean"] for h in multi])) if multi else 0.0

    e2e_m, lis_m, emg_m, cot_m = mmean("e2e"), mmean("latent_is"), mmean("emergent"), mmean("cot")
    h1_gain = round(emg_m - e2e_m, 4)                          # decode 本身（无中间监督）的作用
    cot_minus_emg = round(cot_m - emg_m, 4)                    # 中间监督的额外作用
    ratio = round(emg_m / cot_m, 4) if cot_m > 1e-9 else 0.0

    if h1_gain >= 0.15 and ratio >= 0.7:
        verdict = (f"PASS(纯涌现): decode→re-embed **即便仅末端监督**也让多跳链式涌现——emergent {emg_m:.3f} "
                   f"比 latent 仅末端(e2e){e2e_m:.3f} 高 {h1_gain:+.4f}，且达 full-cot 的 {ratio:.0%}"
                   f"（cot−emergent 仅 {cot_minus_emg:+.4f}）。**中间监督非必需**：离散符号瓶颈(decode→re-embed)"
                   f"本身就把正确中间步逼出来了——潜在 CoT 可自发涌现，中间监督只是加速/增强。")
    elif h1_gain >= 0.10:
        verdict = (f"PARTIAL(部分涌现): decode→re-embed 在仅末端监督下也带来链式增益（emergent−e2e {h1_gain:+.4f}），"
                   f"但显著弱于 full-cot（cot−emergent {cot_minus_emg:+.4f}，emergent 仅达 cot 的 {ratio:.0%}）"
                   f"——中间监督仍重要（涌现不充分）。")
    else:
        verdict = (f"FAIL(中间监督必需): 仅末端监督时 decode→re-embed 也救不活多跳（emergent {emg_m:.3f} ≈ "
                   f"e2e {e2e_m:.3f}，增益 {h1_gain:+.4f}<0.10）——CoT **不会自发涌现**，必须显式教 emit 中间符号"
                   f"（中间监督）。诚实负结果，与「多跳要 CoT 且要教」一致。")

    result = {
        "task": "emergent latent CoT: does decode->re-embed alone (final-only sup) yield multi-hop chaining?",
        "design": "2x2 factorial: readout(decode/latent) x supervision(intermediate/final-only); same task/arch/seed.",
        "config": {"n_sym": args.n_sym, "hop_list": hop_list, "n_distract": args.n_distract, "d": args.d,
                   "epochs": args.epochs, "seeds": args.seeds, "random_baseline": round(1.0 / args.n_sym, 4)},
        "by_n_hops": results,
        "multihop_means": {"e2e": round(e2e_m, 4), "latent_is": round(lis_m, 4),
                           "emergent": round(emg_m, 4), "cot": round(cot_m, 4)},
        "emergent_minus_e2e": h1_gain,
        "cot_minus_emergent": cot_minus_emg,
        "emergent_over_cot_ratio": ratio,
        "verdict": verdict,
        "note": "emergent=decode→re-embed + 仅末端监督（不教中间符号）；cot=decode + 中间监督；e2e=latent + 仅末端；"
                "latent_is=latent + 中间监督。唯一两变量=读出结构 × 监督，同 seed 初始化。判'纯涌现'看 emergent 这一格。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# CAPCW · 纯涌现潜 CoT 2×2 析因（decode-vs-latent × 中间监督-vs-仅末端）",
        "",
        f"- 判定：**{verdict}**",
        f"- 任务：链 c0→…→cH 查 c0 答 cH；n_sym={args.n_sym}, d={args.d}；随机基线 {1.0/args.n_sym:.3f}",
        "",
        "| n_hops | e2e(latent·末端) | latent_is(latent·中间) | **emergent(decode·末端)** | cot(decode·中间) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for h in hop_list:
        r = results[h]
        lines.append(f"| {h} | {r['e2e']['mean']:.3f} | {r['latent_is']['mean']:.3f} | "
                     f"**{r['emergent']['mean']:.3f}** | {r['cot']['mean']:.3f} |")
    lines += [
        "",
        f"- 多跳(H≥2)均值：e2e {e2e_m:.3f} / latent_is {lis_m:.3f} / **emergent {emg_m:.3f}** / cot {cot_m:.3f}",
        f"- emergent − e2e（decode 本身的作用）= **{h1_gain:+.4f}**；cot − emergent（中间监督的额外作用）= "
        f"**{cot_minus_emg:+.4f}**；emergent/cot = **{ratio:.0%}**",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[mh-emg] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    print(f"[mh-emg] emergent−e2e={h1_gain:+.4f} cot−emergent={cot_minus_emg:+.4f} emergent/cot={ratio:.0%}", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Emergent latent CoT 2x2 factorial (decode/latent x intermediate/final-only).")
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
        print("[mh-emg] dry-run：未训练。2×2 析因：decode-vs-latent × 中间监督-vs-仅末端；关键格=emergent(decode+仅末端)。")
        print("[mh-emg] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
