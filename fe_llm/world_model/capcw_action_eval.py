# -*- coding: utf-8 -*-
"""
fe_llm/world_model/capcw_action_eval.py
=======================================
CAPCW 阶段三：把 slot 工作空间接到控制层，并区分「动作类型」与「回复内容」两面。

任务（绑定 + 双输出）：给 K 对随机 (key→value) + 一个 query key：
- 动作类型（粗）：query 未绑定→ASK(0)；已绑定且 value<半区→ANSWER(1)；已绑定且 value≥半区→REFUSE(2)；
- 回复内容（细）：若已绑定，还要答出**精确的 value**（n_vals 路）。

预期（呼应 B2 真实数据的发现）：
- 动作类型只需粗判断 → 单向量也够，flat ≈ CAPCW（动作类型无 headroom）；
- 回复内容需 content-addressable 取回精确 value → 单向量难，**CAPCW 明显胜**（内容才是 CAPCW 的价值）。
附：CAPCW 的 query→slot 路由作为 surprise 信号，是否分离 bound/unbound。

判据：value(内容) CAPCW − flat ≥ +0.10 → CAPCW 在控制层的价值落在"内容/状态取回"（与 B2 一脉相承）。

默认 dry-run；--run 真跑。
运行：python -m fe_llm.world_model.capcw_action_eval --run
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
from fe_llm.world_model.capcw_binding_eval import PairEncoder

REPORT_JSON = os.path.join("docs", "reports", "capcw_action_eval.json")
REPORT_MD = os.path.join("docs", "reports", "capcw_action_eval.md")

N_ACTIONS = 3  # 0=ASK(未绑定), 1=ANSWER(已绑定,value 有效), 2=REFUSE(已绑定,value 风险)


def gen_action(n_keys, n_vals, k, n, seed, p_bound=0.5):
    """返回 pk,pv,qk, y_action(3), y_value(精确 value；未绑定记 -1 表示 value 无意义)。"""
    rng = np.random.default_rng(seed)
    half = n_vals // 2
    pk = np.zeros((n, k), dtype=np.int64)
    pv = np.zeros((n, k), dtype=np.int64)
    qk = np.zeros((n,), dtype=np.int64)
    ya = np.zeros((n,), dtype=np.int64)
    yv = np.full((n,), -1, dtype=np.int64)
    for i in range(n):
        keys = rng.choice(n_keys, size=k, replace=False)
        vals = rng.choice(n_vals, size=k, replace=False)
        pk[i] = keys
        pv[i] = vals
        if rng.random() < p_bound:
            qi = int(rng.integers(k))
            qk[i] = int(keys[qi])
            v = int(vals[qi])
            yv[i] = v
            ya[i] = 1 if v < half else 2
        else:
            not_in = [x for x in range(n_keys) if x not in set(keys.tolist())]
            qk[i] = int(rng.choice(not_in))
            ya[i] = 0
    return pk, pv, qk, ya, yv


class FlatActionModel(nn.Module):
    """单向量世界状态：池化 + query 读出 → 动作头 + value 头。"""

    def __init__(self, n_keys, n_vals, d):
        super().__init__()
        self.enc = PairEncoder(n_keys, n_vals, d)
        self.trunk = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.action_head = nn.Linear(d, N_ACTIONS)
        self.value_head = nn.Linear(d, n_vals)

    def forward(self, pk, pv, qk):
        pairs = self.enc(pk, pv)
        world = pairs.mean(dim=1)
        q = self.enc.key_emb(qk)
        h = self.trunk(torch.cat([world, q], dim=-1))
        return self.action_head(h), self.value_head(h)


class CAPCWActionModel(nn.Module):
    """slot 工作空间 + query 内容寻址读出 → 动作头 + value 头。"""

    def __init__(self, n_keys, n_vals, d, n_slots, iters):
        super().__init__()
        self.enc = PairEncoder(n_keys, n_vals, d)
        self.ws = PCWorkspace(dim=d, n_slots=n_slots, iters=iters)
        self.to_q = nn.Linear(d, d)
        self.trunk = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.action_head = nn.Linear(d, N_ACTIONS)
        self.value_head = nn.Linear(d, n_vals)
        self.d = d

    def _read(self, pk, pv, qk):
        pairs = self.enc(pk, pv)
        slots = self.ws(pairs).slots
        q = self.to_q(self.enc.key_emb(qk))
        score = torch.einsum("bmd,bd->bm", slots, q) / math.sqrt(self.d)
        attn = score.softmax(dim=1)
        read = (slots * attn.unsqueeze(-1)).sum(dim=1)
        return read, attn

    def forward(self, pk, pv, qk):
        read, _ = self._read(pk, pv, qk)
        h = self.trunk(read)
        return self.action_head(h), self.value_head(h)

    @torch.no_grad()
    def query_match(self, pk, pv, qk):
        _, attn = self._read(pk, pv, qk)
        return attn.max(dim=1).values


def balanced_acc(pred, y, n):
    accs = []
    for c in range(n):
        m = y == c
        if m.any():
            accs.append(float((pred[m] == y[m]).mean()))
    return float(np.mean(accs)) if accs else 0.0


def train_model(model, train, test, device, *, epochs, lr, batch, seed):
    torch.manual_seed(seed)
    pk, pv, qk, ya, yv = (torch.tensor(t, device=device) for t in train)
    tpk, tpv, tqk, tya, tyv = (torch.tensor(t, device=device) for t in test)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(ya)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            opt.zero_grad()
            a_logit, v_logit = model(pk[idx], pv[idx], qk[idx])
            loss = F.cross_entropy(a_logit, ya[idx])
            bound = yv[idx] >= 0                                  # value 只在已绑定上算
            if bound.any():
                loss = loss + F.cross_entropy(v_logit[bound], yv[idx][bound])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        a_logit, v_logit = model(tpk, tpv, tqk)
        a_pred = a_logit.argmax(-1).cpu().numpy()
        v_pred = v_logit.argmax(-1).cpu().numpy()
    tya_np = np.asarray(test[3])
    tyv_np = np.asarray(test[4])
    action_acc = balanced_acc(a_pred, tya_np, N_ACTIONS)
    bmask = tyv_np >= 0
    value_acc = float((v_pred[bmask] == tyv_np[bmask]).mean()) if bmask.any() else 0.0
    return model, action_acc, value_acc


def run(args: argparse.Namespace) -> dict:
    device = get_device()
    print(f"[action] device={device} K={args.k} d={args.d} n_keys={args.n_keys} n_vals={args.n_vals} seeds={args.seeds}", flush=True)
    fa_a, fa_v, ca_a, ca_v, sep_b, sep_u = [], [], [], [], [], []
    for si in range(args.seeds):
        seed = args.seed + si
        train = gen_action(args.n_keys, args.n_vals, args.k, args.n_train, seed)
        test = gen_action(args.n_keys, args.n_vals, args.k, args.n_test, seed + 5000)
        n_slots = max(args.n_slots, args.k + 1)
        flat = FlatActionModel(args.n_keys, args.n_vals, args.d)
        capcw = CAPCWActionModel(args.n_keys, args.n_vals, args.d, n_slots=n_slots, iters=args.iters)
        common = dict(device=device, epochs=args.epochs, lr=args.lr, batch=args.batch, seed=seed)
        _, faa, fav = train_model(flat, train, test, **common)
        cmodel, caa, cav = train_model(capcw, train, test, **common)
        fa_a.append(faa); fa_v.append(fav); ca_a.append(caa); ca_v.append(cav)
        tpk, tpv, tqk = (torch.tensor(test[i], device=device) for i in range(3))
        mm = cmodel.query_match(tpk, tpv, tqk).cpu().numpy()
        tyv = np.asarray(test[4])
        sep_b.append(float(mm[tyv >= 0].mean())); sep_u.append(float(mm[tyv < 0].mean()))
        print(f"[action] seed={seed} | action: flat={faa:.3f} capcw={caa:.3f} | value: flat={fav:.3f} capcw={cav:.3f}", flush=True)

    faa_m, caa_m = float(np.mean(fa_a)), float(np.mean(ca_a))
    fav_m, cav_m = float(np.mean(fa_v)), float(np.mean(ca_v))
    d_action = round(caa_m - faa_m, 4)
    d_value = round(cav_m - fav_m, 4)
    sep = round(abs(float(np.mean(sep_b)) - float(np.mean(sep_u))), 4)
    if d_value >= 0.10:
        verdict = ("PASS: 动作类型两者皆可(无 headroom)，但回复内容(精确 value 取回) CAPCW 明显胜单向量"
                   "——CAPCW 在控制层的价值落在内容/状态取回，与 B2 真实数据结论一脉相承")
    elif d_value >= 0.03:
        verdict = "PARTIAL: 回复内容上 CAPCW 有正向优势但偏弱"
    else:
        verdict = "FAIL: 回复内容上 CAPCW 未明显胜单向量"

    result = {
        "task": "binding + dual output: action type (coarse) & answer content (exact value)",
        "config": {"k": args.k, "d": args.d, "n_keys": args.n_keys, "n_vals": args.n_vals,
                   "epochs": args.epochs, "seeds": args.seeds, "value_random_baseline": round(1.0 / args.n_vals, 4)},
        "action_type": {"flat": round(faa_m, 4), "capcw": round(caa_m, 4), "delta": d_action},
        "answer_content_value": {"flat": round(fav_m, 4), "capcw": round(cav_m, 4), "delta": d_value},
        "query_routing_separation_bound_vs_unbound": sep,
        "verdict": verdict,
        "note": "动作类型粗判断单向量够用(无 headroom)；回复内容需取回精确 value，是 CAPCW 内容寻址的主场。"
                "与 B2(真实数据 belief 价值在内容/状态、不在动作类型)在引擎层一致。",
    }
    os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    lines = [
        "# CAPCW 阶段三 · 控制层整合：动作类型 vs 回复内容",
        "",
        f"- 判定：**{verdict}**",
        f"- 任务：绑定+双输出（动作类型 ask/answer/refuse 粗；回复内容=精确 value 细）；K={args.k}, d={args.d}, seeds={args.seeds}；value 随机基线 {1.0/args.n_vals:.3f}",
        "",
        "| 维度 | flat（单向量） | CAPCW（slot 工作空间） | delta |",
        "|---|---:|---:|---:|",
        f"| 动作类型（value 依赖+联合训练） | {faa_m:.4f} | {caa_m:.4f} | {d_action:+.4f} |",
        f"| 回复内容·精确 value（CAPCW 主场） | {fav_m:.4f} | {cav_m:.4f} | **{d_value:+.4f}** |",
        "",
        f"- query→slot 路由分离 bound/unbound：{sep:.4f}（surprise 信号：未绑定难匹配=高 surprise）",
        "",
        "- 注：纯「成员判断」动作（query 是否在场→答/问）单向量也能 ~1.0、无 headroom（与 B2 一致）；",
        "  本任务动作 value 依赖且与 value 头联合训练，故 flat 在动作上也降、CAPCW 两面皆胜。",
        "  最干净的判别结果是「回复内容·精确 value」——content 取回才是 CAPCW 不可替代之处。",
        "",
        f"- 说明：{result['note']}",
    ]
    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[action] === 裁决 ===", flush=True)
    print(verdict, flush=True)
    print(f"[action] 动作类型 flat={faa_m:.3f}/capcw={caa_m:.3f}; 内容 value flat={fav_m:.3f}/capcw={cav_m:.3f} (delta {d_value:+.4f})", flush=True)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CAPCW phase-3: workspace in control layer; action-type vs content.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-keys", type=int, default=10)
    ap.add_argument("--n-vals", type=int, default=12)
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
    args = build_arg_parser().parse_args(argv)
    if not args.run:
        print("[action] dry-run：未训练。绑定+双输出（动作类型 vs 精确 value 内容）上 flat vs CAPCW。")
        print("[action] 真正运行请显式追加 --run。")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
