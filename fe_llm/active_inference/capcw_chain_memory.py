# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/capcw_chain_memory.py —— CAPCW 多跳链式工作记忆（潜在 CoT 接回 controller）
==================================================================================================
把已验证的「多跳链式 = decode→re-embed + 中间监督（潜在思维链 / CoT）」机制
（见 `docs/FE-LLM核心引擎构想.md` 第 23 节、`world_model/capcw_multihop_cot_eval.py`）做成一个
**controller 兼容的多跳工作记忆组件**，落实蓝图的「对内推理」——现场把多个 in-context 绑定**链式
组合**取回（A→B、B→C，问 A 答 C），并把每跳解码的中间符号作为**可溯源的 CoT trace** 输出。

与单跳的关系（诚实、零冗余）
----------------------------
- 单跳 `CAPCWWorkingMemory`（capcw_memory.py）：query 一个已绑定 key → 取回它的 value（1 跳）。
- 本组件 `CAPCWChainMemory`：从 start_key 起**链式**取回 H 跳。关键差异有二：
  1. **key/value 共享一张符号嵌入表**（`sym_emb`）——这样某一跳取回的 value 才能被**再嵌入**成
     下一跳的 query key（链式组合的前提；单跳的 key/value 是两套独立词表，无法相互转化）。
  2. **多跳读出**：每跳 read→head 解码出一个符号，cot 模式下把它 re-embed 成下一跳 query
     （潜在思维链）；latent 模式下把潜读出向量直接当下一跳 query（上轮失败形态，作消融对照）。

为什么是 decode→re-embed 而非潜空间反复 attention（核心结论，第 23 节）
----------------------------------------------------------------------
对**固定 slot** 反复读（潜迭代读）不能链式组合——读回的是纠缠向量，d=32 下无法干净地再注入为
下一跳的键查询（capcw_multihop_eval / _v2 已证 FAIL）。**把每跳中间结论解码成离散符号、再据它
检索**才让链式成立（capcw_multihop_cot_eval 证 +0.30 PASS(机制)，独立复现「LLM 多跳要 CoT」）。
本组件即把该机制接进 controller 决策框架：start 绑定→低 surprise→ANSWER + 链式取回 value；
start 未绑定→高 surprise→ASK_CLARIFICATION（多跳版「知道何时不该答」）。

诚实边界
--------
- 绝对值受 d=32 容量限制（高跳数下滑，与小 d 容量结论一致）；本组件做"机制接回"，不主张高跳数高精度。
- 本组件处理**原子符号链**（c0→c1→c2，取回的 value 直接作下一跳 key）；活文本里"复合所有格"
  （A的经理的工位）那种"value 拼下一属性再查"的链式属于上层 NLU 分解，留作下一步。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fe_llm.active_inference.policy import ActionType
from fe_llm.world_model.capcw import PCWorkspace


class _ChainWorkspace(nn.Module):
    """共享符号嵌入的绑定工作空间 + 多跳链式读出（cot=decode→re-embed / latent=潜读出）。

    与 `capcw_memory._BindingWorkspace` 同构，但 **key/value 共享一张 `sym_emb`**——使某跳取回的
    value 能被 re-embed 成下一跳 query key（链式组合的前提）。`chain_read` 与
    `capcw_multihop_cot_eval.CAPCWChain` 同形（已验证机制），区别仅在这里喂的是**显式 (key,value)
    pair**（工作记忆设置），而非需要序列相邻算子去扫描的 token 序列。
    """

    def __init__(self, n_sym: int, d: int, n_slots: int, iters: int = 3):
        super().__init__()
        self.sym_emb = nn.Embedding(n_sym, d)
        self.pair = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.ws = PCWorkspace(dim=d, n_slots=n_slots, iters=iters)
        self.to_q = nn.Linear(d, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_sym))
        self.d = d

    def _pairs(self, pk, pv):
        return self.pair(torch.cat([self.sym_emb(pk), self.sym_emb(pv)], dim=-1))   # (B,K,d)

    def _slots(self, pk, pv, n_slots=None):
        return self.ws(self._pairs(pk, pv), n_slots=n_slots).slots                  # (B,M,d)

    def _read(self, slots, q):
        """单跳内容寻址读出：返回 (read 向量, 路由匹配度 max-softmax)。匹配度高=命中=低 surprise。"""
        score = torch.einsum("bmd,bd->bm", slots, q) / math.sqrt(self.d)
        attn = score.softmax(dim=1)
        read = (slots * attn.unsqueeze(-1)).sum(dim=1)
        match = attn.max(dim=1).values
        return read, match

    def chain_read(self, pk, pv, q0_sym, n_hops: int, cot: bool = True, n_slots=None):
        """从 q0_sym 起链式读 n_hops 跳。返回 (hop_logits[list of (B,n_sym)], hop_match[list of (B,)])。

        - cot=True：每跳把解码符号(软)重嵌入为下一跳 query（潜在思维链，已验证机制）。
        - cot=False：每跳把潜读出向量直接当下一跳 query（latent，上轮失败形态，作对照/消融）。
        """
        slots = self._slots(pk, pv, n_slots=n_slots)
        q = self.to_q(self.sym_emb(q0_sym))
        hop_logits, hop_match = [], []
        for h in range(n_hops):
            read, match = self._read(slots, q)
            hop_logits.append(self.head(read))
            hop_match.append(match)
            if h < n_hops - 1:
                if cot:
                    soft_emb = hop_logits[-1].softmax(dim=-1) @ self.sym_emb.weight   # 解码→软重嵌入
                    q = self.to_q(soft_emb)
                else:
                    q = self.to_q(read)                                              # latent：潜读出直接当 query
        return hop_logits, hop_match

    def forward(self, pk, pv, q0_sym, n_hops: int, cot: bool = True, n_slots=None):
        return self.chain_read(pk, pv, q0_sym, n_hops, cot=cot, n_slots=n_slots)


@dataclass
class ChainDecision:
    """多跳工作记忆对一次链式查询的裁决（供 controller 决定 ASK/ANSWER + 取回链尾 value）。"""

    action: ActionType            # ANSWER（start 绑定）/ ASK_CLARIFICATION（start 未绑定）——controller 兼容
    value: int | None             # 链尾取回的 value id（ANSWER 时）；ASK 时 None
    chain: list[int] = field(default_factory=list)   # 各跳解码的中间符号 id（可溯源 CoT trace，含链尾）
    surprise: float = 1.0         # 1 - 首跳路由匹配度（start 未绑定→高=该问）
    match: float = 0.0            # 首跳路由匹配度（高=能起链=该答）
    hop_match: list[float] = field(default_factory=list)  # 每跳匹配度（中段断链可由此审计）
    bound: bool = False           # start 是否绑定（surprise 低于阈值）
    n_hops: int = 1


class CAPCWChainMemory:
    """CAPCW 多跳链式工作记忆适配器：绑定 (key→value) + decode→re-embed 链式取回（潜在 CoT）。

    与单跳 `CAPCWWorkingMemory` 并列（不改单跳，零回归）。`cot=True`（默认）用已验证的解码-再嵌入
    机制；`cot=False` 作 latent 消融对照（供 eval contrast）。
    """

    DEFAULT_SESSION = "__default__"

    def __init__(self, n_sym: int, d: int = 32, n_slots: int = 8, iters: int = 3,
                 ask_threshold: float = 0.5, cot: bool = True, device: str | None = None):
        self.n_sym = n_sym
        self.d = d
        self.n_slots = n_slots
        self.iters = iters
        self.ask_threshold = ask_threshold
        self.cot = cot                              # True=decode→re-embed（生产）；False=latent（对照）
        self.device = device or "cpu"
        self.net = _ChainWorkspace(n_sym, d, n_slots, iters).to(self.device)
        # per-session 隔离：每会话维护 pairs(key id→value id) + 共享 str↔id 表（key/value 同一符号空间，
        # 因链式需把 value 再作 key）。
        self._sessions: dict[str, dict] = {}

    def _sess(self, session_id: str | None) -> dict:
        sid = session_id or self.DEFAULT_SESSION
        s = self._sessions.get(sid)
        if s is None:
            s = {"pairs": {}, "sym_ids": {}, "sym_rev": {}}
            self._sessions[sid] = s
        return s

    # ---- 在线工作记忆：累积/重置链式绑定（按会话隔离）----
    def reset(self, session_id: str | None = None) -> None:
        """清空某会话的链式绑定（None=默认会话；'*'=全部会话）。"""
        if session_id == "*":
            self._sessions.clear()
            return
        self._sessions.pop(session_id or self.DEFAULT_SESSION, None)

    def bind(self, key: int, value: int, session_id: str | None = None) -> None:
        """绑定一条边 key→value（后写覆盖）。key/value 同处一个符号空间（链式需 value 可作下一跳 key）。"""
        if not (0 <= key < self.n_sym and 0 <= value < self.n_sym):
            raise ValueError("symbol id 超出词表范围")
        self._sess(session_id)["pairs"][int(key)] = int(value)

    def _packed(self, session_id: str | None = None):
        items = list(self._sess(session_id)["pairs"].items())
        if not items:
            items = [(0, 0)]                          # 占位（任何 query 都将 unbound）
        keys = torch.tensor([[k for k, _ in items]], device=self.device)
        vals = torch.tensor([[v for _, v in items]], device=self.device)
        return keys, vals

    @torch.no_grad()
    def decide_chain(self, start_key: int, n_hops: int, session_id: str | None = None) -> ChainDecision:
        """从 start_key 起链式取回 n_hops 跳：首跳匹配度→surprise 决定 ASK/ANSWER，链尾为取回 value。

        决策口径与单跳一致（首跳=能否起链=start 是否绑定）；中段断链由 hop_match trace 审计（可溯源）。
        """
        self.net.eval()
        pairs = self._sess(session_id)["pairs"]
        if not pairs:
            return ChainDecision(action=ActionType.ASK_CLARIFICATION, value=None, chain=[],
                                 surprise=1.0, match=0.0, hop_match=[], bound=False, n_hops=n_hops)
        pk, pv = self._packed(session_id)
        q0 = torch.tensor([int(start_key)], device=self.device)
        hop_logits, hop_match = self.net.chain_read(pk, pv, q0, n_hops, cot=self.cot)
        chain = [int(logit.argmax(-1)[0]) for logit in hop_logits]
        matches = [float(m[0]) for m in hop_match]
        match = matches[0] if matches else 0.0       # 首跳=能否起链
        surprise = 1.0 - match
        bound = surprise < self.ask_threshold
        value = chain[-1] if (bound and chain) else None
        return ChainDecision(
            action=ActionType.ANSWER if bound else ActionType.ASK_CLARIFICATION,
            value=value, chain=chain, surprise=surprise, match=match,
            hop_match=[round(m, 4) for m in matches], bound=bound, n_hops=n_hops,
        )

    # ---- 字符串接口（key/value 共享一张符号表）----
    def _intern(self, s: str, sess: dict) -> int:
        sym_ids, sym_rev = sess["sym_ids"], sess["sym_rev"]
        s = str(s).strip()
        if s not in sym_ids:
            if len(sym_ids) >= self.n_sym:
                raise ValueError(f"符号词表已满（上限 {self.n_sym}），请 reset 或扩容")
            sym_ids[s] = len(sym_ids)
            sym_rev[sym_ids[s]] = s
        return sym_ids[s]

    def bind_str(self, key: str, value: str, session_id: str | None = None) -> None:
        """绑定一条文本边（key 串→value 串），自动分配会话内共享符号 id。"""
        key, value = str(key).strip(), str(value).strip()
        if not key or not value:
            return
        sess = self._sess(session_id)
        k = self._intern(key, sess)
        v = self._intern(value, sess)
        self.bind(k, v, session_id=session_id)

    def decide_chain_str(self, start_key: str, n_hops: int, session_id: str | None = None):
        """对文本 start_key 链式裁决。返回 (ChainDecision, value 串|None, chain 串列表)。

        从未出现过的 start_key 串：平凡 unbound→ASK（显然起不了链，不必跑引擎）。
        """
        sess = self._sess(session_id)
        key = str(start_key).strip()
        if key not in sess["sym_ids"]:
            return (ChainDecision(action=ActionType.ASK_CLARIFICATION, value=None, chain=[],
                                  surprise=1.0, match=0.0, hop_match=[], bound=False, n_hops=n_hops),
                    None, [])
        dec = self.decide_chain(sess["sym_ids"][key], n_hops, session_id=session_id)
        value_str = sess["sym_rev"].get(dec.value) if dec.value is not None else None
        chain_str = [sess["sym_rev"].get(c) for c in dec.chain]
        return dec, value_str, chain_str

    def decide_path_str(self, base: str, rels: list[str], session_id: str | None = None):
        """复合所有格链式取回（活文本多跳）：从 base 起按关系链逐跳"解码→拼下一属性→再查"。

        这是 decode→re-embed 在**字符串层**的实现（潜在 CoT 的最直接落地）：每跳取回的中间 value
        被**显式解码成离散符号串**、与下一关系拼成下一跳的 key 再检索。返回 (ChainDecision, value 串|None,
        chain 串列表=可溯源 CoT trace)。

        - rels 为空：原子键直查（base 即 key），= 单跳。
        - rels≥1：cur=base，逐跳 key=f"{cur}的{r}" 单跳取回→cur=value；任一跳未绑定→断链→ASK（知道何时不该答）。
        每跳用 `decide_chain_str(key, 1)`（链式工作空间的单跳取回，内容寻址）。决策 surprise=最弱一跳之补。
        """
        if not rels:
            return self.decide_chain_str(base, 1, session_id=session_id)
        cur = base
        trace: list[str] = []
        worst_match = 1.0
        for r in rels:
            key = f"{cur}的{r}"
            dec, val, _ = self.decide_chain_str(key, 1, session_id=session_id)
            worst_match = min(worst_match, dec.match)
            if not dec.bound or val is None:
                # 断链：该跳未绑定→无法完成链式→ASK（多跳版"知道何时不该答"，可溯源到断在哪跳）。
                broken = ChainDecision(action=ActionType.ASK_CLARIFICATION, value=None, chain=trace,
                                       surprise=1.0 - dec.match, match=dec.match, hop_match=[], bound=False,
                                       n_hops=len(rels))
                return broken, None, trace
            trace.append(val)
            cur = val
        done = ChainDecision(action=ActionType.ANSWER, value=None, chain=trace,
                             surprise=1.0 - worst_match, match=worst_match, hop_match=[], bound=True,
                             n_hops=len(rels))
        return done, cur, trace

    # ---- 训练 / 持久化 ----
    def train_on_chain(self, *, max_hops: int = 3, n_pairs: int = 5, n_train: int = 8000,
                       epochs: int = 40, lr: float = 2e-3, batch: int = 128, seed: int = 42) -> float:
        """在链式绑定任务上训练（与 capcw_multihop_cot_eval 同recipe）。返回末步多跳(链尾)准确率。

        每例：随机链 c0→…→c_H（H=max_hops）+ (n_pairs−H) 个干扰边；绑定集=链边+干扰边；query=c0。
        - cot=True：链式前向(解码-再嵌入) + **各跳中间监督**（CoT，已验证胜形态）；
        - cot=False：链式前向(latent 潜读出) + **仅末跳监督**（e2e，已验证败形态，作对照）。
        固定种子并重建 net（d=32 训练高方差，确保可复现、测试不 flaky）。
        """
        H = int(max_hops)
        n_distract = max(0, n_pairs - H)
        need = (H + 1) + 2 * n_distract
        if need > self.n_sym:
            raise ValueError(f"n_sym={self.n_sym} 太小，链+干扰需 {need} 个不同符号")
        if n_pairs > self.n_slots:
            raise ValueError(f"n_pairs={n_pairs} 超过 n_slots={self.n_slots}")
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.net = _ChainWorkspace(self.n_sym, self.d, self.n_slots, self.iters).to(self.device)
        pk = np.zeros((n_train, n_pairs), dtype=np.int64)
        pv = np.zeros((n_train, n_pairs), dtype=np.int64)
        q0 = np.zeros((n_train,), dtype=np.int64)
        chain = np.zeros((n_train, H), dtype=np.int64)
        for i in range(n_train):
            picks = rng.choice(self.n_sym, size=need, replace=False)
            c = picks[: H + 1]                            # c0..cH
            ds = picks[H + 1:]
            d_keys, d_vals = ds[:n_distract], ds[n_distract:]
            keys = [int(c[h]) for h in range(H)] + [int(k) for k in d_keys]      # H 条链边 key + 干扰 key
            vals = [int(c[h + 1]) for h in range(H)] + [int(v) for v in d_vals]
            pk[i, :len(keys)] = keys
            pv[i, :len(vals)] = vals
            if len(keys) < n_pairs:                       # 不足用首边重复填充（不引入新键）
                pk[i, len(keys):] = keys[0]
                pv[i, len(vals):] = vals[0]
            q0[i] = int(c[0])
            chain[i] = [int(c[h + 1]) for h in range(H)]
        pk_t, pv_t, q0_t, ch_t = (torch.tensor(a, device=self.device) for a in (pk, pv, q0, chain))
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(epochs):
            self.net.train()
            perm = torch.randperm(n_train, device=self.device)
            for s in range(0, n_train, batch):
                idx = perm[s:s + batch]
                opt.zero_grad()
                hop_logits, _ = self.net.chain_read(pk_t[idx], pv_t[idx], q0_t[idx], H, cot=self.cot)
                if self.cot:
                    loss = sum(F.cross_entropy(hop_logits[h], ch_t[idx, h]) for h in range(H)) / H
                else:
                    loss = F.cross_entropy(hop_logits[-1], ch_t[idx, H - 1])     # e2e：仅末跳监督
                loss.backward()
                opt.step()
        self.net.eval()
        with torch.no_grad():
            final = self.net.chain_read(pk_t, pv_t, q0_t, H, cot=self.cot)[0][-1].argmax(-1)
            acc = float((final == ch_t[:, H - 1]).float().mean())
        return acc

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "config": {"n_sym": self.n_sym, "d": self.d, "n_slots": self.n_slots, "iters": self.iters,
                       "ask_threshold": self.ask_threshold, "cot": self.cot},
        }, path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "CAPCWChainMemory":
        ckpt = torch.load(path, map_location=device or "cpu")
        cfg = ckpt["config"]
        mem = cls(n_sym=cfg["n_sym"], d=cfg["d"], n_slots=cfg["n_slots"], iters=cfg["iters"],
                  ask_threshold=cfg["ask_threshold"], cot=cfg.get("cot", True), device=device)
        mem.net.load_state_dict(ckpt["state_dict"])
        mem.net.eval()
        return mem
