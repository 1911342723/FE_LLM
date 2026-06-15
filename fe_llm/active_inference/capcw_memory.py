# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/capcw_memory.py —— CAPCW 内容寻址工作记忆（接回 controller 的适配器）
==============================================================================================
把已验证的 CAPCW 核心引擎（内容寻址 slot 工作空间 + query 路由 surprise）做成一个 **controller 兼容
的工作记忆组件**，落实"引擎 surprise → 何时不该答"这一招牌决策（见 `docs/FE-LLM核心引擎构想.md`
第 12/15 节、`经验.md` CAPCW Part3）。

定位（诚实、不与 belief 槽位字典冗余）
--------------------------------------
- controller 的 `BeliefState.known_slots` 是**预定义槽位**（route/city/date…）的精确字典；
- 本组件是 **in-context 任意键值绑定**的工作记忆：把对话里现场出现的 (key→value) 关联绑进 slot
  工作空间，按内容寻址取回，并由 query 路由匹配度的**补值=surprise** 决定动作：
    - bound（匹配到，低 surprise）→ ANSWER + 取回 value；
    - unbound（匹配不到，高 surprise）→ ASK_CLARIFICATION（"知道何时不该答"）。
- 关键价值：这个 ASK/ANSWER 决策**从引擎 surprise 自然涌现**（Part3 已证：无动作监督即可分开 bound/unbound），
  而不是手写 `if key in dict` 规则——这是 FE-LLM "机制从引擎涌现"主张在 controller 上的落地。

诚实边界
--------
- 绑定工作空间需在绑定任务上**训练**才会内容寻址（in-context 泛化到未见绑定，正是 capcw_binding 的结论）；
  本组件训练于符号化绑定（key/value 为 id），适用于**受控小词表**的 in-context 关联。
- 活文本对话要自动把"现场关联"抽成 (key,value) 需要一层 in-context 绑定 NLU（开放词表/容量受限），
  属下一步；本组件先把"引擎→surprise→动作"的桥在 controller 决策框架内打通并可判定。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fe_llm.active_inference.policy import ActionType
from fe_llm.world_model.capcw import PCWorkspace


class _BindingWorkspace(nn.Module):
    """绑定工作空间：pair(key,value) → slot 工作空间(PCWorkspace) + query 内容寻址读出。

    与 `capcw_binding_eval.CAPCWPCModel` 同构（已验证形态），此处内置以使 active_inference 仅依赖引擎
    `capcw.PCWorkspace`、不反向依赖 eval 脚本。`query_match` 给出 query→slot 最大路由权重（=匹配度，
    其补=surprise）。
    """

    def __init__(self, n_keys: int, n_vals: int, d: int, n_slots: int, iters: int = 3):
        super().__init__()
        self.key_emb = nn.Embedding(n_keys, d)
        self.val_emb = nn.Embedding(n_vals, d)
        self.pair = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.ws = PCWorkspace(dim=d, n_slots=n_slots, iters=iters)
        self.to_q = nn.Linear(d, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_vals))
        self.d = d

    def _pairs(self, pk, pv):
        return self.pair(torch.cat([self.key_emb(pk), self.val_emb(pv)], dim=-1))    # (B,K,d)

    def _slots(self, pk, pv, n_slots=None):
        return self.ws(self._pairs(pk, pv), n_slots=n_slots).slots                   # (B,M,d)

    def forward(self, pk, pv, qk, n_slots=None):
        slots = self._slots(pk, pv, n_slots=n_slots)
        q = self.to_q(self.key_emb(qk))
        score = torch.einsum("bmd,bd->bm", slots, q) / math.sqrt(self.d)
        attn = score.softmax(dim=1).unsqueeze(-1)
        read = (slots * attn).sum(dim=1)
        return self.head(read)

    @torch.no_grad()
    def query_match(self, pk, pv, qk, n_slots=None):
        """query→slot 最大路由权重：bound 高 / unbound 低。其补 = surprise。"""
        slots = self._slots(pk, pv, n_slots=n_slots)
        q = self.to_q(self.key_emb(qk))
        score = torch.einsum("bmd,bd->bm", slots, q) / math.sqrt(self.d)
        return score.softmax(dim=1).max(dim=1).values

    @torch.no_grad()
    def free_energy(self, pk, pv, n_slots):
        """当前绑定在 n_slots 个 slot 下弛豫后的自由能（穷则变生长准则用）。"""
        return float(self.ws(self._pairs(pk, pv), n_slots=n_slots).free_energy)


@dataclass
class MemoryDecision:
    """工作记忆对一次查询的裁决（供 controller 决定 ASK/ANSWER）。"""

    action: ActionType          # ActionType.ANSWER（bound）/ ASK_CLARIFICATION（unbound）——controller 兼容
    value: int | None           # bound 时取回的 value id；unbound 时 None
    surprise: float             # 1 - query 路由匹配度（高=该问）
    match: float                # query 路由匹配度（高=该答）
    bound: bool                 # surprise 是否低于阈值
    grew_slots: int | None = None  # 穷则变：本次按绑定负载自校准选用的 slot 数（grow=True 时给出）


class CAPCWWorkingMemory:
    """CAPCW 内容寻址工作记忆适配器：bind 累积绑定、query/decide 由引擎 surprise 驱动 ASK/ANSWER。"""

    def __init__(self, n_keys: int, n_vals: int, d: int = 32, n_slots: int = 6, iters: int = 3,
                 ask_threshold: float = 0.5, device: str | None = None,
                 grow: bool = False, min_rel_gain: float = 0.15):
        self.n_keys = n_keys
        self.n_vals = n_vals
        self.d = d
        self.n_slots = n_slots                      # 工作空间（最大）slot 容量
        self.iters = iters
        self.ask_threshold = ask_threshold          # surprise ≥ 阈值 → ASK；否则 ANSWER
        # 穷则变（自我成长）：grow=True 时按当前绑定负载用"相对边际增益"准则自校准 slot 数（grow_m≤n_slots）。
        # 默认 False = 固定用 n_slots（既有行为，零回归）；grow=True 需用 grow-capable 训练（vary_k=True）。
        self.grow = grow
        self.min_rel_gain = min_rel_gain
        self.device = device or "cpu"
        self.net = _BindingWorkspace(n_keys, n_vals, d, n_slots, iters).to(self.device)
        # per-session 隔离：绑定状态按 session_id 分桶，不同会话互不串话（默认桶 "__default__" 向后兼容）。
        # 每会话维护：bindings(key id→value id) + 字符串↔id 表（活文本任意 key/value 串映射到符号 id；
        # 工作空间学的是符号无关的内容寻址路由，任意串分配不同 id 即可绑定/取回）。
        self._sessions: dict[str, dict] = {}

    DEFAULT_SESSION = "__default__"

    def _sess(self, session_id: str | None) -> dict:
        sid = session_id or self.DEFAULT_SESSION
        s = self._sessions.get(sid)
        if s is None:
            s = {"bindings": {}, "key_ids": {}, "val_ids": {}, "val_rev": {}}
            self._sessions[sid] = s
        return s

    # ---- 在线工作记忆：累积/重置 in-context 绑定（按会话隔离）----
    def reset(self, session_id: str | None = None) -> None:
        """清空某会话的绑定（session_id=None 清默认会话；传 '*' 清全部会话）。"""
        if session_id == "*":
            self._sessions.clear()
            return
        self._sessions.pop(session_id or self.DEFAULT_SESSION, None)

    def bind(self, key: int, value: int, session_id: str | None = None) -> None:
        """绑定一个 in-context 关联（key→value）。后写覆盖（同 key 取新值）。"""
        if not (0 <= key < self.n_keys and 0 <= value < self.n_vals):
            raise ValueError("key/value 超出词表范围")
        self._sess(session_id)["bindings"][int(key)] = int(value)

    def _packed(self, session_id: str | None = None):
        """把会话当前绑定打成 (pk,pv) 张量（至少 1 对，空时用占位 key=0,val=0 但会被判 unbound）。"""
        items = list(self._sess(session_id)["bindings"].items())
        if not items:
            items = [(0, 0)]                          # 占位（任何 query 都将 unbound）
        keys = torch.tensor([[k for k, _ in items]], device=self.device)
        vals = torch.tensor([[v for _, v in items]], device=self.device)
        return keys, vals

    def _pick_grow_m(self, pk, pv) -> int:
        """穷则变自校准：从 m=2 起，若 m→m+1 自由能相对下降 ≥ min_rel_gain 则继续长，否则停（≤ n_slots）。"""
        prev = self.net.free_energy(pk, pv, 2)
        m = 2
        for cand in range(3, self.n_slots + 1):
            cur = self.net.free_energy(pk, pv, cand)
            gain = (prev - cur) / max(prev, 1e-8)
            if gain >= self.min_rel_gain:
                m, prev = cand, cur
            else:
                break
        return m

    @torch.no_grad()
    def decide(self, query_key: int, session_id: str | None = None) -> MemoryDecision:
        """对一个查询键：用引擎 surprise 裁决 ASK/ANSWER（bound 时取回 value）。按会话隔离。

        grow=True 时先按当前绑定负载自校准 slot 数（穷则变/按需分配），再在该 slot 数下做 query 路由。
        """
        self.net.eval()
        bindings = self._sess(session_id)["bindings"]
        pk, pv = self._packed(session_id)
        qk = torch.tensor([int(query_key)], device=self.device)
        # 穷则变：按当前绑定数自校准 slot 数（grow=True 且绑定数 ≥2 才需要长；否则用 n_slots）。
        grew_slots = None
        n_slots = None
        if self.grow and len(bindings) >= 2 and self.n_slots > 2:
            grew_slots = self._pick_grow_m(pk, pv)
            n_slots = grew_slots
        match = float(self.net.query_match(pk, pv, qk, n_slots=n_slots)[0])
        surprise = 1.0 - match
        if not bindings:
            surprise, match = 1.0, 0.0               # 空记忆：必然该问
        bound = surprise < self.ask_threshold
        value = None
        if bound:
            value = int(self.net(pk, pv, qk, n_slots=n_slots).argmax(-1)[0])
        return MemoryDecision(
            action=ActionType.ANSWER if bound else ActionType.ASK_CLARIFICATION,
            value=value, surprise=surprise, match=match, bound=bound, grew_slots=grew_slots,
        )

    # ---- 字符串接口（供 in-context 绑定 NLU / controller 用活文本的 key/value 字符串）----
    def bind_str(self, key: str, value: str, session_id: str | None = None) -> None:
        """绑定一个文本 in-context 关联（key 串→value 串），自动分配该会话的符号 id。"""
        key, value = str(key).strip(), str(value).strip()
        if not key or not value:
            return
        s = self._sess(session_id)
        key_ids, val_ids, val_rev = s["key_ids"], s["val_ids"], s["val_rev"]
        if key not in key_ids:
            if len(key_ids) >= self.n_keys:
                raise ValueError(f"in-context key 词表已满（上限 {self.n_keys}），请 reset 或扩容")
            key_ids[key] = len(key_ids)
        if value not in val_ids:
            if len(val_ids) >= self.n_vals:
                raise ValueError(f"in-context value 词表已满（上限 {self.n_vals}），请 reset 或扩容")
            vid = len(val_ids)
            val_ids[value] = vid
            val_rev[vid] = value
        self.bind(key_ids[key], val_ids[value], session_id=session_id)

    def decide_str(self, query_key: str, session_id: str | None = None) -> tuple[MemoryDecision, str | None]:
        """对文本查询键裁决（按会话隔离）。返回 (MemoryDecision, value 串|None)。

        - 从未在本会话出现过的 key 串：平凡 unbound → ASK（显然不知道，不必跑引擎）；
        - 出现过的 key 串：由引擎 surprise 裁决 bound/unbound，bound 时把取回的 value id 映回字符串。
        """
        key = str(query_key).strip()
        s = self._sess(session_id)
        if key not in s["key_ids"]:
            return MemoryDecision(action=ActionType.ASK_CLARIFICATION, value=None,
                                  surprise=1.0, match=0.0, bound=False), None
        dec = self.decide(s["key_ids"][key], session_id=session_id)
        value_str = s["val_rev"].get(dec.value) if dec.value is not None else None
        return dec, value_str

    # ---- 训练 / 持久化 ----
    def train_on_binding(self, *, k_pairs: int = 5, n_train: int = 8000, epochs: int = 40,
                         lr: float = 2e-3, batch: int = 128, seed: int = 42, vary_k: bool = False) -> float:
        """在符号化绑定任务上训练工作空间（学到内容寻址路由，泛化到未见绑定）。返回末步训练准确率。

        vary_k=True：每例绑定数在 [2, k_pairs] 间随机（少绑定用 0 填充 query 外的占位）——让工作空间
        对**变化的绑定负载**鲁棒，是穷则变(grow=True)按需分配 slot 的训练前提。
        """
        rng = np.random.default_rng(seed)
        # 固定种子并重建 net，使 net 初始化也确定（d=32 训练高方差，否则同 seed 不同 RNG 状态结果漂、测试 flaky）。
        torch.manual_seed(seed)
        self.net = _BindingWorkspace(self.n_keys, self.n_vals, self.d, self.n_slots, self.iters).to(self.device)
        pk = np.zeros((n_train, k_pairs), dtype=np.int64)
        pv = np.zeros((n_train, k_pairs), dtype=np.int64)
        qk = np.zeros((n_train,), dtype=np.int64)
        y = np.zeros((n_train,), dtype=np.int64)
        for i in range(n_train):
            kk = int(rng.integers(2, k_pairs + 1)) if vary_k else k_pairs
            keys = rng.choice(self.n_keys, size=kk, replace=False)
            vals = rng.choice(self.n_vals, size=kk, replace=False)
            # 不足 k_pairs 的位置用第一个 (key,value) 重复填充（不引入新键，query 仍来自真实绑定）。
            pk[i, :kk], pv[i, :kk] = keys, vals
            if kk < k_pairs:
                pk[i, kk:], pv[i, kk:] = keys[0], vals[0]
            qi = int(rng.integers(kk))
            qk[i], y[i] = keys[qi], vals[qi]
        pk_t, pv_t, qk_t, y_t = (torch.tensor(a, device=self.device) for a in (pk, pv, qk, y))
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        acc = 0.0
        for _ in range(epochs):
            self.net.train()
            perm = torch.randperm(n_train, device=self.device)
            for s in range(0, n_train, batch):
                idx = perm[s:s + batch]
                opt.zero_grad()
                logit = self.net(pk_t[idx], pv_t[idx], qk_t[idx])
                loss = F.cross_entropy(logit, y_t[idx])
                loss.backward()
                opt.step()
        self.net.eval()
        with torch.no_grad():
            acc = float((self.net(pk_t, pv_t, qk_t).argmax(-1) == y_t).float().mean())
        return acc

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "config": {"n_keys": self.n_keys, "n_vals": self.n_vals, "d": self.d,
                       "n_slots": self.n_slots, "iters": self.iters, "ask_threshold": self.ask_threshold,
                       "grow": self.grow, "min_rel_gain": self.min_rel_gain},
        }, path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "CAPCWWorkingMemory":
        ckpt = torch.load(path, map_location=device or "cpu")
        cfg = ckpt["config"]
        wm = cls(n_keys=cfg["n_keys"], n_vals=cfg["n_vals"], d=cfg["d"], n_slots=cfg["n_slots"],
                 iters=cfg["iters"], ask_threshold=cfg["ask_threshold"], device=device,
                 grow=cfg.get("grow", False), min_rel_gain=cfg.get("min_rel_gain", 0.15))
        wm.net.load_state_dict(ckpt["state_dict"])
        wm.net.eval()
        return wm
