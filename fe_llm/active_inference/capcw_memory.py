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

    def _slots(self, pk, pv):
        pairs = self.pair(torch.cat([self.key_emb(pk), self.val_emb(pv)], dim=-1))  # (B,K,d)
        return self.ws(pairs).slots                                                  # (B,M,d)

    def forward(self, pk, pv, qk):
        slots = self._slots(pk, pv)
        q = self.to_q(self.key_emb(qk))
        score = torch.einsum("bmd,bd->bm", slots, q) / math.sqrt(self.d)
        attn = score.softmax(dim=1).unsqueeze(-1)
        read = (slots * attn).sum(dim=1)
        return self.head(read)

    @torch.no_grad()
    def query_match(self, pk, pv, qk):
        """query→slot 最大路由权重：bound 高 / unbound 低。其补 = surprise。"""
        slots = self._slots(pk, pv)
        q = self.to_q(self.key_emb(qk))
        score = torch.einsum("bmd,bd->bm", slots, q) / math.sqrt(self.d)
        return score.softmax(dim=1).max(dim=1).values


@dataclass
class MemoryDecision:
    """工作记忆对一次查询的裁决（供 controller 决定 ASK/ANSWER）。"""

    action: ActionType          # ActionType.ANSWER（bound）/ ASK_CLARIFICATION（unbound）——controller 兼容
    value: int | None           # bound 时取回的 value id；unbound 时 None
    surprise: float             # 1 - query 路由匹配度（高=该问）
    match: float                # query 路由匹配度（高=该答）
    bound: bool                 # surprise 是否低于阈值


class CAPCWWorkingMemory:
    """CAPCW 内容寻址工作记忆适配器：bind 累积绑定、query/decide 由引擎 surprise 驱动 ASK/ANSWER。"""

    def __init__(self, n_keys: int, n_vals: int, d: int = 32, n_slots: int = 6, iters: int = 3,
                 ask_threshold: float = 0.5, device: str | None = None):
        self.n_keys = n_keys
        self.n_vals = n_vals
        self.d = d
        self.n_slots = n_slots
        self.iters = iters
        self.ask_threshold = ask_threshold          # surprise ≥ 阈值 → ASK；否则 ANSWER
        self.device = device or "cpu"
        self.net = _BindingWorkspace(n_keys, n_vals, d, n_slots, iters).to(self.device)
        self._bindings: dict[int, int] = {}         # 当前会话累积的 (key id → value id)

    # ---- 在线工作记忆：累积/重置 in-context 绑定 ----
    def reset(self) -> None:
        self._bindings.clear()

    def bind(self, key: int, value: int) -> None:
        """绑定一个 in-context 关联（key→value）。后写覆盖（同 key 取新值）。"""
        if not (0 <= key < self.n_keys and 0 <= value < self.n_vals):
            raise ValueError("key/value 超出词表范围")
        self._bindings[int(key)] = int(value)

    def _packed(self):
        """把当前绑定打成 (pk,pv) 张量（至少 1 对，空时用占位 key=0,val=0 但会被判 unbound）。"""
        items = list(self._bindings.items())
        if not items:
            items = [(0, 0)]                          # 占位（任何 query 都将 unbound）
        keys = torch.tensor([[k for k, _ in items]], device=self.device)
        vals = torch.tensor([[v for _, v in items]], device=self.device)
        return keys, vals

    @torch.no_grad()
    def decide(self, query_key: int) -> MemoryDecision:
        """对一个查询键：用引擎 surprise 裁决 ASK/ANSWER（bound 时取回 value）。"""
        self.net.eval()
        pk, pv = self._packed()
        qk = torch.tensor([int(query_key)], device=self.device)
        match = float(self.net.query_match(pk, pv, qk)[0])
        surprise = 1.0 - match
        if not self._bindings:
            surprise, match = 1.0, 0.0               # 空记忆：必然该问
        bound = surprise < self.ask_threshold
        value = None
        if bound:
            value = int(self.net(pk, pv, qk).argmax(-1)[0])
        return MemoryDecision(
            action=ActionType.ANSWER if bound else ActionType.ASK_CLARIFICATION,
            value=value, surprise=surprise, match=match, bound=bound,
        )

    # ---- 训练 / 持久化 ----
    def train_on_binding(self, *, k_pairs: int = 5, n_train: int = 8000, epochs: int = 40,
                         lr: float = 2e-3, batch: int = 128, seed: int = 42) -> float:
        """在符号化绑定任务上训练工作空间（学到内容寻址路由，泛化到未见绑定）。返回末步训练准确率。"""
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        pk = np.zeros((n_train, k_pairs), dtype=np.int64)
        pv = np.zeros((n_train, k_pairs), dtype=np.int64)
        qk = np.zeros((n_train,), dtype=np.int64)
        y = np.zeros((n_train,), dtype=np.int64)
        for i in range(n_train):
            keys = rng.choice(self.n_keys, size=k_pairs, replace=False)
            vals = rng.choice(self.n_vals, size=k_pairs, replace=False)
            pk[i], pv[i] = keys, vals
            qi = int(rng.integers(k_pairs))
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
                       "n_slots": self.n_slots, "iters": self.iters, "ask_threshold": self.ask_threshold},
        }, path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "CAPCWWorkingMemory":
        ckpt = torch.load(path, map_location=device or "cpu")
        cfg = ckpt["config"]
        wm = cls(n_keys=cfg["n_keys"], n_vals=cfg["n_vals"], d=cfg["d"], n_slots=cfg["n_slots"],
                 iters=cfg["iters"], ask_threshold=cfg["ask_threshold"], device=device)
        wm.net.load_state_dict(ckpt["state_dict"])
        wm.net.eval()
        return wm
