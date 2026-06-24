# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/growth.py —— 后天成长（synapse-only）的一等能力模块
=====================================================================
把"冻结 backbone、只学可学突触 S 实现后天成长"从散落的 eval 脚本，提升为**原型可复用的一等能力**：
demo / 控制器 / 评测脚本都从这里调，而不是各写一份。

两种成长模式（对应辨识点 #2 的不同用法）：
  - in-place（共享）：在当前突触上持续学新经验——会与旧经验在同一张共享矩阵上相互干扰（会遗忘）。
  - isolated（加容量+隔离·穷则变）：每个技能从底座起独立学一块、**冻结快照**存为 adapter；
    评估某技能用它自己那块冻结快照 → 学后续技能**不可能改它**（数学保证不遗忘），参数随技能增长。

机制要点（已在 evaluation/code_growth_*.py 实证）：
  - 只训 `block.synapse`（softplus 调制内容路由的"突触基底"），其余参数冻结。
  - 用**经验回放**（在已见样本小批上更新）稳定在线学习——naive 单样本更新会发散。
  - 例子用 (prefix, completion) 对，损失只算 completion 区（要模型学会的那段）。

诚实边界：synapse-only 表达力有限（学新输出偏弱、靠路由）；in-place 抗遗忘弱；isolated 不遗忘但
无前向迁移、参数线性增长（可低秩压缩）。详见 docs/reports/per_code_growth*.md。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from fe_llm.energy_lm.training.code_train import _logits, generate as _generate

Example = tuple[str, str]   # (prefix, completion)


class GrowthLearner:
    """SeqEnergyNet 的后天成长器：冻结 backbone、只学可学突触 S。"""

    def __init__(self, net, tok, device: str = "cpu", lr: float = 0.02, amp_dtype=None):
        self.net = net
        self.tok = tok
        self.device = device
        self.lr = lr
        self.amp_dtype = amp_dtype
        for p in net.parameters():
            p.requires_grad_(False)
        self.syn_blocks = [b for b in net.blocks if getattr(b, "use_synapse", False)]
        if not self.syn_blocks:
            raise ValueError("该模型无可学突触（use_synapse=False），无法做 synapse-only 成长。")
        self.base_syn = [b.synapse.detach().clone() for b in self.syn_blocks]
        self.adapters: dict[str, list[torch.Tensor]] = {}     # 技能名 -> 突触快照（isolated）
        self.syn_per_block = int(sum(b.synapse.numel() for b in self.syn_blocks))

    # ------------------------------------------------------------------ 突触状态
    def _set_syn(self, snap: list[torch.Tensor]) -> None:
        for b, s in zip(self.syn_blocks, snap):
            b.synapse.data = s.clone()

    def reset_to_base(self) -> None:
        self._set_syn(self.base_syn)

    def snapshot(self) -> list[torch.Tensor]:
        return [b.synapse.detach().clone() for b in self.syn_blocks]

    def activate(self, name: str) -> None:
        """切到某个已学技能的冻结突触快照。"""
        self._set_syn(self.adapters[name])

    def synapse_delta(self, layer: int = -1) -> np.ndarray:
        """当前突触相对底座的变化量 softplus(S)-softplus(S0)（给可视化）。"""
        cur = F.softplus(self.syn_blocks[layer].synapse.detach())
        base = F.softplus(self.base_syn[layer])
        return (cur - base).cpu().numpy()

    def total_params(self) -> int:
        """部署态总突触参数：底座 + 每技能一块（isolated 下随技能增长）。"""
        return self.syn_per_block * (1 + len(self.adapters))

    # ------------------------------------------------------------------ 损失 / 生成
    def example_loss(self, prefix: str, completion: str) -> torch.Tensor:
        ids = self.tok.encode(prefix + completion)
        p = len(self.tok.encode(prefix))
        seq = torch.tensor([ids], device=self.device)
        logits = _logits(self.net, seq)[0].float()
        tgt = torch.tensor(ids[p:], device=self.device)
        return F.cross_entropy(logits[p - 1: len(ids) - 1], tgt) / np.log(2)

    @torch.no_grad()
    def eval_loss(self, examples: list[Example]) -> float:
        self.net.eval()
        v = float(np.mean([float(self.example_loss(p, c)) for p, c in examples]))
        self.net.train()
        return v

    @torch.no_grad()
    def generate(self, prefix: str, max_new: int = 80, temperature: float = 0.0,
                 top_k: int = 0, top_p: float = 0.0, rep: float = 1.0) -> str:
        self.net.eval()
        out = _generate(self.net, self.tok, prefix, self.net.max_len, self.device,
                        max_new=max_new, temperature=temperature, top_k=top_k, top_p=top_p,
                        repetition_penalty=rep, amp_dtype=self.amp_dtype)
        self.net.train()
        return out

    # ------------------------------------------------------------------ 成长
    def teach(self, examples: list[Example], rounds: int = 1, steps: int = 4, replay: int = 4,
              seed: int = 0, on_round: Callable[[int, float], None] | None = None) -> list[float]:
        """在**当前**突触上学 examples（synapse-only + 经验回放）。返回每轮 eval_loss 历史。

        rounds=每轮喂入一个（轮转）样本并加入回放缓冲；steps=每轮梯度步数；replay=每步采样的已见样本数。
        """
        rng = np.random.default_rng(seed)
        for b in self.syn_blocks:
            b.synapse.requires_grad_(True)
        opt = torch.optim.AdamW([b.synapse for b in self.syn_blocks], lr=self.lr)
        self.net.train()
        hist: list[float] = []
        buf: list[Example] = []
        for r in range(rounds):
            ex = examples[r % len(examples)]
            if ex not in buf:
                buf.append(ex)
            for _ in range(steps):
                k = min(len(buf), replay)
                idx = rng.choice(len(buf), size=k, replace=False) if len(buf) > 1 else [0]
                batch = [buf[i] for i in idx]
                loss = torch.stack([self.example_loss(p, c) for p, c in batch]).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_([b.synapse for b in self.syn_blocks], 1.0)
                opt.step()
            cur = self.eval_loss(examples)
            hist.append(cur)
            if on_round is not None:
                on_round(r, cur)
        for b in self.syn_blocks:
            b.synapse.requires_grad_(False)
        return hist

    def add_skill(self, name: str, examples: list[Example], rounds: int = 40, steps: int = 4,
                  replay: int = 4, seed: int = 0) -> list[float]:
        """加容量+隔离（穷则变）：从底座起独立学一块新突触、冻结存为 adapter[name]。

        学完后旧技能各自的 adapter 不受影响 → 数学上不会被覆盖。
        """
        self.reset_to_base()
        hist = self.teach(examples, rounds=rounds, steps=steps, replay=replay, seed=seed)
        self.adapters[name] = self.snapshot()
        return hist
