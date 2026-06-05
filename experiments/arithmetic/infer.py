# -*- coding: utf-8 -*-
"""
arithmetic/infer.py —— 加载权重做推理，直观展示 FE-LLM 核心思想
==============================================================
运行：python experiments/arithmetic/infer.py

展示两件事：
    1) 解题 = 能量下降到吸引子：对一道题，在所有候选答案上算能量，
       取能量最低者作为答案（不是 softmax 概率预测）。
    2) 惊奇 = 错误答案的高能量：给定题目，正确答案能量极低(不惊奇)，
       错误答案能量高(惊奇)，偏离越远能量越高——这正是"最小自由能"的体现。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from fe_llm.config import CHECKPOINT_DIR, get_device
from experiments.arithmetic.encoding import AnswerSpace, encode_question
from experiments.arithmetic.model import AnswerCodebook, EnergyNet, SolverNet


class ArithmeticFE:
    """加载固化权重的算术认知体。"""

    def __init__(self, ckpt_path: str | None = None, device: str | None = None):
        self.device = device or get_device()
        path = ckpt_path or os.path.join(CHECKPOINT_DIR, "arithmetic", "arith_fe.pt")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.lo, self.hi = ckpt["lo"], ckpt["hi"]
        self.scale = ckpt["answer_scale"]
        self.space = AnswerSpace(self.lo, self.hi)
        dim = ckpt["embed_dim"]

        self.codebook = AnswerCodebook(self.space.size, dim,
                                       learnable=ckpt["learnable"], lo=self.lo
                                       ).to(self.device)
        self.codebook.load_state_dict(ckpt["codebook"])
        self.energy = EnergyNet(dim).to(self.device)
        self.energy.load_state_dict(ckpt["energy_net"])
        self.solver = SolverNet(dim).to(self.device)
        self.solver.load_state_dict(ckpt["solver_net"])
        self.energy.eval(); self.solver.eval(); self.codebook.eval()

    @torch.no_grad()
    def _q(self, a: int, b: int, op: str) -> torch.Tensor:
        v = encode_question(a, b, op)
        return torch.tensor(v, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def solve_by_energy(self, a: int, b: int, op: str) -> int:
        """能量法：在所有答案吸引子上取能量最低者（滚落到最深的谷）。"""
        energies = self.energy.energy_over_all(self._q(a, b, op), self.codebook)
        idx = int(energies.argmin(dim=1).item())
        return self.space.to_value(idx)

    @torch.no_grad()
    def solve_by_decoder(self, a: int, b: int, op: str) -> int:
        """解码器法：回归归一化标量答案，× scale 后取整滚落到最近整数吸引子。"""
        _, scalar = self.solver(self._q(a, b, op))
        val = round(float(scalar.item()) * self.scale)
        return max(self.lo, min(self.hi, val))

    @torch.no_grad()
    def surprise_of(self, a: int, b: int, op: str, candidate: int) -> float:
        """返回系统对"这道题=某候选答案"的惊奇能量。"""
        if not self.space.contains(candidate):
            return float("inf")
        vec = self.codebook(torch.tensor([self.space.to_index(candidate)],
                                         device=self.device))
        e = self.energy(self._q(a, b, op), vec)
        return float(e.item())


def demo():
    print("\n加载算术认知体（固化权重）...")
    fe = ArithmeticFE()
    print(f"设备：{fe.device}  答案吸引子数：{fe.space.size}\n")

    # —— 演示 1：两种方式解题 ——
    print("=" * 60)
    print("演示一：解题 = 能量下降到吸引子")
    print("=" * 60)
    cases = [(23, 45, "+"), (37, 19, "-"), (8, 7, "+"), (50, 50, "+"), (12, 40, "-")]
    for a, b, op in cases:
        truth = a + b if op == "+" else a - b
        e_ans = fe.solve_by_energy(a, b, op)
        s_ans = fe.solve_by_decoder(a, b, op)
        mark = "✓" if (e_ans == truth and s_ans == truth) else "✗"
        print(f"  {a} {op} {b} = {truth:<4d} | 能量法={e_ans:<4d} 解码器={s_ans:<4d} {mark}")

    # —— 演示 2：惊奇能量曲线 ——
    print("\n" + "=" * 60)
    print("演示二：惊奇 = 错误答案的高能量（23 + 45 = 68）")
    print("=" * 60)
    a, b, op = 23, 45, "+"
    truth = 68
    cands = [truth - 10, truth - 2, truth - 1, truth, truth + 1, truth + 2, truth + 10]
    raw = {c: fe.surprise_of(a, b, op, c) for c in cands}
    base = min(raw.values())  # 以最低能量(正确答案)为基准，看"额外惊奇"
    print(f"  题目：{a} + {b}，正确答案 {truth}")
    print("  候选答案 → 相对惊奇度（正确答案=0，偏离越远越惊奇）：")
    for c in cands:
        delta = raw[c] - base
        bar = "█" * int(delta / 4) if delta > 0 else ""
        flag = "  ← 能量谷底，不惊奇" if c == truth else ""
        print(f"    {c:>4d} : 惊奇+{delta:6.2f} {bar}{flag}")


if __name__ == "__main__":
    demo()
