# -*- coding: utf-8 -*-
"""
arithmetic/teacher.py —— DeepSeek 教师：生成算术训练数据
=========================================================
两种数据来源：
    1) 本地程序生成（默认，免费、海量、即时）：随机出题并用 Python 算标准答案。
    2) DeepSeek 教师校验（可选）：抽样让 DeepSeek 解题，验证教师管道可用，
       并为将来"不可程序验证的任务"预留同样的蒸馏接口。

设计理念：学生网络不直接学"加法规则"，而是学"教师给出的(题目→答案)映射"。
这正是蒸馏——把教师的能力压缩进我们的小网络。
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class ArithProblem:
    """一道算术题。"""

    a: int
    b: int
    op: str           # '+', '-', '*'
    answer: int

    @property
    def question(self) -> str:
        return f"{self.a} {self.op} {self.b}"


def _compute(a: int, b: int, op: str) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError(op)


def generate_problems(
    n: int,
    max_val: int = 50,
    ops: tuple[str, ...] = ("+", "-"),
    seed: int | None = None,
) -> list[ArithProblem]:
    """
    本地程序生成 n 道题（标准答案由 Python 计算，等价于"绝对正确的教师"）。

    为快速验证，默认只用加减、操作数 0~50，答案范围可控，便于构造离散答案吸引子。
    """
    rng = random.Random(seed)
    problems: list[ArithProblem] = []
    for _ in range(n):
        op = rng.choice(ops)
        a = rng.randint(0, max_val)
        b = rng.randint(0, max_val)
        problems.append(ArithProblem(a, b, op, _compute(a, b, op)))
    return problems


def verify_with_deepseek(problems: list[ArithProblem], sample: int = 5) -> float:
    """
    抽样让 DeepSeek 解题，返回教师与标准答案的一致率。
    用于确认教师管道可用（DeepSeek 真能当算术教师）。失败则抛异常。
    """
    from openai import OpenAI

    from fe_llm.config import get_teacher_config

    cfg = get_teacher_config()
    if not cfg.api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY。")
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    picked = problems[:sample]
    correct = 0
    for p in picked:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system",
                 "content": "你是算术老师，只输出最终整数答案，不要任何解释。"},
                {"role": "user", "content": f"{p.question} = ?"},
            ],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        try:
            got = int("".join(ch for ch in text if ch.isdigit() or ch == "-"))
        except ValueError:
            got = None
        ok = (got == p.answer)
        correct += int(ok)
        print(f"  教师: {p.question} = {text}  (标准={p.answer})  {'✓' if ok else '✗'}")
    return correct / len(picked)
