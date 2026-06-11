# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/teacher_gen.py —— 教师生成对话语料（落盘保存，不重复花钱）
==========================================================================
用 DeepSeek 教师批量生成**高质量、短、有重复模式**的中文对话对，作为能量对话
模型的"经验"。生成结果落盘 data/dialogue/dialogues.jsonl，断点续生成、自动去重，
**已生成的绝不重复请求**（省钱）。

设计：
    - 短对话（用户1句 + 回应1句，各≤12字），契合字级定长能量坍缩。
    - 覆盖高频日常场景，且要求"同一意图多种说法"——制造重复模式，让能量地貌有深沟。
    - 落盘 jsonl，每行 {"prompt": "...", "response": "..."}。

运行：
    python -m fe_llm.energy_lm.data.teacher_gen --total 1500
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.config import get_teacher_config

DATA_DIR = os.path.join("data", "dialogue")
PATH = os.path.join(DATA_DIR, "dialogues.jsonl")

TOPICS = [
    "日常问候与寒暄", "询问状态与心情", "感谢与道歉", "告别与祝福",
    "简单同意与拒绝", "询问天气", "吃饭与饮食", "休息与睡觉",
    "工作与学习", "鼓励与安慰", "简单事实问答", "请求帮助",
    "确认与回应", "情感表达", "时间与日程", "出行与回家",
]


def _client():
    from openai import OpenAI
    cfg = get_teacher_config()
    if not cfg.api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY。")
    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url), cfg.model


def _parse(text: str) -> list[dict]:
    text = re.sub(r"^```[a-zA-Z]*", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    out = []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for it in data:
                if isinstance(it, dict) and "prompt" in it and "response" in it:
                    p = str(it["prompt"]).strip()
                    r = str(it["response"]).strip()
                    if 1 <= len(p) <= 24 and 1 <= len(r) <= 24:
                        out.append({"prompt": p, "response": r})
    except json.JSONDecodeError:
        pass
    return out


def _gen_topic(client, model, topic, n):
    prompt = (
        f"请围绕「{topic}」生成 {n} 组简短的中文日常对话。"
        f"每组是'用户说一句话 + 助手回应一句话'，两句都要简短自然、口语化"
        f"（用户≤16字，助手回应≤24字，回应可以是完整的一两句话）。"
        f"**同一种意图请给多种不同说法**（如'你好/您好/嗨/在吗'都对应问候），"
        f"以制造常见模式的重复。严格只输出一个 JSON 数组，元素形如 "
        f'{{"prompt":"用户说的话","response":"助手回应"}}，不要任何额外说明或代码块标记。'
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system",
                   "content": "你是中文日常对话语料生成专家，输出严格 JSON 数组。"},
                  {"role": "user", "content": prompt}],
        temperature=0.9,
    )
    return _parse(resp.choices[0].message.content)


def generate(target_total: int = 1500, per_topic: int = 30) -> int:
    os.makedirs(DATA_DIR, exist_ok=True)
    existing = set()
    if os.path.exists(PATH):
        with open(PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing.add(json.loads(line)["prompt"])
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"[teacher] 已有对话 {len(existing)} 条，目标 {target_total}")
    if len(existing) >= target_total:
        print("[teacher] 已达目标，无需生成（省钱）。")
        return len(existing)

    client, model = _client()
    total = len(existing)
    rnd = 0
    with open(PATH, "a", encoding="utf-8") as f:
        while total < target_total:
            topic = TOPICS[rnd % len(TOPICS)]; rnd += 1
            try:
                pairs = _gen_topic(client, model, topic, per_topic)
            except Exception as exc:
                print(f"[teacher] 「{topic}」失败重试：{exc}")
                time.sleep(2); continue
            added = 0
            for p in pairs:
                if p["prompt"] in existing:
                    continue
                existing.add(p["prompt"])
                f.write(json.dumps(p, ensure_ascii=False) + "\n"); f.flush()
                added += 1
            total += added
            print(f"[teacher] [{topic}] +{added}，累计 {total}/{target_total}")
    print(f"[teacher] 完成，共 {total} 条 → {PATH}")
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=1500)
    ap.add_argument("--per-topic", type=int, default=30)
    args = ap.parse_args()
    generate(args.total, args.per_topic)
