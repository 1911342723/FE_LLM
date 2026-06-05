# -*- coding: utf-8 -*-
"""
translation/teacher.py —— DeepSeek 教师：生成中英平行语料
==========================================================
让 DeepSeek 围绕多个日常主题，批量生成高质量中英句对，作为蒸馏训练数据。

特性：
    - 主题驱动：覆盖问候/购物/旅行/工作/情感等场景，保证句式多样性。
    - 批量产出：一次请求生成多条，降低 API 往返。
    - 断点续生成：已生成的写入 jsonl，重跑时自动跳过，可分多次攒够数据量。
    - 容错解析：DeepSeek 返回的 JSON 偶有格式偏差，做了健壮解析与去重。

输出：data/translation/pairs.jsonl，每行 {"zh": "...", "en": "..."}。
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fe_llm.config import get_teacher_config

# 平行语料输出路径
DATA_DIR = os.path.join("data", "translation")
PAIRS_PATH = os.path.join(DATA_DIR, "pairs.jsonl")

# 生成主题：覆盖日常高频场景，让模型见过多样句式
TOPICS = [
    "日常问候与寒暄", "购物与价格", "餐厅点餐与美食", "旅行与问路",
    "工作与会议", "学习与教育", "天气与季节", "健康与就医",
    "家庭与亲情", "情感与心情", "科技与电脑", "运动与健身",
    "时间与日程安排", "交通与出行", "电话与沟通", "兴趣爱好",
    "节日与庆祝", "金钱与银行", "租房与居住", "网络与社交媒体",
    "动物与宠物", "自然与环境", "新闻与时事", "梦想与计划",
]


def _build_client():
    from openai import OpenAI

    cfg = get_teacher_config()
    if not cfg.api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY。")
    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url), cfg.model


def _parse_pairs(text: str) -> list[dict]:
    """从 DeepSeek 回复中健壮地解析出句对列表。"""
    # 优先尝试直接 JSON 数组
    text = text.strip()
    # 去掉可能的 ```json ``` 包裹
    text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            out = []
            for item in data:
                if isinstance(item, dict) and "zh" in item and "en" in item:
                    out.append({"zh": str(item["zh"]).strip(),
                                "en": str(item["en"]).strip()})
            return out
    except json.JSONDecodeError:
        pass
    return []


def _gen_for_topic(client, model: str, topic: str, n: int) -> list[dict]:
    """让 DeepSeek 为一个主题生成 n 条中英句对。"""
    prompt = (
        f"请围绕「{topic}」这个场景，生成 {n} 条自然、地道、长度不一的中英文平行句子，"
        f"涵盖陈述句、疑问句、祈使句等多种句式，难度从简单到中等。"
        f"严格只输出一个 JSON 数组，每个元素形如 "
        f'{{"zh": "中文句子", "en": "English sentence"}}，不要任何额外说明或代码块标记。'
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "你是专业的中英双语语料生成专家，输出严格的 JSON 数组。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,  # 适当随机以增加句式多样性
    )
    return _parse_pairs(resp.choices[0].message.content)


def generate(target_total: int = 3000, per_topic: int = 25) -> int:
    """
    生成平行语料直到累计达到 target_total 条（断点续生成）。
    返回最终语料总条数。
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # 载入已有语料用于去重与计数
    existing: set[str] = set()
    if os.path.exists(PAIRS_PATH):
        with open(PAIRS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing.add(json.loads(line)["zh"])
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"[teacher] 已有语料 {len(existing)} 条，目标 {target_total} 条。")
    if len(existing) >= target_total:
        print("[teacher] 已达目标，无需生成。")
        return len(existing)

    client, model = _build_client()
    total = len(existing)
    round_idx = 0

    with open(PAIRS_PATH, "a", encoding="utf-8") as f:
        while total < target_total:
            topic = TOPICS[round_idx % len(TOPICS)]
            round_idx += 1
            try:
                pairs = _gen_for_topic(client, model, topic, per_topic)
            except Exception as exc:
                print(f"[teacher] 主题「{topic}」生成失败，重试中：{exc}")
                time.sleep(2)
                continue

            added = 0
            for p in pairs:
                if not p["zh"] or not p["en"] or p["zh"] in existing:
                    continue
                existing.add(p["zh"])
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
                added += 1
            f.flush()
            total += added
            print(f"[teacher] [{topic}] 新增 {added} 条，累计 {total}/{target_total}")

    print(f"[teacher] 完成，语料总数 {total} 条 → {PAIRS_PATH}")
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=3000, help="目标语料总条数")
    parser.add_argument("--per-topic", type=int, default=25, help="每主题每轮条数")
    args = parser.parse_args()
    generate(args.total, args.per_topic)
