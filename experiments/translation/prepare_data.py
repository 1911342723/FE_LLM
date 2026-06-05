# -*- coding: utf-8 -*-
"""
translation/prepare_data.py —— 从公开数据集准备大规模中英平行语料
==================================================================
比让 DeepSeek 逐句生成快几个数量级。从 HuggingFace 公开数据集下载中英平行语料，
清洗后写成统一的 pairs.jsonl（与 teacher.py 输出格式一致，训练脚本无需改动）。

默认数据集：Helsinki-NLP/opus-100 (en-zh)，约 100 万句对，覆盖面广、质量稳定。

国内网络加速：自动设置 HF_ENDPOINT=https://hf-mirror.com 镜像站。
若已能直连 huggingface.co，可用 --no-mirror 关闭。

清洗规则：
    - 去空、去重
    - 中文侧必须含中文字符，英文侧必须含拉丁字母
    - 控制中英长度比，过滤明显错位的脏对
    - 长度上限过滤超长句
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# 输出路径（与 teacher.py 保持一致）
DATA_DIR = os.path.join("data", "translation")
PAIRS_PATH = os.path.join(DATA_DIR, "pairs.jsonl")

_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[a-zA-Z]")


def _clean_pair(zh: str, en: str, max_chars: int = 200) -> dict | None:
    """清洗单个句对，不合格返回 None。"""
    zh = (zh or "").strip()
    en = (en or "").strip()
    if not zh or not en:
        return None
    # 中文侧需含中文，英文侧需含字母
    if not _CJK.search(zh) or not _LATIN.search(en):
        return None
    # 长度上限
    if len(zh) > max_chars or len(en) > max_chars:
        return None
    # 长度比过滤明显错位（英文字符数 / 中文字符数 落在合理区间）
    ratio = len(en) / max(1, len(zh))
    if ratio < 0.4 or ratio > 8.0:
        return None
    return {"zh": zh, "en": en}


def prepare(dataset: str = "Helsinki-NLP/opus-100",
            config: str = "en-zh",
            limit: int = 200000,
            use_mirror: bool = True,
            append: bool = False) -> int:
    """
    下载并清洗数据集，写入 pairs.jsonl。返回写入的句对数。

    limit : 最多取多少条（清洗后）。opus-100 全量 100 万，按需截取以控制训练时长。
    append: True 则追加到现有 pairs.jsonl（与 DeepSeek 语料合并），False 则覆盖。
    """
    if use_mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        print(f"[prepare] 使用 HF 镜像：{os.environ['HF_ENDPOINT']}")

    from datasets import load_dataset

    print(f"[prepare] 加载数据集 {dataset} [{config}] ...（首次需下载，请稍候）")
    ds = load_dataset(dataset, config, split="train")
    print(f"[prepare] 原始句对：{len(ds)}，目标清洗后取 {limit} 条")

    os.makedirs(DATA_DIR, exist_ok=True)
    mode = "a" if append else "w"

    # 去重集合（若追加，先载入已有 zh）
    seen: set[str] = set()
    if append and os.path.exists(PAIRS_PATH):
        with open(PAIRS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["zh"])
                except (json.JSONDecodeError, KeyError):
                    continue

    written = 0
    with open(PAIRS_PATH, mode, encoding="utf-8") as out:
        for row in ds:
            if written >= limit:
                break
            tr = row.get("translation", row)
            pair = _clean_pair(tr.get("zh"), tr.get("en"))
            if pair is None or pair["zh"] in seen:
                continue
            seen.add(pair["zh"])
            out.write(json.dumps(pair, ensure_ascii=False) + "\n")
            written += 1
            if written % 20000 == 0:
                print(f"[prepare] 已写入 {written} 条 ...")

    print(f"[prepare] 完成，写入 {written} 条 → {PAIRS_PATH}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Helsinki-NLP/opus-100")
    parser.add_argument("--config", default="en-zh")
    parser.add_argument("--limit", type=int, default=200000, help="清洗后最多取多少条")
    parser.add_argument("--no-mirror", action="store_true", help="关闭HF镜像，直连官方")
    parser.add_argument("--append", action="store_true", help="追加到现有语料而非覆盖")
    args = parser.parse_args()

    prepare(args.dataset, args.config, args.limit,
            use_mirror=not args.no_mirror, append=args.append)
