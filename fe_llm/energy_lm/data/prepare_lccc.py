# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/prepare_lccc.py —— 从 LCCC 公开数据集构建对话语料（不用教师）
==============================================================================
LCCC（Large-scale Cleaned Chinese Conversation）是清洗过的真实中文口语对话。
原始每行是一段多轮对话的 JSON 数组，词间以空格分隔。

本脚本：
    1. 流式解压（不全量载入内存）。
    2. 把每段多轮对话拆成相邻的 (prompt, response) 单轮对。
    3. 去词间空格、清洗，按长度筛选（适合字级定长窗口的"短一点对话"）。
    4. 去重后落盘到 data/dialogue/dialogues.jsonl（覆盖旧的教师数据）。

运行：
    python -m fe_llm.energy_lm.data.prepare_lccc --max-pairs 60000 --resp-max 24
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

OUT_DIR = os.path.join("data", "dialogue")
OUT_PATH = os.path.join(OUT_DIR, "dialogues.jsonl")

# 只保留以中文/常见标点为主的字符，剔除表情、生僻符号、URL 残留
_KEEP = re.compile(r"[\u4e00-\u9fff，。！？、…~,.!?]")


def _find_lccc() -> str | None:
    pat = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--silver--lccc/snapshots/*/"
        "lccc_base_train.jsonl.gz")
    hits = glob.glob(pat)
    return hits[0] if hits else None


def _clean(s: str) -> str:
    """去掉词间空格，只保留中文与基本标点。"""
    s = s.replace(" ", "")
    s = "".join(ch for ch in s if _KEEP.match(ch))
    return s


def _ok(text: str, lo: int, hi: int) -> bool:
    if not (lo <= len(text) <= hi):
        return False
    # 中文占比要高（剔除纯标点/乱码）
    n_han = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return n_han >= max(1, int(0.6 * len(text)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=60000, help="最多保留多少对")
    ap.add_argument("--prompt-min", type=int, default=2)
    ap.add_argument("--prompt-max", type=int, default=24)
    ap.add_argument("--resp-min", type=int, default=2)
    ap.add_argument("--resp-max", type=int, default=24, help="回应最长字数")
    ap.add_argument("--max-scan", type=int, default=2_000_000,
                    help="最多扫描多少段对话（控制耗时）")
    args = ap.parse_args()

    src = _find_lccc()
    if src is None:
        print("未找到 LCCC 文件，请先下载："
              "python -c \"from huggingface_hub import hf_hub_download; "
              "hf_hub_download('silver/lccc','lccc_base_train.jsonl.gz',"
              "repo_type='dataset')\"")
        return

    print(f"[lccc] 源文件：{src}")
    print(f"[lccc] 筛选：prompt {args.prompt_min}-{args.prompt_max} 字，"
          f"response {args.resp_min}-{args.resp_max} 字")

    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    n_dialogue = 0
    resp_lens: dict[int, int] = {}

    with gzip.open(src, "rt", encoding="utf-8") as f:
        for line in f:
            n_dialogue += 1
            if n_dialogue > args.max_scan or len(pairs) >= args.max_pairs:
                break
            try:
                turns = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(turns, list) or len(turns) < 2:
                continue
            turns = [_clean(t) for t in turns]
            # 相邻轮组成 (prompt, response)
            for i in range(len(turns) - 1):
                p, r = turns[i], turns[i + 1]
                if not _ok(p, args.prompt_min, args.prompt_max):
                    continue
                if not _ok(r, args.resp_min, args.resp_max):
                    continue
                key = (p, r)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)
                resp_lens[len(r)] = resp_lens.get(len(r), 0) + 1
                if len(pairs) >= args.max_pairs:
                    break

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for p, r in pairs:
            f.write(json.dumps({"prompt": p, "response": r},
                               ensure_ascii=False) + "\n")

    # 统计
    chars = set()
    for p, r in pairs:
        chars.update(p); chars.update(r)
    avg_r = sum(len(r) for _, r in pairs) / max(1, len(pairs))
    print(f"[lccc] 扫描对话段：{n_dialogue}")
    print(f"[lccc] 落盘对数：{len(pairs)} → {OUT_PATH}")
    print(f"[lccc] 字表规模：{len(chars)} 字")
    print(f"[lccc] 平均回应长度：{avg_r:.1f} 字")
    # 回应长度分布（每 4 字一桶）
    buckets: dict[int, int] = {}
    for L, c in resp_lens.items():
        buckets[L // 4 * 4] = buckets.get(L // 4 * 4, 0) + c
    print("[lccc] 回应长度分布：", {k: buckets[k] for k in sorted(buckets)})


if __name__ == "__main__":
    main()
