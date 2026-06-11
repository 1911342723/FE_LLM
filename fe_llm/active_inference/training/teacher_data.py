"""Teacher-data utilities for active inference policy samples."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.config import get_teacher_config
from fe_llm.energy_lm.data.corpus import load_dialogues

from fe_llm.active_inference.observation import Observation

DATA_DIR = os.path.join("data", "active_inference")
POLICY_TEACHER_PATH = os.path.join(DATA_DIR, "policy_teacher.jsonl")
ACTION_TYPES = {"answer", "ask_clarification", "retrieve", "refuse", "update_memory"}
NON_ANSWER_ACTION_TYPES = ACTION_TYPES - {"answer"}

HARD_NEGATIVE_ANSWER_SAMPLES = [
    ("我喜欢这个故事的开头", "这个开头确实有吸引力，可以继续分析它的表达效果。"),
    ("我喜欢这段音乐的节奏", "这个节奏听起来比较鲜明，可以从速度和重音来分析。"),
    ("我不喜欢这个标题", "可以换成更准确、更有吸引力的标题。"),
    ("我喜欢这个方案的方向", "这个方向可以继续细化目标、约束和执行步骤。"),
    ("我不喜欢这句话的语气", "可以把语气改得更自然、更克制。"),
    ("今天我想写一首诗", "可以，从主题、意象和语气开始构思。"),
    ("今天我们聊聊学习计划", "可以，先确定目标、时间和复盘方式。"),
    ("今天适合整理项目思路", "适合，可以先列出目标、风险和下一步动作。"),
    ("现在我有点累", "可以先休息一下，再处理最重要的一件事。"),
    ("现在我想讨论架构", "可以，我们可以从模块边界和数据流开始。"),
    ("最新这一版论文更清楚了", "是的，现在主线更集中，可以继续打磨实验部分。"),
    ("这个价格有点高", "可以比较价值、预算和替代方案后再决定。"),
    ("记忆力是什么", "记忆力是保存、提取和使用信息的能力。"),
    ("你记住了吗", "我会根据当前上下文继续跟进，但这句话本身不需要写入长期记忆。"),
    ("你还记得刚才的问题吗", "记得，我们刚才在讨论主动推理和可溯源生成。"),
    ("我喜欢简洁的代码风格", "简洁的代码风格通常更容易维护和审查。"),
    ("我喜欢严谨的论证", "严谨论证需要清楚的定义、证据和反例边界。"),
    ("我不喜欢太复杂的图", "图可以减少装饰，突出变量关系和机制结构。"),
    ("以后这个方向可能很重要", "这个方向确实值得继续投入，但这句话不是记忆指令。"),
    ("我的名字出现在论文里合适吗", "是否合适取决于署名贡献、项目规范和公开意愿。"),
]


def _existing_prompts(path: str) -> set[str]:
    prompts: set[str] = set()
    if not os.path.exists(path):
        return prompts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                prompt = json.loads(line).get("prompt")
            except json.JSONDecodeError:
                continue
            if isinstance(prompt, str):
                prompts.add(prompt)
    return prompts


def convert_dialogues_to_answer_samples(limit: int = 0, path: str = POLICY_TEACHER_PATH) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = _existing_prompts(path)
    dialogues = load_dialogues()
    if limit > 0:
        dialogues = dialogues[:limit]
    count = 0
    with open(path, "a", encoding="utf-8") as f:
        for prompt, response in dialogues:
            if prompt in existing:
                continue
            item = {
                "prompt": prompt,
                "response": response,
                "action_type": "answer",
                "surprise_components": {
                    "semantic_error": 0.1,
                    "intent_error": 0.1,
                    "consistency_error": 0.0,
                    "uncertainty_error": 0.1,
                    "safety_error": 0.0,
                },
                "rationale": "Routine dialogue with low uncertainty and low risk.",
                "memory_update": False,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            existing.add(prompt)
            count += 1
    return count


def add_hard_negative_answer_samples(path: str = POLICY_TEACHER_PATH) -> int:
    """Add ANSWER samples that contain memory/external-looking cues but should not trigger actions."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = _existing_prompts(path)
    written = 0
    with open(path, "a", encoding="utf-8") as f:
        for prompt, response in HARD_NEGATIVE_ANSWER_SAMPLES:
            if prompt in existing:
                continue
            item = {
                "prompt": prompt,
                "response": response,
                "action_type": "answer",
                "surprise_components": {
                    "semantic_error": 0.15,
                    "intent_error": 0.1,
                    "consistency_error": 0.0,
                    "uncertainty_error": 0.1,
                    "safety_error": 0.0,
                },
                "rationale": "Hard negative: contains memory-like or external-looking words, but the correct action is a direct answer.",
                "memory_update": False,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            existing.add(prompt)
            written += 1
    return written


def normalize_existing_samples(path: str = POLICY_TEACHER_PATH, dry_run: bool = False) -> dict[str, object]:
    """规范化默认 ANSWER 转换带来的高置信外部事实标签冲突。"""

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    rows: list[dict] = []
    changed: list[tuple[str, str, str, str]] = []
    action_counts_before: Counter[str] = Counter()
    action_counts_after: Counter[str] = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            before = str(item.get("action_type", ""))
            action_counts_before[before] += 1
            prompt = str(item.get("prompt", ""))
            features = Observation.from_text(prompt).features
            # 只修正明确需要外部/实时信息的 ANSWER；记忆和安全类标签暂不自动改写。
            if before == "answer" and features.get("needs_external_info"):
                item = dict(item)
                item["action_type"] = "retrieve"
                comps = dict(item.get("surprise_components") or {})
                comps["semantic_error"] = max(float(comps.get("semantic_error", 0.1)), 0.1)
                comps["intent_error"] = max(float(comps.get("intent_error", 0.1)), 0.1)
                comps["consistency_error"] = float(comps.get("consistency_error", 0.0))
                comps["uncertainty_error"] = max(float(comps.get("uncertainty_error", 0.0)), 0.75)
                comps["safety_error"] = float(comps.get("safety_error", 0.0))
                item["surprise_components"] = comps
                item["memory_update"] = False
                item["rationale"] = (
                    "Normalized from routine ANSWER: prompt requires external or real-time information; "
                    + str(item.get("rationale", "")).strip()
                ).strip()
                changed.append((prompt, before, "retrieve", str(features.get("external_info_kind", "none"))))
            action_counts_after[str(item.get("action_type", ""))] += 1
            rows.append(item)

    if changed and not dry_run:
        backup_path = path + ".bak"
        shutil.copyfile(path, backup_path)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for item in rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)

    return {
        "path": path,
        "dry_run": dry_run,
        "total_rows": len(rows),
        "changed": len(changed),
        "action_counts_before": dict(action_counts_before),
        "action_counts_after": dict(action_counts_after),
        "examples": [
            {"prompt": prompt, "from": before, "to": after, "external_info_kind": kind}
            for prompt, before, after, kind in changed[:20]
        ],
    }


def generate_teacher_samples(
    total: int = 2000,
    batch_size: int = 20,
    path: str = POLICY_TEACHER_PATH,
    concurrency: int = 16,
    retries: int = 3,
    timeout: float = 90.0,
    include_answer: bool = False,
) -> int:
    """Generate policy samples concurrently and append valid, deduplicated rows."""

    cfg = get_teacher_config()
    if not cfg.api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured.")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = _existing_prompts(path)
    written = 0
    submitted = 0
    completed = 0
    failed = 0
    action_counts: Counter[str] = Counter()
    started = time.time()
    requests_needed = math.ceil(total / max(1, batch_size))
    max_requests = max(requests_needed * 6, concurrency)
    concurrency = max(1, concurrency)

    print(
        f"[teacher-data] start generate target_add={total} batch_size={batch_size} "
        f"concurrency={concurrency} include_answer={include_answer} existing={len(existing)}",
        flush=True,
    )

    with open(path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = set()

            def desired_inflight() -> int:
                remaining = max(0, total - written)
                remaining_batches = math.ceil(remaining / max(1, batch_size))
                return max(0, min(concurrency, remaining_batches))

            def submit_next() -> None:
                nonlocal submitted
                if submitted >= max_requests or written >= total:
                    return
                n = min(batch_size, max(1, total - written))
                submitted += 1
                futures.add(
                    pool.submit(
                        _generate_batch,
                        cfg.api_key,
                        cfg.base_url,
                        cfg.model,
                        n,
                        submitted,
                        include_answer,
                        retries,
                        timeout,
                    )
                )

            for _ in range(desired_inflight()):
                submit_next()

            while futures and written < total:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    completed += 1
                    try:
                        items, attempts = fut.result()
                    except Exception as exc:
                        failed += 1
                        print(f"[teacher-data] batch failed: {exc}", flush=True)
                        items, attempts = [], retries

                    added = 0
                    for item in items:
                        if not _valid_item(item, include_answer=include_answer):
                            continue
                        if item["prompt"] in existing:
                            continue
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        existing.add(item["prompt"])
                        action_counts[item["action_type"]] += 1
                        written += 1
                        added += 1
                        if written >= total:
                            break
                    f.flush()
                    elapsed = max(1e-6, time.time() - started)
                    print(
                        f"[teacher-data] progress {written}/{total} "
                        f"added={added} parsed={len(items)} attempts={attempts} "
                        f"completed={completed} submitted={submitted} inflight={len(futures)} "
                        f"failed={failed} speed={written / elapsed:.2f}/s "
                        f"actions={dict(action_counts)}",
                        flush=True,
                    )

                    while len(futures) < desired_inflight() and written < total and submitted < max_requests:
                        submit_next()

            for pending in futures:
                pending.cancel()

    if written < total:
        print(
            f"[teacher-data] warning: target not reached, wrote={written}/{total}; "
            f"increase --retries or reduce duplicate rate.",
            flush=True,
        )
    return written


def _generate_batch(
    api_key: str,
    base_url: str,
    model: str,
    n: int,
    batch_id: int,
    include_answer: bool,
    retries: int,
    timeout: float,
) -> tuple[list[dict], int]:
    from openai import OpenAI

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
            prompt = _teacher_prompt(n, batch_id=batch_id, include_answer=include_answer)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate strict JSON arrays for Chinese active-inference policy data.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.85,
            )
            return _parse_items(resp.choices[0].message.content), attempt
        except Exception as exc:
            last_exc = exc
            time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError(f"request failed after {retries} attempts: {last_exc}")


def _teacher_prompt(n: int, batch_id: int = 0, include_answer: bool = False) -> str:
    actions = (
        "answer|ask_clarification|retrieve|refuse|update_memory"
        if include_answer
        else "ask_clarification|retrieve|refuse|update_memory"
    )
    answer_rule = (
        "You may include answer samples when the prompt is fully specified."
        if include_answer
        else "Do not output action_type=answer; local dialogue data already covers answer samples."
    )
    return f"""
Generate {n} Chinese policy-training samples for an active-inference language prototype.
Return only a JSON array. Each object must follow this schema:
{{
  "prompt": "...",
  "response": "...",
  "action_type": "{actions}",
  "surprise_components": {{
    "semantic_error": 0.0,
    "intent_error": 0.0,
    "consistency_error": 0.0,
    "uncertainty_error": 0.0,
    "safety_error": 0.0
  }},
  "rationale": "...",
  "memory_update": false
}}
Batch id: {batch_id}.
{answer_rule}
Cover these cases evenly:
- underspecified requests -> ask_clarification
- logical/time contradictions -> ask_clarification
- external real-time facts -> retrieve
- unsafe or abusive requests -> refuse
- stable user preferences or identity facts -> update_memory
Scores must be floats between 0 and 1.
Prompts must be diverse, natural Chinese, and should not be routine greetings.
"""


def _parse_items(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    if "[" in text and "]" in text:
        text = text[text.find("[") : text.rfind("]") + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        out = []
        for line in text.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                out.append(item)
        return out
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def _valid_item(item: dict, include_answer: bool = True) -> bool:
    actions = ACTION_TYPES if include_answer else NON_ANSWER_ACTION_TYPES
    required = {"prompt", "response", "action_type", "surprise_components", "rationale", "memory_update"}
    if not (isinstance(item, dict) and required.issubset(item) and item["action_type"] in actions):
        return False
    comps = item.get("surprise_components")
    if not isinstance(comps, dict):
        return False
    for key in ("semantic_error", "intent_error", "consistency_error", "uncertainty_error", "safety_error"):
        try:
            value = float(comps[key])
        except (KeyError, TypeError, ValueError):
            return False
        comps[key] = max(0.0, min(1.0, value))
    item["prompt"] = str(item["prompt"]).strip()
    item["response"] = str(item["response"]).strip()
    item["rationale"] = str(item["rationale"]).strip()
    item["memory_update"] = bool(item["memory_update"])
    return bool(item["prompt"] and item["response"] and item["rationale"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert-dialogues", action="store_true")
    ap.add_argument("--add-hard-negatives", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 means all local dialogue samples.")
    ap.add_argument("--generate", type=int, default=0)
    ap.add_argument("--normalize-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--include-answer", action="store_true")
    ap.add_argument("--path", default=POLICY_TEACHER_PATH)
    args = ap.parse_args()

    total = 0
    if args.convert_dialogues:
        total += convert_dialogues_to_answer_samples(limit=args.limit, path=args.path)
    if args.add_hard_negatives:
        total += add_hard_negative_answer_samples(path=args.path)
    if args.normalize_existing:
        result = normalize_existing_samples(path=args.path, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.generate > 0:
        total += generate_teacher_samples(
            total=args.generate,
            batch_size=args.batch_size,
            path=args.path,
            concurrency=args.concurrency,
            retries=args.retries,
            timeout=args.timeout,
            include_answer=args.include_answer,
        )
    print(f"[teacher-data] wrote {total} samples to {args.path}", flush=True)


if __name__ == "__main__":
    main()
