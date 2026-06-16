# -*- coding: utf-8 -*-
"""统一任务 taxonomy —— controller 规则意图 与 教师 9 领域 的单一真相源。

背景（见 `docs/FE-LLM预训练底座N2执行方案.md` / `经验.md`）：
- controller 侧（`slot_intent_nlu.INTENT_REQUIRED_SLOTS`）历史上只有 3 个简化意图：
  `booking→[route]`、`hotel→[city,date]`、`reminder→[time]`。
- 教师任务语料（`teacher_corpus_gen.DOMAINS`）有 9 个领域、槽位更完整：
  `flight/train→[route,date]`、`hotel→[city,date]`、`restaurant→[people,time]` …
- 两套口径长期不一致（最典型：controller 的 `booking` 只要 route，而教师 `flight/train`
  需 route+date）。本模块把「领域 → 必需槽位」收敛成一份 canonical 真相，并显式记录
  legacy controller 意图 → canonical 领域 的映射与「已知简化」差异。

纪律（加法式、行为保持）：
- 本模块是**单一真相源**，不改变 controller 既有运行期行为。
- `LEGACY_REQUIRED_SLOTS` 的值与历史完全一致，由 `slot_intent_nlu` 重导出、并经
  `tests/test_taxonomy.py` 锁定，保证重构不漂。
- 是否把 controller 的 `booking` 升级为 `route+date`（与教师对齐）是一个**行为决策**，
  不在本模块擅自改动；这里只把差异显式化，留待显式拍板。
"""

from __future__ import annotations

# ── canonical：以教师 9 领域为超集，槽位以教师语料为准 ───────────────────────────
CANONICAL_DOMAINS: dict[str, list[str]] = {
    "flight": ["route", "date"],
    "train": ["route", "date"],
    "hotel": ["city", "date"],
    "restaurant": ["people", "time"],
    "appointment": ["dept", "date"],
    "delivery": ["item", "addr"],
    "food": ["dish", "addr"],
    "topup": ["phone", "amount"],
    "repair": ["item", "time"],
}

# ── legacy：controller 历史使用的简化意图 → 必需槽位（值保持不变，测试锁定）────────
LEGACY_REQUIRED_SLOTS: dict[str, list[str]] = {
    "none": [],
    "booking": ["route"],
    "hotel": ["city", "date"],
    "reminder": ["time"],
}

# legacy controller 意图 → canonical 领域（用于把两套口径对齐到同一命名空间）。
LEGACY_INTENT_TO_DOMAIN: dict[str, str | None] = {
    "none": None,
    "booking": "flight",   # 简化别名：controller booking 仅取 route（见 LEGACY_SIMPLIFICATIONS）
    "hotel": "hotel",      # 完全一致
    "reminder": "repair",  # 近似：reminder 仅取 time，是 repair=[item,time] 的 time 子集
}

# canonical 与 legacy 的「已知简化/差异」——仅用于文档与后续行为决策，不参与运行期。
LEGACY_SIMPLIFICATIONS: dict[str, dict] = {
    "booking": {
        "canonical_domain": "flight",
        "canonical_slots": ["route", "date"],
        "legacy_slots": ["route"],
        "note": "controller booking 省略 date；教师 flight/train 需 route+date。升级=行为变更，待定。",
    },
    "reminder": {
        "canonical_domain": "repair",
        "canonical_slots": ["item", "time"],
        "legacy_slots": ["time"],
        "note": "reminder 只取 time 子集，非完整 repair（无 item）。",
    },
}

# 全部 canonical 槽位词表。
SLOT_VOCAB: list[str] = sorted({slot for slots in CANONICAL_DOMAINS.values() for slot in slots})

# controller 规则抽取层（`observation.extract_prompt_features`）目前能从文本检测的槽位。
RULE_EXTRACTABLE_SLOTS: list[str] = ["city", "date", "route", "time"]
# canonical 里规则层尚不能抽取的槽位（开放实体/数值，需学习式或 gazetteer）。
RULE_GAP_SLOTS: list[str] = sorted(set(SLOT_VOCAB) - set(RULE_EXTRACTABLE_SLOTS))


def canonical_domains() -> list[str]:
    """canonical 领域名（稳定排序）。"""
    return sorted(CANONICAL_DOMAINS)


def required_slots(domain: str) -> list[str]:
    """canonical 领域的必需槽位。未知领域抛 KeyError（调用方应先校验）。"""
    return list(CANONICAL_DOMAINS[domain])


def legacy_required_slots(intent: str) -> list[str]:
    """legacy controller 意图的必需槽位（与历史一致）。"""
    return list(LEGACY_REQUIRED_SLOTS[intent])


def legacy_to_canonical(intent: str) -> str | None:
    """legacy controller 意图 → canonical 领域；none → None。"""
    return LEGACY_INTENT_TO_DOMAIN.get(intent)


def is_known_domain(domain: str) -> bool:
    return domain in CANONICAL_DOMAINS
