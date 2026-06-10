"""Observation parsing and prompt-level feature extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


QUESTION_TERMS = (
    "吗", "怎么", "如何", "为什么", "多少", "哪", "谁", "什么", "?", "？",
)
REQUEST_TERMS = ("帮我", "请", "写一下", "做一下", "弄一下", "生成", "整理")
EXTERNAL_INFO_TERMS = ("天气", "现在", "最新", "新闻", "价格", "股票", "汇率", "几点", "时间")
WEATHER_TERMS = (
    "天气", "气温", "温度", "湿度", "空气", "下雨", "降雨", "降温", "台风", "大风", "沙尘",
    "雾", "霜冻", "阴天", "晴天", "冷吗", "热吗", "带伞", "防晒", "晒衣服", "海滩能去",
)
CURRENT_TIME_TERMS = ("现在几点", "当前时间", "现在的时间", "今天星期几", "今天周几")
MARKET_NEWS_TERMS = ("新闻", "股价", "股票", "汇率", "价格走势", "最新", "热搜", "票房")
PUBLIC_SCHEDULE_TERMS = ("航班", "火车", "高铁", "班车", "电影", "比赛", "演唱会")
PUBLIC_SCHEDULE_CUES = ("几点", "时间", "什么时候", "到站", "起飞", "出发", "发车", "开场", "开赛", "门票")
DATE_TERMS = ("今天", "明天", "昨天")
LOOKUP_REQUEST_TERMS = ("查一下", "查查", "搜索", "检索", "告诉我", "是多少", "怎么样")
SOCIAL_OR_SELF_TERMS = (
    "你今天", "你现在", "你几点", "今天过得", "今天精神", "今天开心", "今天高兴", "今天心情",
    "今天有什么计划", "今天有什么安排", "现在怎么样", "现在出门还来得及", "现在开车吗",
    "现在心情", "最新版本",
)
MEMORY_COMMAND_TERMS = ("请记得", "帮我记住", "你要记得", "请记下", "帮我记一下")
MEMORY_IDENTITY_TERMS = (
    "我的名字叫", "我的名字是", "我叫", "叫我", "我的生日", "我住在", "我的地址", "我公司地址",
    "我的职业", "血型", "我是", "我住", "我家住", "我家地址", "我生日", "我的电子邮箱", "邮箱",
    "今年", "儿子叫", "女朋友叫", "弟弟叫", "我的公司", "公司全称", "身高", "体重",
    "我的家住在", "社保号码", "联系电话", "手机号", "上班时间", "老家在", "我名叫",
)
MEMORY_PREFERENCE_MARKERS = (
    "我喜欢", "我不喜欢", "我更喜欢", "我的偏好", "我偏爱", "偏爱", "我讨厌", "讨厌", "最喜欢",
    "最爱", "特别喜欢", "喜歡", "不喜歡", "其实不喜欢", "不吃", "不喝", "吃不了", "过敏",
    "素食主义者", "只用",
)
MEMORY_SCOPE_TERMS = ("回答", "回复", "称呼", "风格", "语气", "中文", "英文", "简短", "详细", "格式")
MEMORY_HABIT_TERMS = ("我习惯", "我比较习惯", "平时", "每天", "每周", "每个月", "经常", "通常", "一般", "通勤", "每次")
MEMORY_UPDATE_TERMS = ("改主意", "之前说", "上次说", "不对，我之前", "从现在起")
MEMORY_STABLE_FACT_TERMS = ("每个周末", "以后想", "养了", "最近在学", "新房子", "家里有", "家有")
TRANSIENT_CONTEXT_TERMS = ("这个故事", "这个开头", "这篇", "这段", "这次", "当前", "刚才")
UNSAFE_TERMS = (
    "炸药", "爆炸物", "自杀", "杀人", "伤害", "毒品", "诈骗", "盗号", "黑客攻击", "绕过密码",
    "武器制作", "开锁", "勒索",
)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _has_time_conflict(text: str) -> bool:
    patterns = (
        r"昨天.*明天|明天.*昨天",
        r"已经.*还没|还没.*已经",
        r"同时.*不是|不是.*同时",
        r"既.*又不|又不.*既",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _is_current_time_query(text: str) -> bool:
    stripped = text.strip()
    exact_queries = {"几点", "几点？", "几点?", "几点了", "几点了？", "几点了?"}
    return stripped in exact_queries or _contains_any(stripped, CURRENT_TIME_TERMS)


def _has_public_schedule_query(text: str) -> bool:
    return _contains_any(text, PUBLIC_SCHEDULE_TERMS) and _contains_any(text, PUBLIC_SCHEDULE_CUES)


def _is_ambiguous_request(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) <= 8 and _contains_any(stripped, REQUEST_TERMS):
        return True
    vague_patterns = ("帮我写一下", "帮我做一下", "帮我弄一下", "这个怎么弄", "写点东西")
    return stripped in vague_patterns


def _has_memory_request(text: str, has_question: bool) -> bool:
    """Detect stable memory-update intent without treating every preference mention as memory."""

    stripped = text.strip()
    if not stripped:
        return False
    if "记住" in stripped and not stripped.startswith("你记住"):
        return True
    if _contains_any(stripped, MEMORY_COMMAND_TERMS):
        return True
    if has_question and not _contains_any(stripped, ("我比较习惯", "我习惯", "我更喜欢", "我的偏好")):
        return False
    if (
        _contains_any(stripped, TRANSIENT_CONTEXT_TERMS)
        and not _contains_any(stripped, ("以后", "每次", "从现在起"))
        and not _contains_any(stripped, MEMORY_UPDATE_TERMS)
    ):
        return False

    has_identity = _contains_any(stripped, MEMORY_IDENTITY_TERMS)
    has_stable_preference = _contains_any(stripped, MEMORY_PREFERENCE_MARKERS)
    has_habit = _contains_any(stripped, MEMORY_HABIT_TERMS)
    has_update = _contains_any(stripped, MEMORY_UPDATE_TERMS)
    has_stable_fact = _contains_any(stripped, MEMORY_STABLE_FACT_TERMS)
    has_scope_preference = _contains_any(stripped, MEMORY_SCOPE_TERMS) and _contains_any(stripped, MEMORY_PREFERENCE_MARKERS)
    has_future_scope = "以后" in stripped and _contains_any(
        stripped,
        MEMORY_SCOPE_TERMS + MEMORY_PREFERENCE_MARKERS + MEMORY_HABIT_TERMS,
    )
    has_future_preference = "以后" in stripped and _contains_any(stripped, ("都", "不再", "不要", "自动", "推荐"))
    has_schedule_update = _contains_any(stripped, ("闹钟", "会议")) and _contains_any(stripped, ("设置", "改到", "明天"))
    if any(
        (
            has_identity,
            has_stable_preference,
            has_habit,
            has_update,
            has_stable_fact,
            has_scope_preference,
            has_future_scope,
            has_future_preference,
            has_schedule_update,
        )
    ):
        return True
    return False


def _external_info_signal(text: str, has_question: bool, has_conflict: bool) -> tuple[bool, str, float]:
    stripped = text.strip()
    if has_conflict:
        return False, "none", 0.0
    if _contains_any(stripped, SOCIAL_OR_SELF_TERMS):
        return False, "none", 0.0

    # 外部信息只在实时事实、公共事实或显式检索意图足够清楚时触发。
    has_external_term = _contains_any(stripped, EXTERNAL_INFO_TERMS)
    has_lookup_request = _contains_any(stripped, LOOKUP_REQUEST_TERMS)
    has_weather = _contains_any(stripped, WEATHER_TERMS)
    has_current_time = _is_current_time_query(stripped)
    has_market_news = _contains_any(stripped, MARKET_NEWS_TERMS)
    has_public_schedule = _has_public_schedule_query(stripped)
    short_external_endings = ("天气", "新闻", "价格", "股价", "汇率", "几点", "时间")
    short_external_query = len(stripped) <= 8 and stripped.endswith(short_external_endings)
    short_weather_query = len(stripped) <= 10 and stripped.endswith(("天气", "气温", "温度", "湿度", "空气"))
    if has_weather and (has_question or has_lookup_request or short_weather_query):
        return True, "weather", 0.95
    if has_current_time:
        return True, "current_time", 0.95
    if has_market_news and (has_question or has_lookup_request or short_external_query):
        return True, "market_or_news", 0.90
    if has_public_schedule and (has_question or has_lookup_request):
        return True, "public_schedule", 0.75
    if has_lookup_request and has_external_term:
        return True, "lookup", 0.70
    return False, "none", 0.0


@dataclass
class Observation:
    """A user utterance entering the active inference loop."""

    text: str
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls,
        text: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Observation":
        obs = cls(text=text.strip(), session_id=session_id, metadata=dict(metadata or {}))
        obs.metadata.setdefault("features", extract_prompt_features(obs.text))
        return obs

    @property
    def features(self) -> dict[str, Any]:
        return self.metadata.setdefault("features", extract_prompt_features(self.text))

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }


def extract_prompt_features(text: str) -> dict[str, Any]:
    """Return stable, rule-based prompt features used by v1 policy layers."""

    stripped = text.strip()
    has_question = _contains_any(stripped, QUESTION_TERMS)
    has_conflict = _has_time_conflict(stripped)
    needs_external, external_kind, external_confidence = _external_info_signal(stripped, has_question, has_conflict)
    return {
        "length": len(stripped),
        "has_question": has_question,
        "has_request": _contains_any(stripped, REQUEST_TERMS),
        "is_ambiguous_request": _is_ambiguous_request(stripped),
        "needs_external_info": needs_external,
        "external_info_kind": external_kind,
        "external_info_confidence": external_confidence,
        "has_memory_request": _has_memory_request(stripped, has_question),
        "has_safety_risk": _contains_any(stripped, UNSAFE_TERMS),
        "has_consistency_conflict": has_conflict,
        "is_greeting": stripped in {"你好", "您好", "嗨", "在吗", "早上好", "晚上好"},
        "is_thanks": stripped in {"谢谢", "谢谢你", "太感谢了", "麻烦你了"},
    }
