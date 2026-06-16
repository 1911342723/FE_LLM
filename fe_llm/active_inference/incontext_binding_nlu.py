# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/incontext_binding_nlu.py —— in-context 绑定 NLU
=====================================================================
把活文本对话里的"现场关联"抽成结构化事件，喂给 CAPCW 内容寻址工作记忆
（见 `capcw_memory.py`、`docs/FE-LLM核心引擎构想.md` 第 19 节）：

- **bind**（陈述一个关联）：记住X是Y / X对应Y / X设为Y / X等于Y / X的密码是Y …… → (key=X, value=Y)
- **query**（查询一个关联）：X是多少 / X是什么 / X对应什么 / X等于几 …… → key=X
- **none**：其余（寒暄/任务/闲聊）——**不触发**，避免劫持既有对话。

设计纪律（仿学习式 NLU 的"窄触发"教训，见 经验.md）：**高精度优先**——只在带明确标记词
（记住/对应/设为/等于/的{密码|编号|…}是 + 查询词 多少/什么/几/啥）时才判定，**裸"X是Y"不触发**
（中文"今天是周一""我有点累"满天飞，会误伤）。查询模式先于绑定匹配，消解"X的密码是多少"这类歧义。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 可作为"属性键"的词（"X的<属性>是Y" / "X的<属性>是多少"）。聚焦高频、低歧义的属性。
_ATTR = r"值|密码|编号|号码|账号|账户|地址|名字|姓名|生日|电话|手机号|座位|房间|房间号|工号"

# 查询模式（先匹配；命中即 query）。group(1)=被查询的 key。
_QUERY_PATTERNS = [
    re.compile(r"^(.+?)是多少[？?]?$"),
    re.compile(r"^(.+?)是什么[？?]?$"),
    re.compile(r"^(.+?)是几[？?]?$"),
    re.compile(r"^(.+?)等于(?:几|多少|什么)[？?]?$"),
    re.compile(r"^(.+?)对应(?:什么|啥|多少)[？?]?$"),
    re.compile(rf"^(.+?的(?:{_ATTR}))是多少[？?]?$"),
]

# 绑定模式（查询不命中后匹配）。group(1)=key, group(2)=value。
_BIND_PATTERNS = [
    re.compile(r"^记住[，,：: ]*(.+?)(?:是|为|对应|等于|=)(.+)$"),
    re.compile(r"^(.+?)对应(.+)$"),
    re.compile(r"^(.+?)设(?:为|成)(.+)$"),
    re.compile(r"^(.+?)等于(.+)$"),
    re.compile(rf"^(.+?的(?:{_ATTR}))是(.+)$"),
]

# 末尾语气/标点清理。
_STRIP = " 　\t\r\n，,。.！!？?；;：:"


@dataclass
class BindingEvent:
    """一次解析结果：kind ∈ {bind, query, none}。"""

    kind: str
    key: str | None = None
    value: str | None = None


def _clean(s: str | None) -> str:
    return (s or "").strip(_STRIP).strip()


class InContextBindingNLU:
    """高精度中文 in-context 绑定/查询解析器（无学习权重，规则即接口）。"""

    def parse(self, text: str) -> BindingEvent:
        t = _clean(text)
        if not t:
            return BindingEvent(kind="none")
        # 先查询（消解"X的密码是多少"这类与绑定的歧义）。
        for pat in _QUERY_PATTERNS:
            m = pat.match(t)
            if m:
                key = _clean(m.group(1))
                if key:
                    return BindingEvent(kind="query", key=key)
        # 再绑定。
        for pat in _BIND_PATTERNS:
            m = pat.match(t)
            if m:
                key, value = _clean(m.group(1)), _clean(m.group(2))
                if key and value:
                    return BindingEvent(kind="bind", key=key, value=value)
        return BindingEvent(kind="none")


# ---- 多跳（复合所有格）绑定 NLU：把"对内多步推理"的活文本接进 CAPCW 链式工作记忆 ----
# 复合所有格查询「X的R1的R2…的Rn是多少/是谁/是什么」（n≥2）→ base=X、rels=[R1..Rn]，逐跳链式取回。
# 高精度结构匹配：要求 ≥2 个"的<关系>"段 + 明确查询词收尾；rel 段不含 的/是/?（避免吞掉查询词）。
_MULTIHOP_QUERY = re.compile(r"^(.+?)((?:的[^的是？?]+){2,})是(?:多少|谁|什么|啥|几)[？?]?$")

# 开放关系绑定「X的R是Y」（免"记住"标记，单跳关系事实）：key=X的R（**恰一个 的**=原子关系边），value=Y。
# 高精度窄触发（仿 B2e/学习式 NLU 教训，避免劫持寒暄/形容词谓语）：
#   ① key 必须是"X的R"（恰 1 个 的，2~12 字）——把"今天是周一"(无 的) 这类 copula 排除；
#   ② value 须"实体化"：1~10 字、不含 的/是/标点、不以语气词结尾——把"…是好的/对的/错的/不错的"等形容词谓语排除；
#   ③ 查询词收尾的（…是多少/谁/什么）由查询模式先行截获，不会落到这里。
_OPEN_REL_BIND = re.compile(r"^([^是，。！？、\s]+的[^是的，。！？、\s]+)是([^是的，。！？、\s]{1,10})$")
_VALUE_TAIL_PARTICLES = ("了", "吗", "呢", "啊", "吧", "嘛", "哦", "呀", "啦", "诶")


@dataclass
class MultiHopEvent:
    """多跳解析结果：kind ∈ {bind, query, none}。

    - bind：key/value（复用单跳绑定语义，含"记住X的R是Y"标记式）。
    - query：base + rels（关系链）。rels 为空=原子直查；len≥1=逐跳链式（"cur的r"逐跳取回）。
    """

    kind: str
    key: str | None = None
    value: str | None = None
    base: str | None = None
    rels: list[str] | None = None


class MultiHopBindingNLU:
    """复合所有格多跳 NLU：先识别 ≥2 跳查询，其余复用单跳 NLU（绑定 + 单跳/原子查询统一成 base+rels）。"""

    def __init__(self) -> None:
        self._base = InContextBindingNLU()

    def parse(self, text: str) -> MultiHopEvent:
        t = _clean(text)
        if not t:
            return MultiHopEvent(kind="none")
        # 先匹配 ≥2 跳复合所有格查询（X的R1的R2…是多少）。
        m = _MULTIHOP_QUERY.match(t)
        if m:
            base = _clean(m.group(1))
            rels = [r for r in m.group(2).split("的") if r]
            if base and len(rels) >= 2:
                return MultiHopEvent(kind="query", base=base, rels=rels)
        # 其余复用单跳 NLU：绑定原样；单跳/原子查询拆成 base+rels（统一由 decide_path_str 处理）。
        e = self._base.parse(t)
        if e.kind == "bind":
            return MultiHopEvent(kind="bind", key=e.key, value=e.value)
        if e.kind == "query" and e.key:
            head, sep, tail = e.key.partition("的")
            if sep and tail and "的" not in tail:           # "X的R" → base=X, rels=[R]（单跳关系查询）
                return MultiHopEvent(kind="query", base=head, rels=[tail])
            return MultiHopEvent(kind="query", base=e.key, rels=[])   # 原子键直查（无关系链）
        # 单跳 NLU 未命中 → 尝试开放关系绑定「X的R是Y」（免"记住"标记，高精度窄触发）。
        rel = self._open_relation_bind(t)
        if rel is not None:
            return rel
        return MultiHopEvent(kind="none")

    def _open_relation_bind(self, t: str) -> "MultiHopEvent | None":
        """高精度开放关系绑定：key=X的R（恰 1 个 的）、value=实体化 Y（非形容词谓语/语气词）。否则 None。"""
        m = _OPEN_REL_BIND.match(t)
        if not m:
            return None
        key, value = _clean(m.group(1)), _clean(m.group(2))
        if not key or not value:
            return None
        if key.count("的") != 1:                              # 恰一个 的=原子关系边；多/零 的 不在此触发
            return None
        if value.endswith(_VALUE_TAIL_PARTICLES):             # 形容词谓语/语气收尾（…是好的已被 的 排除；这里挡 …了/吗）
            return None
        return MultiHopEvent(kind="bind", key=key, value=value)
