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
