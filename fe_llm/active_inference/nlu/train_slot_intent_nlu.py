# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/nlu/train_slot_intent_nlu.py —— 训练并保存稳健的学习式意图 NLU
=====================================================================================
为"启动自动加载"准备一个稳健 checkpoint：意图样本（booking/hotel/reminder）+ 充分的
none 样本（涵盖控制器里该直接回答/非槽位的陈述），高置信下才生效，配合感知层"无其它信号"
门控，确保不误改 weather→retrieve / 记忆 / 拒答 / 寒暄 等既有行为。

运行：python -m fe_llm.active_inference.nlu.train_slot_intent_nlu
输出：checkpoints/active_inference/slot_intent_nlu.pt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.active_inference.nlu.slot_intent_nlu import SlotIntentNLU

CKPT = os.path.join("checkpoints", "active_inference", "slot_intent_nlu.pt")

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安"]
TIMES = ["明天8点", "周末", "下午3点", "后天", "晚上9点", "10点", "早上7点"]

BOOKING = [
    "帮我订票", "订一下票", "买票去{c}", "预订车票", "订张机票", "帮我订{c}的票", "买张到{c}的票",
    "想坐高铁去{c}", "来一张到{c}的车", "搞张去{c}的高铁", "怎么购到{c}的车次", "去{c}的火车帮我安排",
    "我要去{c}坐车", "订{c}到{c2}的票",
]
HOTEL = [
    "帮我订酒店", "订间房", "预订旅馆", "订个民宿", "帮我订{c}的酒店", "订{c}的房",
    "在{c}住一晚", "找个{c}的住处", "{c}有合适的客栈吗", "安排个{c}的宾馆", "{c}过夜的地方", "订{c}的宾馆",
]
REMINDER = [
    "提醒我{t}开会", "设个提醒", "定个闹钟{t}", "提醒一下{t}", "加个提醒{t}",
    "{t}叫我起床", "别让我忘了{t}的事", "到{t}喊我一声", "记得{t}通知我", "{t}催我一下",
]
# none：必须涵盖控制器里"该直接回答/非槽位"的陈述，避免学习式误触发追问。
NONE = [
    "今天心情不错", "讲个笑话吧", "我有点累", "解释一下相对论", "最近工作压力好大", "陪我聊聊天",
    "今天我想写一首诗", "最新这一版论文更清楚了", "这个价格有点高", "我喜欢这个故事的开头",
    "你记住了吗", "给我讲讲自由能原理", "讲讲自由能", "你好啊", "随便聊聊", "你在干嘛",
    "早上好呀", "今天过得不错", "你今天开心吗", "今天心情怎么样", "解释一下原理", "说点什么",
    "我想写点东西", "今天天气真舒服", "帮我看看这段话", "这句话什么意思", "我们聊聊吧",
]


def _expand(templates: list[str]) -> list[str]:
    out = []
    for tpl in templates:
        if "{c2}" in tpl:
            for c in CITIES[:4]:
                for c2 in CITIES[4:8]:
                    out.append(tpl.format(c=c, c2=c2))
        elif "{c}" in tpl:
            for c in CITIES:
                out.append(tpl.format(c=c))
        elif "{t}" in tpl:
            for t in TIMES:
                out.append(tpl.format(t=t))
        else:
            out.append(tpl)
    return out


def main() -> int:
    texts, labels = [], []
    for tpls, lab in [(BOOKING, "booking"), (HOTEL, "hotel"), (REMINDER, "reminder"), (NONE, "none")]:
        for t in _expand(tpls):
            texts.append(t)
            labels.append(lab)
    nlu = SlotIntentNLU().fit(texts, labels, epochs=400)
    # 自检：none 样本不应被误判为意图
    wrong = [t for t, l in zip(texts, labels) if l == "none" and nlu.predict(t) != "none"]
    nlu.save(CKPT)
    print(f"[train-nlu] 样本 {len(texts)}（none 误判 {len(wrong)} 条）→ {CKPT}")
    if wrong:
        print(f"[train-nlu] 警告 none 误判：{wrong[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
