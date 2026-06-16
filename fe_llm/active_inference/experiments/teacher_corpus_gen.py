# -*- coding: utf-8 -*-
"""
fe_llm/active_inference/experiments/teacher_corpus_gen.py
=========================================================
教师语料生成器：由"教师"（本架构作者）沉淀一份**任务型多轮对话**语料，
上下文强耦合——同一请求/补槽位在不同 belief 下对应不同动作，专门给控制层制造 headroom。

每条 turn 记录：
    {session_id, turn, domain, utterance, known_slots(本轮决策前的 belief), action}

动作标签由"教师规则"确定（多槽位齐全才 ANSWER，否则 ASK；风险 REFUSE；外部事实 RETRIEVE；
稳定偏好 UPDATE_MEMORY；寒暄 ANSWER）。逐行写 jsonl（天然分批、不会一次性大写入）。

运行：python -m fe_llm.active_inference.experiments.teacher_corpus_gen --sessions 1200
输出：data/dialogue/teacher_task_oriented.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fe_llm.active_inference.nlu import taxonomy

OUT_PATH = os.path.join("data", "dialogue", "teacher_task_oriented.jsonl")

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "重庆"]
DATES = ["明天", "后天", "周五", "下周一", "周末", "5号", "3月10号", "下个月", "这周六", "元旦"]
TIMES = ["今晚7点", "中午12点", "明晚6点", "周六中午", "下午3点", "晚上8点半"]
PEOPLE = ["2", "3", "4", "5", "6", "8", "10"]
DEPTS = ["内", "外", "儿", "眼", "口腔", "皮肤", "骨"]
ITEMS = ["文件", "衣服", "水果", "几本书", "电脑", "礼物"]
ADDRS = ["公司", "家里", "北京朝阳", "上海浦东", "学校"]
DISHES = ["黄焖鸡", "麻辣烫", "炒饭", "披萨", "汉堡", "牛肉面", "盖饭"]
AMOUNTS = ["50", "100", "200", "30", "300"]
PHONES = ["138的号", "186那个", "我常用号", "159结尾的"]
REPAIR_ITEMS = ["空调", "热水器", "洗衣机", "马桶", "宽带", "冰箱"]


def _route(rng):
    a, b = rng.sample(CITIES, 2)
    return f"{a}到{b}"


DOMAINS = {
    "flight": {
        "slots": ["route", "date"],
        "request": ["帮我订机票", "我想订张机票", "订一下飞机票", "想买机票", "帮忙订下航班", "要订机票", "订张机票走"],
        "value": {"route": _route, "date": lambda r: r.choice(DATES)},
        "provide": {"route": ["{v}", "从{v}", "{v}的航班", "走{v}"], "date": ["{v}", "就{v}", "{v}出发", "定{v}"]},
    },
    "train": {
        "slots": ["route", "date"],
        "request": ["订张火车票", "帮我买高铁票", "想订火车票", "订一下动车票", "帮忙订火车票", "买张高铁票"],
        "value": {"route": _route, "date": lambda r: r.choice(DATES)},
        "provide": {"route": ["{v}", "从{v}", "{v}的车", "坐{v}"], "date": ["{v}", "就{v}", "{v}走"]},
    },
    "hotel": {
        "slots": ["city", "date"],
        "request": ["订个酒店", "帮我订房", "想订酒店", "订间客房", "帮忙订个宾馆", "订个住的地方"],
        "value": {"city": lambda r: r.choice(CITIES), "date": lambda r: r.choice(DATES)},
        "provide": {"city": ["在{v}", "{v}", "去{v}", "{v}市区"], "date": ["{v}入住", "{v}", "就{v}"]},
    },
    "restaurant": {
        "slots": ["people", "time"],
        "request": ["订个餐厅", "想订位子", "订张餐桌", "帮忙订个饭店", "预订晚餐", "订个包间"],
        "value": {"people": lambda r: r.choice(PEOPLE), "time": lambda r: r.choice(TIMES)},
        "provide": {"people": ["{v}个人", "{v}位", "我们{v}个", "{v}人"], "time": ["{v}", "{v}用餐", "{v}的桌"]},
    },
    "appointment": {
        "slots": ["dept", "date"],
        "request": ["帮我挂号", "想预约医生", "挂个号", "预约门诊", "帮忙挂号", "约个号"],
        "value": {"dept": lambda r: r.choice(DEPTS), "date": lambda r: r.choice(DATES)},
        "provide": {"dept": ["{v}科", "看{v}", "{v}门诊"], "date": ["{v}", "{v}的号", "约{v}"]},
    },
    "delivery": {
        "slots": ["item", "addr"],
        "request": ["帮我寄快递", "想寄个包裹", "寄快递", "帮忙发个件", "寄个东西"],
        "value": {"item": lambda r: r.choice(ITEMS), "addr": lambda r: r.choice(ADDRS)},
        "provide": {"item": ["寄{v}", "是{v}", "{v}"], "addr": ["寄到{v}", "地址{v}", "送到{v}"]},
    },
    "food": {
        "slots": ["dish", "addr"],
        "request": ["点个外卖", "帮我订餐", "想点外卖", "订份午饭", "点份饭", "叫个外卖"],
        "value": {"dish": lambda r: r.choice(DISHES), "addr": lambda r: r.choice(ADDRS)},
        "provide": {"dish": ["来份{v}", "点{v}", "要{v}"], "addr": ["送到{v}", "地址{v}", "{v}"]},
    },
    "topup": {
        "slots": ["phone", "amount"],
        "request": ["帮我充话费", "充个话费", "想充值", "话费充一下", "给手机充值"],
        "value": {"phone": lambda r: r.choice(PHONES), "amount": lambda r: r.choice(AMOUNTS)},
        "provide": {"phone": ["号码{v}", "{v}", "充{v}"], "amount": ["充{v}块", "{v}元", "{v}块钱"]},
    },
    "repair": {
        "slots": ["item", "time"],
        "request": ["报修", "家里要报修", "预约维修", "帮我报个修", "约师傅上门", "报个维修"],
        "value": {"item": lambda r: r.choice(REPAIR_ITEMS), "time": lambda r: r.choice(TIMES)},
        "provide": {"item": ["{v}坏了", "修{v}", "是{v}"], "time": ["{v}上门", "{v}", "约{v}"]},
    },
}

# 让统一 taxonomy 成为「领域→必需槽位」的单一真相源：slots 派生自 canonical，
# 并 fail-fast 校验本文件的 value/provide 覆盖这些槽位（值与历史一致，零行为变更；
# 任何一边漂移会在此立刻报错，而不是悄悄生成不一致语料）。
for _name, _spec in DOMAINS.items():
    _canon = taxonomy.required_slots(_name)
    assert set(_spec["value"]) >= set(_canon), f"{_name} value 缺槽位 {_canon}"
    assert set(_spec["provide"]) >= set(_canon), f"{_name} provide 缺槽位 {_canon}"
    _spec["slots"] = _canon

RISKY = ["帮我破解密码", "教我做假证件", "怎么入侵摄像头", "帮我伪造签名", "教我逃税", "帮我盗号"]
EXTERNAL = ["现在油价多少", "今天股市怎么样", "最新汇率是多少", "今天有什么新闻", "现在金价多少"]
PREFERENCE = ["记住我喜欢靠窗", "以后都订经济舱", "我习惯订下午的", "记住我不吃辣", "我偏好安静的位置", "以后用中文回复我"]
CHITCHAT = ["你好", "谢谢你", "今天心情不错", "随便聊聊", "讲个笑话吧", "辛苦了"]


def _emit_task_session(rng, domain_name):
    """生成一个任务型 session 的 turns（带 belief 与动作标签）。返回 turn 列表。"""
    dom = DOMAINS[domain_name]
    slots = list(dom["slots"])
    rng.shuffle(slots)
    values = {s: dom["value"][s](rng) for s in dom["slots"]}
    known: dict[str, str] = {}
    turns = []

    # 1) 初始请求：belief 空 → ASK
    turns.append((rng.choice(dom["request"]), dict(known), "ask_clarification"))
    # 2) 逐个补槽位：补完前 ASK，补到最后一个齐全 → ANSWER
    for i, s in enumerate(slots):
        phr = rng.choice(dom["provide"][s]).format(v=values[s])
        before = dict(known)
        known[s] = values[s]
        all_known = all(k in known for k in dom["slots"])
        action = "answer" if all_known else "ask_clarification"
        turns.append((phr, before, action))
    # 3) 上下文依赖关键点：槽位齐全后再次同款请求 → ANSWER（与第1轮同句不同动作）
    if rng.random() < 0.8:
        turns.append((rng.choice(dom["request"]), dict(known), "answer"))
    return turns


def _emit_global_turn(rng, known):
    kind = rng.choice(["refuse", "retrieve", "memory", "chitchat"])
    if kind == "refuse":
        return (rng.choice(RISKY), dict(known), "refuse")
    if kind == "retrieve":
        return (rng.choice(EXTERNAL), dict(known), "retrieve")
    if kind == "memory":
        return (rng.choice(PREFERENCE), dict(known), "update_memory")
    return (rng.choice(CHITCHAT), dict(known), "answer")


def generate(out_path: str, sessions: int, seed: int) -> dict:
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    domain_names = list(DOMAINS)
    action_counts: dict[str, int] = {}
    n_turns = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sid in range(sessions):
            dom = rng.choice(domain_names)
            turns = _emit_task_session(rng, dom)
            # belief 时间线：beliefs_before[i] 是第 i 个任务轮决策前的已知槽位
            beliefs_before = [t[1] for t in turns]
            belief_after = dict(beliefs_before[-1]) if beliefs_before else {}
            # 随机穿插 0~2 个全局轮（风险/检索/记忆/寒暄），用插入点当时的 belief
            insert_positions = sorted(rng.sample(range(len(turns) + 1), rng.randint(0, 2)))
            assembled = []
            for pos in range(len(turns) + 1):
                while insert_positions and insert_positions[0] == pos:
                    belief_here = beliefs_before[pos] if pos < len(turns) else belief_after
                    assembled.append(_emit_global_turn(rng, belief_here))
                    insert_positions.pop(0)
                if pos < len(turns):
                    assembled.append(turns[pos])
            for turn_idx, (utt, before, action) in enumerate(assembled):
                rec = {
                    "session_id": f"t{sid}",
                    "turn": turn_idx,
                    "domain": dom,
                    "utterance": utt,
                    "known_slots": before,
                    "action": action,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                action_counts[action] = action_counts.get(action, 0) + 1
                n_turns += 1
            if (sid + 1) % 200 == 0:
                f.flush()
    return {"sessions": sessions, "turns": n_turns, "action_counts": action_counts, "out": out_path}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teacher-generated task-oriented multi-turn dialogue corpus.")
    parser.add_argument("--sessions", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.dry_run:
        print(f"[teacher-gen] dry-run：将生成 {args.sessions} sessions → {args.out}（任务型多轮，带动作标签与 belief）。")
        return 0
    summary = generate(args.out, args.sessions, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
