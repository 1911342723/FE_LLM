# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/seq_demo.py —— 双轴自由能生成器演示（v4 验证）
================================================================
验证 设计v4 第五节的判据：
  1. 能成句（训练样本）   2. 泛化（近义未见输入）
  3. 可溯源（逐字思考轨迹）   4. 经验=省电（熟悉vs陌生的确定性/能量）

运行：python -m fe_llm.energy_lm.seq_demo
"""
from __future__ import annotations
import os, sys, json, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fe_llm.config import get_device
from fe_llm.energy_lm.seq_net import SeqEnergyNet
from fe_llm.energy_lm.tokenizer import CharTokenizer
from fe_llm.energy_lm.seq_collapse import SeqFreeEnergyChat
from fe_llm.energy_lm.seq_train import CKPT_NET, CKPT_TOK


def main():
    if not os.path.exists(CKPT_NET):
        print("未找到权重，请先 python -m fe_llm.energy_lm.seq_train")
        return
    d = get_device()
    net = SeqEnergyNet.load(CKPT_NET, map_location=d)
    tok = CharTokenizer.load(CKPT_TOK)
    chat = SeqFreeEnergyChat(net, tok, device=d)

    print("=" * 60)
    print("双轴自由能生成器 v4 —— 内容轴顺序 × 思考轴弛豫")
    print("=" * 60)

    print("\n【判据1】训练经验内：连贯成句")
    seen = ["你好", "谢谢", "在吗", "今天天气怎么样", "晚安", "我有点累"]
    for p in seen:
        out, _ = chat.respond(p)
        print(f"  你：{p}　→　{out}")

    print("\n【判据2】泛化：近义/未逐字训练过的输入")
    probe = ["你好啊", "您好呀", "在么", "谢谢您", "天气不错", "好困啊"]
    for p in probe:
        out, info = chat.respond(p, certainty_floor=0.5)
        flag = "（思考不确定→可追问）" if info["vague"] else ""
        print(f"  你：{p}　→　{out}{flag}")

    print("\n【判据3】可溯源：'今天天气怎么样'逐字思考轨迹")
    out, info = chat.respond("今天天气怎么样", record=True)
    print(f"  → {out}")
    for tr in info["trace"]:
        print(f"    第{tr['step']}字 '{tr['char']}'  能量={tr['energy']}  确定性={tr['certainty']}")

    print("\n【判据4】经验=省电：熟悉 vs 陌生 的平均确定性")
    dias = [json.loads(l) for l in open('data/dialogue/dialogues.jsonl', encoding='utf-8')]
    rng = random.Random(0)
    familiar = [x['prompt'] for x in rng.sample(dias, 30)]
    chars = [c for c in tok.id_to_tok if not tok.is_special(tok.tok_to_id[c])]
    strange = ["".join(rng.choice(chars) for _ in range(rng.randint(3, 6))) for _ in range(30)]

    def avg_cert(prompts):
        s = 0.0
        for p in prompts:
            _, info = chat.respond(p, record=True)
            if info["trace"]:
                s += sum(t["certainty"] for t in info["trace"]) / len(info["trace"])
        return s / len(prompts)

    fc, sc = avg_cert(familiar), avg_cert(strange)
    print(f"  熟悉输入(30条) 平均首步确定性：{fc:.3f}")
    print(f"  陌生乱串(30条) 平均首步确定性：{sc:.3f}")
    print(f"  → 熟悉>陌生 成立：{fc > sc}（经验使思考更笃定、更省力）")


if __name__ == "__main__":
    main()
