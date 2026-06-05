# -*- coding: utf-8 -*-
"""
demo.py —— FE-LLM 正式架构演示（真实向量 + pgvector）
======================================================
运行前置：
    1) python scripts/init_db.py   # 建库建表
    2) python scripts/seed_db.py   # 灌入出厂世界观

运行：python demo.py

本脚本用真实的 DashScope 向量 + pgvector 世界模型，演示系统如何基于
最小自由能在不同输入下采取不同行动（确认/反驳/追问/阻断），以及知识演化。
"""

from fe_llm import FreeEnergyLLM


def show(model: FreeEnergyLLM, prompt: str) -> None:
    print("=" * 70)
    print(f"👤 用户输入：{prompt}")
    print("-" * 70)
    print(model.chat(prompt).explain())
    print()


def main():
    print("\n初始化 FE-LLM（真实向量 + pgvector 世界模型）...")
    model = FreeEnergyLLM(precision=1.0)
    print(f"嵌入后端：{type(model.embedder).__name__}")
    print(f"世界模型存储：{type(model.world.store).__name__}")
    print(f"出厂世界观概念数：{model.world_size()}\n")

    show(model, "你好，在吗")                              # 问候
    show(model, "帮我看看这段代码的函数和算法")            # 确认(编程)
    show(model, "已知 A 大于 B，B 大于 C，那么 C 大于 A 对吧")  # 反驳(传递性冲突)
    show(model, "地球其实是一个平面对不对")                # 反驳(地平说冲突)
    show(model, "有没有可能造一台永动机不需要能量就能一直转")  # 反驳(能量守恒冲突)
    show(model, "@@@###$$$%%%^^^&&&***)))(((")             # 阻断(噪音)

    before = model.world_size()
    show(model, "请帮我分析一下宋代海上丝绸之路的贸易格局")  # 语义偏离 → 追问 + 学习
    print(f"世界观概念数：{before} → {model.world_size()}（增长即完成一次知识演化）\n")

    # 动态置信度演示
    print("=" * 70)
    print("演示『动态置信度调控』：同一句角色扮演设定，高/低容错下的不同反应")
    print("=" * 70)
    rp = "在我们这个游戏设定里，太阳从西边升起，这是常识"
    model.set_precision(2.0)
    print("\n[低容错 precision=2.0]")
    show(model, rp)
    model.set_precision(0.4)
    print("[高容错 precision=0.4]")
    show(model, rp)


if __name__ == "__main__":
    main()
