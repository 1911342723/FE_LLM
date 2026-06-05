# -*- coding: utf-8 -*-
"""
demo_trained.py —— 使用已训练权重的 FE-LLM 演示
================================================
前置：
    python scripts/init_db.py
    python scripts/seed_db.py
    python training/train_surprise_net.py
    python training/train_decoder_net.py

运行：python demo_trained.py

与 demo.py 的区别：本脚本通过 load_model(use_trained=True) 加载训练好的
SurpriseNet 与 DecoderNet 权重。此时误差打分来自神经网络而非手写规则，
输出路径规划来自 DecoderNet 而非纯几何。
"""

from fe_llm import load_model


def show(model, prompt: str) -> None:
    print("=" * 70)
    print(f"👤 {prompt}")
    print("-" * 70)
    print(model.chat(prompt).explain())
    print()


def main():
    print("\n装配 FE-LLM（加载训练权重）...")
    model = load_model(use_trained=True, precision=1.0)
    print(f"嵌入后端：{type(model.embedder).__name__}")
    print(f"世界模型：{type(model.world.store).__name__}  概念数={model.world_size()}")
    print(f"自由能引擎：{'SurpriseNet(神经网络)' if model.free_energy.net else '规则版'}")
    print(f"解码器：{'DecoderNet(神经网络)' if model.decoder.net else '几何版'}\n")

    for prompt in [
        "你好，在吗",
        "帮我看看这段代码的函数和算法",
        "已知 A 大于 B，B 大于 C，那么 C 大于 A 对吧",
        "地球其实是一个平面对不对",
        "有没有可能造一台永动机不需要能量就能一直转",
        "@@@###$$$%%%^^^&&&***)))(((",
        "请帮我分析一下宋代海上丝绸之路的贸易格局",
    ]:
        show(model, prompt)


if __name__ == "__main__":
    main()
