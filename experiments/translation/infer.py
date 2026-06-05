# -*- coding: utf-8 -*-
"""
translation/infer.py —— 加载权重做中英双向翻译
===============================================
运行：
    python experiments/translation/infer.py              # 跑内置示例
    python experiments/translation/infer.py --interactive # 交互模式

自动判断翻译方向：输入含中文字符则中译英，否则英译中（也可显式指定）。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from fe_llm.config import get_device
from experiments.translation.decoding import translate
from experiments.translation.model import TranslationModel
from experiments.translation.tokenizer import MODEL_DIR, SPM_MODEL, Tokenizer

CKPT_PATH = os.path.join(MODEL_DIR, "translation.pt")
_CJK = re.compile(r"[\u4e00-\u9fff]")


class Translator:
    """加载固化权重的双向翻译器。"""

    def __init__(self, device: str | None = None):
        self.device = device or get_device()
        if not os.path.exists(CKPT_PATH):
            raise FileNotFoundError(f"未找到模型 {CKPT_PATH}，请先训练。")
        ckpt = torch.load(CKPT_PATH, map_location=self.device, weights_only=False)
        self.tok = Tokenizer(SPM_MODEL)
        self.model = TranslationModel(**ckpt["config"]).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @staticmethod
    def detect_target(text: str) -> str:
        """含中文 → 目标英文；否则 → 目标中文。"""
        return "en" if _CJK.search(text) else "zh"

    def translate(self, text: str, target_lang: str | None = None,
                  beam_size: int = 5) -> str:
        tgt = target_lang or self.detect_target(text)
        return translate(self.model, self.tok, text, tgt, self.device, beam_size)


def run_samples(tr: Translator):
    samples = [
        "今天天气很好，我们去公园散步吧。",
        "请问最近的地铁站怎么走？",
        "我最喜欢的运动是打篮球。",
        "这家餐厅的菜很好吃，但是有点贵。",
        "I would like to book a table for two.",
        "Could you please help me with this problem?",
        "The weather is getting colder these days.",
        "What time does the meeting start tomorrow?",
    ]
    print("=" * 64)
    print("中英双向翻译演示（能量递减解码 / 束搜索）")
    print("=" * 64)
    for s in samples:
        tgt = tr.detect_target(s)
        out = tr.translate(s, tgt)
        arrow = "中→英" if tgt == "en" else "英→中"
        print(f"[{arrow}] {s}\n        => {out}\n")


def run_interactive(tr: Translator):
    print("交互翻译模式（输入 q 退出）。自动判断方向：含中文→英文，否则→中文。")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("q", "quit", "exit"):
            break
        if not text:
            continue
        print("  =>", tr.translate(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    tr = Translator()
    if args.interactive:
        run_interactive(tr)
    else:
        run_samples(tr)


if __name__ == "__main__":
    main()
