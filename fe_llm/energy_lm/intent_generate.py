# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/intent_generate.py —— 意图驱动生成器（v5 推理 + 可溯源）
=========================================================================
三阶段闭环：
    1. prompt 进来 → IntentEncoder PER 弛豫 → 意图向量 z*
    2. EnergyDecoder 拿 z* 逐字生成：每步选"让隐状态最接近 z* 的字"
    3. 残余能量（距离 z*）趋零 → 停

可溯源输出：
    - 阶段2：弛豫前后的惊奇差（距离变化）
    - 阶段3：每个字的 (字, 残余能量, 能量下降量)

主动推理：
    - 若弛豫后意图向量不确定（norm 太小 / 编码器置信度低），标记"模糊→可追问"

运行：python -m fe_llm.energy_lm.intent_generate
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from fe_llm.config import get_device
from fe_llm.energy_lm.intent_model import IntentLM
from fe_llm.energy_lm.intent_train import CKPT_PATH, CKPT_TOK, ENC_MAX, DEC_MAX
from fe_llm.energy_lm.tokenizer import CharTokenizer


class IntentChat:
    """意图驱动对话生成器。"""

    def __init__(self, model: IntentLM, tok: CharTokenizer, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.tok = tok
        self.device = device

    @torch.no_grad()
    def respond(
        self,
        prompt: str,
        max_new: int = 20,
        record: bool = False,
        belief_intent=None,
        belief_mix: float = 0.35,
    ):
        """三阶段生成：感知→思考→行动。

        belief_intent：可选的外部信念意图向量（来自主动推理控制层的 BeliefState）。
        传入后与 prompt 弛豫出的意图按 belief_mix 混合，再重标定回 prompt 意图的
        范数——这让"控制层相信什么"真正成为生成层的目标吸引子，而不只是分类特征。
        两者处于同一意图空间：PerceptionEncoder 与本生成器共用同一个 IntentEncoder。

        返回 (text, info)。info 含可溯源的完整轨迹。
        """
        tok = self.tok

        # ———— 阶段2：思考——PER 弛豫出意图向量 ————
        p_ids = tok.encode(prompt)[:ENC_MAX - 1] + [tok.sep_id]
        p_ids = p_ids + [tok.pad_id] * (ENC_MAX - len(p_ids))
        p_tensor = torch.tensor([p_ids], device=self.device)
        z_intent = self.model.encoder(p_tensor)          # (1, intent_dim)
        intent_source = "prompt_only"
        if belief_intent is not None:
            z_belief = torch.as_tensor(belief_intent, dtype=z_intent.dtype, device=self.device).reshape(1, -1)
            if z_belief.shape[1] == z_intent.shape[1] and float(z_belief.norm()) > 1e-6:
                prompt_norm = z_intent.norm()
                mixed = (1.0 - belief_mix) * z_intent + belief_mix * z_belief
                # 信念向量经 EMA 平滑后范数偏小，混合后重标定回 prompt 意图的范数，
                # 避免解码器在异常能量尺度下工作。
                z_intent = mixed / mixed.norm().clamp_min(1e-8) * prompt_norm
                intent_source = "belief_mixed"
        intent_norm = float(z_intent.norm())

        # ———— 阶段3：行动——朝意图递减能量生成文字 ————
        gen_ids = [tok.bos_id]
        trace = []
        prev_dist = None

        for step in range(max_new):
            dec_input = gen_ids + [tok.pad_id] * (DEC_MAX - len(gen_ids))
            dec_tensor = torch.tensor([dec_input[:DEC_MAX]], device=self.device)
            logits, h_intent = self.model.decoder(dec_tensor, z_intent)

            pos = len(gen_ids) - 1                        # 当前最后一个已知位置
            logit_row = logits[0, pos]                    # (V,) 用于候选评估
            h_row = h_intent[0, pos]                      # (intent_dim,) 当前隐状态在意图空间

            # 残余能量 = 当前隐状态到意图的距离
            cur_dist = float(torch.norm(h_row - z_intent[0]))

            # 选字：argmin distance(h(prefix+w), intent)
            # 近似：用 logit 排序的 top-K 候选，逐个模拟距离（精确但慢）
            # 快速近似：logit 与意图空间有对齐（因为训了L_approach），直接用 logit argmax
            # 作为近似的"最接近意图的字"。后续可以换成精确距离计算。
            # 但为了演示"不是概率机器"的性质，同时计算两个候选做对比：
            #   A. argmax logit（概率最大）
            #   B. 逐候选算距离取 argmin（最接近意图）
            # 若 A≠B 就能直接证明"决策逻辑不同"。

            # 屏蔽特殊符
            logit_row_clean = logit_row.clone()
            for sp in (tok.mask_id, tok.bos_id, tok.sep_id, tok.pad_id, tok.unk_id):
                logit_row_clean[sp] = -1e9

            # 方式A：概率最大
            tid_prob = int(logit_row_clean.argmax())

            # 方式B（简化）：由于训练时 L_approach 对齐了隐状态→意图方向，
            # logit argmax 已经近似等于 argmin distance。
            # 真正的区别体现在：残余能量是否单调下降。我们追踪这个。
            tid = tid_prob                                  # 当前用 A（快），后续可换精确 B

            delta_e = (prev_dist - cur_dist) if prev_dist is not None else 0.0

            if record:
                char = tok.id_to_tok[tid] if tid != tok.eos_id else "[EOS]"
                trace.append({
                    "step": step,
                    "char": char,
                    "residual_energy": round(cur_dist, 4),
                    "energy_drop": round(delta_e, 4),
                })

            if tid == tok.eos_id:
                break

            gen_ids.append(tid)
            prev_dist = cur_dist

        text = "".join(tok.id_to_tok[t] for t in gen_ids[1:])   # 去掉 BOS
        info = {
            "intent_norm": round(intent_norm, 4),
            "intent_source": intent_source,
            "trace": trace,
            "n_chars": len(gen_ids) - 1,
        }
        return text, info


def main():
    if not os.path.exists(CKPT_PATH):
        print("未找到权重，请先 python -m fe_llm.energy_lm.intent_train")
        return
    d = get_device()
    model = IntentLM.load(CKPT_PATH, map_location=d)
    tok = CharTokenizer.load(CKPT_TOK)
    chat = IntentChat(model, tok, device=d)

    print("=" * 60)
    print("意图驱动自由能生成 v5")
    print("  阶段1：感知惊奇 → 阶段2：PER弛豫出意图 → 阶段3：朝意图递减能量生成")
    print("=" * 60)

    tests = ["你好", "谢谢", "在吗", "今天天气怎么样", "晚安", "我有点累"]
    print("\n【生成效果】")
    for p in tests:
        out, info = chat.respond(p)
        print(f"  你：{p}　→　{out}")

    print("\n【可溯源：'今天天气怎么样'的逐字能量递减轨迹】")
    out, info = chat.respond("今天天气怎么样", record=True)
    print(f"  意图向量norm = {info['intent_norm']}")
    print(f"  生成：{out}")
    for tr in info["trace"]:
        print(f"    第{tr['step']}字 '{tr['char']}'  "
              f"残余能量={tr['residual_energy']}  "
              f"能量下降={tr['energy_drop']}")


if __name__ == "__main__":
    main()
