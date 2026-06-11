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

运行：python -m fe_llm.energy_lm.generation.intent_generate
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from fe_llm.config import get_device
from fe_llm.energy_lm.models.intent_model import IntentLM
from fe_llm.energy_lm.training.intent_train import CKPT_PATH, CKPT_TOK, ENC_MAX, DEC_MAX
from fe_llm.energy_lm.models.tokenizer import CharTokenizer


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
        decode_mode: str = "hybrid",
        top_k: int = 8,
        decode_alpha: float = 1.0,
    ):
        """三阶段生成：感知→思考→行动。

        belief_intent：可选的外部信念意图向量（来自主动推理控制层的 BeliefState）。
        传入后与 prompt 弛豫出的意图按 belief_mix 混合，再重标定回 prompt 意图的
        范数——这让"控制层相信什么"真正成为生成层的目标吸引子，而不只是分类特征。
        两者处于同一意图空间：PerceptionEncoder 与本生成器共用同一个 IntentEncoder。

        decode_mode：
            "hybrid"：复合评分选字（默认）——score = log P(w) - α·归一化残余能量。
                      对应设计草案"打分 = 语言可读性 + 距离 z* 的残余能量"：
                      语言能力由 logit 承载，意图收敛由能量承载。
            "energy"：纯 argmin distance（实验对照用——当前训练强度下会牺牲语言性，
                      纯距离贪心容易退化成循环字）。
            "logit" ：纯 argmax logit（GPT 式决策，实验对照用）。

        返回 (text, info)。info 含可溯源的完整轨迹与决策分歧统计。
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
        disagreement_steps = 0
        total_steps = 0

        for step in range(max_new):
            dec_input = gen_ids + [tok.pad_id] * (DEC_MAX - len(gen_ids))
            dec_tensor = torch.tensor([dec_input[:DEC_MAX]], device=self.device)
            logits, h_intent = self.model.decoder(dec_tensor, z_intent)

            pos = len(gen_ids) - 1                        # 当前最后一个已知位置
            logit_row = logits[0, pos]                    # (V,) 用于候选评估
            h_row = h_intent[0, pos]                      # (intent_dim,) 当前隐状态在意图空间

            # 残余能量 = 当前隐状态到意图的距离
            cur_dist = float(torch.norm(h_row - z_intent[0]))

            # 屏蔽特殊符（保留 EOS：停止也是一个候选行动）
            logit_row_clean = logit_row.clone()
            for sp in (tok.mask_id, tok.bos_id, tok.sep_id, tok.pad_id, tok.unk_id):
                logit_row_clean[sp] = -1e9

            # 方式A：概率最大（GPT 式决策）
            tid_prob = int(logit_row_clean.argmax())

            if decode_mode in ("energy", "hybrid") and len(gen_ids) < DEC_MAX:
                # 对 top-K 候选逐个模拟"说出该字后"的隐状态，得到各候选的残余能量。
                tid = self._candidate_choice(
                    gen_ids, z_intent, logit_row_clean, top_k,
                    mode=decode_mode, alpha=decode_alpha,
                )
            else:
                tid = tid_prob
            total_steps += 1
            if tid != tid_prob:
                disagreement_steps += 1

            delta_e = (prev_dist - cur_dist) if prev_dist is not None else 0.0

            if record:
                char = tok.id_to_tok[tid] if tid != tok.eos_id else "[EOS]"
                entry = {
                    "step": step,
                    "char": char,
                    "residual_energy": round(cur_dist, 4),
                    "energy_drop": round(delta_e, 4),
                }
                if tid != tid_prob:
                    # 记录决策分歧：概率会选什么字、能量选了什么字。
                    entry["prob_char"] = tok.id_to_tok[tid_prob] if tid_prob != tok.eos_id else "[EOS]"
                trace.append(entry)

            if tid == tok.eos_id:
                break

            gen_ids.append(tid)
            prev_dist = cur_dist

        text = "".join(tok.id_to_tok[t] for t in gen_ids[1:])   # 去掉 BOS
        info = {
            "intent_norm": round(intent_norm, 4),
            "intent_source": intent_source,
            "decode_mode": decode_mode,
            "disagreement_steps": disagreement_steps,
            "total_steps": total_steps,
            "trace": trace,
            "n_chars": len(gen_ids) - 1,
        }
        return text, info

    @torch.no_grad()
    def _candidate_choice(self, gen_ids: list[int], z_intent: torch.Tensor,
                          logit_row_clean: torch.Tensor, top_k: int,
                          mode: str = "hybrid", alpha: float = 1.0) -> int:
        """对 top-K 候选批量模拟一步，按 mode 决策：

        - energy：argmin 残余能量（纯距离，可能牺牲语言性）；
        - hybrid：argmax( log P(w) - α·归一化残余能量 )——
                  语言可读性与意图收敛复合打分。
        """

        tok = self.tok
        topk = torch.topk(logit_row_clean, k=min(top_k, logit_row_clean.numel()))
        cand_ids = topk.indices.tolist()
        pos = len(gen_ids)                                 # 候选字将出现的位置
        batch = []
        for cand in cand_ids:
            seq = gen_ids + [cand]
            seq = seq + [tok.pad_id] * (DEC_MAX - len(seq))
            batch.append(seq[:DEC_MAX])
        batch_tensor = torch.tensor(batch, device=self.device)
        z_batch = z_intent.expand(len(cand_ids), -1)
        _, h_intent = self.model.decoder(batch_tensor, z_batch)
        h_at = h_intent[:, pos]                            # (K, intent_dim) 说出候选字后的隐状态
        dists = torch.norm(h_at - z_batch, dim=-1)         # (K,) 各候选的残余能量
        if mode == "energy":
            return int(cand_ids[int(dists.argmin())])
        # hybrid：能量在候选内做 min-max 归一化（量纲 ~[0,1]），与 log prob 同尺度复合。
        log_probs = torch.log_softmax(logit_row_clean, dim=-1)[topk.indices]
        dist_norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)
        scores = log_probs - alpha * dist_norm
        return int(cand_ids[int(scores.argmax())])


def main():
    if not os.path.exists(CKPT_PATH):
        print("未找到权重，请先 python -m fe_llm.energy_lm.training.intent_train")
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
