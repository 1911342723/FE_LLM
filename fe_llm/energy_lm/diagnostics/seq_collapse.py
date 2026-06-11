# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/seq_collapse.py —— 双轴自由能生成器（v4，可溯源）
==================================================================
对应 设计v4 第三节。生成 = 内容轴顺序 × 思考轴弛豫：

  内容轴（顺序）：逐字生成 r₁ r₂ … 直到 [EOS]。每个字依赖真实前缀 → 服从
                 链式依赖，根治字序乱 / mode collapse。
  思考轴（弛豫）：生成每个字时，网络的 depth 层 CausalPERBlock 即"思考时间"上的
                 多轮预测-误差弛豫。我们读出弛豫终态的能量分布，并量化"确定性"
                 （1 - 归一化熵）。确定性高 = 想清楚了；低 = 该多想 / 主动追问。

可溯源：对每个字，记录 (思考是否确定, 能量, 确定性, top候选)。整段生成的轨迹可打印。
主动推理：若某字确定性低于阈值 → 标记"模糊"，可触发追问而非硬落字。
经验=省电：熟悉上文确定性高、能量低（弛豫"省力"）。

注：本实现把 SeqEnergyNet 的 depth 层弛豫视为"思考轴的一次完整弛豫"，读末位置
能量。后续可把单字弛豫显式展开成 T 个 tick（CTM 式提前停止），此处先验证核心闭环。
"""

from __future__ import annotations

import math

import torch

from fe_llm.energy_lm.models.seq_net import SeqEnergyNet
from fe_llm.energy_lm.models.tokenizer import CharTokenizer


class SeqFreeEnergyChat:
    """双轴自由能对话生成器。"""

    def __init__(self, net: SeqEnergyNet, tok: CharTokenizer, device: str = "cpu"):
        self.net = net.to(device).eval()
        self.tok = tok
        self.device = device
        self.max_len = net.max_len

    @torch.no_grad()
    def _next_energy(self, ids: list[int]):
        """给当前序列，返回最后位置的"下一字"能量行 (V,)。"""
        x = torch.tensor([ids], device=self.device)
        e = self.net(x)[0]              # (L, V)
        return e[len(ids) - 1]          # 最后一个已知位置 → 预测下一字

    @staticmethod
    def _certainty(energy_row: torch.Tensor) -> float:
        """确定性 = 1 - 归一化熵。能量=-logit，softmax(-energy)=softmax(logit)。"""
        p = torch.softmax(-energy_row, dim=-1)
        ent = -(p * (p + 1e-12).log()).sum()
        return float(1.0 - ent / math.log(p.numel()))

    @torch.no_grad()
    def respond(self, prompt: str, max_new: int = 24, temperature: float = 0.0,
                certainty_floor: float = 0.0, record: bool = False):
        """逐字生成。temperature=0 取能量最低；>0 按玻尔兹曼采样。
        certainty_floor>0 时，确定性低于它的字会被标记（主动推理：可追问）。"""
        tok = self.tok
        ids = tok.encode(prompt) + [tok.sep_id, tok.bos_id]
        out, trace = [], []
        vague = False
        for step in range(max_new):
            if len(ids) >= self.max_len:
                break
            e_row = self._next_energy(ids).clone()
            # 屏蔽不该生成的特殊符（不让它吐 MASK/BOS/SEP/PAD/UNK 作内容）
            for sp in (tok.mask_id, tok.bos_id, tok.sep_id, tok.pad_id, tok.unk_id):
                e_row[sp] = 1e9
            cert = self._certainty(e_row)
            if temperature <= 1e-6:
                tid = int(torch.argmin(e_row))
            else:
                probs = torch.softmax(-e_row / temperature, dim=-1)
                tid = int(torch.multinomial(probs, 1))
            ev = float(e_row[tid])
            if record:
                cand = tok.id_to_tok[tid] if tid != tok.eos_id else "[EOS]"
                trace.append({"step": step, "char": cand,
                              "energy": round(ev, 3), "certainty": round(cert, 3)})
            if cert < certainty_floor:
                vague = True
            if tid == tok.eos_id:
                break
            out.append(tid)
            ids.append(tid)
        text = "".join(tok.id_to_tok[t] for t in out)
        info = {"trace": trace, "vague": vague,
                "n_chars": len(out)}
        return text, info


if __name__ == "__main__":
    import os
    from fe_llm.config import get_device
    from fe_llm.energy_lm.training.seq_train import CKPT_NET, CKPT_TOK
    if not os.path.exists(CKPT_NET):
        print("未找到权重，请先 python -m fe_llm.energy_lm.training.seq_train")
    else:
        d = get_device()
        net = SeqEnergyNet.load(CKPT_NET, map_location=d)
        tok = CharTokenizer.load(CKPT_TOK)
        chat = SeqFreeEnergyChat(net, tok, device=d)
        for p in ["你好", "谢谢", "在吗", "今天天气怎么样"]:
            t, _ = chat.respond(p)
            print(f"{p} -> {t}")
