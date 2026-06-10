# -*- coding: utf-8 -*-
"""
fe_llm/energy_lm/collapse.py —— 能量坍缩生成（去掩码 + 退火 + 空稳态）
======================================================================
生成 = 让整句（含"何处该是空"）坍缩到能量最低的稳态。

根因解决：回应窗口固定 RESP_MAX 位，从全 [MASK] 出发，能量网络反复把每个空位
填成"让整句最稳定"的 token——**包括把多余位置填成 [EOS]/[PAD]（"空"也是低能稳态）**。
于是回应长度由能量自然决定，不再硬塞字。

两种坍缩：
    collapse_greedy   ：置信优先一次性填满（快）。
    collapse_annealed ：填满后多轮退火，把最不稳位置打回重填（治字序错位）。
读出：从回应窗口起点逐位取字，遇 [EOS] 即停（[EOS] = 内容结束的稳态）。
"""

from __future__ import annotations

import torch

from fe_llm.energy_lm.energy_net import DialogueEnergyNet
from fe_llm.energy_lm.tokenizer import CharTokenizer

RESP_MAX = 12       # 与 train.py 一致：回应窗口固定长度


# ==================================================================
# 经验记忆库（零重训成长：新经验= 能量地貌上新刻的一条低能沟，不动权重）
# ==================================================================
class MemoryBank:
    """
    外挂经验记忆：存网络**从没训练过**的 (prompt, response) 对。
    生成时若当前输入与某条记忆的 prompt 足够相似，就给该记忆 response 的字
    在对应位置叠加"能量奖励"（刻一条低能沟），引导坍缩走向它。
    这实现"零重训成长"——权重冻结，只往记忆库加一条，模型立刻会用。
    """

    def __init__(self):
        self.items: list[tuple[str, str]] = []

    def add(self, prompt: str, response: str):
        self.items.append((prompt.strip(), response.strip()))

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _sim(a: str, b: str) -> float:
        """字面相似度（字集合 Jaccard），够用来判"是不是问的同一件事"。"""
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def recall(self, prompt: str, threshold: float = 0.5):
        """召回最相似的记忆。返回 (response, 相似度) 或 None。"""
        best, best_s = None, 0.0
        for p, r in self.items:
            s = self._sim(prompt, p)
            if s > best_s:
                best, best_s = r, s
        if best is not None and best_s >= threshold:
            return best, best_s
        return None


def _build_input(tok: CharTokenizer, prompt: str, seq_len: int):
    """拼 [上文][SEP][BOS][MASK*RESP_MAX]，返回 ids 与回应窗口下标。"""
    p = tok.encode(prompt)
    seq = p + [tok.sep_id, tok.bos_id] + [tok.mask_id] * RESP_MAX
    resp_start = len(p) + 2
    resp_idx = list(range(resp_start, resp_start + RESP_MAX))
    if len(seq) > seq_len:
        seq = seq[:seq_len]
        resp_idx = [i for i in resp_idx if i < seq_len]
    seq = seq + [tok.pad_id] * (seq_len - len(seq))
    return seq, resp_idx


class EnergyCollapseChat:
    """能量坍缩对话生成器。"""

    def __init__(self, net: DialogueEnergyNet, tok: CharTokenizer,
                 device: str = "cpu", memory: "MemoryBank | None" = None):
        self.net = net.to(device).eval()
        self.tok = tok
        self.device = device
        self.seq_len = net.seq_len
        self.memory = memory                     # 外挂经验记忆（零重训成长）
        self._recall_ids: list[int] | None = None  # 当前命中的记忆 response 的字 id

    # ----------------------------------------------------------
    @torch.no_grad()
    def respond(self, prompt: str, record: bool = False, annealed: bool = True,
                rounds: int = 10, mem_bonus: float = 4.0, mem_threshold: float = 0.5,
                temp0: float = 0.7):
        """生成回应。annealed=True 用退火坍缩（治字序），否则贪心一次性。
        temp0：退火起始温度。0≈纯贪心（低熵强对应数据够用）；越高越随机探索
        （高熵数据破 mode collapse 用）。若挂了记忆库且命中相似经验，给该经验
        的字叠加能量奖励(mem_bonus)。"""
        # —— 召回经验记忆：命中则把其 response 的字 id 作为低能引导 ——
        self._recall_ids = None
        self._mem_bonus = mem_bonus
        if self.memory is not None:
            hit = self.memory.recall(prompt, threshold=mem_threshold)
            if hit is not None:
                resp, _ = hit
                self._recall_ids = self.tok.encode(resp)
        seq, resp_idx = _build_input(self.tok, prompt, self.seq_len)
        ids = torch.tensor([seq], device=self.device)
        if annealed:
            return self._annealed(ids, resp_idx, rounds, record, temp0=temp0)
        return self._greedy(ids, resp_idx, record)

    # ----------------------------------------------------------
    @torch.no_grad()
    def _fill_pos(self, ids, pos, allow_special=True, eos_bias: float = 1.5,
                  slot: int | None = None, temp: float = 0.0):
        """给某位置选 token。temp=0 取能量最低（argmin）；temp>0 按玻尔兹曼分布
        softmax(-E/temp) 随机采样——这才是真正的"随机退火"：高温探索、低温锁定。
        贪心 argmin 会让所有位置一起滚进同一个"通用高频字盆地"（mode collapse，
        如对什么都答"你是了"）；随机采样让坍缩**承诺到某一个具体回应**而非所有
        合理回应的平均。allow_special 时允许 EOS/PAD（空稳态）。"""
        e_row = self.net(ids)[0][pos].clone()
        if not allow_special:
            e_row[:6] = 1e9
        else:
            for sp in (self.tok.mask_id, self.tok.bos_id, self.tok.sep_id,
                       self.tok.unk_id):
                e_row[sp] = 1e9
            e_row[self.tok.eos_id] = e_row[self.tok.eos_id] + eos_bias
            e_row[self.tok.pad_id] = e_row[self.tok.pad_id] + eos_bias
        # —— 经验记忆引导：命中记忆 = 最强吸引子，刻出全行最深的低能沟 ——
        # 关键：能量地貌量级随位置变化（本网络单行 min≈-22, max≈+18, std≈6），
        # 固定加性 bonus 治标不治本；改为"压到当前行最低能再减 margin"——
        # 与地貌量级无关，保证命中字必成 argmin（坍缩必然流入此沟）。
        if self._recall_ids is not None and slot is not None:
            floor = float(e_row.min()) - self._mem_bonus     # margin 复用 mem_bonus 字段
            if slot < len(self._recall_ids):
                tid_mem = self._recall_ids[slot]
                if not self.tok.is_special(tid_mem):
                    e_row[tid_mem] = floor
            else:
                # 记忆已说完，该位置压向 EOS（让回应长度也对齐记忆）
                e_row[self.tok.eos_id] = floor
        if temp <= 1e-6:
            tid = int(torch.argmin(e_row))
        else:
            # 玻尔兹曼采样：低能=高概率。能量=-logit，故 logit/temp = -e_row/temp
            probs = torch.softmax(-e_row / temp, dim=-1)
            tid = int(torch.multinomial(probs, 1))
        return tid, float(e_row[tid])

    @torch.no_grad()
    def _greedy(self, ids, resp_idx, record):
        remaining = list(resp_idx); trace = []; esum = 0.0; steps = 0
        pos2slot = {p: s for s, p in enumerate(resp_idx)}
        while remaining:
            steps += 1
            best_pos, best_tok, best_e = None, None, 1e9
            for pos in remaining:
                tid, ev = self._fill_pos(ids, pos, slot=pos2slot[pos])
                if ev < best_e:
                    best_e, best_pos, best_tok = ev, pos, tid
            ids[0, best_pos] = best_tok
            remaining.remove(best_pos); esum += best_e
            if record:
                trace.append((steps, self._read(ids[0].tolist(), resp_idx)))
        return self._finish(ids, resp_idx, steps, esum, trace)

    @torch.no_grad()
    def _annealed(self, ids, resp_idx, rounds, record, temp0: float = 1.2):
        """随机退火坍缩。temp0 是起始温度，按 1/rounds 线性降到 0。
        高温轮：玻尔兹曼采样，让整句承诺到某个具体回应（破 mode collapse）；
        低温轮：趋于 argmin，锁定并自洽。"""
        pos2slot = {p: s for s, p in enumerate(resp_idx)}
        # 初稿：高温采样填满（含可能的 EOS/PAD），先承诺一个方向
        for pos in resp_idx:
            ids[0, pos], _ = self._fill_pos(ids, pos, slot=pos2slot[pos], temp=temp0)
        L = len(resp_idx); trace = []
        for r in range(rounds):
            frac = r / max(1, rounds - 1)
            temp = temp0 * (1.0 - frac)              # 线性降温到 0
            n_refill = max(1, int(round((1.0 - frac) * L)))
            energy = self.net(ids)[0]
            cur = sorted(((float(energy[pos, ids[0, pos]]), pos) for pos in resp_idx),
                         reverse=True)
            refill = [p for _, p in cur[:n_refill]]
            for pos in refill:
                ids[0, pos] = self.tok.mask_id
            for pos in refill:
                ids[0, pos], _ = self._fill_pos(ids, pos, slot=pos2slot[pos], temp=temp)
            if record:
                trace.append((r, round(temp, 2), self._read(ids[0].tolist(), resp_idx)))
        energy = self.net(ids)[0]
        esum = sum(float(energy[pos, ids[0, pos]]) for pos in resp_idx)
        return self._finish(ids, resp_idx, rounds, esum, trace)

    # ----------------------------------------------------------
    def _read(self, ids: list[int], resp_idx: list[int]) -> str:
        """从回应窗口读字，遇 EOS/PAD 停（内容结束的稳态），过滤特殊符。"""
        tok = self.tok
        out = []
        for i in resp_idx:
            t = ids[i]
            if t in (tok.eos_id, tok.pad_id):
                break
            if tok.is_special(t):
                continue
            out.append(tok.id_to_tok[t])
        return "".join(out)

    def _finish(self, ids, resp_idx, steps, esum, trace):
        text = self._read(ids[0].tolist(), resp_idx)
        # 有效内容长度（到 EOS 为止）
        clen = len(text) if text else 1
        info = {"steps": steps, "final_energy": esum / len(resp_idx),
                "content_energy": esum / clen, "trace": trace}
        return text, info
