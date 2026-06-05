# -*- coding: utf-8 -*-
"""
engine.py —— FE-LLM 顶层引擎（总装）
=====================================
把所有分层模块串成完整的自由能闭环：

    用户输入
       │ [感知层] 穿过马尔可夫毯 → 真实向量(DashScope) + 自由能计算 → 抽象惊奇信号
       │ [分层预测编码] 自上而下预测 / 自下而上误差 → 坍缩出稳定意图向量
       │ [主动推理] 意图定型 + EFE → 选策略(确认/反驳/追问/阻断/问候)
       │ [可选 内部更新] 惊奇可消化 → 把新认知作为浅吸引子写回 pgvector(知识演化)
       │ [能量递减解码器] 沿能量梯度逐元滚落 → 生成文字
       │ [行动层] 穿过马尔可夫毯输出 → 改变用户下一步输入
    系统恢复平静(最小自由能)

可插拔的两个训练权重：
    - SurpriseNet (free_energy)  : 给误差打分。传入 surprise_net 即启用。
    - DecoderNet  (generation)   : 规划输出词汇路径。传入 decoder_net 即启用。
    两者均可缺省，缺省时用解析/几何版，保证未训练也能跑通。
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import get_device
from .embedding import Embedder, get_embedder
from .free_energy import FreeEnergyEngine, SurpriseReport
from .generation import (ActionPlan, ActiveInferenceEngine,
                         EnergyDescentDecoder, build_default_tokenizer)
from .perception import (ActiveLayer, CodingResult, PredictiveCodingHierarchy,
                         SensoryLayer, SensorySignal)
from .world_model import WorldModel, build_world_model


@dataclass
class Response:
    text: str
    strategy: str
    surprise: SurpriseReport
    coding: CodingResult
    plan: ActionPlan
    learned: bool

    def explain(self) -> str:
        return "\n".join([
            f"【输出】{self.text}",
            f"【策略】{self.strategy}",
            f"【惊奇】总自由能={self.surprise.total:.3f} "
            f"(语义={self.surprise.semantic:.3f}, 因果={self.surprise.causal:.3f}, "
            f"噪音={self.surprise.noise:.3f})"
            f"{' [已阻断]' if self.surprise.blocked else ''}",
            f"【诊断】{self.surprise.reason}",
            f"【坍缩】{self.coding.iterations} 轮迭代，残余能量={self.coding.residual_energy:.3f}",
            f"【推理】{self.plan.rationale}",
            f"【学习】{'是，已写回世界模型' if self.learned else '否'}",
        ])


class FreeEnergyLLM:
    """FE-LLM 主类。对外只暴露 chat()，内部跑完整的最小自由能认知闭环。"""

    def __init__(
        self,
        embedder: Embedder | None = None,
        world: WorldModel | None = None,
        world_backend: str = "auto",
        precision: float = 1.0,
        semantic_threshold: float = 0.85,
        noise_threshold: float = 2.2,
        enable_learning: bool = True,
        learn_threshold: float = 1.5,
        surprise_net=None,
        decoder_net=None,
        device: str | None = None,
    ):
        self.device = device or get_device()

        # 嵌入层（真实 DashScope，缺密钥自动降级哈希）
        self.embedder = embedder or get_embedder()

        # ① 世界模型（pgvector，连接失败自动降级内存）
        self.world = world or build_world_model(self.embedder, backend=world_backend)

        # ② 自由能引擎（可选 SurpriseNet 权重）
        self.free_energy = FreeEnergyEngine(
            self.world, self.embedder, precision=precision,
            semantic_threshold=semantic_threshold,
            noise_threshold=noise_threshold,
            surprise_net=surprise_net, device=self.device,
        )

        # ③ 马尔可夫毯 + 分层预测编码
        self.sensory = SensoryLayer(self.free_energy, self.embedder)
        self.coder = PredictiveCodingHierarchy(self.world)

        # ④ 主动推理 + 概念分词器 + 能量解码器(可选 DecoderNet) + 行动层
        self.inference = ActiveInferenceEngine(self.coder)
        self.tokenizer = build_default_tokenizer(self.embedder)
        self.decoder = EnergyDescentDecoder(
            self.tokenizer, decoder_net=decoder_net, device=self.device)
        self.active = ActiveLayer(self.decoder)

        self.enable_learning = enable_learning
        self.learn_threshold = learn_threshold
        self._turn = 0

    # ============================================================
    def chat(self, prompt: str) -> Response:
        """处理一轮用户输入，返回完整 Response（含内省）。"""
        self._turn += 1

        signal: SensorySignal = self.sensory.perceive(prompt)        # 感知
        plan, coding = self.inference.infer(signal)                  # 推理(含坍缩)
        learned = self._maybe_learn(signal)                          # 内部更新
        text = self.active.act(plan.intent_vector,                   # 行动(生成)
                               {"strategy": plan.strategy})

        return Response(text=text, strategy=plan.strategy,
                        surprise=signal.report, coding=coding,
                        plan=plan, learned=learned)

    # ============================================================
    def _maybe_learn(self, signal: SensorySignal) -> bool:
        """
        主动推理的"内部更新"路径：惊奇处于"中等可消化"区间(高于学习阈值、
        未阻断、非硬性因果冲突)时，把新认知作为浅吸引子写回世界模型(零微调演化)。
        因果冲突与噪音不学习，守护核心公理不被污染。
        """
        if not self.enable_learning:
            return False
        r = signal.report
        if r.blocked or r.causal > 1.0 or r.total < self.learn_threshold:
            return False
        self.world.learn(name=f"经验记忆#{self._turn}", text=signal.raw_text)
        return True

    # ============================================================
    def set_precision(self, precision: float) -> None:
        """动态置信度调控：高=低容错(数理严苛)，低=高容错(闲聊/角色扮演)。"""
        self.free_energy.precision = precision

    def world_size(self) -> int:
        return len(self.world)
