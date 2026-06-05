# -*- coding: utf-8 -*-
"""
generation/active_inference.py —— 主动推理引擎
================================================
对应文档：「生成机制的重构：作为主动推理的文本输出」。

系统输出文字不是为完成任务，而是为「改变用户下一步输入，让未来惊奇度降低」。
三步（导弹制导比喻）：
    1) 意图定型   ：预测编码坍缩出的稳定状态 = 潜在意图向量（目标吸引子）
    2) 预期自由能 ：EFE = 认识不确定性 - 实现目标的外部效用，挑 EFE 最低的策略
    3) 交解码器   ：把策略 + 意图向量交给能量递减解码器逐元生成
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..free_energy import SurpriseReport
from ..perception import CodingResult, PredictiveCodingHierarchy, SensorySignal

# 行动策略 → 允许使用的语义元标签
STRATEGY_TAGS = {
    "确认": ["连接", "结论", "解释"],
    "反驳": ["反驳", "解释", "追问"],
    "追问": ["连接", "追问"],
    "阻断": ["阻断"],
    "问候": ["问候"],
}


@dataclass
class ActionPlan:
    strategy: str                 # 选定策略（确认/反驳/追问/阻断/问候）
    intent_vector: np.ndarray     # 潜在意图向量（目标吸引子）
    expected_free_energy: float   # 预期自由能 EFE
    rationale: str                # 决策理由（可解释性）


class ActiveInferenceEngine:
    def __init__(self, coder: PredictiveCodingHierarchy):
        self.coder = coder

    def infer(self, signal: SensorySignal) -> tuple[ActionPlan, CodingResult]:
        # 1) 意图定型
        coding = self.coder.encode(signal.state_vector)
        intent = coding.settled_state
        # 2) 策略选择 + EFE
        strategy = self._select_strategy(signal.report)
        efe = self._expected_free_energy(signal.report, coding.residual_energy)
        rationale = self._explain(strategy, signal.report, coding)
        return ActionPlan(strategy, intent, efe, rationale), coding

    @staticmethod
    def _select_strategy(report: SurpriseReport) -> str:
        """决策优先级：阻断 > 反驳(因果冲突) > 追问(语义偏离) > 问候/确认。"""
        if report.blocked:
            return "阻断"
        if report.causal > 1.0:
            return "反驳"
        if report.semantic > 1.5:
            return "追问"
        if "问候" in report.nearest_concept:
            return "问候"
        return "确认"

    @staticmethod
    def _expected_free_energy(report: SurpriseReport, residual: float) -> float:
        """EFE = 认识不确定性 - 外部效用。越低代表越能让系统回归平静。"""
        epistemic = residual + report.total * 0.5
        utility = 0.6
        return epistemic - utility

    @staticmethod
    def _explain(strategy: str, report: SurpriseReport, coding: CodingResult) -> str:
        base = (f"残余能量={coding.residual_energy:.3f}，"
                f"经 {coding.iterations} 轮预测编码坍缩；")
        mapping = {
            "阻断": "总惊奇含噪音爆表，采取阻断以节能。",
            "反驳": f"检测到与公理「{report.conflict_concept}」的因果冲突，反驳并索取约束。",
            "追问": f"输入语义偏离已知概念(最近为「{report.nearest_concept}」)，追问以收窄环境。",
            "确认": f"输入贴合「{report.nearest_concept}」，系统自洽，直接确认。",
            "问候": "识别为日常问候，低惊奇寒暄。",
        }
        return base + mapping.get(strategy, "")
