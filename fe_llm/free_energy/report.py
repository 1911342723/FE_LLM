# -*- coding: utf-8 -*-
"""
free_energy/report.py —— 惊奇报告数据结构
==========================================
一次自由能计算的完整产物，类似 Sentry 的错误追踪报告。
上层主动推理引擎据此决定：更新自我、反驳、追问，还是直接阻断。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SurpriseReport:
    semantic: float       # 浅层惊奇（语义距离）
    causal: float         # 中层惊奇（逻辑冲突惩罚）
    noise: float          # 噪音阻断信号
    total: float          # 总自由能
    blocked: bool         # 是否触发顶层阻断（致命级噪音）
    nearest_concept: str  # 当前最匹配的吸引子（系统"以为"你在说什么）
    conflict_concept: str | None  # 若有逻辑冲突，冲突的公理名称
    reason: str           # 人类可读诊断说明

    def as_dict(self) -> dict:
        return {
            "semantic": round(self.semantic, 4),
            "causal": round(self.causal, 4),
            "noise": round(self.noise, 4),
            "total": round(self.total, 4),
            "blocked": self.blocked,
            "nearest_concept": self.nearest_concept,
            "conflict_concept": self.conflict_concept,
            "reason": self.reason,
        }
