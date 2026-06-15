"""FE-LLM 世界模型引擎（从 0 自建）。

- 分层预测编码（纵向 z_1..z_L）：已经组合泛化裁决判为本规模过度设计、封存，引擎留档。
- CAPCW 内容寻址预测编码工作空间（横向 slot）：阶段一已验证内容寻址 > 单向量（绑定任务），
  当前核心引擎方向。见 `docs/FE-LLM核心引擎构想.md`。
"""

from .capcw import PCWorkspace, WorkspaceState
from .hierarchical_encoder import HierarchicalPredictiveEncoder, HierarchicalState
from .hierarchical_lm import HierarchicalIntentLM

__all__ = [
    "HierarchicalIntentLM",
    "HierarchicalPredictiveEncoder",
    "HierarchicalState",
    "PCWorkspace",
    "WorkspaceState",
]
