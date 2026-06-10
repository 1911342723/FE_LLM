"""Action realization layer."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .observation import Observation
from .policy import ActionType, CandidateAction
from .state import BeliefState
from .surprise import SurpriseScore

if TYPE_CHECKING:
    from .memory import MemoryCandidate


class ActionRealizer:
    """Turns selected actions into user-facing text."""

    def __init__(self, use_energy_decoder: bool = False):
        self._energy_chat = None
        if use_energy_decoder:
            self._try_load_energy_chat()

    def realize(
        self,
        selected_action: CandidateAction,
        observation: Observation,
        posterior_belief: BeliefState,
        surprise: SurpriseScore,
        recalled_memories: list["MemoryCandidate"] | None = None,
    ) -> str:
        del posterior_belief
        action = selected_action.action_type
        if action == ActionType.ASK_CLARIFICATION:
            if surprise.components.consistency_error > 0:
                return "这句话里有时间或逻辑冲突，请先澄清你真正想表达的情况。"
            return "信息还不够，请补充你想让我具体做什么。"
        if action == ActionType.RETRIEVE:
            return "这个问题需要外部信息或实时数据，我需要先检索后再回答。"
        if action == ActionType.REFUSE:
            return "这个请求可能带来风险，我不能帮助执行或提供具体方法。"
        if action == ActionType.UPDATE_MEMORY:
            return "我已记录这个偏好，后续会尽量按这个方式回应。"
        return self._answer(observation.text, recalled_memories=recalled_memories)

    def _answer(self, text: str, recalled_memories: list["MemoryCandidate"] | None = None) -> str:
        answer = self._base_answer(text)
        # 记忆读回的行为闭环：历史偏好真实影响当前输出，而不是只躺在 jsonl 里。
        preference = self._applicable_preference(recalled_memories)
        if preference:
            answer = f"{answer}（已按你之前的偏好：{preference}）"
        return answer

    def _base_answer(self, text: str) -> str:
        rule = self._rule_answer(text)
        if rule:
            return rule
        if self._energy_chat is not None:
            try:
                out, _ = self._energy_chat.respond(text)
                if self._looks_usable(out):
                    return out
            except Exception:
                pass
        return "我会尽量帮你处理这个问题。"

    @staticmethod
    def _applicable_preference(recalled_memories: list["MemoryCandidate"] | None) -> str | None:
        """从召回记忆中挑出可直接作用于回答风格的偏好。"""

        if not recalled_memories:
            return None
        style_markers = ("简短", "详细", "中文", "英文", "称呼", "语气", "风格", "格式")
        for memory in reversed(recalled_memories):
            if any(marker in memory.text for marker in style_markers):
                return memory.text
        return None

    @staticmethod
    def _rule_answer(text: str) -> str | None:
        stripped = text.strip()
        if stripped in {"你好", "您好", "嗨", "在吗"}:
            return "你好，我在。"
        if stripped in {"谢谢", "谢谢你", "太感谢了"}:
            return "不客气。"
        if "一加一" in stripped:
            return "等于二。"
        if "你是谁" in stripped or "你叫什么" in stripped:
            return "我是 FE-LLM 的主动推理原型。"
        return None

    @staticmethod
    def _looks_usable(text: str) -> bool:
        if not text or len(text.strip()) == 0:
            return False
        if "[UNK]" in text or "[MASK]" in text:
            return False
        return len(text) <= 80

    def _try_load_energy_chat(self) -> None:
        try:
            from fe_llm.config import get_device
            from fe_llm.energy_lm.intent_generate import IntentChat
            from fe_llm.energy_lm.intent_model import IntentLM
            from fe_llm.energy_lm.intent_train import CKPT_PATH, CKPT_TOK
            from fe_llm.energy_lm.tokenizer import CharTokenizer

            if not (os.path.exists(CKPT_PATH) and os.path.exists(CKPT_TOK)):
                return
            device = get_device()
            model = IntentLM.load(CKPT_PATH, map_location=device)
            tok = CharTokenizer.load(CKPT_TOK)
            self._energy_chat = IntentChat(model, tok, device=device)
        except Exception:
            self._energy_chat = None
