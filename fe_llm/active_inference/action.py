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
    """Turns selected actions into user-facing text.

    v3：answer 动作默认走 EnergyDecoder（能量递减生成），控制层的信念意图向量
    作为生成层的目标吸引子注入；逐字能量轨迹随 realization 返回，进入 trace，
    使"可溯源生成"真正发生在生成层而不只在策略层。
    """

    # 能量轨迹质量门控：整体残余能量必须下降（末值 < 初值 * 该比例），否则视为
    # 生成未收敛到意图，回退到规则/模板，避免把发散输出交给用户。
    ENERGY_DESCENT_RATIO = 0.999

    def __init__(self, use_energy_decoder: bool = True):
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
    ) -> tuple[str, dict]:
        action = selected_action.action_type
        if action == ActionType.ASK_CLARIFICATION:
            if surprise.components.consistency_error > 0:
                return "这句话里有时间或逻辑冲突，请先澄清你真正想表达的情况。", {"engine": "template"}
            return "信息还不够，请补充你想让我具体做什么。", {"engine": "template"}
        if action == ActionType.RETRIEVE:
            return "这个问题需要外部信息或实时数据，我需要先检索后再回答。", {"engine": "template"}
        if action == ActionType.REFUSE:
            return "这个请求可能带来风险，我不能帮助执行或提供具体方法。", {"engine": "template"}
        if action == ActionType.UPDATE_MEMORY:
            return "我已记录这个偏好，后续会尽量按这个方式回应。", {"engine": "template"}
        return self._answer(observation.text, posterior_belief, recalled_memories=recalled_memories)

    def _answer(
        self,
        text: str,
        posterior_belief: BeliefState,
        recalled_memories: list["MemoryCandidate"] | None = None,
    ) -> tuple[str, dict]:
        answer, realization = self._base_answer(text, posterior_belief)
        # 记忆读回的行为闭环：历史偏好真实影响当前输出，而不是只躺在 jsonl 里。
        preference = self._applicable_preference(recalled_memories)
        if preference:
            answer = f"{answer}（已按你之前的偏好：{preference}）"
            realization["applied_preference"] = preference
        return answer, realization

    def _base_answer(self, text: str, posterior_belief: BeliefState) -> tuple[str, dict]:
        # 固定寒暄/身份白名单仍走规则快速路径：这类输出属于人格设定而非生成任务，
        # 且当前小模型对寒暄的生成质量不稳定。其余 answer 一律走能量解码。
        rule = self._rule_answer(text)
        if rule:
            return rule, {"engine": "rule"}
        rejected_realization: dict | None = None
        if self._energy_chat is not None:
            out, realization = self._energy_answer(text, posterior_belief)
            if out is not None:
                return out, realization
            rejected_realization = realization
        fallback: dict = {"engine": "template"}
        if rejected_realization is not None:
            # 保留被门控拒绝的生成轨迹，回退本身也可溯源。
            fallback["rejected_generation"] = rejected_realization
        return "我会尽量帮你处理这个问题。", fallback

    def _energy_answer(self, text: str, posterior_belief: BeliefState) -> tuple[str | None, dict]:
        """EnergyDecoder 生成 + 能量轨迹质量门控。文本为 None 表示需要回退。"""

        try:
            out, info = self._energy_chat.respond(
                text,
                record=True,
                belief_intent=posterior_belief.intent_vector,
            )
        except Exception as exc:
            return None, {"engine": "energy_decoder", "rejected": f"exception: {exc}"}
        trace = info.get("trace", [])
        realization = {
            "engine": "energy_decoder",
            "intent_source": info.get("intent_source", "prompt_only"),
            "intent_norm": info.get("intent_norm"),
            "decode_mode": info.get("decode_mode"),
            # 与纯 argmax logit 决策不同的步数：能量信号真实参与了选字的证据。
            "decode_disagreement_steps": info.get("disagreement_steps"),
            "decode_total_steps": info.get("total_steps"),
            "n_chars": info.get("n_chars"),
            "energy_start": trace[0]["residual_energy"] if trace else None,
            "energy_end": trace[-1]["residual_energy"] if trace else None,
            "energy_descent_steps": sum(1 for item in trace if item["energy_drop"] > 0),
            "energy_trace": trace,
        }
        if not self._looks_usable(out):
            realization["rejected"] = "text_not_usable"
            return None, realization
        if len(trace) >= 2 and trace[-1]["residual_energy"] > trace[0]["residual_energy"] * self.ENERGY_DESCENT_RATIO:
            # 残余能量未整体下降：生成没有收敛向意图，可溯源地拒绝该输出。
            realization["rejected"] = "energy_not_descending"
            return None, realization
        return out, realization

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
