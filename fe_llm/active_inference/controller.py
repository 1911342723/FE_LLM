"""Top-level active inference controller."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .action import ActionRealizer
from .belief_update import BeliefUpdater
from .free_energy import ExpectedFreeEnergyScore, FreeEnergyScorer
from .memory import MemoryCandidate, MemoryManager
from .observation import Observation
from .perception import PerceptionEncoder
from .policy import ActionType, PolicyGenerator, PolicySelector
from .prediction import Predictor
from .state import BeliefStateStore
from .surprise import (
    PredictionError,
    PredictionErrorEstimator,
    SurpriseEstimator,
    SurpriseScore,
    is_clarification_fulfilled,
)
from .trace import InferenceTrace, TraceConsistencyScorer, TraceRecorder


@dataclass
class ModelResponse:
    text: str
    selected_action_type: ActionType
    surprise_score: SurpriseScore
    prediction_error: PredictionError
    action_scores: dict[ActionType, ExpectedFreeEnergyScore]
    trace: InferenceTrace
    memory_candidate: MemoryCandidate | None = None
    recalled_memories: list[MemoryCandidate] | None = None
    # CAPCW in-context 工作记忆取回的 value 串（query 命中已绑定时），否则 None。
    incontext_value: str | None = None
    # CAPCW in-context query 的引擎 surprise（=1-路由匹配度；query 时给出，可溯源）。
    incontext_surprise: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "selected_action_type": self.selected_action_type.value,
            "surprise_score": self.surprise_score.to_dict(),
            "prediction_error": self.prediction_error.to_dict(),
            "action_scores": {key.value: value.to_dict() for key, value in self.action_scores.items()},
            "trace": self.trace.to_dict(),
            "memory_candidate": self.memory_candidate.to_dict() if self.memory_candidate else None,
            "recalled_memories": [item.to_dict() for item in self.recalled_memories or []],
            "incontext_value": self.incontext_value,
            "incontext_surprise": self.incontext_surprise,
        }


class ActiveInferenceController:
    """Runs the v1 observation -> surprise -> policy -> action -> trace loop."""

    def __init__(
        self,
        use_intent_model: bool = True,
        use_energy_decoder: bool = True,
        policy_classifier_path: str | None = os.path.join("checkpoints", "active_inference", "policy_selector.pt"),
        policy_classifier_weight: float = 0.5,
        free_energy_calibration_path: str | None = os.path.join(
            "checkpoints", "active_inference", "free_energy_weights.json"
        ),
        memory_candidate_path: str | None = os.path.join("data", "active_inference", "memory_candidates.jsonl"),
        use_learned_nlu: bool = True,
        learned_nlu_path: str | None = os.path.join("checkpoints", "active_inference", "slot_intent_nlu.pt"),
        context_policy_path: str | None = None,
        capcw_memory_path: str | None = None,
    ):
        self.perception_encoder = PerceptionEncoder(use_intent_model=use_intent_model)
        self.state_store = BeliefStateStore(vector_dim=self.perception_encoder.vector_dim)
        self.predictor = Predictor()
        self.error_estimator = PredictionErrorEstimator()
        self.surprise_estimator = SurpriseEstimator()
        self.belief_updater = BeliefUpdater()
        self.policy_generator = PolicyGenerator()
        self.free_energy_scorer = FreeEnergyScorer(calibration_path=free_energy_calibration_path)
        self.action_selector = PolicySelector(
            classifier_path=policy_classifier_path,
            classifier_weight=policy_classifier_weight,
        )
        self.action_realizer = ActionRealizer(use_energy_decoder=use_energy_decoder)
        self.trace_recorder = TraceRecorder()
        self.trace_consistency = TraceConsistencyScorer()
        self.memory_manager = MemoryManager(candidate_path=memory_candidate_path)
        # 启动自动加载学习式意图 NLU（若 checkpoint 存在）：高置信 + 感知层"无其它信号"门控，
        # 仅在关键词漏判时补意图，不改既有 weather/记忆/拒答/寒暄 行为。
        if use_learned_nlu and learned_nlu_path and os.path.exists(learned_nlu_path):
            try:
                from fe_llm.active_inference.nlu.slot_intent_nlu import SlotIntentNLU
                from fe_llm.active_inference.observation import set_learned_nlu

                set_learned_nlu(SlotIntentNLU.load(learned_nlu_path), conf_threshold=0.9)
            except Exception:
                pass
        # 可选：学习式上下文感知 policy（utterance + belief → action）。默认 None 不启用，
        # 既有 EFE+分类器+规则管线不变；启用时由它覆盖动作选择（任务型多轮）。
        self._context_policy = None
        if context_policy_path and os.path.exists(context_policy_path):
            try:
                from fe_llm.active_inference.context_policy import ContextAwarePolicy

                self._context_policy = ContextAwarePolicy.load(context_policy_path)
            except Exception:
                self._context_policy = None
        # 可选：CAPCW 内容寻址工作记忆（in-context 任意键值绑定 + 引擎 surprise 驱动 ASK/ANSWER）。
        # 默认 None 不启用 → 既有管线零影响。启用时由 bind_working_memory / working_memory_decision 使用，
        # 让"知道何时不该答 + 内容取回"从 CAPCW 引擎 surprise 涌现（见 capcw_memory.py / capcw_controller_integration_eval）。
        # 诚实边界：活文本自动把"现场关联"抽成 (key,value) 需 in-context 绑定 NLU（下一步），故为显式 API 接口。
        self.capcw_memory = None
        if capcw_memory_path and os.path.exists(capcw_memory_path):
            try:
                from fe_llm.active_inference.capcw_memory import CAPCWWorkingMemory

                self.capcw_memory = CAPCWWorkingMemory.load(capcw_memory_path)
            except Exception:
                self.capcw_memory = None
        # in-context 绑定 NLU（轻量规则，无权重）：把活文本"现场关联"抽成 bind/query 事件喂工作记忆。
        # 仅在 capcw_memory 已加载时于 respond() 内使用；高精度模式触发，避免劫持既有对话。
        from fe_llm.active_inference.incontext_binding_nlu import InContextBindingNLU

        self._incontext_nlu = InContextBindingNLU()

    def respond(self, text: str, session_id: str | None = None) -> ModelResponse:
        observation = Observation.from_text(text, session_id=session_id)
        observation_state = self.perception_encoder.encode(observation)
        prior_belief = self.state_store.load(session_id)
        prediction = self.predictor.predict(prior_belief)
        prediction_error = self.error_estimator.compare(observation_state, prediction)
        surprise = self.surprise_estimator.score(prediction_error)
        # 多轮闭环：判断上一轮的澄清预期是否被当前观测满足。
        clarification_fulfilled = prior_belief.pending_clarification and is_clarification_fulfilled(
            observation.features
        )
        posterior_belief = self.belief_updater.update(
            prior_belief,
            observation_state,
            prediction_error,
            clarification_fulfilled=clarification_fulfilled,
        )
        # 记忆读回：相关记忆注入信念假设，使历史 update_memory 影响当前行为。
        recalled_memories = self.memory_manager.recall(observation.text, session_id=session_id)
        for memory in recalled_memories:
            note = f"memory: {memory.text}"
            if note not in posterior_belief.assumptions:
                posterior_belief.assumptions.append(note)
        candidates = self.policy_generator.generate(posterior_belief, surprise)
        scores = self.free_energy_scorer.score(candidates, posterior_belief, surprise, observation)
        selected_action = self.action_selector.select(
            candidates,
            scores,
            observation=observation,
            observation_state=observation_state,
            prediction_error=prediction_error,
            # 澄清刚被满足时，单轮 MLP 缺乏多轮上下文，交由可解释公式主导。
            fusion_scale=0.0 if clarification_fulfilled else 1.0,
        )
        # 槽位级 belief 决策：所需槽位已知→直接回答，未知→追问；仅在请求带 requires_slot 时生效。
        selected_action = self._apply_slot_belief(selected_action, candidates, observation, posterior_belief)
        # 可选学习式上下文 policy 覆盖（启用时）：用 utterance + belief 直接选动作。
        if self._context_policy is not None:
            try:
                pred = self._context_policy.predict(observation.text, posterior_belief.known_slots)
                cand = self._pick_candidate(candidates, ActionType(pred))
                if cand is not None:
                    selected_action = cand
            except Exception:
                pass
        # 可选 CAPCW in-context 工作记忆（默认未加载→跳过，既有管线零影响）：
        # 绑定 NLU 解析活文本 → bind 存入工作记忆；query 由**引擎 surprise** 裁决 ASK/ANSWER 并取回 value。
        # 这是把已验证的 CAPCW 引擎接回 controller 招牌决策"知道何时不该答"的活文本闭环（见 capcw_memory）。
        incontext_value: str | None = None
        incontext_surprise: float | None = None  # WM query 的引擎 surprise（可溯源，主动推理用）
        incontext_reply: str | None = None       # 由引擎取回内容生成的 grounded 回答（可溯源）
        if self.capcw_memory is not None:
            try:
                event = self._incontext_nlu.parse(observation.text)
                if event.kind == "bind":
                    self.capcw_memory.bind_str(event.key, event.value, session_id=session_id)
                    cand = self._pick_candidate(candidates, ActionType.ANSWER)  # 确认已记住
                    if cand is not None:
                        selected_action = cand
                    incontext_reply = f"好的，已记住{event.key}是{event.value}"
                elif event.kind == "query":
                    dec, value_str = self.capcw_memory.decide_str(event.key, session_id=session_id)
                    cand = self._pick_candidate(candidates, dec.action)  # 引擎 surprise: bound→ANSWER/unbound→ASK
                    if cand is not None:
                        selected_action = cand
                    incontext_value = value_str
                    incontext_surprise = dec.surprise        # 暴露 surprise：未绑定高→追问、绑定后降→回答
                    # grounded 生成：bound 时回答扎根于引擎取回的 value（可溯源），unbound 走常规追问文本。
                    if value_str is not None:
                        incontext_reply = f"{event.key}是{value_str}"
            except Exception:
                incontext_value = None
                incontext_surprise = None
                incontext_reply = None
        # 行动回写：selected action 决定下一轮 Predictor 的预期（如等待澄清）。
        posterior_belief = self.belief_updater.apply_action_feedback(
            posterior_belief, selected_action.action_type.value
        )
        # 生成层：返回文本 + 可溯源 realization（含 EnergyDecoder 逐字能量轨迹）。
        text_output, realization = self.action_realizer.realize(
            selected_action,
            observation,
            posterior_belief,
            surprise,
            recalled_memories=recalled_memories,
        )
        # grounded 生成：in-context 绑定/取回命中时，回答扎根于引擎取回的内容（可溯源），覆盖通用模板文本。
        if incontext_reply is not None:
            text_output = incontext_reply
        trace = self.trace_recorder.record(
            observation=observation,
            observation_state=observation_state,
            prior_belief=prior_belief,
            prediction=prediction,
            prediction_error=prediction_error,
            surprise=surprise,
            candidate_actions=candidates,
            action_scores=scores,
            selected_action=selected_action,
            posterior_belief=posterior_belief,
            realization=realization,
        )
        ok, missing = self.trace_consistency.validate(trace)
        if not ok:
            raise RuntimeError(f"Incomplete inference trace: {missing}")
        memory_candidate = self.memory_manager.update_if_needed(trace)
        self.state_store.save(posterior_belief, session_id=session_id)
        return ModelResponse(
            text=text_output,
            selected_action_type=selected_action.action_type,
            surprise_score=surprise,
            prediction_error=prediction_error,
            action_scores=scores,
            trace=trace,
            memory_candidate=memory_candidate,
            recalled_memories=recalled_memories,
            incontext_value=incontext_value,
            incontext_surprise=incontext_surprise,
        )

    # ---- CAPCW 内容寻址工作记忆接口（可选组件；默认未加载时为 no-op；按 session 隔离）----
    def bind_working_memory(self, key: int, value: int, session_id: str | None = None) -> bool:
        """把一个 in-context (key→value) 绑定写入 CAPCW 工作记忆（按会话）。未加载时返回 False（no-op）。"""
        if self.capcw_memory is None:
            return False
        self.capcw_memory.bind(key, value, session_id=session_id)
        return True

    def reset_working_memory(self, session_id: str | None = None) -> None:
        """清空 CAPCW 工作记忆某会话的绑定（换话题时用；session_id='*' 清全部）。未加载时为 no-op。"""
        if self.capcw_memory is not None:
            self.capcw_memory.reset(session_id=session_id)

    def working_memory_decision(self, query_key: int, session_id: str | None = None):
        """用 CAPCW 引擎 surprise 裁决一个查询键：bound→ANSWER+取回 value / unbound→ASK_CLARIFICATION。

        返回 `MemoryDecision`（含 action: ActionType、value、surprise）；未加载工作记忆时返回 None。
        这是把已验证的 CAPCW 引擎 surprise 接回 controller 招牌决策"知道何时不该答"的接口（见 capcw_memory）。
        """
        if self.capcw_memory is None:
            return None
        return self.capcw_memory.decide(query_key, session_id=session_id)

    @staticmethod
    def _pick_candidate(candidates, action_type: ActionType):
        for action in candidates:
            if action.action_type == action_type:
                return action
        return None

    def _apply_slot_belief(self, selected_action, candidates, observation, belief):
        """槽位级 belief 决策：请求需要的槽位已知→ANSWER，未知→ASK_CLARIFICATION。

        仅在观测带 requires_slot（如订票需要 route）时触发，不影响 weather/greeting
        等非槽位输入——因此不改动既有 weather→retrieve 等已验证行为。
        """
        # B2d 优先：领域未明示的"裸槽位值跟进句"——本轮没有领域关键词、但有活跃领域且提供了槽位值
        # （且无其它信号），用活跃领域的必需槽位跨轮检查完整性，实现"不必复述领域"的多槽位主动补全。
        # 必须前置于 required 分支：因为学习式 NLU 会把裸值（"北京"/"明天"）高置信误判出 required_slots，
        # 这里用活跃领域语境覆盖那个误判。（真实数据 CrossWOZ 验证 belief 价值在领域追踪，见 经验.md B2 系列。）
        inherited = self._inherited_required_slots(observation, belief)
        if inherited:
            missing = [slot for slot in inherited if slot not in belief.known_slots]
            target = ActionType.ASK_CLARIFICATION if missing else ActionType.ANSWER
            return self._pick_candidate(candidates, target) or selected_action
        # 既有：显式请求自带 required_slots（如订票需要 route）。不影响 weather/greeting 等。
        required = observation.features.get("required_slots") or []
        if required:
            # 多槽位：所有必需槽位都已知才回答；任一缺失则追问。
            missing = [slot for slot in required if slot not in belief.known_slots]
            target = ActionType.ASK_CLARIFICATION if missing else ActionType.ANSWER
            return self._pick_candidate(candidates, target) or selected_action
        return selected_action

    @staticmethod
    def _inherited_required_slots(observation, belief) -> list[str]:
        """B2d 窄门控：仅当「有活跃领域 + 本轮是裸槽位值跟进句 + 无其它信号」时，
        返回活跃领域的必需槽位；否则返回空（不接管）。

        三重保险（仿 learned-NLU 门控）守住既有行为：寒暄/天气/记忆/安全/逻辑冲突
        任一信号在场即不触发；非裸值（显式领域关键词请求）也不触发，避免劫持。
        """
        active = getattr(belief, "active_domain", None)
        if not active:
            return []
        feats = observation.features
        if not feats.get("is_bare_slot_value"):
            return []
        other_signal = (
            feats.get("needs_external_info")
            or feats.get("has_memory_request")
            or feats.get("has_safety_risk")
            or feats.get("has_consistency_conflict")
            or feats.get("is_greeting")
            or feats.get("is_thanks")
        )
        if other_signal:
            return []
        from fe_llm.active_inference.nlu.taxonomy import LEGACY_REQUIRED_SLOTS

        return list(LEGACY_REQUIRED_SLOTS.get(active, []))
