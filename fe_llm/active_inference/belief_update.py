"""Belief update layer."""

from __future__ import annotations

import numpy as np

from .perception import ObservationState
from .state import BeliefState
from .surprise import PredictionError

# 澄清请求在 unresolved_questions 里使用的统一条目，便于满足后精确移除。
CLARIFICATION_QUESTION = "Waiting for user to supply missing task details"
UNDERSPECIFIED_ASSUMPTION = "Input is underspecified"


class BeliefUpdater:
    """Updates beliefs with a small precision-weighted latent step."""

    def __init__(self, update_rate: float = 0.35):
        self.update_rate = update_rate

    def update(
        self,
        prior_belief: BeliefState,
        observation_state: ObservationState,
        prediction_error: PredictionError,
        clarification_fulfilled: bool = False,
    ) -> BeliefState:
        old = prior_belief.intent_vector
        obs = observation_state.vector
        dim = min(len(old), len(obs))
        old_part = old[:dim]
        obs_part = obs[:dim]
        uncertainty = prediction_error.uncertainty_error
        safety = prediction_error.safety_error
        rate = self.update_rate * max(0.1, 1.0 - 0.4 * safety)
        new_vec = old.copy()
        new_vec[:dim] = (1.0 - rate) * old_part + rate * obs_part
        context = prior_belief.context_vector.copy()
        context[:dim] = new_vec[:dim]

        assumptions = list(prior_belief.assumptions)
        unresolved = list(prior_belief.unresolved_questions)
        if clarification_fulfilled:
            # 澄清得到满足：撤销"输入欠规范"的假设并清掉等待澄清的未决问题。
            assumptions = [item for item in assumptions if item != UNDERSPECIFIED_ASSUMPTION]
            unresolved = [item for item in unresolved if item != CLARIFICATION_QUESTION]
        if uncertainty >= 0.75 and UNDERSPECIFIED_ASSUMPTION not in assumptions:
            assumptions.append(UNDERSPECIFIED_ASSUMPTION)
        if prediction_error.consistency_error > 0 and "Resolve internal consistency conflict" not in unresolved:
            unresolved.append("Resolve internal consistency conflict")

        confidence = float(np.clip(1.0 - max(uncertainty, safety, prediction_error.consistency_error), 0.0, 1.0))
        # 槽位级 belief：观测提供的所有槽位值写入/更新 known_slots（跨轮记忆来源，支持多槽位）。
        known_slots = dict(prior_belief.known_slots)
        for slot_key, slot_val in (observation_state.features.get("provided_slots") or {}).items():
            known_slots[slot_key] = slot_val
        # B2d：仅当本轮由关键词可靠检出领域时更新活跃领域，否则沿用上一活跃领域。
        # 关键：裸槽位值（"北京"/"明天"）即便被 NLU 高置信猜成 booking/reminder，也不更新活跃领域——
        # 否则会把语境污染掉，跟进句无法挂回真正在进行的领域。
        kw_domain = observation_state.features.get("keyword_domain")
        active_domain = kw_domain if kw_domain else prior_belief.active_domain
        return BeliefState(
            intent_vector=new_vec.astype(np.float32),
            context_vector=context.astype(np.float32),
            confidence=round(confidence, 4),
            assumptions=assumptions,
            unresolved_questions=unresolved,
            turn_index=prior_belief.turn_index,
            last_action=prior_belief.last_action,
            pending_clarification=prior_belief.pending_clarification and not clarification_fulfilled,
            # 槽位记忆跨轮持久化（含本轮新填）；澄清被满足时清掉正在等待的槽位。
            known_slots=known_slots,
            pending_slot=None if clarification_fulfilled else prior_belief.pending_slot,
            active_domain=active_domain,
        )

    @staticmethod
    def apply_action_feedback(belief: BeliefState, selected_action_value: str) -> BeliefState:
        """行动选定后回写信念状态：这是 Predictor 下一轮预期的来源。"""

        belief.turn_index += 1
        belief.last_action = selected_action_value
        if selected_action_value == "ask_clarification":
            # 发出澄清请求 == 模型对下一观测做出"会补充信息"的预测。
            belief.pending_clarification = True
            if CLARIFICATION_QUESTION not in belief.unresolved_questions:
                belief.unresolved_questions.append(CLARIFICATION_QUESTION)
        else:
            belief.pending_clarification = False
        return belief

