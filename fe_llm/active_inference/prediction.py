"""Prediction layer for expected latent states.

v2：Predictor 不再只是把 prior belief 原样拷贝，而是基于"上一轮行动 + 当前信念"
对下一观测形成可被验证或违背的预期（expected_action_outcome）。这是多轮闭环里
"行动改变环境，从而降低未来自由能"能被度量的前提：
- 上一轮发出澄清请求 -> 预期下一观测补充具体信息（expects_clarification=True）。
- 预期被满足 -> prediction error 下降 -> surprise 下降。
- 预期被违背（用户仍然模糊）-> 误差维持高位。
"""

from __future__ import annotations

from .state import BeliefState, PredictionState


class Predictor:
    """基于信念状态生成对下一观测的结构化预期。"""

    def predict(self, prior_belief: BeliefState) -> PredictionState:
        return PredictionState(
            # 潜变量层面的预期：信念向量是对"下一轮意图应当落在哪"的最优猜测。
            expected_intent=prior_belief.intent_vector.copy(),
            expected_observation=prior_belief.context_vector.copy(),
            expected_action_outcome={
                "assumptions": list(prior_belief.assumptions),
                "unresolved_questions": list(prior_belief.unresolved_questions),
                # 结构化预期：上一轮若是澄清请求，则预测用户将补充信息。
                "expects_clarification": bool(prior_belief.pending_clarification),
                "last_action": prior_belief.last_action,
                "turn_index": int(prior_belief.turn_index),
            },
        )
