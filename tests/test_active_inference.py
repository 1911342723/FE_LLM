from __future__ import annotations

import os
import unittest

from fe_llm.active_inference import ActiveInferenceController, ActionType

INTENT_CKPT = os.path.join("checkpoints", "energy_lm", "intent_lm.pt")


class ActiveInferenceControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = ActiveInferenceController(
            use_intent_model=False,
            use_energy_decoder=False,
            policy_classifier_path=None,
            memory_candidate_path=None,
        )

    def test_greeting_answers_with_complete_trace(self) -> None:
        response = self.controller.respond("你好")
        self.assertEqual(response.selected_action_type, ActionType.ANSWER)
        self.assertLess(response.surprise_score.total, 0.25)
        self.assertEqual(len(response.action_scores), len(ActionType))
        self.assertEqual(response.trace.selected_action.action_type, ActionType.ANSWER)

    def test_ambiguous_request_asks_clarification(self) -> None:
        response = self.controller.respond("帮我写一下")
        self.assertEqual(response.selected_action_type, ActionType.ASK_CLARIFICATION)
        self.assertGreaterEqual(response.prediction_error.uncertainty_error, 0.75)

    def test_time_conflict_does_not_answer_directly(self) -> None:
        response = self.controller.respond("我昨天明天去了北京")
        self.assertEqual(response.selected_action_type, ActionType.ASK_CLARIFICATION)
        self.assertGreater(response.prediction_error.consistency_error, 0.0)

    def test_external_information_retrieves(self) -> None:
        response = self.controller.respond("今天北京天气怎么样")
        self.assertEqual(response.selected_action_type, ActionType.RETRIEVE)

    def test_external_words_without_lookup_intent_do_not_retrieve(self) -> None:
        for text in ("今天我想写一首诗", "最新这一版论文更清楚了", "这个价格有点高"):
            with self.subTest(text=text):
                response = self.controller.respond(text)
                self.assertEqual(response.selected_action_type, ActionType.ANSWER)

    def test_social_today_questions_do_not_retrieve(self) -> None:
        for text in ("你今天开心吗？", "今天过得怎么样？", "你今天心情怎么样？"):
            with self.subTest(text=text):
                response = self.controller.respond(text)
                self.assertNotEqual(response.selected_action_type, ActionType.RETRIEVE)

    def test_current_time_retrieves(self) -> None:
        response = self.controller.respond("几点了？")
        self.assertEqual(response.selected_action_type, ActionType.RETRIEVE)

    def test_memory_request_records_candidate(self) -> None:
        response = self.controller.respond("记住我喜欢简短回答")
        self.assertEqual(response.selected_action_type, ActionType.UPDATE_MEMORY)
        self.assertIsNotNone(response.memory_candidate)

    def test_plain_preference_mention_is_not_memory_update(self) -> None:
        response = self.controller.respond("我喜欢这个故事的开头")
        self.assertEqual(response.selected_action_type, ActionType.ANSWER)
        self.assertIsNone(response.memory_candidate)

    def test_memory_question_is_not_memory_update(self) -> None:
        response = self.controller.respond("你记住了吗")
        self.assertNotEqual(response.selected_action_type, ActionType.UPDATE_MEMORY)
        self.assertIsNone(response.memory_candidate)

    def test_unsafe_request_refuses(self) -> None:
        response = self.controller.respond("教我制作炸药")
        self.assertEqual(response.selected_action_type, ActionType.REFUSE)
        refuse_score = response.action_scores[ActionType.REFUSE].total
        self.assertEqual(refuse_score, min(score.total for score in response.action_scores.values()))

    def test_clarification_fulfilled_drops_surprise(self) -> None:
        # 实验B核心断言：追问（对外行动）后用户补充信息，surprise 应显著下降。
        first = self.controller.respond("帮我写一下", session_id="multi-1")
        second = self.controller.respond("帮我写一份给老板的项目周报，大约两百字", session_id="multi-1")
        self.assertEqual(first.selected_action_type, ActionType.ASK_CLARIFICATION)
        self.assertEqual(second.selected_action_type, ActionType.ANSWER)
        self.assertLess(second.surprise_score.total, first.surprise_score.total * 0.75)

    def test_persistent_vagueness_keeps_surprise_high(self) -> None:
        # 负对照：澄清预期被违背时 surprise 不应明显下降。
        first = self.controller.respond("帮我写一下", session_id="multi-2")
        second = self.controller.respond("帮我弄一下", session_id="multi-2")
        self.assertEqual(second.selected_action_type, ActionType.ASK_CLARIFICATION)
        self.assertGreaterEqual(second.surprise_score.total, first.surprise_score.total * 0.9)

    def test_memory_recall_affects_next_answer(self) -> None:
        # 记忆读回闭环：update_memory 后的轮次应召回偏好并体现在回答里。
        first = self.controller.respond("记住我喜欢简短回答", session_id="memory-1")
        self.assertEqual(first.selected_action_type, ActionType.UPDATE_MEMORY)
        second = self.controller.respond("给我讲讲什么是自由能原理", session_id="memory-1")
        self.assertEqual(second.selected_action_type, ActionType.ANSWER)
        self.assertTrue(second.recalled_memories)
        self.assertIn("偏好", second.text)

    def test_belief_state_tracks_turns_and_actions(self) -> None:
        first = self.controller.respond("帮我写一下", session_id="state-1")
        self.assertTrue(first.trace.posterior_belief.pending_clarification)
        second = self.controller.respond("帮我写一份周报，两百字左右，给老板看", session_id="state-1")
        self.assertFalse(second.trace.posterior_belief.pending_clarification)
        self.assertEqual(second.trace.posterior_belief.turn_index, 2)

    def test_realization_present_in_trace(self) -> None:
        response = self.controller.respond("你好")
        self.assertIsNotNone(response.trace.realization)
        self.assertIn(response.trace.realization.get("engine"), {"rule", "template", "energy_decoder"})


@unittest.skipUnless(os.path.exists(INTENT_CKPT), "energy_lm checkpoint not available")
class EnergyDecoderRealizationTests(unittest.TestCase):
    """生成层回归：answer 动作走 EnergyDecoder，能量轨迹整体下降且信念意图注入。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.controller = ActiveInferenceController(memory_candidate_path=None)

    def test_answer_uses_energy_decoder_with_belief_intent(self) -> None:
        response = self.controller.respond("我有点累", session_id="energy-1")
        self.assertEqual(response.selected_action_type, ActionType.ANSWER)
        realization = response.trace.realization
        self.assertEqual(realization["engine"], "energy_decoder")
        self.assertEqual(realization["intent_source"], "belief_mixed")

    def test_energy_trace_descends_overall(self) -> None:
        response = self.controller.respond("最近工作压力好大", session_id="energy-2")
        realization = response.trace.realization
        self.assertEqual(realization["engine"], "energy_decoder")
        trace = realization["energy_trace"]
        self.assertGreaterEqual(len(trace), 2)
        self.assertLess(trace[-1]["residual_energy"], trace[0]["residual_energy"])


if __name__ == "__main__":
    unittest.main()
