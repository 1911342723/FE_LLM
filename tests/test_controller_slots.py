from __future__ import annotations

import argparse
import os
import unittest

from fe_llm.active_inference.controller import ActiveInferenceController
from fe_llm.active_inference.experiments.contextual_controller_eval import main as ctrl_eval_main
from fe_llm.active_inference.experiments.fe_llm_cli import main as cli_main
from fe_llm.active_inference.experiments.fe_llm_cli import run_inputs as cli_run_inputs
from fe_llm.active_inference.experiments.fe_llm_demo import main as demo_main
from fe_llm.active_inference.experiments.fe_llm_demo import run as demo_run
from fe_llm.active_inference.experiments.fe_llm_multidomain_demo import main as md_demo_main
from fe_llm.active_inference.experiments.fe_llm_multidomain_demo import run as md_demo_run
from fe_llm.active_inference.experiments.fe_llm_demo_web import build_html
from fe_llm.active_inference.experiments.fe_llm_demo_web import main as web_main
from fe_llm.active_inference.experiments.fe_llm_web_server import page_html, respond_payload
from fe_llm.active_inference.experiments.real_data_validation import main as realval_main
from fe_llm.active_inference.experiments.teacher_corpus_eval import main as teacher_eval_main
from fe_llm.active_inference.experiments.teacher_corpus_gen import main as teacher_gen_main
from fe_llm.active_inference.experiments.train_task_nlu import main as task_nlu_main
from fe_llm.active_inference.policy import ActionType


class ControllerSlotHeadroomTests(unittest.TestCase):
    """FE-LLM 标准回归基准：真实 controller 的 belief 在 headroom 轮上的行为锁定。"""

    def test_remembers_route_to_answer_repeat_booking(self) -> None:
        controller = ActiveInferenceController()
        first = controller.respond("帮我订票", session_id="slot-1")
        self.assertEqual(first.selected_action_type, ActionType.ASK_CLARIFICATION)
        controller.respond("北京到上海", session_id="slot-1")
        repeat = controller.respond("帮我订票", session_id="slot-1")
        # belief 记住 route → 同一句"帮我订票"由追问变为回答（headroom）。
        self.assertEqual(repeat.selected_action_type, ActionType.ANSWER)
        self.assertEqual(repeat.trace.posterior_belief.known_slots.get("route"), "北京到上海")

    def test_memoryless_asks_again(self) -> None:
        controller = ActiveInferenceController()
        controller.respond("帮我订票", session_id="mem-a")
        controller.respond("北京到上海", session_id="mem-b")
        # 换新 session（无跨轮记忆）→ 同一句"帮我订票"仍只能追问。
        repeat = controller.respond("帮我订票", session_id="mem-c")
        self.assertEqual(repeat.selected_action_type, ActionType.ASK_CLARIFICATION)

    def test_weather_still_retrieves(self) -> None:
        # 回归保护：槽位机制不接管 weather，天气仍走 retrieve。
        controller = ActiveInferenceController()
        response = controller.respond("今天北京天气怎么样", session_id="w-1")
        self.assertEqual(response.selected_action_type, ActionType.RETRIEVE)

    def test_unknown_slot_raises_surprise_and_asks(self) -> None:
        # 未知必需槽位 → 高 uncertainty → 主动追问（即便句子本身不属于 ambiguous 模板）。
        controller = ActiveInferenceController()
        response = controller.respond("我要订票", session_id="su-1")
        self.assertEqual(response.selected_action_type, ActionType.ASK_CLARIFICATION)
        self.assertGreaterEqual(response.prediction_error.uncertainty_error, 0.75)

    def test_surprise_drops_after_route_provided(self) -> None:
        # 提供槽位（作为澄清回复）后 surprise 显著下降，可溯源记录在 trace。
        controller = ActiveInferenceController()
        first = controller.respond("帮我订票", session_id="sd-1")
        second = controller.respond("北京到上海", session_id="sd-1")
        self.assertLess(second.surprise_score.total, first.surprise_score.total)

    def test_route_rewrite_phrasings_detected(self) -> None:
        from fe_llm.active_inference.observation import extract_prompt_features

        for text in ("从北京去上海", "北京飞上海", "北京至上海"):
            with self.subTest(text=text):
                feats = extract_prompt_features(text)
                self.assertEqual(feats["provides_slot"], "route")
                self.assertEqual(feats["slot_value"], "北京到上海")

    def test_memory_survives_distractor_turn(self) -> None:
        # 记忆跨干扰轮持久：提供 route 后插一句无关话，仍能在下次订票时回答。
        controller = ActiveInferenceController()
        controller.respond("帮我订票", session_id="dist-1")
        controller.respond("从北京去上海", session_id="dist-1")
        controller.respond("你好", session_id="dist-1")
        final = controller.respond("帮我订票", session_id="dist-1")
        self.assertEqual(final.selected_action_type, ActionType.ANSWER)
        self.assertEqual(final.trace.posterior_belief.known_slots.get("route"), "北京到上海")

    def test_hotel_multi_slot_requires_all_slots(self) -> None:
        # 多槽位：订酒店需要 city + date，全部凑齐才回答；缺任一则追问。
        controller = ActiveInferenceController()
        sid = "hotel-1"
        a = controller.respond("帮我订酒店", session_id=sid)
        self.assertEqual(a.selected_action_type, ActionType.ASK_CLARIFICATION)
        controller.respond("北京", session_id=sid)            # 提供 city
        b = controller.respond("帮我订酒店", session_id=sid)
        self.assertEqual(b.selected_action_type, ActionType.ASK_CLARIFICATION)  # 仍缺 date
        controller.respond("明天", session_id=sid)            # 提供 date
        c = controller.respond("帮我订酒店", session_id=sid)
        self.assertEqual(c.selected_action_type, ActionType.ANSWER)            # 两槽位齐 → 回答
        self.assertEqual(c.trace.posterior_belief.known_slots.get("city"), "北京")
        self.assertEqual(c.trace.posterior_belief.known_slots.get("date"), "明天")

    def test_reminder_domain_requires_time(self) -> None:
        # 新槽位领域：提醒需要 time 槽位，验证槽位框架可直接扩到新领域。
        controller = ActiveInferenceController()
        sid = "rem-1"
        a = controller.respond("提醒我", session_id=sid)
        self.assertEqual(a.selected_action_type, ActionType.ASK_CLARIFICATION)
        controller.respond("明天8点", session_id=sid)
        b = controller.respond("提醒我", session_id=sid)
        self.assertEqual(b.selected_action_type, ActionType.ANSWER)
        self.assertIn("time", b.trace.posterior_belief.known_slots)

    def test_b2d_proactive_multislot_without_restating_domain(self) -> None:
        # B2d：领域未明示的跟进句——提供 city 后无需复述"订酒店"，controller 凭 active_domain
        # 主动追问缺失的 date；再补 date 后自动回答。这是 B2b/B2c 真实数据洞察的落地。
        controller = ActiveInferenceController()
        sid = "b2d-hotel"
        a = controller.respond("帮我订酒店", session_id=sid)
        self.assertEqual(a.selected_action_type, ActionType.ASK_CLARIFICATION)
        self.assertEqual(a.trace.posterior_belief.active_domain, "hotel")
        b = controller.respond("北京", session_id=sid)  # 仅提供 city，不复述领域
        self.assertEqual(b.selected_action_type, ActionType.ASK_CLARIFICATION)  # 凭 active_domain 知缺 date
        c = controller.respond("明天", session_id=sid)   # 仅提供 date，不复述领域
        self.assertEqual(c.selected_action_type, ActionType.ANSWER)             # 两槽位齐 → 回答
        self.assertEqual(c.trace.posterior_belief.known_slots.get("city"), "北京")
        self.assertEqual(c.trace.posterior_belief.known_slots.get("date"), "明天")

    def test_b2d_does_not_hijack_greeting(self) -> None:
        # B2d 门控回归：活跃领域存在时，寒暄/无槽位轮不被劫持成追问。
        controller = ActiveInferenceController()
        sid = "b2d-guard"
        controller.respond("帮我订酒店", session_id=sid)
        greet = controller.respond("你好", session_id=sid)
        self.assertNotEqual(greet.selected_action_type, ActionType.ASK_CLARIFICATION)

    def test_b2e_multi_domain_switch_followups_attach_to_active_domain(self) -> None:
        # B2e：端到端多域切换——切到新领域后，裸槽位值跟进句正确挂到「当前」活跃领域，
        # 而非被学习式 NLU 对裸值的误判（"上海"→booking）带偏。
        controller = ActiveInferenceController()
        sid = "b2e-switch"
        controller.respond("帮我订票", session_id=sid)
        controller.respond("北京到上海", session_id=sid)        # 完成 booking 的 route
        sw = controller.respond("帮我订酒店", session_id=sid)    # 切到 hotel
        self.assertEqual(sw.selected_action_type, ActionType.ASK_CLARIFICATION)
        self.assertEqual(sw.trace.posterior_belief.active_domain, "hotel")
        city = controller.respond("上海", session_id=sid)        # 裸 city，不复述领域
        # 关键：仍挂在 hotel（不被 NLU 误判成 booking→route 已知→answer），且因缺 date 仍追问。
        self.assertEqual(city.trace.posterior_belief.active_domain, "hotel")
        self.assertEqual(city.selected_action_type, ActionType.ASK_CLARIFICATION)
        done = controller.respond("后天", session_id=sid)        # 裸 date
        self.assertEqual(done.selected_action_type, ActionType.ANSWER)            # hotel 凑齐 city+date

    def test_multidomain_demo_dry_run(self) -> None:
        self.assertEqual(md_demo_main([]), 0)

    def test_multidomain_demo_key_actions(self) -> None:
        # 锁定 B2d/B2e 展示能力：不复述领域补全 + 领域切换 + 门控。
        tmp = os.path.join("docs", "reports", "_md_demo_test.md")
        result = md_demo_run(argparse.Namespace(session="md-demo-test", transcript=tmp))
        rows = result["transcript"]
        actions = [r["action"] for r in rows]
        domains = [r["active_domain"] for r in rows]
        self.assertEqual(actions[0], "ask_clarification")      # 订酒店 缺 city+date
        self.assertEqual(actions[1], "ask_clarification")      # 只给 city → 仍缺 date（B2d）
        self.assertEqual(domains[1], "hotel")
        self.assertEqual(actions[2], "answer")                 # 给 date → 凑齐（B2d，不复述领域）
        self.assertEqual(domains[3], "booking")                # 切到订票（B2e）
        self.assertEqual(actions[4], "answer")                 # 只给 route → booking 凑齐（不被误判回 hotel）
        self.assertEqual(actions[6], "refuse")                 # 高风险 → 拒答（门控）
        if os.path.exists(tmp):
            os.remove(tmp)

    def test_eval_dry_run(self) -> None:
        self.assertEqual(ctrl_eval_main([]), 0)

    def test_demo_dry_run(self) -> None:
        self.assertEqual(demo_main([]), 0)

    def test_cli_inputs_mode(self) -> None:
        self.assertEqual(cli_main(["--inputs", "帮我订票||北京到上海||帮我订票", "--session", "cli-t"]), 0)

    def test_cli_remembers_route(self) -> None:
        out = cli_run_inputs(["帮我订票", "北京到上海", "帮我订票"], "cli-mem")
        self.assertEqual(len(out), 3)
        self.assertIn("ask_clarification", out[0])
        self.assertIn("answer", out[2])  # 记住路线后同句改为回答

    def test_web_demo_dry_run(self) -> None:
        self.assertEqual(web_main([]), 0)

    def test_web_server_payload_remembers(self) -> None:
        controller = ActiveInferenceController(memory_candidate_path=None)
        first = respond_payload(controller, "帮我订票", "ws-1")
        self.assertEqual(first["action"], "ask_clarification")
        self.assertEqual(first["requires_slot"], "route")
        self.assertIn("channels", first)
        respond_payload(controller, "北京到上海", "ws-1")
        third = respond_payload(controller, "帮我订票", "ws-1")
        self.assertEqual(third["action"], "answer")  # 记住路线后改答

    def test_web_server_page_html(self) -> None:
        page = page_html()
        self.assertIn("<!DOCTYPE html>", page)
        self.assertIn("/api/respond", page)

    def test_real_validation_dry_run(self) -> None:
        self.assertEqual(realval_main([]), 0)

    def test_teacher_gen_dry_run(self) -> None:
        self.assertEqual(teacher_gen_main(["--dry-run"]), 0)

    def test_teacher_eval_dry_run(self) -> None:
        self.assertEqual(teacher_eval_main([]), 0)

    def test_task_nlu_dry_run(self) -> None:
        self.assertEqual(task_nlu_main([]), 0)

    def test_controller_with_context_policy(self) -> None:
        ckpt = os.path.join("checkpoints", "active_inference", "context_policy.pt")
        if not os.path.exists(ckpt):
            self.skipTest("context_policy checkpoint not available")
        controller = ActiveInferenceController(memory_candidate_path=None, context_policy_path=ckpt)
        sid = "cp-1"
        first = controller.respond("帮我订票", session_id=sid)
        self.assertEqual(first.selected_action_type, ActionType.ASK_CLARIFICATION)
        controller.respond("北京到上海", session_id=sid)
        controller.respond("明天", session_id=sid)
        final = controller.respond("帮我订票", session_id=sid)
        self.assertEqual(final.selected_action_type, ActionType.ANSWER)

    def test_build_html_renders(self) -> None:
        turns = [{
            "turn": 1, "input": "帮我订票", "intent": "缺信息→追问", "action": "ask_clarification",
            "surprise": 0.3, "channels": {"语义": 0.1, "意图": 0.2, "逻辑": 0.0, "不确定": 0.9, "安全": 0.0},
            "known_slots": {}, "requires_slot": "route", "recalled": [], "output": "信息不够",
        }]
        page = build_html(turns)
        self.assertIn("FE-LLM", page)
        self.assertIn("ask_clarification", page)
        self.assertIn("帮我订票", page)
        self.assertIn("<!DOCTYPE html>", page)

    def test_demo_key_actions(self) -> None:
        tmp = os.path.join("docs", "reports", "_demo_test_transcript.md")
        result = demo_run(argparse.Namespace(session="demo-test", transcript=tmp))
        actions = [row["action"] for row in result["transcript"]]
        self.assertEqual(actions[0], "ask_clarification")   # 帮我订票（缺路线）
        self.assertEqual(actions[2], "answer")              # 帮我订票（已记住路线）
        self.assertEqual(actions[3], "refuse")              # 教我做炸药
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    unittest.main()
