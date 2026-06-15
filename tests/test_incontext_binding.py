from __future__ import annotations

import os
import tempfile
import unittest

from fe_llm.active_inference.capcw_memory import CAPCWWorkingMemory
from fe_llm.active_inference.incontext_binding_nlu import InContextBindingNLU
from fe_llm.active_inference.policy import ActionType


class InContextBindingNLUTests(unittest.TestCase):
    """in-context 绑定 NLU：高精度解析 bind/query，寒暄/裸'X是Y'不触发，查询先于绑定消歧。"""

    def setUp(self) -> None:
        self.nlu = InContextBindingNLU()

    def test_bind_remember(self) -> None:
        e = self.nlu.parse("记住会议室是B302")
        self.assertEqual((e.kind, e.key, e.value), ("bind", "会议室", "B302"))

    def test_bind_correspond(self) -> None:
        e = self.nlu.parse("项目代号对应X9")
        self.assertEqual((e.kind, e.key, e.value), ("bind", "项目代号", "X9"))

    def test_bind_set_as(self) -> None:
        e = self.nlu.parse("门禁卡设成A05")
        self.assertEqual((e.kind, e.key, e.value), ("bind", "门禁卡", "A05"))

    def test_bind_attribute(self) -> None:
        e = self.nlu.parse("我的工号是1024")
        self.assertEqual((e.kind, e.key, e.value), ("bind", "我的工号", "1024"))

    def test_query_how_much(self) -> None:
        e = self.nlu.parse("会议室是多少")
        self.assertEqual((e.kind, e.key), ("query", "会议室"))

    def test_query_what(self) -> None:
        e = self.nlu.parse("项目代号是什么")
        self.assertEqual((e.kind, e.key), ("query", "项目代号"))

    def test_query_correspond(self) -> None:
        e = self.nlu.parse("项目代号对应什么")
        self.assertEqual((e.kind, e.key), ("query", "项目代号"))

    def test_query_before_bind_disambiguation(self) -> None:
        # "我的密码是多少" 既像 "X的{attr}是Y"(bind) 又是查询 → 查询优先。
        e = self.nlu.parse("我的密码是多少")
        self.assertEqual((e.kind, e.key), ("query", "我的密码"))

    def test_chitchat_not_triggered(self) -> None:
        for text in ("你好", "我有点累", "今天是周一", "谢谢你", "帮我订票"):
            self.assertEqual(self.nlu.parse(text).kind, "none", text)

    def test_empty_none(self) -> None:
        self.assertEqual(self.nlu.parse("").kind, "none")
        self.assertEqual(self.nlu.parse("   ").kind, "none")


class WorkingMemoryStringInterfaceTests(unittest.TestCase):
    """字符串接口：bind_str/decide_str 的 per-session str↔id 表与取回。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.wm = CAPCWWorkingMemory(n_keys=8, n_vals=10, d=32, n_slots=5, ask_threshold=0.5)
        cls.wm.train_on_binding(k_pairs=3, n_train=3000, epochs=25, seed=0)

    def test_unseen_key_trivially_asks(self) -> None:
        self.wm.reset()
        self.wm.bind_str("会议室", "B302")
        dec, val = self.wm.decide_str("从未提过的东西")
        self.assertEqual(dec.action, ActionType.ASK_CLARIFICATION)
        self.assertIsNone(val)

    def test_bound_str_answers_with_value(self) -> None:
        self.wm.reset()
        self.wm.bind_str("会议室", "B302")
        dec, val = self.wm.decide_str("会议室")
        self.assertEqual(dec.action, ActionType.ANSWER)
        self.assertEqual(val, "B302")

    def test_reset_clears_string_tables(self) -> None:
        self.wm.reset()
        self.wm.bind_str("会议室", "B302")
        self.wm.reset()
        self.assertEqual(len(self.wm._key_ids), 0)
        self.assertEqual(len(self.wm._val_ids), 0)


class ControllerInContextLoopTests(unittest.TestCase):
    """controller 活文本闭环：绑定 NLU→工作记忆→引擎 surprise 驱动 ASK/ANSWER+取回，不劫持寒暄。"""

    @classmethod
    def setUpClass(cls) -> None:
        wm = CAPCWWorkingMemory(n_keys=10, n_vals=12, d=32, n_slots=6, ask_threshold=0.5)
        wm.train_on_binding(k_pairs=4, n_train=4000, epochs=25, seed=0)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt = os.path.join(cls._tmp.name, "wm.pt")
        wm.save(cls.ckpt)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_default_controller_no_incontext(self) -> None:
        from fe_llm.active_inference.controller import ActiveInferenceController

        controller = ActiveInferenceController()       # 默认不加载工作记忆
        resp = controller.respond("记住会议室是B302", session_id="x")
        self.assertIsNone(resp.incontext_value)        # 工作记忆未加载 → 不触发，既有管线不受影响

    def test_live_text_bind_query_loop(self) -> None:
        from fe_llm.active_inference.controller import ActiveInferenceController

        controller = ActiveInferenceController(capcw_memory_path=self.ckpt)
        controller.reset_working_memory()
        controller.respond("记住会议室是B302", session_id="t")
        bound = controller.respond("会议室是多少", session_id="t")
        self.assertEqual(bound.selected_action_type, ActionType.ANSWER)
        self.assertEqual(bound.incontext_value, "B302")
        self.assertIn("B302", bound.text)              # grounded 生成：回答扎根于取回的 value
        unbound = controller.respond("门禁卡是多少", session_id="t")
        self.assertEqual(unbound.selected_action_type, ActionType.ASK_CLARIFICATION)
        chit = controller.respond("你好", session_id="t")   # 寒暄不被劫持
        self.assertIsNone(chit.incontext_value)

    def test_active_inference_surprise_drop_after_clarification(self) -> None:
        # 主动推理闭环：问未绑定(高 surprise→ASK) → 用户补绑定 → 再问(surprise 降→grounded 答)。
        # 体现蓝图"对外行动(追问)改变环境从而降低未来自由能"。
        from fe_llm.active_inference.controller import ActiveInferenceController

        controller = ActiveInferenceController(capcw_memory_path=self.ckpt)
        controller.reset_working_memory()
        ask = controller.respond("工号是多少", session_id="ai")          # 未绑定 → 该问
        self.assertEqual(ask.selected_action_type, ActionType.ASK_CLARIFICATION)
        controller.respond("记住工号是1024", session_id="ai")            # 用户补绑定（满足追问）
        ans = controller.respond("工号是多少", session_id="ai")          # 再问 → 该答
        self.assertEqual(ans.selected_action_type, ActionType.ANSWER)
        self.assertEqual(ans.incontext_value, "1024")
        self.assertIsNotNone(ask.incontext_surprise)
        self.assertIsNotNone(ans.incontext_surprise)
        self.assertLess(ans.incontext_surprise, ask.incontext_surprise)  # surprise 下降=自由能平复


if __name__ == "__main__":
    unittest.main()
