from __future__ import annotations

import unittest

from fe_llm.active_inference.capcw_memory import CAPCWWorkingMemory, MemoryDecision
from fe_llm.active_inference.policy import ActionType


class CAPCWWorkingMemoryTests(unittest.TestCase):
    """CAPCW 内容寻址工作记忆：训练学会绑定取值 + 引擎 surprise 驱动 ASK/ANSWER。"""

    @classmethod
    def setUpClass(cls) -> None:
        # 训一个小工作记忆（固定 seed，快）：够学出内容寻址路由即可。
        cls.wm = CAPCWWorkingMemory(n_keys=6, n_vals=8, d=32, n_slots=5, ask_threshold=0.5)
        cls.acc = cls.wm.train_on_binding(k_pairs=3, n_train=3000, epochs=25, seed=0)

    def test_training_learns_binding(self) -> None:
        # 训练后绑定取值应明显高于随机(1/8=0.125)。
        self.assertGreater(self.acc, 0.5)

    def test_bound_lower_surprise_than_unbound(self) -> None:
        # 引擎 surprise 的核心：已绑定 key 的匹配度高(surprise 低)、未绑定 key 匹配度低(surprise 高)。
        self.wm.reset()
        self.wm.bind(2, 7)
        self.wm.bind(4, 3)
        self.wm.bind(1, 5)
        bound = self.wm.decide(2)
        unbound = self.wm.decide(0)            # key 0 未绑定
        self.assertLess(bound.surprise, unbound.surprise)
        self.assertGreater(bound.match, unbound.match)

    def test_bound_answers_unbound_asks(self) -> None:
        self.wm.reset()
        self.wm.bind(2, 7)
        self.wm.bind(4, 3)
        self.wm.bind(1, 5)
        bound = self.wm.decide(2)
        self.assertEqual(bound.action, ActionType.ANSWER)
        self.assertTrue(bound.bound)
        self.assertIsNotNone(bound.value)
        unbound = self.wm.decide(0)
        self.assertEqual(unbound.action, ActionType.ASK_CLARIFICATION)
        self.assertFalse(unbound.bound)

    def test_empty_memory_asks(self) -> None:
        self.wm.reset()
        dec = self.wm.decide(3)
        self.assertEqual(dec.action, ActionType.ASK_CLARIFICATION)
        self.assertIsNone(dec.value)
        self.assertEqual(dec.surprise, 1.0)

    def test_bind_overwrites_same_key(self) -> None:
        self.wm.reset()
        self.wm.bind(2, 7)
        self.wm.bind(2, 1)                     # 同 key 后写覆盖
        self.assertEqual(self.wm._bindings[2], 1)

    def test_bind_validation(self) -> None:
        self.wm.reset()
        with self.assertRaises(ValueError):
            self.wm.bind(99, 0)
        with self.assertRaises(ValueError):
            self.wm.bind(0, 99)

    def test_decision_is_dataclass(self) -> None:
        self.wm.reset()
        self.wm.bind(2, 7)
        dec = self.wm.decide(2)
        self.assertIsInstance(dec, MemoryDecision)
        self.assertIn(dec.action, (ActionType.ANSWER, ActionType.ASK_CLARIFICATION))

    def test_save_load_roundtrip(self) -> None:
        import os
        import tempfile

        self.wm.reset()
        self.wm.bind(2, 7)
        before = self.wm.decide(2)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "wm.pt")
            self.wm.save(path)
            loaded = CAPCWWorkingMemory.load(path)
        loaded.bind(2, 7)
        after = loaded.decide(2)
        self.assertEqual(before.action, after.action)
        self.assertAlmostEqual(before.match, after.match, places=4)


class ControllerWorkingMemoryHookTests(unittest.TestCase):
    """CAPCW 工作记忆接回 controller：默认关零回归 + 加载后经 controller 端到端裁决。"""

    def test_default_controller_has_no_working_memory(self) -> None:
        # 默认不加载 → 既有管线零影响（钩子全为 no-op）。
        from fe_llm.active_inference.controller import ActiveInferenceController

        controller = ActiveInferenceController()
        self.assertIsNone(controller.capcw_memory)
        self.assertIsNone(controller.working_memory_decision(0))
        self.assertFalse(controller.bind_working_memory(0, 0))

    def test_loaded_working_memory_drives_controller_decision(self) -> None:
        import os
        import tempfile

        from fe_llm.active_inference.controller import ActiveInferenceController

        wm = CAPCWWorkingMemory(n_keys=6, n_vals=8, d=32, n_slots=5, ask_threshold=0.5)
        wm.train_on_binding(k_pairs=3, n_train=3000, epochs=25, seed=0)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "wm.pt")
            wm.save(path)
            controller = ActiveInferenceController(capcw_memory_path=path)
            self.assertIsNotNone(controller.capcw_memory)
            controller.reset_working_memory()
            self.assertTrue(controller.bind_working_memory(2, 7))
            controller.bind_working_memory(4, 3)
            controller.bind_working_memory(1, 5)
            bound = controller.working_memory_decision(2)
            self.assertEqual(bound.action, ActionType.ANSWER)        # bound→该答
            unbound = controller.working_memory_decision(0)
            self.assertEqual(unbound.action, ActionType.ASK_CLARIFICATION)  # unbound→该问


if __name__ == "__main__":
    unittest.main()
