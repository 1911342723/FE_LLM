from __future__ import annotations

import os
import tempfile
import unittest

from fe_llm.active_inference.capcw_chain_memory import CAPCWChainMemory, ChainDecision
from fe_llm.active_inference.policy import ActionType


class CAPCWChainMemoryTests(unittest.TestCase):
    """CAPCW 多跳链式工作记忆：decode→re-embed 链式取回 + 首跳 surprise 驱动 ASK/ANSWER + 可溯源 CoT trace。"""

    @classmethod
    def setUpClass(cls) -> None:
        # 训一个 cot 链式工作记忆（固定 seed，可复现；配置对齐烟测可分离区间）。
        cls.mem = CAPCWChainMemory(n_sym=12, d=32, n_slots=6, ask_threshold=0.5, cot=True)
        cls.acc = cls.mem.train_on_chain(max_hops=2, n_pairs=4, n_train=4000, epochs=30, seed=0)

    def _bind_chain(self) -> None:
        # 链 0→1→2（+干扰边 5→6、7→8），凑满 n_pairs=4。绑定具体链=泛化（训练是随机链）。
        self.mem.reset()
        self.mem.bind(0, 1)
        self.mem.bind(1, 2)
        self.mem.bind(5, 6)
        self.mem.bind(7, 8)

    def test_training_learns_chaining(self) -> None:
        # 训练后多跳(链尾)取回应明显高于随机(1/12≈0.083)。
        self.assertGreater(self.acc, 0.5)

    def test_scripted_chain_retrieval(self) -> None:
        # 链式组合：从 0 起 2 跳 → 链尾 value=2，CoT trace=[1,2]，start 绑定→ANSWER。
        self._bind_chain()
        dec = self.mem.decide_chain(0, 2)
        self.assertEqual(dec.action, ActionType.ANSWER)
        self.assertTrue(dec.bound)
        self.assertEqual(dec.value, 2)
        self.assertEqual(dec.chain, [1, 2])

    def test_cot_trace_has_n_hops(self) -> None:
        # 可溯源：CoT trace（各跳解码的中间符号）长度=跳数。
        self._bind_chain()
        self.assertEqual(len(self.mem.decide_chain(0, 2).chain), 2)
        self.assertEqual(len(self.mem.decide_chain(0, 1).chain), 1)

    def test_single_hop_is_first_link(self) -> None:
        # sanity：只读 1 跳=取首边的 value（0→1），与多跳的首个中间符号一致。
        self._bind_chain()
        self.assertEqual(self.mem.decide_chain(0, 1).value, 1)

    def test_unbound_start_asks(self) -> None:
        # start 未绑定→起不了链→高 surprise→ASK（多跳版"知道何时不该答"）。
        self._bind_chain()
        bound = self.mem.decide_chain(0, 2)
        unbound = self.mem.decide_chain(9, 2)             # 9 不是任何边的 key
        self.assertEqual(unbound.action, ActionType.ASK_CLARIFICATION)
        self.assertFalse(unbound.bound)
        self.assertIsNone(unbound.value)
        self.assertGreater(unbound.surprise, bound.surprise)

    def test_empty_memory_asks(self) -> None:
        self.mem.reset()
        dec = self.mem.decide_chain(0, 2)
        self.assertEqual(dec.action, ActionType.ASK_CLARIFICATION)
        self.assertIsNone(dec.value)
        self.assertEqual(dec.surprise, 1.0)
        self.assertEqual(dec.chain, [])

    def test_decision_is_dataclass(self) -> None:
        self._bind_chain()
        dec = self.mem.decide_chain(0, 2)
        self.assertIsInstance(dec, ChainDecision)
        self.assertEqual(dec.n_hops, 2)
        self.assertEqual(len(dec.hop_match), 2)

    def test_bind_validation(self) -> None:
        self.mem.reset()
        with self.assertRaises(ValueError):
            self.mem.bind(99, 0)
        with self.assertRaises(ValueError):
            self.mem.bind(0, 99)

    def test_string_interface_chain(self) -> None:
        # 字符串接口：key/value 共享符号表，取回的 value 串能作下一跳 key。
        self.mem.reset()
        self.mem.bind_str("张三", "李四")
        self.mem.bind_str("李四", "王五")
        self.mem.bind_str("甲", "乙")
        self.mem.bind_str("丙", "丁")
        dec, val, chain = self.mem.decide_chain_str("张三", 2)
        self.assertEqual(dec.action, ActionType.ANSWER)
        self.assertEqual(val, "王五")
        self.assertEqual(chain, ["李四", "王五"])

    def test_unseen_str_start_trivially_asks(self) -> None:
        self.mem.reset()
        self.mem.bind_str("张三", "李四")
        dec, val, chain = self.mem.decide_chain_str("从未提过", 2)
        self.assertEqual(dec.action, ActionType.ASK_CLARIFICATION)
        self.assertIsNone(val)

    def test_bounded_working_memory_evicts_oldest(self) -> None:
        # 有界工作记忆：字符串词表满（>n_sym=12）时 FIFO 淘汰最旧符号，不崩；最近可取、最旧被忘。
        mem = CAPCWChainMemory(n_sym=8, d=32, n_slots=6, ask_threshold=0.5, cot=True)
        mem.train_on_chain(max_hops=2, n_pairs=4, n_train=2000, epochs=15, seed=0)
        mem.reset("b")
        for i in range(1, 6):                         # 5 条边=10 符号 > n_sym=8 → 必触发淘汰
            mem.bind_str(f"a{i}", f"v{i}", session_id="b")
        sess = mem._sess("b")
        self.assertLessEqual(len(sess["sym_ids"]), 8)  # 词表有界，不超 n_sym（不崩）
        self.assertNotIn("a1", sess["sym_ids"])        # 最旧被淘汰（FIFO）
        self.assertIn("a5", sess["sym_ids"])           # 最近的保留（未被淘汰）
        self.assertIn("v5", sess["sym_ids"])
        # 已忘的最旧 key → 平凡 unseen → ASK（不崩；不依赖取回训练质量）。
        d_old, val_old, _ = mem.decide_chain_str("a1", 1, session_id="b")
        self.assertEqual(d_old.action, ActionType.ASK_CLARIFICATION)
        self.assertIsNone(val_old)

    def test_session_isolation(self) -> None:
        # per-session 隔离：会话 s1 的链不串到 s2。
        self.mem.reset("*")
        self.mem.bind(0, 1, session_id="s1")
        self.mem.bind(1, 2, session_id="s1")
        s2 = self.mem.decide_chain(0, 2, session_id="s2")       # s2 空 → 该问
        self.assertEqual(s2.action, ActionType.ASK_CLARIFICATION)
        s1 = self.mem.decide_chain(0, 2, session_id="s1")       # s1 有链 → 该答
        self.assertEqual(s1.action, ActionType.ANSWER)

    def test_save_load_roundtrip(self) -> None:
        self._bind_chain()
        before = self.mem.decide_chain(0, 2)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "chain.pt")
            self.mem.save(path)
            loaded = CAPCWChainMemory.load(path)
        loaded.bind(0, 1)
        loaded.bind(1, 2)
        loaded.bind(5, 6)
        loaded.bind(7, 8)
        after = loaded.decide_chain(0, 2)
        self.assertEqual(before.value, after.value)
        self.assertEqual(before.chain, after.chain)


class ControllerChainHookTests(unittest.TestCase):
    """CAPCW 多跳链式工作记忆接回 controller：默认关零回归 + 加载后端到端链式裁决。"""

    @classmethod
    def setUpClass(cls) -> None:
        mem = CAPCWChainMemory(n_sym=12, d=32, n_slots=6, ask_threshold=0.5, cot=True)
        mem.train_on_chain(max_hops=2, n_pairs=4, n_train=4000, epochs=30, seed=0)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt = os.path.join(cls._tmp.name, "chain.pt")
        mem.save(cls.ckpt)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_default_controller_has_no_chain_memory(self) -> None:
        # 默认不加载 → 钩子全 no-op，既有管线零影响。
        from fe_llm.active_inference.controller import ActiveInferenceController

        controller = ActiveInferenceController()
        self.assertIsNone(controller.capcw_chain_memory)
        self.assertIsNone(controller.chain_working_memory_decision(0, 2))
        self.assertFalse(controller.bind_chain_working_memory(0, 1))

    def test_loaded_chain_memory_drives_controller(self) -> None:
        from fe_llm.active_inference.controller import ActiveInferenceController

        controller = ActiveInferenceController(capcw_chain_memory_path=self.ckpt)
        self.assertIsNotNone(controller.capcw_chain_memory)
        controller.reset_chain_working_memory()
        controller.bind_chain_working_memory(0, 1)
        controller.bind_chain_working_memory(1, 2)
        controller.bind_chain_working_memory(5, 6)
        controller.bind_chain_working_memory(7, 8)
        bound = controller.chain_working_memory_decision(0, 2)   # 链式组合 0→1→2
        self.assertEqual(bound.action, ActionType.ANSWER)
        self.assertEqual(bound.value, 2)
        self.assertEqual(bound.chain, [1, 2])                    # 可溯源 CoT trace
        unbound = controller.chain_working_memory_decision(9, 2)
        self.assertEqual(unbound.action, ActionType.ASK_CLARIFICATION)


if __name__ == "__main__":
    unittest.main()
