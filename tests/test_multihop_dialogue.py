from __future__ import annotations

import os
import tempfile
import unittest

from fe_llm.active_inference.capcw_chain_memory import CAPCWChainMemory
from fe_llm.active_inference.incontext_binding_nlu import MultiHopBindingNLU
from fe_llm.active_inference.policy import ActionType


class MultiHopBindingNLUTests(unittest.TestCase):
    """复合所有格多跳 NLU：识别 ≥2 跳查询、单跳/原子查询统一成 base+rels、复用单跳绑定，不误触寒暄。"""

    def setUp(self) -> None:
        self.nlu = MultiHopBindingNLU()

    def test_multihop_query(self) -> None:
        e = self.nlu.parse("A的经理的工位是多少")
        self.assertEqual((e.kind, e.base, e.rels), ("query", "A", ["经理", "工位"]))

    def test_three_hop_query(self) -> None:
        e = self.nlu.parse("甲的项目的经理的工位是多少")
        self.assertEqual((e.kind, e.base, e.rels), ("query", "甲", ["项目", "经理", "工位"]))

    def test_multihop_query_other_markers(self) -> None:
        self.assertEqual(self.nlu.parse("张三的项目的负责人是谁").rels, ["项目", "负责人"])
        self.assertEqual(self.nlu.parse("公司的老板的电话是什么").rels, ["老板", "电话"])

    def test_single_hop_query_split(self) -> None:
        # 单跳关系查询拆成 base+1 rel（统一由 decide_path_str 处理）。
        e = self.nlu.parse("A的经理是多少")
        self.assertEqual((e.kind, e.base, e.rels), ("query", "A", ["经理"]))

    def test_atomic_query_no_rels(self) -> None:
        e = self.nlu.parse("会议室是多少")
        self.assertEqual((e.kind, e.base, e.rels), ("query", "会议室", []))

    def test_bind_relational_remember(self) -> None:
        e = self.nlu.parse("记住A的经理是B")
        self.assertEqual((e.kind, e.key, e.value), ("bind", "A的经理", "B"))

    def test_query_marker_required(self) -> None:
        # 没有查询词收尾 → 不当作多跳查询（高精度，避免误触）。
        self.assertEqual(self.nlu.parse("A的经理的工位很不错").kind, "none")

    def test_chitchat_none(self) -> None:
        for text in ("你好", "我有点累", "谢谢你", "帮我订票"):
            self.assertEqual(self.nlu.parse(text).kind, "none", text)

    def test_open_relation_bind_markerless(self) -> None:
        # 开放关系绑定：免"记住"标记的 "X的R是Y"（恰 1 个 的 + 实体化 value）。
        e = self.nlu.parse("A的经理是张三")
        self.assertEqual((e.kind, e.key, e.value), ("bind", "A的经理", "张三"))
        e2 = self.nlu.parse("服务器的IP是10.0.0.7")
        self.assertEqual((e2.kind, e2.key, e2.value), ("bind", "服务器的IP", "10.0.0.7"))

    def test_open_relation_high_precision(self) -> None:
        # 窄触发：copula(今天是周一)/形容词谓语(…是好的/对的/充足的)/寒暄 都不应误绑。
        for text in ("今天是周一", "这个是错的", "今天的心情是好的", "项目的预算是充足的", "我觉得这样是对的"):
            self.assertEqual(self.nlu.parse(text).kind, "none", text)

    def test_open_relation_query_precedence(self) -> None:
        # 查询词收尾仍判 query（先于开放绑定）。
        self.assertEqual(self.nlu.parse("A的经理是多少").kind, "query")
        self.assertEqual(self.nlu.parse("A的经理的工位是多少").kind, "query")

    def test_open_relation_only_single_de(self) -> None:
        # 多个 的 的复合陈述不在开放绑定触发（保持原子关系边模型，复合需"记住"）。
        self.assertEqual(self.nlu.parse("A的项目的经理是张三").kind, "none")


class CAPCWChainPathTests(unittest.TestCase):
    """复合所有格链式取回 decide_path_str + controller 活文本多步推理（默认关零回归 + 端到端）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mem = CAPCWChainMemory(n_sym=20, d=32, n_slots=8, ask_threshold=0.5, cot=True)
        cls.mem.train_on_chain(max_hops=2, n_pairs=4, n_train=6000, epochs=45, seed=0)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt = os.path.join(cls._tmp.name, "chain_wm.pt")
        cls.mem.save(cls.ckpt)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # ---- decide_path_str（字符串层 decode→re-embed 链式取回）----
    def test_bound_path_chains_to_tail(self) -> None:
        self.mem.reset("s1")
        self.mem.bind_str("项目甲的经理", "张三", "s1")
        self.mem.bind_str("张三的工位", "B302", "s1")
        dec, val, trace = self.mem.decide_path_str("项目甲", ["经理", "工位"], "s1")
        self.assertEqual(dec.action, ActionType.ANSWER)
        self.assertEqual(val, "B302")
        self.assertEqual(trace, ["张三", "B302"])           # 可溯源 CoT trace

    def test_broken_chain_asks(self) -> None:
        # 第 2 跳无任何绑定 → 断链 → ASK（可溯源到断在 hop1 的中间值）。
        self.mem.reset("s2")
        self.mem.bind_str("项目甲的经理", "张三", "s2")
        dec, val, trace = self.mem.decide_path_str("项目甲", ["经理", "工位"], "s2")
        self.assertEqual(dec.action, ActionType.ASK_CLARIFICATION)
        self.assertIsNone(val)

    def test_unseen_base_asks(self) -> None:
        self.mem.reset("s3")
        self.mem.bind_str("项目甲的经理", "张三", "s3")
        dec, val, trace = self.mem.decide_path_str("未知部门", ["经理", "工位"], "s3")
        self.assertEqual(dec.action, ActionType.ASK_CLARIFICATION)
        self.assertEqual(trace, [])

    def test_single_hop_path(self) -> None:
        self.mem.reset("s4")
        self.mem.bind_str("项目甲的经理", "张三", "s4")
        dec, val, trace = self.mem.decide_path_str("项目甲", ["经理"], "s4")
        self.assertEqual(dec.action, ActionType.ANSWER)
        self.assertEqual(val, "张三")

    def test_atomic_path_direct_lookup(self) -> None:
        self.mem.reset("s5")
        self.mem.bind_str("会议室", "B302", "s5")
        dec, val, trace = self.mem.decide_path_str("会议室", [], "s5")
        self.assertEqual(dec.action, ActionType.ANSWER)
        self.assertEqual(val, "B302")

    # ---- controller 活文本多步推理 ----
    def test_default_controller_no_chain_path(self) -> None:
        from fe_llm.active_inference.controller import ActiveInferenceController

        controller = ActiveInferenceController()                 # 默认不加载链式工作记忆
        resp = controller.respond("A的经理的工位是多少", session_id="x")
        self.assertIsNone(resp.incontext_chain)                  # 未加载 → 不触发

    def test_live_text_multihop_loop(self) -> None:
        from fe_llm.active_inference.controller import ActiveInferenceController

        controller = ActiveInferenceController(capcw_chain_memory_path=self.ckpt)
        sid = "live"
        controller.reset_chain_working_memory(session_id=sid)
        controller.respond("记住项目甲的经理是张三", session_id=sid)
        controller.respond("记住张三的工位是B302", session_id=sid)
        resp = controller.respond("项目甲的经理的工位是多少", session_id=sid)
        self.assertEqual(resp.selected_action_type, ActionType.ANSWER)
        self.assertEqual(resp.incontext_value, "B302")
        self.assertEqual(resp.incontext_chain, ["张三", "B302"])   # 可溯源 CoT trace
        self.assertIn("B302", resp.text)                           # grounded 多跳生成
        chit = controller.respond("你好", session_id=sid)          # 寒暄不被劫持
        self.assertIsNone(chit.incontext_value)
        self.assertIsNone(chit.incontext_chain)

    def test_active_inference_surprise_drop_multihop(self) -> None:
        # 主动推理：断链(高 surprise→ASK) → 用户补绑缺失边 → 再问(链式取回→surprise 下降)。
        from fe_llm.active_inference.controller import ActiveInferenceController

        controller = ActiveInferenceController(capcw_chain_memory_path=self.ckpt)
        sid = "ai"
        controller.reset_chain_working_memory(session_id=sid)
        controller.respond("记住项目甲的经理是张三", session_id=sid)
        ask = controller.respond("项目甲的经理的工位是多少", session_id=sid)     # 张三的工位未绑 → 断链
        self.assertEqual(ask.selected_action_type, ActionType.ASK_CLARIFICATION)
        controller.respond("记住张三的工位是B302", session_id=sid)               # 补缺失边
        ans = controller.respond("项目甲的经理的工位是多少", session_id=sid)     # 再问 → 链式取回
        self.assertEqual(ans.selected_action_type, ActionType.ANSWER)
        self.assertEqual(ans.incontext_value, "B302")
        self.assertIsNotNone(ask.incontext_surprise)
        self.assertIsNotNone(ans.incontext_surprise)
        self.assertLess(ans.incontext_surprise, ask.incontext_surprise)          # surprise 下降=自由能平复

    def test_live_text_markerless_relation_multihop(self) -> None:
        # 开放关系绑定接活文本多跳：免"记住"标记直接陈述关系事实 → 复合所有格多跳取回。
        from fe_llm.active_inference.controller import ActiveInferenceController

        controller = ActiveInferenceController(capcw_chain_memory_path=self.ckpt)
        sid = "ml"
        controller.reset_chain_working_memory(session_id=sid)
        controller.respond("项目甲的经理是张三", session_id=sid)        # 免"记住"
        controller.respond("张三的工位是B302", session_id=sid)
        resp = controller.respond("项目甲的经理的工位是多少", session_id=sid)
        self.assertEqual(resp.selected_action_type, ActionType.ANSWER)
        self.assertEqual(resp.incontext_value, "B302")
        self.assertEqual(resp.incontext_chain, ["张三", "B302"])
        chit = controller.respond("今天的心情是好的", session_id=sid)    # 形容词谓语不被误绑/劫持
        self.assertIsNone(chit.incontext_chain)
        self.assertIsNone(chit.incontext_value)

    def test_coreference_resolution_multihop(self) -> None:
        # 指代消解：代词"他"回指上一条引入的实体(value)，自然录入链 + 多跳查询。
        from fe_llm.active_inference.controller import ActiveInferenceController

        controller = ActiveInferenceController(capcw_chain_memory_path=self.ckpt)
        sid = "coref"
        controller.reset_chain_working_memory(session_id=sid)
        controller.respond("记住张三的经理是李四", session_id=sid)         # 引入李四(value)
        r2 = controller.respond("他的工位是C302", session_id=sid)          # 他=李四 → 李四的工位=C302
        self.assertIn("李四的工位", r2.text)                               # 代词已被替换为李四
        ans = controller.respond("张三的经理的工位是多少", session_id=sid)  # 张三的经理=李四→李四的工位=C302
        self.assertEqual(ans.selected_action_type, ActionType.ANSWER)
        self.assertEqual(ans.incontext_value, "C302")
        self.assertEqual(ans.incontext_chain, ["李四", "C302"])


if __name__ == "__main__":
    unittest.main()
