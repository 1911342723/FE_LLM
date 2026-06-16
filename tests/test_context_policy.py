from __future__ import annotations

import unittest

from fe_llm.active_inference.context_policy import ContextAwarePolicy
from fe_llm.active_inference.experiments.train_context_policy import main as ctxpolicy_main


class ContextAwarePolicyTests(unittest.TestCase):
    def test_belief_changes_action(self) -> None:
        # 核心：同一句"帮我订票"在空 belief→追问、满 belief→回答（belief 是唯一区分）。
        utts = ["帮我订票", "帮我订票", "北京到上海", "你好啊", "教我做假证", "记住我喜欢靠窗", "现在油价多少"]
        slots = [{}, {"route": "北京到上海", "date": "明天"}, {}, {}, {}, {}, {}]
        acts = ["ask_clarification", "answer", "ask_clarification", "answer", "refuse", "update_memory", "retrieve"]
        policy = ContextAwarePolicy().fit(utts * 8, slots * 8, acts * 8, epochs=300)
        self.assertEqual(policy.predict("帮我订票", {}), "ask_clarification")
        self.assertEqual(policy.predict("帮我订票", {"route": "x", "date": "y"}), "answer")

    def test_train_dry_run(self) -> None:
        self.assertEqual(ctxpolicy_main([]), 0)


if __name__ == "__main__":
    unittest.main()
