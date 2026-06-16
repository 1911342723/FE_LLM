from __future__ import annotations

import unittest

from fe_llm.active_inference.experiments.contextual_action_eval import (
    WEATHER_NO_SLOT,
    generate_samples,
    main,
)


class ContextualActionTests(unittest.TestCase):
    def test_dry_run(self) -> None:
        self.assertEqual(main([]), 0)

    def test_same_utterance_depends_on_belief(self) -> None:
        # headroom 前提：同一句天气询问，已知城市→answer，未知城市→ask_clarification。
        samples = generate_samples(n_sessions=500, seed=0)
        weather = [s for s in samples if s["utterance"] in WEATHER_NO_SLOT]
        self.assertTrue(weather, "should generate weather-without-slot turns")
        known_actions = {s["action"] for s in weather if s["known_city"] == 1.0}
        unknown_actions = {s["action"] for s in weather if s["known_city"] == 0.0}
        self.assertIn("answer", known_actions)
        self.assertIn("ask_clarification", unknown_actions)

    def test_ambiguous_flagged(self) -> None:
        samples = generate_samples(n_sessions=300, seed=1)
        amb = [s for s in samples if s["ambiguous"]]
        self.assertTrue(amb, "ambiguous turns must exist (that is the headroom)")
        # 歧义轮次的同一文本应至少出现两种动作（只看当前句无法区分）。
        from collections import defaultdict

        text_actions = defaultdict(set)
        for s in amb:
            text_actions[s["utterance"]].add(s["action"])
        self.assertTrue(any(len(v) >= 2 for v in text_actions.values()))


if __name__ == "__main__":
    unittest.main()
