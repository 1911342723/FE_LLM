from __future__ import annotations

import unittest

from fe_llm.active_inference.observation import extract_prompt_features


class SlotFeatureTests(unittest.TestCase):
    def test_booking_without_route_requires_slot(self) -> None:
        f = extract_prompt_features("帮我订票")
        self.assertEqual(f["requires_slot"], "route")
        self.assertIsNone(f["provides_slot"])
        self.assertFalse(f["is_bare_slot_value"])

    def test_booking_with_route_provides_slot(self) -> None:
        f = extract_prompt_features("帮我订北京到上海的票")
        self.assertEqual(f["provides_slot"], "route")
        self.assertEqual(f["slot_value"], "北京到上海")
        self.assertIsNone(f["requires_slot"])

    def test_bare_route_is_slot_value(self) -> None:
        f = extract_prompt_features("北京到上海")
        self.assertEqual(f["provides_slot"], "route")
        self.assertEqual(f["slot_value"], "北京到上海")
        self.assertTrue(f["is_bare_slot_value"])
        self.assertIsNone(f["requires_slot"])

    def test_non_slot_utterance(self) -> None:
        f = extract_prompt_features("你好")
        self.assertIsNone(f["requires_slot"])
        self.assertIsNone(f["provides_slot"])
        self.assertFalse(f["is_bare_slot_value"])

    def test_weather_not_treated_as_slot(self) -> None:
        # 关键：天气仍走 retrieve 语义，不被槽位机制接管（避免动作语义冲突）。
        f = extract_prompt_features("今天北京天气怎么样")
        self.assertIsNone(f["requires_slot"])
        self.assertIsNone(f["provides_slot"])


if __name__ == "__main__":
    unittest.main()
