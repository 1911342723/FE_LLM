from __future__ import annotations

import unittest

from fe_llm.active_inference.experiments.nlu_intent_eval import keyword_intent
from fe_llm.active_inference.experiments.nlu_intent_eval import main as nlu_eval_main
from fe_llm.active_inference.experiments.nlu_value_eval import main as value_eval_main
from fe_llm.active_inference.nlu.slot_intent_nlu import INTENT_REQUIRED_SLOTS, SlotIntentNLU
from fe_llm.active_inference.nlu.slot_value_tagger import SlotValueTagger
from fe_llm.active_inference.observation import extract_prompt_features, set_learned_nlu


class SlotIntentNLUTests(unittest.TestCase):
    def test_overfits_training(self) -> None:
        texts = ["帮我订票", "买票", "你好啊", "讲个笑话", "提醒我开会", "定个闹钟", "订酒店", "订间房"]
        labels = ["booking", "booking", "none", "none", "reminder", "reminder", "hotel", "hotel"]
        nlu = SlotIntentNLU().fit(texts, labels, epochs=300)
        for text, label in zip(texts, labels):
            self.assertEqual(nlu.predict(text), label)

    def test_required_slots_mapping(self) -> None:
        self.assertEqual(INTENT_REQUIRED_SLOTS["hotel"], ["city", "date"])
        self.assertEqual(INTENT_REQUIRED_SLOTS["booking"], ["route"])
        self.assertEqual(INTENT_REQUIRED_SLOTS["none"], [])

    def test_keyword_baseline(self) -> None:
        self.assertEqual(keyword_intent("帮我订票"), "booking")
        self.assertEqual(keyword_intent("订酒店"), "hotel")
        self.assertEqual(keyword_intent("提醒我"), "reminder")
        self.assertEqual(keyword_intent("你好"), "none")

    def test_eval_dry_run(self) -> None:
        self.assertEqual(nlu_eval_main([]), 0)

    def test_value_tagger_extracts_trained_span(self) -> None:
        texts = ["提醒我8点开会", "北京天气", "帮我订明天的酒店"]
        labels = [
            [0, 0, 0, 3, 3, 0, 0],          # 8点 -> TIME
            [1, 1, 0, 0],                   # 北京 -> CITY
            [0, 0, 0, 2, 2, 0, 0, 0],       # 明天 -> DATE
        ]
        tagger = SlotValueTagger().fit(texts, labels, epochs=300)
        spans = tagger.extract_spans("提醒我8点开会")
        self.assertIn(("TIME", "8点"), spans)

    def test_value_eval_dry_run(self) -> None:
        self.assertEqual(value_eval_main([]), 0)

    def test_perception_uses_injected_nlu(self) -> None:
        nlu = SlotIntentNLU().fit(
            ["想坐高铁去北京", "帮我订票", "买票", "你好啊", "讲个笑话", "订酒店", "提醒我开会"],
            ["booking", "booking", "booking", "none", "none", "hotel", "reminder"],
            epochs=400,
        )
        # 未注入：关键词命中不到该改写 → 无 required_slot
        set_learned_nlu(None)
        self.assertIsNone(extract_prompt_features("想坐高铁去北京")["requires_slot"])
        # 注入后：学习式补出 booking → route；none 不误触
        set_learned_nlu(nlu, conf_threshold=0.3)
        try:
            self.assertEqual(extract_prompt_features("想坐高铁去北京")["requires_slot"], "route")
            self.assertIsNone(extract_prompt_features("你好啊")["requires_slot"])
        finally:
            set_learned_nlu(None)  # 复位，避免污染其它用例


if __name__ == "__main__":
    unittest.main()
