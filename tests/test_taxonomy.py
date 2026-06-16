from __future__ import annotations

import unittest

from fe_llm.active_inference.experiments.teacher_corpus_gen import DOMAINS
from fe_llm.active_inference.nlu import taxonomy
from fe_llm.active_inference.nlu.slot_intent_nlu import INTENT_REQUIRED_SLOTS, INTENTS
from fe_llm.active_inference.observation import extract_prompt_features


class TaxonomyConsistencyTests(unittest.TestCase):
    """统一 taxonomy 的回归锁：保证单一真相源不漂、行为保持、差异显式。"""

    def test_legacy_values_unchanged(self) -> None:
        # 行为保持：slot_intent_nlu 的 required_slots 与历史值完全一致（重导出自 taxonomy）。
        self.assertIs(INTENT_REQUIRED_SLOTS, taxonomy.LEGACY_REQUIRED_SLOTS)
        self.assertEqual(INTENT_REQUIRED_SLOTS["none"], [])
        self.assertEqual(INTENT_REQUIRED_SLOTS["booking"], ["route"])
        self.assertEqual(INTENT_REQUIRED_SLOTS["hotel"], ["city", "date"])
        self.assertEqual(INTENT_REQUIRED_SLOTS["reminder"], ["time"])

    def test_legacy_intents_cover_required_slots(self) -> None:
        # INTENTS 与 legacy required_slots 的键一一对应（顺序锁定单独保证）。
        self.assertEqual(set(INTENTS), set(taxonomy.LEGACY_REQUIRED_SLOTS))
        self.assertEqual(INTENTS, ["none", "booking", "hotel", "reminder"])

    def test_canonical_matches_teacher_domains(self) -> None:
        # canonical 必须与教师语料 9 领域的槽位逐一致（任何一边改动都会被这条测试抓到）。
        self.assertEqual(set(taxonomy.CANONICAL_DOMAINS), set(DOMAINS))
        for domain, spec in DOMAINS.items():
            self.assertEqual(
                taxonomy.required_slots(domain),
                list(spec["slots"]),
                msg=f"canonical 与教师 {domain} 槽位不一致",
            )

    def test_hotel_exact_alignment(self) -> None:
        # hotel 是 legacy 与 canonical 完全一致的领域（无简化）。
        self.assertEqual(taxonomy.legacy_required_slots("hotel"), taxonomy.required_slots("hotel"))
        self.assertEqual(taxonomy.legacy_required_slots("hotel"), ["city", "date"])

    def test_legacy_to_canonical_mapping(self) -> None:
        self.assertIsNone(taxonomy.legacy_to_canonical("none"))
        self.assertEqual(taxonomy.legacy_to_canonical("booking"), "flight")
        self.assertEqual(taxonomy.legacy_to_canonical("hotel"), "hotel")
        self.assertEqual(taxonomy.legacy_to_canonical("reminder"), "repair")
        # 非 none 的 legacy 意图都映射到已知 canonical 领域。
        for intent, domain in taxonomy.LEGACY_INTENT_TO_DOMAIN.items():
            if domain is not None:
                self.assertTrue(taxonomy.is_known_domain(domain), msg=f"{intent}→{domain} 非法")

    def test_simplifications_are_documented_subsets(self) -> None:
        # 每条「已知简化」必须：legacy 槽位是 canonical 的真子集，且记录值自洽。
        for intent, info in taxonomy.LEGACY_SIMPLIFICATIONS.items():
            self.assertEqual(info["legacy_slots"], taxonomy.legacy_required_slots(intent))
            self.assertEqual(info["canonical_slots"], taxonomy.required_slots(info["canonical_domain"]))
            self.assertTrue(
                set(info["legacy_slots"]).issubset(set(info["canonical_slots"])),
                msg=f"{intent} legacy 槽位不是 canonical 子集",
            )
            self.assertLess(
                len(info["legacy_slots"]),
                len(info["canonical_slots"]),
                msg=f"{intent} 标了简化但槽位并未减少",
            )

    def test_non_simplified_legacy_equals_canonical(self) -> None:
        # 未列入简化表的 legacy 意图（none/hotel），其槽位应与 canonical 完全相等。
        for intent in taxonomy.LEGACY_REQUIRED_SLOTS:
            if intent in taxonomy.LEGACY_SIMPLIFICATIONS or intent == "none":
                continue
            domain = taxonomy.legacy_to_canonical(intent)
            self.assertEqual(taxonomy.legacy_required_slots(intent), taxonomy.required_slots(domain))

    def test_slot_vocab_and_gaps(self) -> None:
        union = sorted({s for slots in taxonomy.CANONICAL_DOMAINS.values() for s in slots})
        self.assertEqual(taxonomy.SLOT_VOCAB, union)
        # 规则可抽取槽位必须都在词表内。
        self.assertTrue(set(taxonomy.RULE_EXTRACTABLE_SLOTS).issubset(set(taxonomy.SLOT_VOCAB)))
        # gap = 词表 − 规则可抽取（开放实体/数值，规则层抽不出，需学习式/gazetteer）。
        self.assertEqual(taxonomy.RULE_GAP_SLOTS, sorted(set(taxonomy.SLOT_VOCAB) - set(taxonomy.RULE_EXTRACTABLE_SLOTS)))
        self.assertIn("addr", taxonomy.RULE_GAP_SLOTS)
        self.assertNotIn("route", taxonomy.RULE_GAP_SLOTS)

    def test_perception_required_slots_derive_from_canonical(self) -> None:
        # B1 阶段二（安全部分）：感知层关键词领域的 required_slots 派生自统一 taxonomy。
        # 单一真相源贯通 感知/NLU/教师 三层，任一漂移这条测试抓到。
        cases = {"帮我订票": "booking", "帮我订酒店": "hotel", "提醒我": "reminder"}
        for utterance, intent in cases.items():
            with self.subTest(utterance=utterance):
                feats = extract_prompt_features(utterance)
                self.assertEqual(feats["keyword_domain"], intent)
                self.assertEqual(feats["required_slots"], taxonomy.legacy_required_slots(intent))

    def test_helpers_return_copies(self) -> None:
        # required_slots / legacy_required_slots 返回拷贝，外部 mutate 不污染真相源。
        a = taxonomy.required_slots("flight")
        a.append("x")
        self.assertEqual(taxonomy.required_slots("flight"), ["route", "date"])
        b = taxonomy.legacy_required_slots("hotel")
        b.append("y")
        self.assertEqual(taxonomy.legacy_required_slots("hotel"), ["city", "date"])
        self.assertEqual(taxonomy.canonical_domains(), sorted(taxonomy.CANONICAL_DOMAINS))


if __name__ == "__main__":
    unittest.main()
