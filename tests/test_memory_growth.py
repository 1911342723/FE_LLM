from __future__ import annotations

import unittest

from fe_llm.active_inference.experiments.memory_distill import distill_confirmed
from fe_llm.active_inference.experiments.memory_distill import main as distill_main
from fe_llm.active_inference.experiments.offline_retrain_eval import main as retrain_main
from fe_llm.active_inference.experiments.memory_growth_eval import main as growth_main
from fe_llm.active_inference.memory import MemoryCandidate, MemoryManager


class MemoryGrowthTests(unittest.TestCase):
    def test_audit_summary_promotes_repeated(self) -> None:
        mm = MemoryManager(candidate_path=None)
        for i in range(3):
            mm.candidates.append(MemoryCandidate(text="我喜欢简短", session_id=f"s{i}", reason="pref"))
        mm.candidates.append(MemoryCandidate(text="我住北京", session_id="x", reason="fact"))
        summary = mm.audit_summary(confirm_threshold=2)
        by_text = {row["text"]: row for row in summary}
        self.assertEqual(by_text["我喜欢简短"]["count"], 3)
        self.assertEqual(by_text["我喜欢简短"]["status"], "confirmed")
        self.assertEqual(by_text["我喜欢简短"]["distinct_sessions"], 3)
        self.assertEqual(by_text["我住北京"]["status"], "candidate")

    def test_audit_summary_confidence_capped(self) -> None:
        mm = MemoryManager(candidate_path=None)
        for i in range(5):
            mm.candidates.append(MemoryCandidate(text="重复偏好", session_id=f"s{i}", reason="pref"))
        row = mm.audit_summary(full_confidence_count=3)[0]
        self.assertEqual(row["confidence"], 1.0)

    def test_growth_eval_dry_run(self) -> None:
        self.assertEqual(growth_main([]), 0)

    def test_distill_only_confirmed(self) -> None:
        audit = [
            {"text": "我喜欢简短", "status": "confirmed", "count": 3, "confidence": 1.0, "distinct_sessions": 3},
            {"text": "我住北京", "status": "candidate", "count": 1, "confidence": 0.33, "distinct_sessions": 1},
        ]
        dataset = distill_confirmed(audit)
        texts = {row["prompt"] for row in dataset}
        self.assertIn("我喜欢简短", texts)
        self.assertNotIn("我住北京", texts)
        self.assertEqual(dataset[0]["action_type"], "update_memory")

    def test_distill_dry_run(self) -> None:
        self.assertEqual(distill_main([]), 0)

    def test_offline_retrain_dry_run(self) -> None:
        self.assertEqual(retrain_main([]), 0)


if __name__ == "__main__":
    unittest.main()
