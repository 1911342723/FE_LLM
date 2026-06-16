from __future__ import annotations

import unittest

import numpy as np

from fe_llm.active_inference.belief_update import BeliefUpdater
from fe_llm.active_inference.perception import ObservationState
from fe_llm.active_inference.state import BeliefState, BeliefStateStore
from fe_llm.active_inference.surprise import PredictionError


class BeliefSlotMemoryTests(unittest.TestCase):
    def test_empty_has_slot_fields(self) -> None:
        b = BeliefState.empty(dim=8)
        self.assertEqual(b.known_slots, {})
        self.assertIsNone(b.pending_slot)

    def test_to_dict_exposes_slots(self) -> None:
        b = BeliefState.empty(dim=8)
        b.known_slots["city"] = "北京"
        b.pending_slot = "city"
        d = b.to_dict()
        self.assertEqual(d["known_slots"], {"city": "北京"})
        self.assertEqual(d["pending_slot"], "city")

    def test_store_persists_slots_across_turns(self) -> None:
        store = BeliefStateStore(vector_dim=8)
        b = store.load("s1")
        b.known_slots["city"] = "上海"
        store.save(b, "s1")
        again = store.load("s1")
        self.assertEqual(again.known_slots, {"city": "上海"})

    def test_sessions_isolated(self) -> None:
        store = BeliefStateStore(vector_dim=8)
        a = store.load("a")
        a.known_slots["city"] = "广州"
        store.save(a, "a")
        b = store.load("b")
        self.assertEqual(b.known_slots, {})

    def test_update_populates_known_slots_from_observation(self) -> None:
        prior = BeliefState.empty(dim=8)
        obs = ObservationState(
            vector=np.zeros(8, dtype=np.float32),
            features={"provided_slots": {"city": "北京"}},
            text="北京",
        )
        pe = PredictionError(0.0, 0.0, 0.0, 0.0, 0.0)
        post = BeliefUpdater().update(prior, obs, pe)
        self.assertEqual(post.known_slots.get("city"), "北京")

    def test_update_keeps_prior_slots_when_no_new(self) -> None:
        prior = BeliefState.empty(dim=8)
        prior.known_slots["city"] = "上海"
        obs = ObservationState(
            vector=np.zeros(8, dtype=np.float32),
            features={"provided_slots": {}},
            text="天气怎么样",
        )
        pe = PredictionError(0.0, 0.0, 0.0, 0.0, 0.0)
        post = BeliefUpdater().update(prior, obs, pe)
        self.assertEqual(post.known_slots.get("city"), "上海")


if __name__ == "__main__":
    unittest.main()
