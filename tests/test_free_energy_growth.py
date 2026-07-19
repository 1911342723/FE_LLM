from __future__ import annotations

import torch

from fe_llm.energy_lm.free_energy_growth import (
    FreeEnergyGrowthSystem,
    StructuralFreeEnergyStabilizer,
)
from fe_llm.energy_lm.models.free_energy_lm import FreeEnergyLM
from fe_llm.energy_lm.models.low_rank_transition import LowRankGenerativeTransition


def _system() -> FreeEnergyGrowthSystem:
    torch.manual_seed(11)
    core = FreeEnergyLM(17, 10, dim=12, relaxation_steps=4, tolerance=1e-4)
    return FreeEnergyGrowthSystem(core)


def test_adding_pathway_preserves_frozen_base_behavior() -> None:
    system = _system().eval()
    ids = torch.randint(0, 17, (3, 8))
    before = system.forward_pathway(ids, 0)

    new_index = system.add_pathway(noise_std=0.01)
    after = system.forward_pathway(ids, 0)

    assert new_index == 1
    assert system.pathway_count == 2
    assert system.pathway_costs.shape == (2,)
    assert system.added_parameter_count() > 0
    torch.testing.assert_close(after, before)


def test_threshold_calibration_and_energy_routing_have_expected_shapes() -> None:
    system = _system().eval()
    ids = torch.randint(0, 17, (20, 9))
    threshold = system.calibrate_threshold(ids, quantile=0.9)
    system.add_pathway(noise_std=0.02)

    pressure, best = system.growth_pressure(ids)
    choices, scores = system.route(ids)

    assert threshold > 0
    assert pressure.shape == (20,)
    assert best.shape == (20,)
    assert choices.shape == (20,)
    assert scores.shape == (20, 2)
    assert int(choices.min()) >= 0 and int(choices.max()) < 2


def test_training_new_pathway_freezes_core_and_old_pathways() -> None:
    system = _system()
    new_index = system.add_pathway(noise_std=0.0)
    trainable = system.train_only_pathway(new_index)

    assert trainable
    assert all(parameter.requires_grad for parameter in trainable)
    assert all(not parameter.requires_grad for parameter in system.core.parameters())


def test_provisional_pathway_does_not_grow_until_committed() -> None:
    system = _system()
    provisional = system.create_provisional_pathway(noise_std=0.0)
    params = system.train_only_provisional(provisional)

    assert system.pathway_count == 1
    assert params and all(parameter.requires_grad for parameter in params)
    index = system.commit_pathway(provisional, complexity_cost=0.03)
    assert index == 1
    assert system.pathway_count == 2
    assert abs(float(system.pathway_costs[1]) - 0.03) < 1e-6


def test_complexity_cost_is_calibrated_from_false_advantage_on_stable_data() -> None:
    system = _system().eval()
    ids = torch.randint(0, 17, (24, 9))
    provisional = system.create_provisional_pathway(noise_std=0.02)

    cost = system.calibrate_complexity_cost(ids, provisional, quantile=0.9)

    assert cost >= 0


def test_committed_pathway_can_have_its_own_readout_without_changing_base() -> None:
    system = _system().eval()
    ids = torch.randint(0, 17, (3, 8))
    base_before = system.forward_pathway(ids, 0)
    transition = system.create_provisional_pathway(noise_std=0.0)
    head = system.create_provisional_head()
    with torch.no_grad():
        head.bias.add_(0.25)

    index = system.commit_pathway(transition, head=head)
    grown_logits = system.forward_pathway(ids, index)
    base_after = system.forward_pathway(ids, 0)

    assert not torch.allclose(grown_logits, base_before)
    torch.testing.assert_close(base_after, base_before)


def test_removing_pathway_reindexes_costs_and_private_heads_together() -> None:
    system = _system().eval()
    ids = torch.randint(0, 17, (3, 8))
    transitions = []
    for cost, shift in ((0.1, 0.1), (0.2, 0.2), (0.3, 0.3)):
        transition = system.create_provisional_pathway(noise_std=0.0)
        head = system.create_provisional_head()
        with torch.no_grad():
            head.bias.add_(shift)
        system.commit_pathway(transition, complexity_cost=cost, head=head)
        transitions.append(transition)

    old_last_logits = system.forward_pathway(ids, 3)
    mapping = system.remove_pathway(2)

    assert mapping == {0: 0, 1: 1, 3: 2}
    assert system.pathway_count == 3
    assert system.transition_for(2) is transitions[2]
    torch.testing.assert_close(system.pathway_costs, torch.tensor([0.0, 0.1, 0.3]))
    torch.testing.assert_close(system.forward_pathway(ids, 2), old_last_logits)


def test_energy_equivalent_duplicate_pathway_is_merged() -> None:
    system = _system().eval()
    ids = torch.randint(0, 17, (24, 9))
    first = system.add_pathway(noise_std=0.0)
    duplicate = system.add_pathway(noise_std=0.0)

    merged, audit = system.merge_pathways_if_redundant(
        first, duplicate, ids, energy_tolerance=1e-8)

    assert merged
    assert audit["covered_fraction"] == 1.0
    assert audit["survivor"] == first
    assert system.pathway_count == 2


def test_high_cost_inactive_pathway_is_recycled_without_changing_base() -> None:
    system = _system().eval()
    ids = torch.randint(0, 17, (24, 9))
    base_before = system.forward_pathway(ids, 0)
    provisional = system.create_provisional_pathway(noise_std=0.0)
    pathway = system.commit_pathway(provisional, complexity_cost=100.0)

    retired, audit = system.retire_pathway_if_inactive(
        pathway, ids, max_route_fraction=0.0)

    assert retired
    assert audit["route_fraction"] == 0.0
    assert system.pathway_count == 1
    torch.testing.assert_close(system.forward_pathway(ids, 0), base_before)


def test_base_pathway_cannot_be_removed() -> None:
    system = _system()
    try:
        system.remove_pathway(0)
    except ValueError as error:
        assert "基础通路" in str(error)
    else:
        raise AssertionError("remove_pathway(0) 应拒绝删除共享稳定态")


def test_structural_free_energy_is_monotonic_within_each_window() -> None:
    stabilizer = StructuralFreeEnergyStabilizer()
    _, _, trace = stabilizer.observe(0.7, return_trace=True)

    assert trace is not None
    energy = trace["free_energy"]
    assert isinstance(energy, torch.Tensor)
    assert torch.all(energy[1:] <= energy[:-1] + 1e-8)
    assert float(energy[-1]) < float(energy[0])


def test_single_burst_dissipates_but_sustained_instability_crosses_barrier() -> None:
    stabilizer = StructuralFreeEnergyStabilizer()
    for _ in range(5):
        stabilizer.observe(0.01)
    burst_state, burst_active, _ = stabilizer.observe(0.40)
    assert not burst_active and float(burst_state) < stabilizer.activation_barrier
    for _ in range(6):
        stabilizer.observe(0.01)
    assert float(stabilizer.state) < float(burst_state)

    activated = False
    for _ in range(10):
        _, activated, _ = stabilizer.observe(0.60)
        if activated:
            break
    assert activated


def test_structural_hysteresis_resets_only_below_lower_barrier() -> None:
    stabilizer = StructuralFreeEnergyStabilizer()
    stabilizer.reset(0.30, active=True)
    state, active, _ = stabilizer.observe(0.20)
    assert active and stabilizer.reset_barrier < float(state) < stabilizer.activation_barrier + 0.1

    for _ in range(20):
        state, active, _ = stabilizer.observe(0.0)
        if not active:
            break
    assert not active and float(state) <= stabilizer.reset_barrier


def test_free_energy_cascade_with_full_shortlist_matches_exhaustive_route() -> None:
    system = _system().eval()
    system.add_pathway(noise_std=0.01)
    system.add_pathway(noise_std=0.02)
    ids = torch.randint(0, 17, (12, 9))

    exhaustive_choices, exhaustive_scores = system.route(ids)
    choices, exact_scores, shortlist, screen_scores = system.route_cascade(
        ids, screen_relax_steps=1, shortlist_size=system.pathway_count)

    torch.testing.assert_close(choices, exhaustive_choices)
    torch.testing.assert_close(exact_scores, exhaustive_scores)
    assert shortlist.shape == screen_scores.shape == (12, system.pathway_count)


def test_free_energy_cascade_final_choice_always_comes_from_shortlist() -> None:
    system = _system().eval()
    for noise in (0.01, 0.02, 0.03):
        system.add_pathway(noise_std=noise)
    ids = torch.randint(0, 17, (16, 9))

    choices, exact_scores, shortlist, _ = system.route_cascade(
        ids, screen_relax_steps=1, shortlist_size=2)

    assert torch.all((shortlist == choices[:, None]).any(dim=1))
    assert torch.all(torch.isfinite(exact_scores.gather(1, shortlist)))
    assert torch.all(torch.isinf(exact_scores).sum(dim=1) == system.pathway_count - 2)


def test_batched_low_rank_scores_match_sequential_energy_solves() -> None:
    system = _system().eval()
    for rank_seed in range(3):
        torch.manual_seed(30 + rank_seed)
        pathway = LowRankGenerativeTransition(system.core.transition, dim=12, rank=3)
        with torch.no_grad():
            pathway.up.weight.normal_(std=0.03)
        system.commit_pathway(pathway, complexity_cost=0.001 * rank_seed)
    ids = torch.randint(0, 17, (10, 9))

    sequential = system.score_all(ids)
    batched = system.score_all_low_rank_batched(ids)
    cascade_choices, cascade_scores, _, _ = system.route_cascade(
        ids,
        screen_relax_steps=1,
        shortlist_size=system.pathway_count,
        batched_low_rank=True,
    )

    torch.testing.assert_close(batched, sequential, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(cascade_scores, sequential, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(cascade_choices, sequential.argmin(dim=1))
