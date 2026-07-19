from __future__ import annotations

import torch

from fe_llm.energy_lm.free_energy_growth import FreeEnergyGrowthSystem
from fe_llm.energy_lm.models.free_energy_lm import FreeEnergyLM


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
