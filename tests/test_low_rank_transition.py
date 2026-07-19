from __future__ import annotations

import torch
import torch.nn as nn

from fe_llm.energy_lm.models.low_rank_transition import (
    IndexedLowRankGenerativeTransition,
    LowRankGenerativeTransition,
    LowRankReadout,
)


def test_low_rank_transition_starts_as_exact_base_and_counts_only_delta() -> None:
    torch.manual_seed(3)
    base = nn.Sequential(nn.Linear(12, 24), nn.Tanh(), nn.Linear(24, 12))
    low_rank = LowRankGenerativeTransition(base, dim=12, rank=3)
    state = torch.randn(5, 12)

    torch.testing.assert_close(low_rank(state), base(state))
    assert low_rank.added_parameter_count() == 2 * 12 * 3
    assert all(id(parameter) not in {id(p) for p in base.parameters()}
               for parameter in low_rank.parameters())


def test_training_delta_does_not_modify_frozen_base() -> None:
    torch.manual_seed(4)
    base = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 8))
    before = [parameter.detach().clone() for parameter in base.parameters()]
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    low_rank = LowRankGenerativeTransition(base, dim=8, rank=2)
    opt = torch.optim.Adam(low_rank.parameters(), lr=0.05)
    state = torch.randn(16, 8)
    target = torch.randn(16, 8)

    for _ in range(5):
        loss = (low_rank(state) - target).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()

    assert float(low_rank.up.weight.detach().abs().sum()) > 0
    for old, current in zip(before, base.parameters()):
        torch.testing.assert_close(current, old)


def test_low_rank_readout_starts_as_exact_base_and_counts_delta_only() -> None:
    torch.manual_seed(8)
    base = nn.Linear(10, 7)
    readout = LowRankReadout(base, in_dim=10, out_dim=7, rank=3)
    state = torch.randn(4, 10)

    torch.testing.assert_close(readout(state), base(state))
    assert readout.added_parameter_count() == 3 * (10 + 7)
    assert all(id(parameter) not in {id(p) for p in base.parameters()}
               for parameter in readout.parameters())


def test_indexed_low_rank_transition_matches_individual_deltas() -> None:
    torch.manual_seed(9)
    base = nn.Sequential(nn.Linear(6, 12), nn.Tanh(), nn.Linear(12, 6))
    first = LowRankGenerativeTransition(base, dim=6, rank=2)
    second = LowRankGenerativeTransition(base, dim=6, rank=2)
    with torch.no_grad():
        first.up.weight.normal_(std=0.1)
        second.up.weight.normal_(std=0.1)
    state = torch.randn(4, 6)
    down = torch.stack((first.down.weight, second.down.weight,
                        first.down.weight, second.down.weight))
    up = torch.stack((first.up.weight, second.up.weight,
                      first.up.weight, second.up.weight))
    scale = torch.stack((first.scale, second.scale, first.scale, second.scale))
    indexed = IndexedLowRankGenerativeTransition(base, down, up, scale)
    expected = torch.stack((
        first(state[0]), second(state[1]), first(state[2]), second(state[3])))

    torch.testing.assert_close(indexed(state), expected)
