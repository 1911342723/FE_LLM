from __future__ import annotations

import torch
import torch.nn.functional as F

from fe_llm.energy_lm.models.free_energy_lm import FreeEnergyLM


def _model(*, tolerance: float = 1e-6, steps: int = 8) -> FreeEnergyLM:
    torch.manual_seed(7)
    return FreeEnergyLM(
        vocab_size=29,
        max_len=12,
        dim=16,
        relaxation_steps=steps,
        tolerance=tolerance,
    )


def test_relaxation_monotonically_lowers_explicit_free_energy() -> None:
    net = _model().eval()
    ids = torch.randint(0, net.vocab_size, (3, 10))

    _, trace = net(ids, return_trace=True)
    energy = trace["free_energy"]

    assert len(energy) >= 2
    assert torch.all(energy[1:] <= energy[:-1] + 1e-5)
    assert energy[-1] < energy[0]


def test_future_tokens_cannot_change_past_logits_or_stopping_time() -> None:
    net = _model().eval()
    prefix = torch.tensor([[2, 4, 6, 8, 10]])
    first = torch.cat((prefix, torch.tensor([[1, 3, 5, 7]])), dim=1)
    second = torch.cat((prefix, torch.tensor([[9, 11, 13, 15]])), dim=1)

    logits_a, trace_a = net(first, return_trace=True)
    logits_b, trace_b = net(second, return_trace=True)

    torch.testing.assert_close(logits_a[:, : prefix.size(1)], logits_b[:, : prefix.size(1)])
    torch.testing.assert_close(
        trace_a["steps_per_position"][:, : prefix.size(1)],
        trace_b["steps_per_position"][:, : prefix.size(1)],
    )


def test_adaptive_stopping_is_local_and_respects_max_steps() -> None:
    net = _model(tolerance=1.0, steps=9).eval()
    ids = torch.randint(0, net.vocab_size, (2, 7))

    _, trace = net(ids, return_trace=True)
    steps = trace["steps_per_position"]

    assert int(steps.min()) >= 1
    assert int(steps.max()) < net.relaxation_steps
    assert 0.0 < float(trace["converged_fraction"]) <= 1.0


def test_language_loss_trains_shared_generative_transition() -> None:
    net = _model().train()
    ids = torch.randint(0, net.vocab_size, (4, 9))

    logits = net(ids)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, net.vocab_size), ids[:, 1:].reshape(-1))
    assert net.last_free_energy_loss is not None
    (loss + 0.1 * net.last_free_energy_loss).backward()

    first_transition = net.transition[0].weight
    last_transition = net.transition[2].weight
    assert first_transition.grad is not None
    assert float(first_transition.grad.abs().sum()) > 0
    assert last_transition.grad is not None
    assert float(last_transition.grad.abs().sum()) > 0


def test_outer_loop_free_energy_loss_is_differentiable() -> None:
    net = _model().train()
    ids = torch.randint(0, net.vocab_size, (3, 7))

    net(ids)
    assert net.last_free_energy_loss is not None
    assert net.last_position_free_energy is not None
    assert net.last_prediction_surprise is not None
    assert net.last_position_free_energy.shape == ids.shape
    assert net.last_prediction_surprise.shape == ids.shape
    net.last_free_energy_loss.backward()

    assert net.transition[0].weight.grad is not None
    assert float(net.transition[0].weight.grad.abs().sum()) > 0


def test_zero_step_ablation_removes_contextual_relaxation() -> None:
    net = _model().eval()
    ids = torch.randint(0, net.vocab_size, (2, 8))

    logits, trace = net(ids, return_trace=True, max_relax_steps=0)

    assert logits.shape == (2, 8, net.vocab_size)
    assert trace["free_energy"].numel() == 1
    assert int(trace["steps_per_position"].max()) == 0
    assert trace["max_relax_steps"] == 0


def test_checkpoint_round_trip_preserves_dynamics(tmp_path) -> None:
    net = _model(tolerance=2e-4, steps=5).eval()
    ids = torch.randint(0, net.vocab_size, (2, 6))
    expected = net(ids)
    path = tmp_path / "free_energy.pt"

    net.save(str(path), step=13)
    restored = FreeEnergyLM.load(str(path)).eval()

    assert restored.relaxation_steps == 5
    assert restored.tolerance == 2e-4
    torch.testing.assert_close(restored(ids), expected)
