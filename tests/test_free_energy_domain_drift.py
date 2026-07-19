from __future__ import annotations

import torch

from fe_llm.energy_lm.evaluation.free_energy_domain_drift_eval import (
    make_mixture,
    normalize_code,
)


def test_code_normalization_preserves_structure_but_removes_unicode() -> None:
    text = "def 加法(x_1: int):\n\treturn x_1 + 2  # 说明\n"
    normalized = normalize_code(text)

    assert "def " in normalized
    assert "(x_1: int)" in normalized
    assert "return x_1 + 2 #" in normalized
    assert "加" not in normalized and "说" not in normalized
    assert "\n" in normalized


def test_mixture_has_exact_requested_source_counts() -> None:
    base = torch.arange(40).view(10, 4)
    shifted = torch.arange(40, 80).view(10, 4)

    mixed, source = make_mixture(base, shifted, 0.3)

    assert mixed.shape == base.shape
    assert int((source == 0).sum()) == 7
    assert int((source == 1).sum()) == 3
    torch.testing.assert_close(mixed[:7], base[:7])
    torch.testing.assert_close(mixed[7:], shifted[:3])
