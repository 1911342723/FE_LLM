from __future__ import annotations

import unittest

import torch

from fe_llm.world_model import (
    HierarchicalIntentLM,
    HierarchicalPredictiveEncoder,
    HierarchicalState,
)
from fe_llm.world_model.m1_train_eval import main as m1_train_eval_main
from fe_llm.world_model.m1_judgments import main as m1_judgments_main
from fe_llm.world_model.action_select_eval import main as action_ablation_main


def _encoder(relax_steps: int = 5, alpha: float = 0.2) -> HierarchicalPredictiveEncoder:
    torch.manual_seed(0)
    return HierarchicalPredictiveEncoder(
        vocab_size=32,
        max_len=8,
        dim=32,
        n_heads=4,
        intent_dim=16,
        depth=2,
        relax_steps=relax_steps,
        alpha=alpha,
        precision=1.0,
    )


class HierarchicalPredictiveEncoderTests(unittest.TestCase):
    def test_output_shapes(self) -> None:
        enc = _encoder(relax_steps=4)
        ids = torch.randint(0, 32, (2, 5))
        state = enc(ids)
        self.assertIsInstance(state, HierarchicalState)
        self.assertEqual(state.z_global.shape, (2, 16))
        self.assertEqual(state.z_local.shape, (2, 5, 16))
        self.assertEqual(state.final_error.shape, (2, 5, 16))
        self.assertEqual(len(state.free_energy_trace), 5)  # relax_steps + 1

    def test_free_energy_monotonically_decreases(self) -> None:
        # 无 padding（mask 全 1）时弛豫是凸二次的精确梯度下降，F 应单调不增且整体下降。
        enc = _encoder(relax_steps=6, alpha=0.2)
        ids = torch.randint(0, 32, (2, 5))
        trace = enc(ids).free_energy_trace
        for earlier, later in zip(trace, trace[1:]):
            self.assertLessEqual(later, earlier + 1e-6)
        self.assertLess(trace[-1], trace[0])

    def test_relaxation_changes_global(self) -> None:
        enc = _encoder(relax_steps=5)
        ids = torch.randint(0, 32, (2, 5))
        no_relax = enc(ids, relax_steps=0).z_global
        relaxed = enc(ids, relax_steps=5).z_global
        self.assertFalse(torch.allclose(no_relax, relaxed))

    def test_zero_relax_returns_prior_mean(self) -> None:
        enc = _encoder(relax_steps=0)
        ids = torch.randint(0, 32, (2, 5))
        state = enc(ids, relax_steps=0)
        # 不弛豫时 z_global 应等于 z_local 的均值（先验），且 trace 只有初始 F。
        self.assertTrue(torch.allclose(state.z_global, state.z_local.mean(dim=1), atol=1e-5))
        self.assertEqual(len(state.free_energy_trace), 1)

    def test_mask_zeroes_padded_error(self) -> None:
        # M1 的 mask 作用于池化与误差聚合（不作用于注意力；padding 级注意力屏蔽留待后续）。
        # 这里验证 M1 真实提供的保证：padded 位置的预测误差被 mask 置零。
        enc = _encoder(relax_steps=3)
        ids = torch.randint(0, 32, (1, 6))
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
        state = enc(ids, attention_mask=mask)
        padded_error = state.final_error[:, 4:]
        self.assertTrue(torch.allclose(padded_error, torch.zeros_like(padded_error)))


class HierarchicalIntentLMTests(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        torch.manual_seed(0)
        lm = HierarchicalIntentLM(
            vocab_size=32, enc_max=8, dec_max=8, dim=32,
            enc_depth=2, dec_depth=2, n_heads=4, intent_dim=16, relax_steps=3,
        )
        prompt = torch.randint(0, 32, (2, 6))
        resp = torch.randint(0, 32, (2, 5))
        logits, h_intent, state = lm(prompt, resp)
        self.assertEqual(logits.shape, (2, 5, 32))
        self.assertEqual(h_intent.shape, (2, 5, 16))
        self.assertEqual(state.z_global.shape, (2, 16))
        self.assertEqual(state.z_local.shape, (2, 6, 16))

    def test_tiny_overfit_decreases_loss(self) -> None:
        import torch.nn.functional as F

        torch.manual_seed(0)
        lm = HierarchicalIntentLM(
            vocab_size=16, enc_max=8, dec_max=8, dim=32,
            enc_depth=2, dec_depth=2, n_heads=4, intent_dim=16, relax_steps=2,
        )
        prompt = torch.randint(0, 16, (1, 5))
        resp_in = torch.randint(0, 16, (1, 5))
        target = torch.randint(0, 16, (1, 5))
        opt = torch.optim.AdamW(lm.parameters(), lr=1e-2)
        first = last = None
        for step in range(60):
            logits, _, _ = lm(prompt, resp_in)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step == 0:
                first = float(loss.detach())
            last = float(loss.detach())
        # 整条图（弛豫展开 + 解码器）可导可训练：tiny overfit 损失应下降。
        self.assertLess(last, first)

    def test_m1_train_eval_dry_run(self) -> None:
        self.assertEqual(m1_train_eval_main([]), 0)

    def test_m1_judgments_dry_run(self) -> None:
        self.assertEqual(m1_judgments_main([]), 0)

    def test_action_ablation_dry_run(self) -> None:
        self.assertEqual(action_ablation_main([]), 0)


if __name__ == "__main__":
    unittest.main()
