from __future__ import annotations

import unittest

import torch

from fe_llm.world_model.capcw import PCWorkspace, SequenceAdjacency, WorkspaceState


class PCWorkspaceTests(unittest.TestCase):
    """CAPCW 核心引擎 PCWorkspace 的不变量：自由能平复 + 可溯源 + 生长钩子。"""

    def _inputs(self, b=4, p=5, d=16, seed=0):
        torch.manual_seed(seed)
        # 构造 p 个明显不同的输入向量（每个样本一组），便于 slot 去解释。
        return torch.randn(b, p, d)

    def test_free_energy_decreases(self) -> None:
        # 自由能平复：弛豫后自由能应明显低于初始（感知=降自由能）。
        torch.manual_seed(0)
        ws = PCWorkspace(dim=16, n_slots=6, iters=5, alpha=0.5)
        x = self._inputs()
        state = ws(x)
        self.assertIsInstance(state, WorkspaceState)
        self.assertEqual(len(state.free_energy_trace), 6)  # iters+1
        self.assertLess(state.free_energy_trace[-1], state.free_energy_trace[0])
        # 末值应为非负标量。
        self.assertGreaterEqual(float(state.free_energy.detach()), 0.0)

    def test_shapes_and_traceability(self) -> None:
        ws = PCWorkspace(dim=16, n_slots=6, iters=3)
        x = self._inputs(b=3, p=5, d=16)
        state = ws(x)
        self.assertEqual(tuple(state.slots.shape), (3, 6, 16))
        self.assertEqual(tuple(state.responsibilities.shape), (3, 6, 5))   # (B,M,P)
        self.assertEqual(tuple(state.final_error.shape), (3, 5, 16))
        # responsibilities 是 over-slots 的 softmax：对每个输入 p，∑_m r=1。
        col_sum = state.responsibilities.sum(dim=1)                        # (B,P)
        self.assertTrue(torch.allclose(col_sum, torch.ones_like(col_sum), atol=1e-4))

    def test_more_slots_lower_free_energy(self) -> None:
        # 容量更大（更多 slot）应能把自由能降得更低（解释力更强）。
        x = self._inputs(b=4, p=8, d=16, seed=1)
        torch.manual_seed(1)
        ws_small = PCWorkspace(dim=16, n_slots=2, iters=5, alpha=0.5)
        torch.manual_seed(1)
        ws_big = PCWorkspace(dim=16, n_slots=10, iters=5, alpha=0.5)
        f_small = float(ws_small(x).free_energy)
        f_big = float(ws_big(x).free_energy)
        self.assertLess(f_big, f_small)

    def test_grow_hook_returns_in_range(self) -> None:
        # 穷则变钩子：返回的建议 slot 数在 [n_slots, max_slots] 内。
        ws = PCWorkspace(dim=16, n_slots=2, iters=4)
        x = self._inputs(b=2, p=8, d=16, seed=2)
        m = ws.grow_if_unexplained(x, threshold=1e-6, max_slots=8)  # 极小阈值 → 倾向生长
        self.assertGreaterEqual(m, 2)
        self.assertLessEqual(m, 8)

    def test_n_slots_override(self) -> None:
        # forward 可覆盖 slot 数（供生长）。
        ws = PCWorkspace(dim=16, n_slots=4, iters=2)
        x = self._inputs(b=2, p=6, d=16, seed=3)
        state = ws(x, n_slots=7)
        self.assertEqual(state.slots.shape[1], 7)


class SequenceAdjacencyTests(unittest.TestCase):
    """序列相邻算子（induction head 的 previous-token channel）：形状不变 + 位置0用BOS + 前驱敏感。"""

    def test_shape_preserved(self) -> None:
        torch.manual_seed(0)
        adj = SequenceAdjacency(dim=16)
        x = torch.randn(3, 7, 16)
        out = adj(x)
        self.assertEqual(tuple(out.shape), (3, 7, 16))

    def test_position0_uses_bos_not_later_tokens(self) -> None:
        # 位置 0 的前驱是可学 BOS，与后续 token 无关：改后续 token 不应改变 out[:,0]。
        torch.manual_seed(1)
        adj = SequenceAdjacency(dim=16)
        x = torch.randn(2, 6, 16)
        out = adj(x)
        x2 = x.clone()
        x2[:, 1:] = torch.randn(2, 5, 16)        # 只改位置 >=1
        out2 = adj(x2)
        self.assertTrue(torch.allclose(out[:, 0], out2[:, 0], atol=1e-6))

    def test_predecessor_sensitivity(self) -> None:
        # 相邻算子确实编码"前驱身份"：改 x[:,2] 应改变 out[:,3]（其前驱=位置2），
        # 但不改变 out[:,4]（前驱=位置3、当前=位置4 均未变）。
        torch.manual_seed(2)
        adj = SequenceAdjacency(dim=16)
        x = torch.randn(2, 6, 16)
        out = adj(x)
        x2 = x.clone()
        x2[:, 2] = x2[:, 2] + 1.0
        out2 = adj(x2)
        self.assertFalse(torch.allclose(out[:, 3], out2[:, 3], atol=1e-5))
        self.assertTrue(torch.allclose(out[:, 4], out2[:, 4], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
