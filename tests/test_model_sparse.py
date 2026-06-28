import unittest

import torch

from moe_topk.model import TopKMoE


class SparseMoETest(unittest.TestCase):
    def test_sparse_matches_dense_without_dropout(self):
        torch.manual_seed(7)
        sparse = TopKMoE(d_model=12, n_experts=8, expert_hidden=24, dropout=0.0, sparse_dispatch=True)
        dense = TopKMoE(d_model=12, n_experts=8, expert_hidden=24, dropout=0.0, sparse_dispatch=False)
        dense.load_state_dict(sparse.state_dict())
        sparse.eval()
        dense.eval()

        x = torch.randn(3, 5, 12)
        sparse_out, sparse_aux, sparse_route = sparse(x, top_k=3, collect_routes=True)
        dense_out, dense_aux, dense_route = dense(x, top_k=3, collect_routes=True)

        torch.testing.assert_close(sparse_out, dense_out, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(sparse_aux, dense_aux, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(sparse_route["gate_logits"], dense_route["gate_logits"])
        torch.testing.assert_close(sparse_route["selected_ids"], dense_route["selected_ids"])
        torch.testing.assert_close(sparse_route["selected_weights"], dense_route["selected_weights"])

    def test_sparse_backward_runs(self):
        torch.manual_seed(11)
        moe = TopKMoE(d_model=10, n_experts=6, expert_hidden=20, dropout=0.0, sparse_dispatch=True)
        x = torch.randn(4, 6, 10, requires_grad=True)
        out, aux, _ = moe(x, top_k=2, collect_routes=False)
        loss = out.pow(2).mean() + 0.01 * aux
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(moe.router.weight.grad)
        self.assertGreater(float(moe.router.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
