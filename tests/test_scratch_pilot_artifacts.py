import unittest

import torch
from torch import nn

from moe_topk.scratch_pilot import checkpoint_state_dict, task_loss_rows


class _CheckpointFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moe = nn.Module()
        self.moe.router = nn.Linear(3, 2, bias=False)
        self.shared = nn.Linear(3, 3)


class ScratchPilotArtifactTest(unittest.TestCase):
    def test_task_loss_rows_are_ordered_and_normalized(self):
        rows = task_loss_rows(
            {"task_b": 9.0, "task_a": 4.0},
            {"task_b": 3, "task_a": 2},
            {"task_b": 1, "task_a": 0},
        )
        self.assertEqual([row["task_name"] for row in rows], ["task_a", "task_b"])
        self.assertEqual([row["validation_loss"] for row in rows], [2.0, 3.0])

    def test_checkpoint_state_dict_casts_and_filters_router(self):
        model = _CheckpointFixture()
        full = checkpoint_state_dict(model, torch.float16)
        self.assertTrue(full)
        self.assertTrue(all(tensor.dtype == torch.float16 for tensor in full.values()))

        router = checkpoint_state_dict(model, torch.float32, router_only=True)
        self.assertEqual(list(router), ["moe.router.weight"])
        self.assertEqual(router["moe.router.weight"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
