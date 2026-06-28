import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_combined_run import build


class BuildCombinedRunTest(unittest.TestCase):
    def make_source(self, root: Path, seed: int) -> Path:
        source = root / f"source_{seed}"
        (source / "routes").mkdir(parents=True)
        (source / "checkpoints").mkdir()
        config = {
            "experiment_name": f"source_{seed}",
            "seeds": [seed],
            "train_ks": [1],
            "inference_ks": [1],
            "train_steps_by_k": {"1": 10},
            "seq_len": 4,
            "vocab_size": 8,
            "n_layers": 1,
            "d_model": 4,
            "n_heads": 1,
            "n_experts": 2,
            "expert_hidden": 8,
            "dropout": 0.0,
            "sparse_dispatch": True,
            "router_aux_loss_coef": 0.0,
            "batch_size": 1,
            "eval_batch_size": 1,
            "data_seed": 1,
            "data_order_seed": 2,
            "collapse_threshold": 0.9,
        }
        (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (source / "env.txt").write_text("test_env\n", encoding="utf-8")
        route_name = f"seed{seed}_train1_infer1_step10.npz"
        (source / "routes" / route_name).write_bytes(b"route")
        with (source / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["seed", "train_k", "inference_k", "checkpoint_step", "route_file"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "seed": seed,
                    "train_k": 1,
                    "inference_k": 1,
                    "checkpoint_step": 10,
                    "route_file": f"routes/{route_name}",
                }
            )
        with (source / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["seed", "metric_name", "value"])
            writer.writeheader()
            writer.writerow({"seed": seed, "metric_name": "test", "value": seed + 1})
        return source

    def test_combines_disjoint_seeds_without_modifying_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_0 = self.make_source(root, 0)
            source_1 = self.make_source(root, 1)
            output = root / "combined"
            with patch("pathlib.Path.symlink_to") as symlink_to:
                build([source_0, source_1], output, "combined_2seed")

            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["seeds"], [0, 1])
            self.assertEqual(config["experiment_name"], "combined_2seed")
            with (output / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["seed"]) for row in rows], [0, 1])
            with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                metric_rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["seed"]) for row in metric_rows], [0, 1])
            self.assertEqual(symlink_to.call_count, 2)
            self.assertTrue((source_0 / "routes" / "seed0_train1_infer1_step10.npz").exists())


if __name__ == "__main__":
    unittest.main()
