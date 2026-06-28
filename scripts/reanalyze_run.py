from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_topk.scratch_pilot import analyze


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate metrics, summary, and plots for an existing run directory.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    config_path = args.config or run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    analyze(run_dir, config)
    print(f"reanalyzed {run_dir}")


if __name__ == "__main__":
    main()
