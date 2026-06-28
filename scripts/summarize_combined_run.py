#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_topk.scratch_pilot import read_manifest, summarize, write_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate aggregate summary and plots from merged per-seed metrics."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifest = read_manifest(run_dir)
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing merged metrics: {metrics_path}")
    final_step = max(int(row["checkpoint_step"]) for row in manifest)
    summary = summarize(run_dir, config, manifest, metrics_path, final_step)
    write_plots(run_dir, metrics_path)
    print(
        f"summarized combined run {run_dir} "
        f"completed={summary['completed_runs']} collapsed={summary['collapsed_runs']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
