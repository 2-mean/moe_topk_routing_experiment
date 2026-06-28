#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


COMPATIBILITY_KEYS = (
    "train_ks",
    "inference_ks",
    "train_steps_by_k",
    "seq_len",
    "vocab_size",
    "n_layers",
    "d_model",
    "n_heads",
    "n_experts",
    "expert_hidden",
    "dropout",
    "sparse_dispatch",
    "router_aux_loss_coef",
    "batch_size",
    "eval_batch_size",
    "data_seed",
    "data_order_seed",
    "collapse_threshold",
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def concatenate_csv(source_runs: list[Path], name: str, output_path: Path) -> int:
    fieldnames = None
    rows: list[dict[str, str]] = []
    for source_run in source_runs:
        path = source_run / name
        if not path.exists():
            continue
        current_fields, current_rows = read_csv(path)
        if fieldnames is None:
            fieldnames = current_fields
        elif current_fields != fieldnames:
            raise ValueError(f"CSV schema mismatch for {name}: {path}")
        rows.extend(current_rows)
    if fieldnames is None:
        return 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def link_artifacts(source_runs: list[Path], directory: str, output_run: Path) -> int:
    target_dir = output_run / directory
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for source_run in source_runs:
        source_dir = source_run / directory
        if not source_dir.exists():
            continue
        for source_path in sorted(source_dir.iterdir()):
            if not source_path.is_file():
                continue
            target_path = target_dir / source_path.name
            if target_path.exists() or target_path.is_symlink():
                raise ValueError(f"artifact name collision: {target_path.name}")
            target_path.symlink_to(source_path.resolve())
            count += 1
    return count


def compatible_config(source_runs: list[Path], experiment_name: str) -> dict[str, Any]:
    configs = [
        json.loads((source_run / "config.json").read_text(encoding="utf-8"))
        for source_run in source_runs
    ]
    reference = configs[0]
    for config in configs[1:]:
        for key in COMPATIBILITY_KEYS:
            if config.get(key) != reference.get(key):
                raise ValueError(f"incompatible config key {key}: {reference.get(key)!r} != {config.get(key)!r}")
    combined = dict(reference)
    combined["experiment_name"] = experiment_name
    combined["run_mode"] = "full"
    combined["save_final_checkpoints"] = False
    return combined


def build(source_runs: list[Path], output_run: Path, experiment_name: str) -> None:
    source_runs = [path.resolve() for path in source_runs]
    for source_run in source_runs:
        if not (source_run / "manifest.csv").exists():
            raise FileNotFoundError(f"missing source manifest: {source_run}")
    if output_run.exists():
        raise FileExistsError(f"output run already exists: {output_run}")

    temporary = output_run.with_name(f".{output_run.name}.building-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        config = compatible_config(source_runs, experiment_name)
        manifest_count = concatenate_csv(source_runs, "manifest.csv", temporary / "manifest.csv")
        concatenate_csv(source_runs, "metrics.csv", temporary / "metrics.csv")
        concatenate_csv(source_runs, "task_metrics.csv", temporary / "task_metrics.csv")
        concatenate_csv(source_runs, "train_log.csv", temporary / "train_log.csv")
        concatenate_csv(source_runs, "checkpoint_manifest.csv", temporary / "checkpoint_manifest.csv")

        _, manifest = read_csv(temporary / "manifest.csv")
        keys = [
            (
                int(row["seed"]),
                int(row["train_k"]),
                int(row["inference_k"]),
                int(row["checkpoint_step"]),
            )
            for row in manifest
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("combined manifest contains duplicate route keys")
        seeds = sorted({key[0] for key in keys})
        config["seeds"] = seeds
        (temporary / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        source_lines = [str(source_run) for source_run in source_runs]
        (temporary / "source_runs.txt").write_text(
            "\n".join(source_lines) + "\n", encoding="utf-8"
        )
        env_text = (source_runs[0] / "env.txt").read_text(encoding="utf-8")
        (temporary / "env.txt").write_text(
            env_text + "\ncombined_source_runs:\n" + "\n".join(source_lines) + "\n",
            encoding="utf-8",
        )
        route_count = link_artifacts(source_runs, "routes", temporary)
        checkpoint_count = link_artifacts(source_runs, "checkpoints", temporary)
        if route_count != manifest_count:
            raise ValueError(f"route count {route_count} != manifest rows {manifest_count}")
        output_run.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output_run)
    except Exception:
        if temporary.exists():
            for path in sorted(temporary.rglob("*"), reverse=True):
                if path.is_symlink() or path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            temporary.rmdir()
        raise

    print(
        f"built combined run {output_run} seeds={seeds} "
        f"manifest_rows={manifest_count} routes={route_count} checkpoints={checkpoint_count}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a symlinked combined run from disjoint seed runs.")
    parser.add_argument("--source-run", type=Path, action="append", required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    args = parser.parse_args()
    build(args.source_run, args.output_run, args.experiment_name)


if __name__ == "__main__":
    main()
