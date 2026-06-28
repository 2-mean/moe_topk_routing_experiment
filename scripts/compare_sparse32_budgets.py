#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def final_steps(metrics: list[dict[str, str]]) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for row in metrics:
        if row["metric_type"] != "loss" or row["metric_name"] != "validation_loss":
            continue
        seed = int(row["seed"])
        train_k = int(row["train_k"])
        step = int(row["checkpoint_step"])
        result[(seed, train_k)] = max(result.get((seed, train_k), -1), step)
    return result


def final_pair_map(metrics: list[dict[str, str]]) -> dict[tuple[int, str, str], float]:
    return {
        (int(row["seed"]), row["pair"], row["metric_name"]): float(row["value"])
        for row in metrics
        if row["metric_type"] == "matched_train_k_final_pair"
    }


def final_mismatch_map(metrics: list[dict[str, str]]) -> dict[tuple[int, int, int], float]:
    finals = final_steps(metrics)
    result = {}
    for row in metrics:
        if row["metric_type"] != "mismatch_cost" or row["metric_name"] != "validation_loss_delta":
            continue
        seed = int(row["seed"])
        train_k = int(row["train_k"])
        inference_k = int(row["inference_k"])
        if int(row["checkpoint_step"]) == finals[(seed, train_k)]:
            result[(seed, train_k, inference_k)] = float(row["value"])
    return result


def keyed_map(rows: list[dict[str, str]], keys: list[str], value: str = "value") -> dict[tuple[Any, ...], float]:
    result = {}
    for row in rows:
        converted = []
        for key in keys:
            converted.append(int(row[key]) if key in {"seed", "train_k", "layer_id"} else row[key])
        result[tuple(converted)] = float(row[value])
    return result


def paired_rows(
    same: dict[tuple[Any, ...], float],
    fixed: dict[tuple[Any, ...], float],
    key_names: list[str],
) -> list[dict[str, Any]]:
    if set(same) != set(fixed):
        missing_same = sorted(set(fixed) - set(same))[:5]
        missing_fixed = sorted(set(same) - set(fixed))[:5]
        raise ValueError(f"budget keys differ: missing_same={missing_same}, missing_fixed={missing_fixed}")
    return [
        {
            **dict(zip(key_names, key)),
            "same_compute": same[key],
            "fixed_step": fixed[key],
            "fixed_minus_same": fixed[key] - same[key],
        }
        for key in sorted(same)
    ]


def mean_delta(rows: list[dict[str, Any]], **filters: Any) -> float:
    values = [
        float(row["fixed_minus_same"])
        for row in rows
        if all(row[key] == value for key, value in filters.items())
    ]
    return float(np.mean(values)) if values else float("nan")


def draw_pair_heatmap(rows: list[dict[str, Any]], metric: str, out_path: Path) -> None:
    matrix = np.full((8, 8), np.nan, dtype=np.float64)
    for small in range(1, 9):
        for large in range(small + 1, 9):
            matrix[small - 1, large - 1] = mean_delta(rows, pair=f"{small}-{large}", metric_name=metric)
    vmax = float(np.nanmax(np.abs(matrix)))
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    image = ax.imshow(matrix, vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax.set_xticks(range(8), labels=range(1, 9))
    ax.set_yticks(range(8), labels=range(1, 9))
    ax.set_xlabel("larger training k")
    ax.set_ylabel("smaller training k")
    ax.set_title(f"Fixed-step minus same-compute: {metric}")
    for row_id in range(8):
        for column_id in range(8):
            if np.isfinite(matrix[row_id, column_id]):
                ax.text(column_id, row_id, f"{matrix[row_id, column_id]:+.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def draw_mismatch_heatmap(rows: list[dict[str, Any]], out_path: Path) -> None:
    matrix = np.zeros((8, 8), dtype=np.float64)
    for train_k in range(1, 9):
        for inference_k in range(1, 9):
            if train_k != inference_k:
                matrix[train_k - 1, inference_k - 1] = mean_delta(
                    rows,
                    train_k=train_k,
                    inference_k=inference_k,
                )
    vmax = float(np.max(np.abs(matrix)))
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    image = ax.imshow(matrix, vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax.set_xticks(range(8), labels=range(1, 9))
    ax.set_yticks(range(8), labels=range(1, 9))
    ax.set_xlabel("inference k")
    ax.set_ylabel("training k")
    ax.set_title("Fixed-step minus same-compute: mismatch loss delta")
    for row_id in range(8):
        for column_id in range(8):
            ax.text(column_id, row_id, f"{matrix[row_id, column_id]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_summary(
    out_dir: Path,
    pair_rows: list[dict[str, Any]],
    mismatch_rows: list[dict[str, Any]],
    specialization_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Sparse32 Budget Comparison",
        "",
        "All deltas are paired by seed and defined as fixed-step minus same-compute.",
        "",
        "## Key Routing Deltas",
        "",
        "| pair | metric | same_mean | fixed_mean | delta | seeds_same_sign |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for pair in ("1-2", "1-4", "1-8", "2-4", "2-8", "4-8", "7-8"):
        for metric in ("nestedness", "top1_agreement", "spearman"):
            selected = [row for row in pair_rows if row["pair"] == pair and row["metric_name"] == metric]
            same_mean = float(np.mean([row["same_compute"] for row in selected]))
            fixed_mean = float(np.mean([row["fixed_step"] for row in selected]))
            delta = fixed_mean - same_mean
            signs = sum(np.sign(row["fixed_minus_same"]) == np.sign(delta) for row in selected)
            lines.append(f"| {pair} | {metric} | {same_mean:.4f} | {fixed_mean:.4f} | {delta:+.4f} | {signs}/{len(selected)} |")

    lines.extend(
        [
            "",
            "## Key Mismatch Deltas",
            "",
            "| direction | same_mean | fixed_mean | delta | seeds_same_sign |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for train_k, inference_k in ((1, 4), (4, 1), (1, 8), (8, 1), (2, 8), (8, 2), (4, 8), (8, 4)):
        selected = [
            row
            for row in mismatch_rows
            if row["train_k"] == train_k and row["inference_k"] == inference_k
        ]
        same_mean = float(np.mean([row["same_compute"] for row in selected]))
        fixed_mean = float(np.mean([row["fixed_step"] for row in selected]))
        delta = fixed_mean - same_mean
        signs = sum(np.sign(row["fixed_minus_same"]) == np.sign(delta) for row in selected)
        lines.append(f"| {train_k}->{inference_k} | {same_mean:.4f} | {fixed_mean:.4f} | {delta:+.4f} | {signs}/{len(selected)} |")

    lines.extend(
        [
            "",
            "## Specialization Deltas",
            "",
            "| train_k | normalized_mi_delta | purity_delta | task_js_delta |",
            "|---:|---:|---:|---:|",
        ]
    )
    for train_k in range(1, 9):
        values = []
        for metric in ("normalized_mutual_information", "expert_purity", "mean_pairwise_task_js"):
            selected = [row["fixed_minus_same"] for row in specialization_rows if row["train_k"] == train_k and row["metric_name"] == metric]
            values.append(float(np.mean(selected)))
        lines.append(f"| {train_k} | {values[0]:+.4f} | {values[1]:+.4f} | {values[2]:+.4f} |")

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This is a paired three-seed comparison, not an inferential significance test.",
            "- Fixed-step equalizes optimizer updates and token exposure; active expert compute still grows with k.",
            "- Positive routing deltas mean the fixed-step models route more similarly than the same-compute models.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Sparse32 same-compute and fixed-step runs.")
    parser.add_argument("--same-dir", type=Path, required=True)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    same_metrics = read_csv(args.same_dir / "metrics.csv")
    fixed_metrics = read_csv(args.fixed_dir / "metrics.csv")
    pair_rows = paired_rows(
        final_pair_map(same_metrics),
        final_pair_map(fixed_metrics),
        ["seed", "pair", "metric_name"],
    )
    mismatch_rows = paired_rows(
        final_mismatch_map(same_metrics),
        final_mismatch_map(fixed_metrics),
        ["seed", "train_k", "inference_k"],
    )

    same_layers = keyed_map(
        read_csv(args.same_dir / "diagnostics" / "layer_pair_metrics.csv"),
        ["seed", "pair", "layer_id", "metric_name"],
    )
    fixed_layers = keyed_map(
        read_csv(args.fixed_dir / "diagnostics" / "layer_pair_metrics.csv"),
        ["seed", "pair", "layer_id", "metric_name"],
    )
    layer_rows = paired_rows(same_layers, fixed_layers, ["seed", "pair", "layer_id", "metric_name"])

    specialization_metrics = ["normalized_mutual_information", "expert_purity", "mean_pairwise_task_js"]
    same_specialization_raw = read_csv(
        args.same_dir / "diagnostics" / "task_specialization" / "specialization_metrics.csv"
    )
    fixed_specialization_raw = read_csv(
        args.fixed_dir / "diagnostics" / "task_specialization" / "specialization_metrics.csv"
    )
    same_specialization = {
        (int(row["seed"]), int(row["train_k"]), int(row["layer_id"]), metric): float(row[metric])
        for row in same_specialization_raw
        for metric in specialization_metrics
    }
    fixed_specialization = {
        (int(row["seed"]), int(row["train_k"]), int(row["layer_id"]), metric): float(row[metric])
        for row in fixed_specialization_raw
        for metric in specialization_metrics
    }
    specialization_rows = paired_rows(
        same_specialization,
        fixed_specialization,
        ["seed", "train_k", "layer_id", "metric_name"],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    write_csv(
        args.out_dir / "paired_route_metric_deltas.csv",
        pair_rows,
        ["seed", "pair", "metric_name", "same_compute", "fixed_step", "fixed_minus_same"],
    )
    write_csv(
        args.out_dir / "paired_mismatch_deltas.csv",
        mismatch_rows,
        ["seed", "train_k", "inference_k", "same_compute", "fixed_step", "fixed_minus_same"],
    )
    write_csv(
        args.out_dir / "paired_layer_metric_deltas.csv",
        layer_rows,
        ["seed", "pair", "layer_id", "metric_name", "same_compute", "fixed_step", "fixed_minus_same"],
    )
    write_csv(
        args.out_dir / "paired_specialization_deltas.csv",
        specialization_rows,
        ["seed", "train_k", "layer_id", "metric_name", "same_compute", "fixed_step", "fixed_minus_same"],
    )
    for metric in ("nestedness", "top1_agreement", "spearman"):
        draw_pair_heatmap(pair_rows, metric, plot_dir / f"{metric}_delta_heatmap.png")
    draw_mismatch_heatmap(mismatch_rows, plot_dir / "mismatch_delta_difference_heatmap.png")
    write_summary(args.out_dir, pair_rows, mismatch_rows, specialization_rows)
    print(f"wrote budget comparison to {args.out_dir}")


if __name__ == "__main__":
    main()
