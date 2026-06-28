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


def final_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    finals: dict[tuple[int, int], int] = {}
    for row in rows:
        key = (int(row["seed"]), int(row["train_k"]))
        finals[key] = max(finals.get(key, -1), int(row["checkpoint_step"]))
    return [
        row
        for row in rows
        if int(row["checkpoint_step"]) == finals[(int(row["seed"]), int(row["train_k"]))]
    ]


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    losses = {
        (
            int(row["seed"]),
            int(row["train_k"]),
            int(row["inference_k"]),
            int(row["task_id"]),
        ): float(row["validation_loss"])
        for row in rows
    }
    task_names = {int(row["task_id"]): row["task_name"] for row in rows}
    grouped: dict[tuple[int, int, int], list[tuple[float, float]]] = defaultdict(list)
    for (seed, train_k, inference_k, task_id), loss in losses.items():
        matched = losses[(seed, train_k, train_k, task_id)]
        grouped[(task_id, train_k, inference_k)].append((matched, loss))

    result = []
    for (task_id, train_k, inference_k), values in sorted(grouped.items()):
        matched_values = [value[0] for value in values]
        observed_values = [value[1] for value in values]
        deltas = [observed - matched for matched, observed in values]
        result.append(
            {
                "task_id": task_id,
                "task_name": task_names[task_id],
                "train_k": train_k,
                "inference_k": inference_k,
                "matched_loss": float(np.mean(matched_values)),
                "observed_loss": float(np.mean(observed_values)),
                "delta_loss": float(np.mean(deltas)),
                "seed_values": ";".join(f"{value:.6f}" for value in deltas),
                "num_seeds": len(values),
            }
        )
    return result


def draw_matrix(
    matrix: np.ndarray,
    x_labels: list[Any],
    y_labels: list[Any],
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    diverging: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 7.8))
    if diverging:
        vmax = float(np.nanmax(np.abs(matrix)))
        image = ax.imshow(matrix, vmin=-vmax, vmax=vmax, cmap="coolwarm", aspect="auto")
    else:
        image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(x_labels)), labels=x_labels)
    ax.set_yticks(range(len(y_labels)), labels=y_labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    for row_id in range(matrix.shape[0]):
        for column_id in range(matrix.shape[1]):
            value = matrix[row_id, column_id]
            if np.isfinite(value):
                ax.text(column_id, row_id, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def draw_plots(rows: list[dict[str, Any]], out_dir: Path) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    tasks = sorted({(int(row["task_id"]), row["task_name"]) for row in rows})

    matched = np.zeros((len(tasks), 8), dtype=np.float64)
    down_to_one = np.full((len(tasks), 7), np.nan, dtype=np.float64)
    for task_row, (task_id, _) in enumerate(tasks):
        for train_k in range(1, 9):
            row = next(
                row
                for row in rows
                if row["task_id"] == task_id and row["train_k"] == train_k and row["inference_k"] == train_k
            )
            matched[task_row, train_k - 1] = float(row["matched_loss"])
            if train_k >= 2:
                down = next(
                    row
                    for row in rows
                    if row["task_id"] == task_id and row["train_k"] == train_k and row["inference_k"] == 1
                )
                down_to_one[task_row, train_k - 2] = float(down["delta_loss"])

    draw_matrix(
        matched,
        list(range(1, 9)),
        [name for _, name in tasks],
        "Matched validation loss by task and training k",
        "training k",
        "task",
        plot_dir / "matched_task_loss_heatmap.png",
    )
    draw_matrix(
        down_to_one,
        list(range(2, 9)),
        [name for _, name in tasks],
        "Inference k=1 loss delta by task",
        "training k",
        "task",
        plot_dir / "down_to_k1_task_heatmap.png",
    )

    for task_id, task_name in tasks:
        matrix = np.zeros((8, 8), dtype=np.float64)
        for row in rows:
            if row["task_id"] == task_id:
                matrix[int(row["train_k"]) - 1, int(row["inference_k"]) - 1] = float(row["delta_loss"])
        draw_matrix(
            matrix,
            list(range(1, 9)),
            list(range(1, 9)),
            f"Mismatch loss delta: {task_name}",
            "inference k",
            "training k",
            plot_dir / f"mismatch_{task_name}_heatmap.png",
            diverging=True,
        )


def write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    tasks = sorted({(int(row["task_id"]), row["task_name"]) for row in rows})
    lines = [
        "# Fixed-step Task Loss Grid",
        "",
        "Loss deltas are relative to each model's matched inference k and averaged over three seeds.",
        "",
        "| task | matched_k1 | matched_k8 | 1->8 delta | 8->1 delta | asymmetry ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task_id, task_name in tasks:
        by_key = {
            (int(row["train_k"]), int(row["inference_k"])): row
            for row in rows
            if row["task_id"] == task_id
        }
        up = float(by_key[(1, 8)]["delta_loss"])
        down = float(by_key[(8, 1)]["delta_loss"])
        ratio = down / up if abs(up) > 1e-12 else float("nan")
        lines.append(
            f"| {task_name} | {float(by_key[(1, 1)]['matched_loss']):.4f} | "
            f"{float(by_key[(8, 8)]['matched_loss']):.4f} | {up:+.4f} | {down:+.4f} | {ratio:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Tasks are synthetic corpus categories, not external benchmark domains.",
            "- Per-task losses share the same trained models and are not independent observations.",
            "- A task heatmap localizes mismatch cost but does not establish task-specific expert causality.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze per-task matched and mismatch loss grids.")
    parser.add_argument("--task-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = aggregate(final_rows(read_csv(args.task_metrics)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / "task_loss_grid.csv",
        rows,
        [
            "task_id",
            "task_name",
            "train_k",
            "inference_k",
            "matched_loss",
            "observed_loss",
            "delta_loss",
            "seed_values",
            "num_seeds",
        ],
    )
    draw_plots(rows, args.out_dir)
    write_summary(rows, args.out_dir)
    print(f"wrote task loss grid to {args.out_dir}")


if __name__ == "__main__":
    main()
