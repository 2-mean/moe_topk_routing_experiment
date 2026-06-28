#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
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


def final_matched_rows(manifest: list[dict[str, str]]) -> list[dict[str, str]]:
    final_steps: dict[tuple[int, int], int] = {}
    for row in manifest:
        seed = int(row["seed"])
        train_k = int(row["train_k"])
        inference_k = int(row["inference_k"])
        if train_k != inference_k:
            continue
        key = (seed, train_k)
        final_steps[key] = max(final_steps.get(key, -1), int(row["checkpoint_step"]))
    return [
        row
        for row in manifest
        if int(row["train_k"]) == int(row["inference_k"])
        and int(row["checkpoint_step"]) == final_steps[(int(row["seed"]), int(row["train_k"]))]
    ]


def entropy(probabilities: np.ndarray) -> float:
    values = probabilities[probabilities > 0]
    return -float(np.sum(values * np.log(values)))


def specialization_metrics(counts: np.ndarray) -> dict[str, float]:
    total = float(counts.sum())
    if total <= 0:
        return {
            "mutual_information": float("nan"),
            "normalized_mutual_information": float("nan"),
            "expert_purity": float("nan"),
            "mean_pairwise_task_js": float("nan"),
        }

    joint = counts / total
    task_prob = joint.sum(axis=1)
    expert_prob = joint.sum(axis=0)
    expected = task_prob[:, None] * expert_prob[None, :]
    valid = (joint > 0) & (expected > 0)
    mutual_information = float(np.sum(joint[valid] * np.log(joint[valid] / expected[valid])))
    task_entropy = entropy(task_prob)
    expert_entropy = entropy(expert_prob)
    denominator = math.sqrt(task_entropy * expert_entropy)
    normalized_mi = mutual_information / denominator if denominator > 0 else float("nan")
    expert_purity = float(np.max(counts, axis=0).sum() / total)

    distributions = counts / np.maximum(1.0, counts.sum(axis=1, keepdims=True))
    divergences = []
    for left in range(distributions.shape[0] - 1):
        for right in range(left + 1, distributions.shape[0]):
            p = distributions[left]
            q = distributions[right]
            midpoint = 0.5 * (p + q)
            p_valid = p > 0
            q_valid = q > 0
            left_kl = float(np.sum(p[p_valid] * np.log(p[p_valid] / midpoint[p_valid])))
            right_kl = float(np.sum(q[q_valid] * np.log(q[q_valid] / midpoint[q_valid])))
            divergences.append(0.5 * (left_kl + right_kl))

    return {
        "mutual_information": mutual_information,
        "normalized_mutual_information": normalized_mi,
        "expert_purity": expert_purity,
        "mean_pairwise_task_js": float(np.mean(divergences)) if divergences else float("nan"),
    }


def gate_metrics(
    logits: np.ndarray,
    selected_ids: np.ndarray,
    selected_weights: np.ndarray,
) -> dict[str, float]:
    logits = logits.astype(np.float32)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    gate_entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-30)), axis=1)
    gate_entropy /= math.log(probabilities.shape[1])

    top_two = np.partition(logits, -2, axis=1)[:, -2:]
    logit_margin = np.max(top_two, axis=1) - np.min(top_two, axis=1)
    selected_mass = np.take_along_axis(probabilities, selected_ids.astype(np.intp), axis=1).sum(axis=1)

    if selected_weights.shape[1] > 1:
        weights = selected_weights.astype(np.float32)
        weight_entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-30)), axis=1)
        weight_entropy /= math.log(selected_weights.shape[1])
        mean_weight_entropy = float(np.mean(weight_entropy))
    else:
        mean_weight_entropy = 0.0

    return {
        "normalized_gate_entropy": float(np.mean(gate_entropy)),
        "top1_top2_logit_margin": float(np.mean(logit_margin)),
        "selected_probability_mass": float(np.mean(selected_mass)),
        "normalized_selected_weight_entropy": mean_weight_entropy,
    }


def analyze_run(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = read_csv(run_dir / "manifest.csv")
    matched = sorted(final_matched_rows(manifest), key=lambda row: (int(row["seed"]), int(row["train_k"])))
    specialization_rows: list[dict[str, Any]] = []
    task_metric_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []

    for row in matched:
        route_path = run_dir / row["route_file"]
        with np.load(route_path) as data:
            task_ids = data["task_id"].astype(np.intp, copy=False)
            layer_ids = data["layer_id"].astype(np.intp, copy=False)
            selected_ids = data["selected_ids"].astype(np.intp, copy=False)
            selected_weights = data["selected_weights"]
            gate_logits = data["gate_logits"]
            task_names = [str(value) for value in data["task_names"].tolist()]
            n_experts = int(gate_logits.shape[1])

            for layer_id in sorted(np.unique(layer_ids).tolist()):
                layer_mask = layer_ids == layer_id
                counts = np.zeros((len(task_names), n_experts), dtype=np.float64)
                for task_id, task_name in enumerate(task_names):
                    mask = layer_mask & (task_ids == task_id)
                    task_selected = selected_ids[mask]
                    expert_counts = np.bincount(task_selected.reshape(-1), minlength=n_experts).astype(np.float64)
                    counts[task_id] = expert_counts
                    total_assignments = max(1.0, float(expert_counts.sum()))
                    for expert_id, count in enumerate(expert_counts):
                        frequency_rows.append(
                            {
                                "seed": int(row["seed"]),
                                "train_k": int(row["train_k"]),
                                "layer_id": int(layer_id),
                                "task_id": task_id,
                                "task_name": task_name,
                                "expert_id": expert_id,
                                "count": int(count),
                                "share": float(count / total_assignments),
                            }
                        )

                    task_gate_metrics = gate_metrics(
                        gate_logits[mask],
                        task_selected,
                        selected_weights[mask],
                    )
                    task_metric_rows.append(
                        {
                            "seed": int(row["seed"]),
                            "train_k": int(row["train_k"]),
                            "layer_id": int(layer_id),
                            "task_id": task_id,
                            "task_name": task_name,
                            "num_records": int(mask.sum()),
                            **task_gate_metrics,
                        }
                    )

                specialization_rows.append(
                    {
                        "seed": int(row["seed"]),
                        "train_k": int(row["train_k"]),
                        "layer_id": int(layer_id),
                        **specialization_metrics(counts),
                    }
                )

        print(f"[task-specialization] seed={row['seed']} train_k={row['train_k']} route={route_path.name}", flush=True)

    return specialization_rows, task_metric_rows, frequency_rows


def mean_for(rows: list[dict[str, Any]], field: str, **filters: int) -> float:
    values = [
        float(row[field])
        for row in rows
        if all(int(row[key]) == int(value) for key, value in filters.items())
    ]
    return float(np.mean(values)) if values else float("nan")


def draw_heatmaps(
    out_dir: Path,
    specialization_rows: list[dict[str, Any]],
    task_metric_rows: list[dict[str, Any]],
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    train_ks = sorted({int(row["train_k"]) for row in specialization_rows})
    layers = sorted({int(row["layer_id"]) for row in specialization_rows})
    specs = [
        (specialization_rows, "normalized_mutual_information", "Task-expert normalized mutual information"),
        (specialization_rows, "expert_purity", "Expert task purity"),
        (specialization_rows, "mean_pairwise_task_js", "Mean pairwise task JS divergence"),
        (task_metric_rows, "normalized_gate_entropy", "Normalized gate entropy"),
        (task_metric_rows, "top1_top2_logit_margin", "Top1-top2 router logit margin"),
        (task_metric_rows, "selected_probability_mass", "Selected top-k probability mass"),
    ]
    for rows, field, title in specs:
        matrix = np.array(
            [[mean_for(rows, field, train_k=train_k, layer_id=layer) for layer in layers] for train_k in train_ks],
            dtype=np.float64,
        )
        fig, ax = plt.subplots(figsize=(9.2, 7.2))
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(layers)), labels=layers)
        ax.set_yticks(range(len(train_ks)), labels=train_ks)
        ax.set_xlabel("layer")
        ax.set_ylabel("training k")
        ax.set_title(title)
        for row_id in range(matrix.shape[0]):
            for column_id in range(matrix.shape[1]):
                ax.text(column_id, row_id, f"{matrix[row_id, column_id]:.2f}", ha="center", va="center", color="white")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{field}_heatmap.png", dpi=160)
        plt.close(fig)


def write_summary(
    out_dir: Path,
    specialization_rows: list[dict[str, Any]],
    task_metric_rows: list[dict[str, Any]],
) -> None:
    train_ks = sorted({int(row["train_k"]) for row in specialization_rows})
    lines = [
        "# Task Specialization Diagnostics",
        "",
        "Values are averaged over seeds and layers; task-conditioned gate values are also averaged over tasks.",
        "",
        "| train_k | normalized_mi | expert_purity | task_js | gate_entropy | logit_margin | selected_mass |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for train_k in train_ks:
        values = [
            mean_for(specialization_rows, "normalized_mutual_information", train_k=train_k),
            mean_for(specialization_rows, "expert_purity", train_k=train_k),
            mean_for(specialization_rows, "mean_pairwise_task_js", train_k=train_k),
            mean_for(task_metric_rows, "normalized_gate_entropy", train_k=train_k),
            mean_for(task_metric_rows, "top1_top2_logit_margin", train_k=train_k),
            mean_for(task_metric_rows, "selected_probability_mass", train_k=train_k),
        ]
        lines.append(f"| {train_k} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Expert purity has a balanced-task baseline of 1 / number_of_tasks.",
            "- Mutual information and JS divergence describe routing specialization, not task quality.",
            "- Expert identities are only compared within a seed; cross-seed expert labels are permutation-sensitive.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze task-conditioned routing specialization from final matched routes.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else run_dir / "diagnostics" / "task_specialization"
    out_dir.mkdir(parents=True, exist_ok=True)
    specialization_rows, task_metric_rows, frequency_rows = analyze_run(run_dir)
    write_csv(
        out_dir / "specialization_metrics.csv",
        specialization_rows,
        [
            "seed",
            "train_k",
            "layer_id",
            "mutual_information",
            "normalized_mutual_information",
            "expert_purity",
            "mean_pairwise_task_js",
        ],
    )
    write_csv(
        out_dir / "task_gate_metrics.csv",
        task_metric_rows,
        [
            "seed",
            "train_k",
            "layer_id",
            "task_id",
            "task_name",
            "num_records",
            "normalized_gate_entropy",
            "top1_top2_logit_margin",
            "selected_probability_mass",
            "normalized_selected_weight_entropy",
        ],
    )
    write_csv(
        out_dir / "task_expert_frequency.csv",
        frequency_rows,
        ["seed", "train_k", "layer_id", "task_id", "task_name", "expert_id", "count", "share"],
    )
    draw_heatmaps(out_dir, specialization_rows, task_metric_rows)
    write_summary(out_dir, specialization_rows, task_metric_rows)
    print(f"wrote task specialization diagnostics to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
