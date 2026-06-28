#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from moe_topk.ranking import (
    displacement_by_source_rank,
    empirical_null_ceiling_adjusted_overlap,
    inverse_ranks_from_logits,
    kendall_tau_from_ranks,
    mean_absolute_rank_displacement,
    merge_selected_expert_order_stats,
    normalized_footrule_similarity,
    rank_biased_overlap,
    rank_transition_counts,
    selected_bitmasks_from_ranks,
    selected_expert_order_stats,
    summarize_selected_expert_order_stats,
    topm_overlap_curve,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def final_matched_index(manifest: list[dict[str, str]]) -> dict[tuple[int, int], dict[str, str]]:
    result: dict[tuple[int, int], dict[str, str]] = {}
    for row in manifest:
        seed = int(row["seed"])
        train_k = int(row["train_k"])
        inference_k = int(row["inference_k"])
        if train_k != inference_k:
            continue
        key = (seed, train_k)
        if key not in result or int(row["checkpoint_step"]) > int(result[key]["checkpoint_step"]):
            result[key] = row
    return result


def load_seed_ranks(
    run_dir: Path,
    index: dict[tuple[int, int], dict[str, str]],
    seed: int,
    train_ks: list[int],
) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray]]:
    ranks_by_k = {}
    reference_metadata = None
    for train_k in train_ks:
        row = index[(seed, train_k)]
        route_path = run_dir / row["route_file"]
        with np.load(route_path) as data:
            metadata = np.stack(
                [data["sample_index"], data["task_id"], data["layer_id"], data["token_pos"]],
                axis=1,
            )
            if reference_metadata is None:
                reference_metadata = metadata
            elif not np.array_equal(reference_metadata, metadata):
                raise ValueError(f"route records are not aligned for seed={seed}, train_k={train_k}")
            ranks_by_k[train_k] = inverse_ranks_from_logits(data["gate_logits"])
        print(f"[ranking-load] seed={seed} train_k={train_k} route={route_path.name}", flush=True)
    assert reference_metadata is not None
    return ranks_by_k, {
        "sample_ids": reference_metadata[:, 0],
        "task_ids": reference_metadata[:, 1],
        "layer_ids": reference_metadata[:, 2],
        "token_positions": reference_metadata[:, 3],
    }


def layer_indices(layer_ids: np.ndarray, max_records_per_layer: int | None) -> dict[int, np.ndarray]:
    result = {}
    for layer_id in sorted(np.unique(layer_ids).tolist()):
        indices = np.flatnonzero(layer_ids == layer_id)
        if max_records_per_layer is not None:
            indices = indices[:max_records_per_layer]
        result[int(layer_id)] = indices
    return result


def analyze(
    run_dir: Path,
    seeds: list[int],
    train_ks: list[int],
    max_records_per_layer: int | None,
    kendall_samples_per_layer: int,
) -> tuple[
    dict[tuple[str, int], np.ndarray],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    list[dict[str, Any]],
]:
    index = final_matched_index(read_csv(run_dir / "manifest.csv"))
    aggregate_counts: dict[tuple[str, int], np.ndarray] = {}
    topm_rows = []
    summary_rows = []
    raw_counts: dict[str, np.ndarray] = {}
    intersection_rows = []

    for seed in seeds:
        ranks_by_k, metadata = load_seed_ranks(run_dir, index, seed, train_ks)
        layers = metadata["layer_ids"]
        masks_by_k = {
            train_k: selected_bitmasks_from_ranks(ranks_by_k[train_k], train_k)
            for train_k in train_ks
        }
        by_layer = layer_indices(layers, max_records_per_layer)
        full_by_layer = layer_indices(layers, None)
        for small_index, small in enumerate(train_ks):
            for large in train_ks[small_index + 1 :]:
                pair = f"{small}-{large}"
                all_counts = np.zeros((32, 32), dtype=np.int64)
                layer_kendall_values = []
                all_selected_stats = None
                for layer_id, indices in by_layer.items():
                    counts = rank_transition_counts(ranks_by_k[small], ranks_by_k[large], indices)
                    if indices.size > kendall_samples_per_layer:
                        positions = np.linspace(
                            0,
                            indices.size - 1,
                            kendall_samples_per_layer,
                            dtype=np.int64,
                        )
                        kendall_indices = indices[positions]
                    else:
                        kendall_indices = indices
                    kendall_tau = kendall_tau_from_ranks(
                        ranks_by_k[small],
                        ranks_by_k[large],
                        kendall_indices,
                    )
                    layer_kendall_values.append(kendall_tau)
                    selected_stats = selected_expert_order_stats(
                        ranks_by_k[small],
                        ranks_by_k[large],
                        small,
                        large,
                        indices,
                    )
                    selected_summary = summarize_selected_expert_order_stats(
                        selected_stats, small, large
                    )
                    empirical_summary = empirical_null_ceiling_adjusted_overlap(
                        masks_by_k[small],
                        masks_by_k[large],
                        cutoff_a=small,
                        record_indices=full_by_layer[layer_id],
                        **metadata,
                    )
                    histogram = np.asarray(selected_stats["intersection_histogram"])
                    for intersection in range(small + 1):
                        intersection_rows.append(
                            {
                                "seed": seed,
                                "pair": pair,
                                "layer_id": layer_id,
                                "intersection_size": intersection,
                                "count": int(histogram[intersection]),
                                "share": float(histogram[intersection] / histogram.sum()),
                            }
                        )
                    all_selected_stats = merge_selected_expert_order_stats(
                        all_selected_stats, selected_stats
                    )
                    all_counts += counts
                    raw_counts[f"seed{seed}_pair{small}_{large}_layer{layer_id}"] = counts
                    key = (pair, layer_id)
                    aggregate_counts[key] = aggregate_counts.get(key, np.zeros_like(counts)) + counts
                    curve = topm_overlap_curve(counts)
                    for top_m, value in enumerate(curve, start=1):
                        topm_rows.append(
                            {
                                "seed": seed,
                                "pair": pair,
                                "layer_id": layer_id,
                                "top_m": top_m,
                                "overlap_fraction": value,
                            }
                        )
                    summary_rows.append(
                        {
                            "seed": seed,
                            "pair": pair,
                            "layer_id": layer_id,
                            "rbo_p90": rank_biased_overlap(curve, 0.90),
                            "rbo_p95": rank_biased_overlap(curve, 0.95),
                            "kendall_tau": kendall_tau,
                            "mean_absolute_rank_displacement": mean_absolute_rank_displacement(counts),
                            "normalized_footrule_similarity": normalized_footrule_similarity(counts),
                            "top1_overlap": curve[0],
                            "top4_overlap": curve[3],
                            "top8_overlap": curve[7],
                            **selected_summary,
                            **empirical_summary,
                            "num_records": int(counts.sum() // counts.shape[0]),
                        }
                    )

                raw_counts[f"seed{seed}_pair{small}_{large}_layer_all"] = all_counts
                key = (pair, -1)
                aggregate_counts[key] = aggregate_counts.get(key, np.zeros_like(all_counts)) + all_counts
                curve = topm_overlap_curve(all_counts)
                assert all_selected_stats is not None
                selected_summary = summarize_selected_expert_order_stats(
                    all_selected_stats, small, large
                )
                empirical_summary = empirical_null_ceiling_adjusted_overlap(
                    masks_by_k[small],
                    masks_by_k[large],
                    cutoff_a=small,
                    record_indices=np.arange(layers.size),
                    **metadata,
                )
                histogram = np.asarray(all_selected_stats["intersection_histogram"])
                for intersection in range(small + 1):
                    intersection_rows.append(
                        {
                            "seed": seed,
                            "pair": pair,
                            "layer_id": -1,
                            "intersection_size": intersection,
                            "count": int(histogram[intersection]),
                            "share": float(histogram[intersection] / histogram.sum()),
                        }
                    )
                for top_m, value in enumerate(curve, start=1):
                    topm_rows.append(
                        {
                            "seed": seed,
                            "pair": pair,
                            "layer_id": -1,
                            "top_m": top_m,
                            "overlap_fraction": value,
                        }
                    )
                summary_rows.append(
                    {
                        "seed": seed,
                        "pair": pair,
                        "layer_id": -1,
                        "rbo_p90": rank_biased_overlap(curve, 0.90),
                        "rbo_p95": rank_biased_overlap(curve, 0.95),
                        "kendall_tau": float(np.mean(layer_kendall_values)),
                        "mean_absolute_rank_displacement": mean_absolute_rank_displacement(all_counts),
                        "normalized_footrule_similarity": normalized_footrule_similarity(all_counts),
                        "top1_overlap": curve[0],
                        "top4_overlap": curve[3],
                        "top8_overlap": curve[7],
                        **selected_summary,
                        **empirical_summary,
                        "num_records": int(all_counts.sum() // all_counts.shape[0]),
                    }
                )
                print(f"[ranking-pair] seed={seed} pair={pair}", flush=True)
        del ranks_by_k, masks_by_k
    return aggregate_counts, topm_rows, summary_rows, raw_counts, intersection_rows


def aggregate_output_rows(
    aggregate_counts: dict[tuple[str, int], np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transition_rows = []
    displacement_rows = []
    for (pair, layer_id), counts in sorted(aggregate_counts.items()):
        row_totals = counts.sum(axis=1)
        mean_target, mean_absolute = displacement_by_source_rank(counts)
        for source_rank in range(1, counts.shape[0] + 1):
            displacement_rows.append(
                {
                    "pair": pair,
                    "layer_id": layer_id,
                    "source_rank": source_rank,
                    "mean_target_rank": mean_target[source_rank - 1],
                    "mean_absolute_displacement": mean_absolute[source_rank - 1],
                    "count": int(row_totals[source_rank - 1]),
                }
            )
            for target_rank in range(1, counts.shape[1] + 1):
                count = int(counts[source_rank - 1, target_rank - 1])
                transition_rows.append(
                    {
                        "pair": pair,
                        "layer_id": layer_id,
                        "source_rank": source_rank,
                        "target_rank": target_rank,
                        "count": count,
                        "conditional_share": count / max(1, int(row_totals[source_rank - 1])),
                    }
                )
    return transition_rows, displacement_rows


def mean_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(row["pair"], int(row["layer_id"]))].append(row)
    result = []
    for (pair, layer_id), rows in sorted(grouped.items()):
        metric_fields = (
            "rbo_p90",
            "rbo_p95",
            "kendall_tau",
            "mean_absolute_rank_displacement",
            "normalized_footrule_similarity",
            "top1_overlap",
            "top4_overlap",
            "top8_overlap",
            "selected_containment_a",
            "selected_containment_b",
            "selected_jaccard",
            "hypergeometric_calibrated_overlap",
            "empirical_observed_overlap",
            "empirical_null_overlap",
            "empirical_ceiling_overlap",
            "empirical_alignment_headroom",
            "frequency_adjusted_overlap",
            "empirical_ceiling_utilization",
            "empirical_num_samples",
            "selected_union_kendall_tau",
            "common_selected_kendall_tau",
            "common_pairwise_order_agreement",
            "common_selected_exact_order_fraction",
            "full_containment_record_fraction",
            "ordered_containment_record_fraction",
            "contained_exact_order_fraction",
            "common_selected_record_mean_tau",
            "common_selected_valid_record_fraction",
            "common_pair_coverage",
        )
        means = {}
        for field in metric_fields:
            values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
            means[field] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")
        result.append(
            {
                "pair": pair,
                "layer_id": layer_id,
                **means,
                "num_seeds": len(rows),
            }
        )
    return result


def draw_transition(
    counts: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    conditional = counts / np.maximum(1, counts.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(9.2, 8.2))
    image = ax.imshow(conditional, cmap="magma", vmin=0.0, vmax=float(np.quantile(conditional, 0.99)))
    labels = list(range(1, 33))
    ax.set_xticks(range(32), labels=labels, fontsize=6)
    ax.set_yticks(range(32), labels=labels, fontsize=6)
    ax.set_xlabel("target model rank")
    ax.set_ylabel("source model rank")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="P(target rank | source rank)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def draw_plots(
    out_dir: Path,
    counts: dict[tuple[str, int], np.ndarray],
    topm_rows: list[dict[str, Any]],
    summary_mean: list[dict[str, Any]],
    displacement_rows: list[dict[str, Any]],
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (plot_dir / "empirical_adjusted_overlap_pair_layer_heatmap.png").unlink(
        missing_ok=True
    )
    key_pairs = [pair for pair in ("1-4", "1-8", "4-8") if (pair, -1) in counts]
    for pair in key_pairs:
        for layer_id in (-1, 0, 6):
            label = "all_layers" if layer_id == -1 else f"layer_{layer_id}"
            draw_transition(
                counts[(pair, layer_id)],
                f"Rank transition {pair}: {label}",
                plot_dir / f"rank_transition_{pair.replace('-', '_')}_{label}.png",
            )

    fig, ax = plt.subplots(figsize=(9.2, 6.5))
    curve_pairs = [
        pair
        for pair in ("1-2", "1-4", "1-8", "2-4", "2-8", "4-8", "7-8")
        if any(row["pair"] == pair for row in topm_rows)
    ]
    for pair in curve_pairs:
        selected = [row for row in topm_rows if row["pair"] == pair and int(row["layer_id"]) == -1]
        by_m = defaultdict(list)
        for row in selected:
            by_m[int(row["top_m"])].append(float(row["overlap_fraction"]))
        xs = sorted(by_m)
        ys = [float(np.mean(by_m[value])) for value in xs]
        ax.plot(xs, ys, marker="o", markersize=3, label=pair)
    ax.set_xlabel("top-m cutoff")
    ax.set_ylabel("set overlap fraction")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    ax.set_title("Full-ranking top-m overlap curves")
    fig.tight_layout()
    fig.savefig(plot_dir / "topm_overlap_curves.png", dpi=170)
    plt.close(fig)

    pairs = sorted({row["pair"] for row in summary_mean})
    for metric_name, title, filename, vmin in (
        ("rbo_p90", "Rank-biased overlap, p=0.90", "rbo_p90_pair_layer_heatmap.png", 0.0),
        ("kendall_tau", "Kendall tau of all 32 ranks", "kendall_tau_pair_layer_heatmap.png", -1.0),
        (
            "normalized_footrule_similarity",
            "Normalized Spearman footrule similarity",
            "footrule_similarity_pair_layer_heatmap.png",
            0.0,
        ),
        ("selected_jaccard", "Selected-set Jaccard", "selected_jaccard_pair_layer_heatmap.png", 0.0),
        (
            "hypergeometric_calibrated_overlap",
            "Hypergeometric-calibrated selected overlap",
            "hypergeometric_calibrated_overlap_pair_layer_heatmap.png",
            -1.0,
        ),
        (
            "frequency_adjusted_overlap",
            "Frequency-preserving adjusted overlap",
            "frequency_adjusted_overlap_pair_layer_heatmap.png",
            -1.0,
        ),
        (
            "selected_union_kendall_tau",
            "Kendall tau over selected-expert union",
            "selected_union_kendall_pair_layer_heatmap.png",
            -1.0,
        ),
        (
            "common_selected_kendall_tau",
            "Pair-weighted Kendall tau over common selected experts",
            "common_selected_kendall_pair_layer_heatmap.png",
            -1.0,
        ),
        (
            "common_pairwise_order_agreement",
            "Pairwise precedence agreement over common selected experts",
            "common_pairwise_order_agreement_pair_layer_heatmap.png",
            0.0,
        ),
        (
            "common_selected_exact_order_fraction",
            "Exact relative order over common selected experts",
            "common_selected_exact_order_pair_layer_heatmap.png",
            0.0,
        ),
        (
            "ordered_containment_record_fraction",
            "Full small-set containment with exact relative order",
            "ordered_containment_pair_layer_heatmap.png",
            0.0,
        ),
        (
            "common_pair_coverage",
            "Common selected expert-pair coverage",
            "common_pair_coverage_pair_layer_heatmap.png",
            0.0,
        ),
    ):
        metric_pairs = [
            pair
            for pair in pairs
            if any(
                row["pair"] == pair
                and int(row["layer_id"]) >= 0
                and np.isfinite(float(row[metric_name]))
                for row in summary_mean
            )
        ]
        matrix = np.full((len(metric_pairs), 8), np.nan)
        for pair_id, pair in enumerate(metric_pairs):
            for row in summary_mean:
                if row["pair"] == pair and int(row["layer_id"]) >= 0:
                    matrix[pair_id, int(row["layer_id"])] = float(row[metric_name])
        fig_height = max(5.0, min(12.0, 2.0 + 0.32 * len(metric_pairs)))
        fig, ax = plt.subplots(figsize=(10.0, fig_height))
        cmap = plt.colormaps["viridis"].copy()
        cmap.set_bad("#d9d9d9")
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=1.0)
        ax.set_xticks(range(8), labels=range(8))
        ax.set_yticks(range(len(metric_pairs)), labels=metric_pairs, fontsize=8)
        ax.set_xlabel("layer")
        ax.set_ylabel("train-k pair")
        ax.set_title(title)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=170)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 6.5))
    for pair in key_pairs:
        selected = sorted(
            [row for row in displacement_rows if row["pair"] == pair and int(row["layer_id"]) == -1],
            key=lambda row: int(row["source_rank"]),
        )
        ax.plot(
            [int(row["source_rank"]) for row in selected],
            [float(row["mean_absolute_displacement"]) for row in selected],
            marker="o",
            markersize=3,
            label=pair,
        )
    ax.set_xlabel("source model rank")
    ax.set_ylabel("mean absolute target-rank displacement")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Rank displacement by source position")
    fig.tight_layout()
    fig.savefig(plot_dir / "rank_displacement_by_source_rank.png", dpi=170)
    plt.close(fig)


def write_summary(out_dir: Path, summary_mean: list[dict[str, Any]]) -> None:
    lines = [
        "# Full Expert Ranking Analysis",
        "",
        "Every row uses all 32 expert ranks. Layer `-1` aggregates all eight MoE layers.",
        "",
        "| pair | RBO p=0.90 | Kendall tau | Footrule similarity | mean abs displacement | top1 | top4 | top8 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    available_pairs = {row["pair"] for row in summary_mean}
    for pair in ("1-2", "1-4", "1-8", "2-4", "2-8", "4-8", "7-8"):
        if pair not in available_pairs:
            continue
        row = next(row for row in summary_mean if row["pair"] == pair and int(row["layer_id"]) == -1)
        lines.append(
            f"| {pair} | {row['rbo_p90']:.4f} | {row['kendall_tau']:.4f} | "
            f"{row['normalized_footrule_similarity']:.4f} | {row['mean_absolute_rank_displacement']:.4f} | "
            f"{row['top1_overlap']:.4f} | "
            f"{row['top4_overlap']:.4f} | {row['top8_overlap']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Selected Experts Only",
            "",
            "| pair | small containment | full containment | ordered containment | common exact order | pairwise precedence | coverage |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in ("1-2", "1-4", "1-8", "2-4", "2-8", "4-8", "7-8"):
        if pair not in available_pairs:
            continue
        row = next(row for row in summary_mean if row["pair"] == pair and int(row["layer_id"]) == -1)
        common_exact = row["common_selected_exact_order_fraction"]
        pairwise = row["common_pairwise_order_agreement"]
        coverage = row["common_pair_coverage"]
        lines.append(
            f"| {pair} | {row['selected_containment_a']:.4f} | "
            f"{row['full_containment_record_fraction']:.4f} | "
            f"{row['ordered_containment_record_fraction']:.4f} | "
            f"{common_exact:.4f} | {pairwise:.4f} | {coverage:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Cardinality-Calibrated Selected Overlap",
            "",
            "| pair | hypergeometric calibrated | frequency adjusted | ceiling utilization | empirical null | empirical ceiling | headroom |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in ("1-2", "1-4", "1-8", "2-4", "2-8", "4-8", "7-8"):
        if pair not in available_pairs:
            continue
        row = next(row for row in summary_mean if row["pair"] == pair and int(row["layer_id"]) == -1)
        lines.append(
            f"| {pair} | {row['hypergeometric_calibrated_overlap']:.4f} | "
            f"{row['frequency_adjusted_overlap']:.4f} | "
            f"{row['empirical_ceiling_utilization']:.4f} | "
            f"{row['empirical_null_overlap']:.4f} | "
            f"{row['empirical_ceiling_overlap']:.4f} | "
            f"{row['empirical_alignment_headroom']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- RBO weights the top of the ranking more heavily but uses all 32 ranks.",
            "- Kendall tau measures pairwise ordering agreement across all 496 expert pairs and is estimated on an evenly spaced record sample per layer.",
            "- Normalized Spearman footrule is one minus mean absolute rank displacement divided by its permutation maximum.",
            "- Top-m overlap compares equal cutoffs in the two separately trained models; it is different from unequal-k nestedness.",
            "- Selected-set containment and Jaccard use the models' matched operational cutoffs, not all 32 experts.",
            "- Hypergeometric calibration uses the complete random-intersection distribution for the pair's exact set sizes; random maps to zero and perfect nesting to one.",
            "- Frequency adjustment matches sample blocks within task. A no-fixed-point random derangement maps to zero and theoretical perfect containment maps to one.",
            "- Ceiling utilization separately reports how much of the empirically attainable optimal-matching headroom is realized; it is not used as the main k-comparison metric.",
            "- Ceiling utilization is unstable when empirical alignment headroom is near zero, so headroom is always reported.",
            "- Union Kendall compares only experts selected by at least one model; common-selected Kendall compares only experts selected by both.",
            "- Common-selected Kendall is pair-weighted and must be read with common-pair coverage. It is undefined when the smaller cutoff is one.",
            "- Common exact order requires every comparable common-expert pair to preserve precedence; pairwise precedence reports the fraction preserved.",
            "- Ordered containment requires the entire smaller selected set to appear in the larger set in the same relative order.",
            "- Transition matrices aggregate aligned expert IDs within each seed. Cross-seed expert labels are not treated as independently aligned models.",
            "- Token records are repeated observations from the same trained models, not independent statistical samples.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze complete 32-expert ranking transitions.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-ks", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--max-records-per-layer", type=int, default=None)
    parser.add_argument("--kendall-samples-per-layer", type=int, default=2048)
    args = parser.parse_args()

    train_ks = sorted(set(args.train_ks))
    counts, topm_rows, summary_rows, raw_counts, intersection_rows = analyze(
        args.run_dir,
        args.seeds,
        train_ks,
        args.max_records_per_layer,
        args.kendall_samples_per_layer,
    )
    transition_rows, displacement_rows = aggregate_output_rows(counts)
    summary_mean = mean_summary(summary_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "rank_transition_seed_counts.npz", **raw_counts)
    write_csv(
        args.out_dir / "rank_transition_matrix.csv",
        transition_rows,
        ["pair", "layer_id", "source_rank", "target_rank", "count", "conditional_share"],
    )
    write_csv(
        args.out_dir / "topm_overlap_curve.csv",
        topm_rows,
        ["seed", "pair", "layer_id", "top_m", "overlap_fraction"],
    )
    write_csv(
        args.out_dir / "selected_intersection_histogram.csv",
        intersection_rows,
        ["seed", "pair", "layer_id", "intersection_size", "count", "share"],
    )
    write_csv(
        args.out_dir / "rank_displacement.csv",
        displacement_rows,
        ["pair", "layer_id", "source_rank", "mean_target_rank", "mean_absolute_displacement", "count"],
    )
    write_csv(
        args.out_dir / "full_ranking_summary_by_seed.csv",
        summary_rows,
        [
            "seed",
            "pair",
            "layer_id",
            "rbo_p90",
            "rbo_p95",
            "kendall_tau",
            "mean_absolute_rank_displacement",
            "normalized_footrule_similarity",
            "top1_overlap",
            "top4_overlap",
            "top8_overlap",
            "selected_containment_a",
            "selected_containment_b",
            "selected_jaccard",
            "hypergeometric_calibrated_overlap",
            "empirical_observed_overlap",
            "empirical_null_overlap",
            "empirical_ceiling_overlap",
            "empirical_alignment_headroom",
            "frequency_adjusted_overlap",
            "empirical_ceiling_utilization",
            "empirical_num_samples",
            "selected_union_kendall_tau",
            "common_selected_kendall_tau",
            "common_pairwise_order_agreement",
            "common_selected_exact_order_fraction",
            "full_containment_record_fraction",
            "ordered_containment_record_fraction",
            "contained_exact_order_fraction",
            "common_selected_record_mean_tau",
            "common_selected_valid_record_fraction",
            "common_pair_coverage",
            "num_records",
        ],
    )
    write_csv(
        args.out_dir / "full_ranking_summary_mean.csv",
        summary_mean,
        [
            "pair",
            "layer_id",
            "rbo_p90",
            "rbo_p95",
            "kendall_tau",
            "mean_absolute_rank_displacement",
            "normalized_footrule_similarity",
            "top1_overlap",
            "top4_overlap",
            "top8_overlap",
            "selected_containment_a",
            "selected_containment_b",
            "selected_jaccard",
            "hypergeometric_calibrated_overlap",
            "empirical_observed_overlap",
            "empirical_null_overlap",
            "empirical_ceiling_overlap",
            "empirical_alignment_headroom",
            "frequency_adjusted_overlap",
            "empirical_ceiling_utilization",
            "empirical_num_samples",
            "selected_union_kendall_tau",
            "common_selected_kendall_tau",
            "common_pairwise_order_agreement",
            "common_selected_exact_order_fraction",
            "full_containment_record_fraction",
            "ordered_containment_record_fraction",
            "contained_exact_order_fraction",
            "common_selected_record_mean_tau",
            "common_selected_valid_record_fraction",
            "common_pair_coverage",
            "num_seeds",
        ],
    )
    draw_plots(args.out_dir, counts, topm_rows, summary_mean, displacement_rows)
    write_summary(args.out_dir, summary_mean)
    print(f"wrote full ranking analysis to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
