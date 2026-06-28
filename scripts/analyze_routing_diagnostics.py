from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from moe_topk.metrics import coactivation_matrix, metric_rows_for_pair


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for idx in order[i:j]:
            ranks[idx] = rank
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return float("nan")
    return pearson(rank_values(xs), rank_values(ys))


def final_step_by_train(manifest: list[dict[str, str]]) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for row in manifest:
        seed = int(row["seed"])
        train_k = int(row["train_k"])
        infer_k = int(row["inference_k"])
        step = int(row["checkpoint_step"])
        if train_k == infer_k:
            result[(seed, train_k)] = max(result.get((seed, train_k), -1), step)
    return result


def manifest_index(manifest: list[dict[str, str]]) -> dict[tuple[int, int, int, int], dict[str, str]]:
    return {
        (
            int(row["seed"]),
            int(row["train_k"]),
            int(row["inference_k"]),
            int(row["checkpoint_step"]),
        ): row
        for row in manifest
    }


def load_route(run_dir: Path, row: dict[str, str]) -> dict[str, np.ndarray]:
    path = run_dir / row["route_file"]
    with np.load(path) as data:
        return {
            "layer_id": data["layer_id"].astype(np.int16),
            "gate_logits": data["gate_logits"].astype(np.float32),
            "selected_ids": data["selected_ids"].astype(np.int16),
        }


def metric_value_map(metrics: list[dict[str, str]], metric_type: str) -> dict[tuple[int, str, str], float]:
    result: dict[tuple[int, str, str], float] = {}
    for row in metrics:
        if row["metric_type"] != metric_type:
            continue
        key = (int(row["seed"]), row["pair"], row["metric_name"])
        result[key] = float(row["value"])
    return result


def final_mismatch_map(
    metrics: list[dict[str, str]],
    finals: dict[tuple[int, int], int],
) -> dict[tuple[int, int, int], float]:
    result: dict[tuple[int, int, int], float] = {}
    for row in metrics:
        if row["metric_type"] != "mismatch_cost":
            continue
        if row["metric_name"] != "validation_loss_delta":
            continue
        seed = int(row["seed"])
        train_k = int(row["train_k"])
        infer_k = int(row["inference_k"])
        step = int(row["checkpoint_step"])
        if finals.get((seed, train_k)) == step:
            result[(seed, train_k, infer_k)] = float(row["value"])
    return result


def analyze_metric_correlations(
    metrics: list[dict[str, str]],
    finals: dict[tuple[int, int], int],
) -> list[dict[str, Any]]:
    pair_metrics = metric_value_map(metrics, "matched_train_k_final_pair")
    mismatch = final_mismatch_map(metrics, finals)
    records: list[dict[str, Any]] = []
    observed_pairs = sorted({pair for _, pair, _ in pair_metrics})
    observed_metrics = sorted({name for _, _, name in pair_metrics})
    for pair in observed_pairs:
        small, large = [int(x) for x in pair.split("-")]
        for metric_name in observed_metrics:
            xs: list[float] = []
            up: list[float] = []
            down: list[float] = []
            mean_abs: list[float] = []
            for seed in sorted(seed for seed, p, name in pair_metrics if p == pair and name == metric_name):
                metric = pair_metrics.get((seed, pair, metric_name))
                up_delta = mismatch.get((seed, small, large))
                down_delta = mismatch.get((seed, large, small))
                if metric is None or up_delta is None or down_delta is None:
                    continue
                xs.append(metric)
                up.append(up_delta)
                down.append(down_delta)
                mean_abs.append((abs(up_delta) + abs(down_delta)) / 2.0)
            for target_name, ys in [
                ("up_delta_loss", up),
                ("down_delta_loss", down),
                ("mean_abs_delta_loss", mean_abs),
            ]:
                records.append(
                    {
                        "pair": pair,
                        "metric_name": metric_name,
                        "target": target_name,
                        "n": len(xs),
                        "pearson": pearson(xs, ys),
                        "spearman": spearman(xs, ys),
                        "mean_metric": float(np.mean(xs)) if xs else float("nan"),
                        "mean_target": float(np.mean(ys)) if ys else float("nan"),
                    }
                )
    return records


def analyze_layers(
    run_dir: Path,
    manifest: list[dict[str, str]],
    n_experts: int,
) -> list[dict[str, Any]]:
    finals = final_step_by_train(manifest)
    index = manifest_index(manifest)
    train_ks = sorted({int(row["train_k"]) for row in manifest})
    pairs = [(a, b) for i, a in enumerate(train_ks) for b in train_ks[i + 1 :]]
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, int, int, int], dict[str, np.ndarray]] = {}
    coactivation_cache: dict[tuple[tuple[int, int, int, int], int], np.ndarray] = {}

    def route(key: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
        if key not in cache:
            cache[key] = load_route(run_dir, index[key])
        return cache[key]

    def layer_coactivation(key: tuple[int, int, int, int], layer_id: int) -> np.ndarray:
        cache_key = (key, layer_id)
        if cache_key not in coactivation_cache:
            data = route(key)
            mask = data["layer_id"] == layer_id
            coactivation_cache[cache_key] = coactivation_matrix(data["selected_ids"][mask], n_experts)
        return coactivation_cache[cache_key]

    for seed in sorted({int(row["seed"]) for row in manifest}):
        for small, large in pairs:
            step_a = finals.get((seed, small))
            step_b = finals.get((seed, large))
            if step_a is None or step_b is None:
                continue
            key_a = (seed, small, small, step_a)
            key_b = (seed, large, large, step_b)
            if key_a not in index or key_b not in index:
                continue
            a = route(key_a)
            b = route(key_b)
            for layer_id in sorted(set(a["layer_id"].tolist()) & set(b["layer_id"].tolist())):
                mask_a = a["layer_id"] == layer_id
                mask_b = b["layer_id"] == layer_id
                for name, value in metric_rows_for_pair(
                    "layer_matched_train_k_final_pair",
                    a["selected_ids"][mask_a],
                    b["selected_ids"][mask_b],
                    a["gate_logits"][mask_a],
                    b["gate_logits"][mask_b],
                    n_experts,
                    coactivation_a=layer_coactivation(key_a, int(layer_id)) if small >= 2 else None,
                    coactivation_b=layer_coactivation(key_b, int(layer_id)) if large >= 2 else None,
                ):
                    rows.append(
                        {
                            "seed": seed,
                            "pair": f"{small}-{large}",
                            "layer_id": layer_id,
                            "metric_name": name,
                            "value": value,
                            "step_a": step_a,
                            "step_b": step_b,
                        }
                    )
    return rows


def analyze_rank_histograms(
    run_dir: Path,
    manifest: list[dict[str, str]],
) -> list[dict[str, Any]]:
    finals = final_step_by_train(manifest)
    index = manifest_index(manifest)
    train_ks = sorted({int(row["train_k"]) for row in manifest})
    pairs = [(a, b) for i, a in enumerate(train_ks) for b in train_ks[i + 1 :]]
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, int, int, int], dict[str, np.ndarray]] = {}
    inverse_rank_cache: dict[tuple[tuple[int, int, int, int], int], np.ndarray] = {}
    inverse_rank_order: list[tuple[tuple[int, int, int, int], int]] = []
    max_inverse_rank_cache = 16

    def route(key: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
        if key not in cache:
            cache[key] = load_route(run_dir, index[key])
        return cache[key]

    def inverse_ranks_for(key: tuple[int, int, int, int], layer_id: int) -> np.ndarray:
        cache_key = (key, layer_id)
        if cache_key not in inverse_rank_cache:
            data = route(key)
            mask = data["layer_id"] == layer_id
            logits = data["gate_logits"][mask]
            rank_order = np.argsort(-logits.astype(np.float32), axis=1)
            inverse_ranks = np.empty(rank_order.shape, dtype=np.int16)
            record_ids = np.arange(rank_order.shape[0])[:, None]
            inverse_ranks[record_ids, rank_order] = np.arange(1, rank_order.shape[1] + 1, dtype=np.int16)
            inverse_rank_cache[cache_key] = inverse_ranks
            inverse_rank_order.append(cache_key)
            while len(inverse_rank_order) > max_inverse_rank_cache:
                old_key = inverse_rank_order.pop(0)
                inverse_rank_cache.pop(old_key, None)
        return inverse_rank_cache[cache_key]

    for seed in sorted({int(row["seed"]) for row in manifest}):
        for small, large in pairs:
            step_a = finals.get((seed, small))
            step_b = finals.get((seed, large))
            if step_a is None or step_b is None:
                continue
            key_a = (seed, small, small, step_a)
            key_b = (seed, large, large, step_b)
            if key_a not in index or key_b not in index:
                continue
            a = route(key_a)
            b = route(key_b)
            for layer_id in sorted(set(a["layer_id"].tolist()) & set(b["layer_id"].tolist())):
                mask_a = a["layer_id"] == layer_id
                selected = a["selected_ids"][mask_a]
                inverse_ranks = inverse_ranks_for(key_b, int(layer_id))
                ranks = inverse_ranks[np.arange(selected.shape[0]), selected[:, 0]]
                counts = np.bincount(ranks.astype(np.int16), minlength=inverse_ranks.shape[1] + 1)
                total = int(ranks.size)
                for rank in range(1, inverse_ranks.shape[1] + 1):
                    count = int(counts[rank])
                    rows.append(
                        {
                            "seed": seed,
                            "pair": f"{small}-{large}",
                            "source_train_k": small,
                            "target_train_k": large,
                            "layer_id": layer_id,
                            "rank_in_target_logits": rank,
                            "count": count,
                            "share": count / max(1, total),
                            "step_source": step_a,
                            "step_target": step_b,
                        }
                    )
    return rows


def mean_by_key(rows: list[dict[str, Any]], keys: list[str], value_name: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(float(row[value_name]))
    out = []
    for key, values in sorted(grouped.items()):
        entry = {name: key[idx] for idx, name in enumerate(keys)}
        entry["mean"] = float(np.mean(values))
        entry["min"] = float(np.min(values))
        entry["max"] = float(np.max(values))
        entry["n"] = len(values)
        out.append(entry)
    return out


def write_summary(out_dir: Path, correlation_rows: list[dict[str, Any]], layer_rows: list[dict[str, Any]], rank_rows: list[dict[str, Any]]) -> None:
    layer_nested = [
        row for row in layer_rows if row["metric_name"] in {"nestedness", "top1_agreement", "spearman", "coactivation_cosine"}
    ]
    layer_means = mean_by_key(layer_nested, ["pair", "layer_id", "metric_name"], "value")
    rank_means = mean_by_key(rank_rows, ["pair", "layer_id", "rank_in_target_logits"], "share")

    lines = [
        "# Routing Diagnostics",
        "",
        "## Metric Correlations",
        "",
        "| pair | metric | target | n | pearson | spearman |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in correlation_rows:
        lines.append(
            f"| {row['pair']} | {row['metric_name']} | {row['target']} | {row['n']} | "
            f"{float(row['pearson']):.4f} | {float(row['spearman']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Layer Means",
            "",
            "| pair | layer | metric | mean | min | max | n |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in layer_means:
        lines.append(
            f"| {row['pair']} | {row['layer_id']} | {row['metric_name']} | "
            f"{row['mean']:.4f} | {row['min']:.4f} | {row['max']:.4f} | {row['n']} |"
        )

    lines.extend(
        [
            "",
            "## Rank Histogram Means",
            "",
            "For pair `a-b`, this ranks the first selected expert from the matched `train_k=a` run within the router-logit ranking of the matched `train_k=b` run.",
            "",
            "| pair | layer | rank_in_target_logits | mean_share | min | max | n |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rank_means:
        lines.append(
            f"| {row['pair']} | {row['layer_id']} | {row['rank_in_target_logits']} | "
            f"{row['mean']:.4f} | {row['min']:.4f} | {row['max']:.4f} | {row['n']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Correlations are descriptive and have small `n` for three-seed pilot runs.",
            "- `coactivation_cosine` exists only when both compared routes select at least two experts.",
            "- Layer and rank histograms are computed from raw route `.npz` files, so they must be regenerated from `/tmp` raw run directories.",
        ]
    )
    (out_dir / "routing_diagnostics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze routing diagnostics from a completed raw run directory.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-experts", type=int, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    manifest = read_csv(run_dir / "manifest.csv")
    metrics = read_csv(run_dir / "metrics.csv")
    if args.n_experts is None:
        config = {}
        config_path = run_dir / "config.json"
        if config_path.exists():
            import json

            config = json.loads(config_path.read_text(encoding="utf-8"))
        n_experts = int(config.get("n_experts", 8))
    else:
        n_experts = args.n_experts

    finals = final_step_by_train(manifest)
    correlation_rows = analyze_metric_correlations(metrics, finals)
    layer_rows = analyze_layers(run_dir, manifest, n_experts)
    rank_rows = analyze_rank_histograms(run_dir, manifest)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / "metric_correlations.csv",
        correlation_rows,
        ["pair", "metric_name", "target", "n", "pearson", "spearman", "mean_metric", "mean_target"],
    )
    write_csv(
        args.out_dir / "layer_pair_metrics.csv",
        layer_rows,
        ["seed", "pair", "layer_id", "metric_name", "value", "step_a", "step_b"],
    )
    write_csv(
        args.out_dir / "rank_position_histogram.csv",
        rank_rows,
        [
            "seed",
            "pair",
            "source_train_k",
            "target_train_k",
            "layer_id",
            "rank_in_target_logits",
            "count",
            "share",
            "step_source",
            "step_target",
        ],
    )
    write_summary(args.out_dir, correlation_rows, layer_rows, rank_rows)
    print(f"wrote diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
