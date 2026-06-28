#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from moe_topk.data import CATEGORIES, Corpus, build_corpus
from moe_topk.model import ModelConfig, TinyMoETransformer


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def subset_corpus(corpus: Corpus, samples_per_category: int) -> Corpus:
    indices = []
    seen = defaultdict(int)
    for index, task_name in enumerate(corpus.task_types):
        if seen[task_name] < samples_per_category:
            indices.append(index)
            seen[task_name] += 1
    missing = [task for task in CATEGORIES if seen[task] != samples_per_category]
    if missing:
        raise ValueError(f"probe does not contain requested samples for: {missing}")
    return Corpus(
        tokens=corpus.tokens[indices],
        sample_ids=[corpus.sample_ids[index] for index in indices],
        task_types=[corpus.task_types[index] for index in indices],
    )


def checkpoint_path(run_dir: Path, seed: int, train_k: int) -> Path:
    matches = sorted((run_dir / "checkpoints").glob(f"final_seed{seed}_train{train_k}_step*_float16.pt"))
    if len(matches) != 1:
        raise ValueError(f"expected one checkpoint for seed={seed}, train_k={train_k}, found {matches}")
    return matches[0]


def load_model(run_dir: Path, seed: int, train_k: int, device: torch.device) -> TinyMoETransformer:
    payload = torch.load(checkpoint_path(run_dir, seed, train_k), map_location="cpu", weights_only=True)
    config = ModelConfig(**payload["model_config"])
    model = TinyMoETransformer(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    if device.type == "cuda":
        model = model.half()
    return model.to(device)


def run_and_capture(
    model: TinyMoETransformer,
    inputs: torch.Tensor,
    top_k: int,
) -> list[torch.Tensor]:
    captures: list[torch.Tensor | None] = [None] * len(model.blocks)
    handles = []

    for layer_id, block in enumerate(model.blocks):
        def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...], layer: int = layer_id) -> None:
            captures[layer] = args[0].detach()

        handles.append(block.moe.register_forward_pre_hook(hook))

    try:
        with torch.no_grad():
            model(inputs, top_k=top_k, collect_routes=False)
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captures):
        raise RuntimeError("failed to capture every MoE router input")
    return [value for value in captures if value is not None]


def route_state(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    flat = logits.reshape(-1, logits.shape[-1])
    order = torch.argsort(flat, dim=1, descending=True)
    selected = order[:, :top_k]
    ranks = torch.empty_like(order)
    rank_values = torch.arange(order.shape[1], device=order.device).expand_as(order)
    ranks.scatter_(1, order, rank_values)
    return selected, ranks


def compare_route_states(
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
) -> tuple[dict[str, float], int]:
    left_ids, left_ranks = left
    right_ids, right_ranks = right
    count = int(left_ids.shape[0])
    top1 = float((left_ids[:, 0] == right_ids[:, 0]).float().mean())
    matches = left_ids[:, :, None] == right_ids[:, None, :]
    nestedness = float(matches.any(dim=2).float().mean())

    left_centered = left_ranks.float() - left_ranks.float().mean(dim=1, keepdim=True)
    right_centered = right_ranks.float() - right_ranks.float().mean(dim=1, keepdim=True)
    numerator = (left_centered * right_centered).sum(dim=1)
    denominator = torch.sqrt(
        (left_centered.square().sum(dim=1)) * (right_centered.square().sum(dim=1))
    )
    spearman = float((numerator / denominator.clamp_min(1e-12)).mean())
    return {"top1_agreement": top1, "nestedness": nestedness, "spearman": spearman}, count


def accumulate_route_metrics(
    sums: dict[tuple[str, int, str, str], float],
    counts: dict[tuple[str, int, str, str], int],
    pair: str,
    layer_id: int,
    component: str,
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
) -> None:
    values, count = compare_route_states(left, right)
    for metric_name, value in values.items():
        key = (pair, layer_id, component, metric_name)
        sums[key] += value * count
        counts[key] += count


@torch.no_grad()
def analyze_seed_decomposition(
    models: dict[int, TinyMoETransformer],
    probe: Corpus,
    train_ks: list[int],
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    route_sums: dict[tuple[str, int, str, str], float] = defaultdict(float)
    route_counts: dict[tuple[str, int, str, str], int] = defaultdict(int)
    logit_sums: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for start in range(0, probe.tokens.shape[0], batch_size):
        inputs = probe.tokens[start : start + batch_size, :-1].to(device)
        hidden_by_k = {
            train_k: run_and_capture(models[train_k], inputs, train_k)
            for train_k in train_ks
        }
        for small_index, small in enumerate(train_ks):
            for large in train_ks[small_index + 1 :]:
                pair = f"{small}-{large}"
                for layer_id in range(len(models[small].blocks)):
                    hidden_a = hidden_by_k[small][layer_id]
                    hidden_b = hidden_by_k[large][layer_id]
                    weight_a = models[small].blocks[layer_id].moe.router.weight
                    weight_b = models[large].blocks[layer_id].moe.router.weight
                    aa = F.linear(hidden_a, weight_a).float()
                    ab = F.linear(hidden_a, weight_b).float()
                    ba = F.linear(hidden_b, weight_a).float()
                    bb = F.linear(hidden_b, weight_b).float()

                    state_aa = route_state(aa, small)
                    state_ab = route_state(ab, large)
                    state_ba = route_state(ba, small)
                    state_bb = route_state(bb, large)
                    accumulate_route_metrics(route_sums, route_counts, pair, layer_id, "total", state_aa, state_bb)
                    accumulate_route_metrics(
                        route_sums, route_counts, pair, layer_id, "router_on_hidden_a", state_aa, state_ab
                    )
                    accumulate_route_metrics(
                        route_sums, route_counts, pair, layer_id, "router_on_hidden_b", state_ba, state_bb
                    )
                    accumulate_route_metrics(
                        route_sums, route_counts, pair, layer_id, "representation_with_router_a", state_aa, state_ba
                    )
                    accumulate_route_metrics(
                        route_sums, route_counts, pair, layer_id, "representation_with_router_b", state_ab, state_bb
                    )

                    total = bb - aa
                    router = 0.5 * ((ab - aa) + (bb - ba))
                    representation = 0.5 * ((ba - aa) + (bb - ab))
                    interaction = bb - ba - ab + aa
                    key = (pair, layer_id)
                    logit_sums[key]["num_values"] += float(total.numel())
                    logit_sums[key]["total_squared"] += float(total.square().sum())
                    logit_sums[key]["router_projection"] += float((total * router).sum())
                    logit_sums[key]["representation_projection"] += float((total * representation).sum())
                    logit_sums[key]["router_squared"] += float(router.square().sum())
                    logit_sums[key]["representation_squared"] += float(representation.square().sum())
                    logit_sums[key]["interaction_squared"] += float(interaction.square().sum())
                    logit_sums[key]["decomposition_residual"] += float((total - router - representation).abs().sum())

        del hidden_by_k
        if device.type == "cuda":
            torch.cuda.empty_cache()

    route_rows = []
    for key in sorted(route_sums):
        pair, layer_id, component, metric_name = key
        route_rows.append(
            {
                "pair": pair,
                "layer_id": layer_id,
                "component": component,
                "metric_name": metric_name,
                "value": route_sums[key] / route_counts[key],
                "num_records": route_counts[key],
            }
        )

    logit_rows = []
    for (pair, layer_id), values in sorted(logit_sums.items()):
        count = values["num_values"]
        total_mse = values["total_squared"] / count
        router_contribution = values["router_projection"] / count
        representation_contribution = values["representation_projection"] / count
        logit_rows.append(
            {
                "pair": pair,
                "layer_id": layer_id,
                "total_logit_mse": total_mse,
                "router_contribution": router_contribution,
                "representation_contribution": representation_contribution,
                "router_share": router_contribution / total_mse if total_mse > 0 else float("nan"),
                "representation_share": representation_contribution / total_mse if total_mse > 0 else float("nan"),
                "router_component_mse": values["router_squared"] / count,
                "representation_component_mse": values["representation_squared"] / count,
                "interaction_mse": values["interaction_squared"] / count,
                "mean_abs_decomposition_residual": values["decomposition_residual"] / count,
                "num_values": int(count),
            }
        )
    return route_rows, logit_rows


def evaluate_loss(
    model: TinyMoETransformer,
    probe: Corpus,
    top_k: int,
    batch_size: int,
    device: torch.device,
) -> float:
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, probe.tokens.shape[0], batch_size):
            batch = probe.tokens[start : start + batch_size].to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits, _, _ = model(inputs, top_k=top_k, collect_routes=False)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss)
            total_tokens += int(targets.numel())
    return total_loss / max(1, total_tokens)


def copy_router_layers(
    host: TinyMoETransformer,
    donor: TinyMoETransformer,
    layers: list[int],
) -> None:
    with torch.no_grad():
        for layer_id in layers:
            host.blocks[layer_id].moe.router.weight.copy_(donor.blocks[layer_id].moe.router.weight)


@torch.no_grad()
def analyze_seed_transplants(
    models: dict[int, TinyMoETransformer],
    probe: Corpus,
    train_ks: list[int],
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    n_layers = len(next(iter(models.values())).blocks)
    for host_k in train_ks:
        host = models[host_k]
        original = [block.moe.router.weight.detach().clone() for block in host.blocks]
        baseline = evaluate_loss(host, probe, host_k, batch_size, device)
        for donor_k in train_ks:
            if donor_k == host_k:
                rows.append(
                    {
                        "host_k": host_k,
                        "donor_k": donor_k,
                        "swap_scope": "all_layers",
                        "layer_id": -1,
                        "validation_loss": baseline,
                        "baseline_loss": baseline,
                        "delta_loss": 0.0,
                    }
                )
                for layer_id in range(n_layers):
                    rows.append(
                        {
                            "host_k": host_k,
                            "donor_k": donor_k,
                            "swap_scope": "single_layer",
                            "layer_id": layer_id,
                            "validation_loss": baseline,
                            "baseline_loss": baseline,
                            "delta_loss": 0.0,
                        }
                    )
                continue

            donor = models[donor_k]
            copy_router_layers(host, donor, list(range(n_layers)))
            swapped_loss = evaluate_loss(host, probe, host_k, batch_size, device)
            rows.append(
                {
                    "host_k": host_k,
                    "donor_k": donor_k,
                    "swap_scope": "all_layers",
                    "layer_id": -1,
                    "validation_loss": swapped_loss,
                    "baseline_loss": baseline,
                    "delta_loss": swapped_loss - baseline,
                }
            )
            with torch.no_grad():
                for layer_id, value in enumerate(original):
                    host.blocks[layer_id].moe.router.weight.copy_(value)

            for layer_id in range(n_layers):
                copy_router_layers(host, donor, [layer_id])
                layer_loss = evaluate_loss(host, probe, host_k, batch_size, device)
                rows.append(
                    {
                        "host_k": host_k,
                        "donor_k": donor_k,
                        "swap_scope": "single_layer",
                        "layer_id": layer_id,
                        "validation_loss": layer_loss,
                        "baseline_loss": baseline,
                        "delta_loss": layer_loss - baseline,
                    }
                )
                with torch.no_grad():
                    host.blocks[layer_id].moe.router.weight.copy_(original[layer_id])
        print(f"[transplant] host_k={host_k} complete", flush=True)
    return rows


def mean_rows(rows: list[dict[str, Any]], keys: list[str], values: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = []
    for key, selected in sorted(grouped.items()):
        result.append(
            {
                **dict(zip(keys, key)),
                **{value: float(np.mean([float(row[value]) for row in selected])) for value in values},
                "num_seeds": len(selected),
            }
        )
    return result


def draw_matrix(
    matrix: np.ndarray,
    labels: list[int],
    title: str,
    out_path: Path,
    x_label: str,
    y_label: str,
    diverging: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 8.0))
    if diverging:
        vmax = float(np.nanmax(np.abs(matrix)))
        image = ax.imshow(matrix, vmin=-vmax, vmax=vmax, cmap="coolwarm")
    else:
        image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    for row_id in range(matrix.shape[0]):
        for column_id in range(matrix.shape[1]):
            value = matrix[row_id, column_id]
            if np.isfinite(value):
                ax.text(column_id, row_id, f"{value:+.2f}" if diverging else f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_plots(
    out_dir: Path,
    logit_mean: list[dict[str, Any]],
    transplant_mean: list[dict[str, Any]],
    train_ks: list[int],
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    index = {value: idx for idx, value in enumerate(train_ks)}
    router_share = np.full((len(train_ks), len(train_ks)), np.nan)
    for row in logit_mean:
        small, large = [int(value) for value in str(row["pair"]).split("-")]
        value = float(row["router_share"])
        router_share[index[small], index[large]] = value
        router_share[index[large], index[small]] = value
    draw_matrix(
        router_share,
        train_ks,
        "Router share of matched logit difference",
        plot_dir / "router_share_heatmap.png",
        "training k",
        "training k",
    )

    for layer_id in [-1] + list(range(8)):
        scope = "all_layers" if layer_id == -1 else "single_layer"
        matrix = np.full((len(train_ks), len(train_ks)), np.nan)
        for row in transplant_mean:
            if row["swap_scope"] == scope and int(row["layer_id"]) == layer_id:
                matrix[index[int(row["host_k"])], index[int(row["donor_k"])]] = float(row["delta_loss"])
        name = "all_layers" if layer_id == -1 else f"layer{layer_id}"
        draw_matrix(
            matrix,
            train_ks,
            f"Router transplant loss delta: {name}",
            plot_dir / f"router_transplant_{name}_heatmap.png",
            "donor training k",
            "host training k",
            diverging=True,
        )


def write_summary(
    out_dir: Path,
    logit_mean: list[dict[str, Any]],
    transplant_mean: list[dict[str, Any]],
) -> None:
    lines = [
        "# Router and Representation Mechanism Analysis",
        "",
        "The logit decomposition uses the exact identity `total = router_shapley + representation_shapley` for the diagonal change from model A to B.",
        "",
        "## Key Logit Decomposition",
        "",
        "| pair | total_mse | router_share | representation_share | interaction_mse |",
        "|---|---:|---:|---:|---:|",
    ]
    for pair in ("1-2", "1-4", "1-8", "2-4", "2-8", "4-8", "7-8"):
        selected = [row for row in logit_mean if row["pair"] == pair]
        if not selected:
            continue
        lines.append(
            f"| {pair} | {float(np.mean([row['total_logit_mse'] for row in selected])):.6f} | "
            f"{float(np.mean([row['router_share'] for row in selected])):.4f} | "
            f"{float(np.mean([row['representation_share'] for row in selected])):.4f} | "
            f"{float(np.mean([row['interaction_mse'] for row in selected])):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Key All-layer Router Transplants",
            "",
            "| host_k | donor_k | mean_delta_loss |",
            "|---:|---:|---:|",
        ]
    )
    for host_k, donor_k in ((1, 4), (4, 1), (1, 8), (8, 1), (2, 8), (8, 2), (4, 8), (8, 4)):
        selected = [
            row
            for row in transplant_mean
            if row["swap_scope"] == "all_layers"
            and int(row["host_k"]) == host_k
            and int(row["donor_k"]) == donor_k
        ]
        if selected:
            lines.append(f"| {host_k} | {donor_k} | {float(selected[0]['delta_loss']):+.4f} |")

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Router and representation contributions are Shapley path averages in aligned hidden coordinates, not independent causal variables.",
            "- Router shares can fall outside [0, 1] when router and representation changes oppose each other.",
            "- Router transplant loss includes router-expert co-adaptation and downstream hidden-state changes.",
            "- The mechanism probe is a deterministic subset of the synthetic validation corpus.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose routing changes into router and representation effects.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-ks", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--probe-samples-per-category", type=int, default=8)
    parser.add_argument("--probe-batch-size", type=int, default=8)
    parser.add_argument("--transplant-samples-per-category", type=int, default=2)
    parser.add_argument("--transplant-batch-size", type=int, default=14)
    args = parser.parse_args()

    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    full_probe = build_corpus(
        samples_per_category=int(config["probe_samples_per_category"]),
        seq_len=int(config["seq_len"]),
        seed=int(config["data_seed"]) + 99_999,
    )
    mechanism_probe = subset_corpus(full_probe, args.probe_samples_per_category)
    transplant_probe = subset_corpus(full_probe, args.transplant_samples_per_category)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    route_rows: list[dict[str, Any]] = []
    logit_rows: list[dict[str, Any]] = []
    transplant_rows: list[dict[str, Any]] = []
    train_ks = sorted(set(args.train_ks))
    for seed in args.seeds:
        print(f"[load-seed] {seed}", flush=True)
        models = {train_k: load_model(args.run_dir, seed, train_k, device) for train_k in train_ks}
        seed_route, seed_logit = analyze_seed_decomposition(
            models,
            mechanism_probe,
            train_ks,
            args.probe_batch_size,
            device,
        )
        route_rows.extend({"seed": seed, **row} for row in seed_route)
        logit_rows.extend({"seed": seed, **row} for row in seed_logit)
        seed_transplants = analyze_seed_transplants(
            models,
            transplant_probe,
            train_ks,
            args.transplant_batch_size,
            device,
        )
        transplant_rows.extend({"seed": seed, **row} for row in seed_transplants)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[seed-done] {seed}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.out_dir / "route_component_metrics.csv",
        route_rows,
        ["seed", "pair", "layer_id", "component", "metric_name", "value", "num_records"],
    )
    write_csv(
        args.out_dir / "logit_shapley_decomposition.csv",
        logit_rows,
        [
            "seed",
            "pair",
            "layer_id",
            "total_logit_mse",
            "router_contribution",
            "representation_contribution",
            "router_share",
            "representation_share",
            "router_component_mse",
            "representation_component_mse",
            "interaction_mse",
            "mean_abs_decomposition_residual",
            "num_values",
        ],
    )
    write_csv(
        args.out_dir / "router_transplant_loss.csv",
        transplant_rows,
        [
            "seed",
            "host_k",
            "donor_k",
            "swap_scope",
            "layer_id",
            "validation_loss",
            "baseline_loss",
            "delta_loss",
        ],
    )

    logit_mean = mean_rows(
        logit_rows,
        ["pair", "layer_id"],
        [
            "total_logit_mse",
            "router_contribution",
            "representation_contribution",
            "router_share",
            "representation_share",
            "router_component_mse",
            "representation_component_mse",
            "interaction_mse",
            "mean_abs_decomposition_residual",
        ],
    )
    transplant_mean = mean_rows(
        transplant_rows,
        ["host_k", "donor_k", "swap_scope", "layer_id"],
        ["validation_loss", "baseline_loss", "delta_loss"],
    )
    write_csv(
        args.out_dir / "logit_shapley_mean.csv",
        logit_mean,
        [
            "pair",
            "layer_id",
            "total_logit_mse",
            "router_contribution",
            "representation_contribution",
            "router_share",
            "representation_share",
            "router_component_mse",
            "representation_component_mse",
            "interaction_mse",
            "mean_abs_decomposition_residual",
            "num_seeds",
        ],
    )
    write_csv(
        args.out_dir / "router_transplant_loss_mean.csv",
        transplant_mean,
        [
            "host_k",
            "donor_k",
            "swap_scope",
            "layer_id",
            "validation_loss",
            "baseline_loss",
            "delta_loss",
            "num_seeds",
        ],
    )
    write_plots(args.out_dir, logit_mean, transplant_mean, train_ks)
    write_summary(args.out_dir, logit_mean, transplant_mean)
    print(f"wrote router/representation analysis to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
