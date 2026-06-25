from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from moe_topk.data import CATEGORIES, Corpus, batch_indices, build_corpus, build_jsonl_corpus
from moe_topk.metrics import expert_frequency, metric_rows_for_pair, topk_ids_from_logits
from moe_topk.model import ModelConfig, TinyMoETransformer


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_env(run_dir: Path) -> None:
    lines = [
        f"python={sys.version}",
        f"torch={torch.__version__}",
        f"torch_cuda={torch.version.cuda}",
        f"cuda_available={torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"cuda_device={torch.cuda.get_device_name(0)}")
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        lines.append("nvidia_smi_stdout=")
        lines.append(smi.stdout)
        if smi.stderr:
            lines.append("nvidia_smi_stderr=")
            lines.append(smi.stderr)
    except Exception as exc:  # pragma: no cover - environment specific
        lines.append(f"nvidia_smi_error={type(exc).__name__}: {exc}")
    (run_dir / "env.txt").write_text("\n".join(lines), encoding="utf-8")


def model_config_from_dict(config: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        vocab_size=int(config["vocab_size"]),
        seq_len=int(config["seq_len"]),
        n_layers=int(config["n_layers"]),
        d_model=int(config["d_model"]),
        n_heads=int(config["n_heads"]),
        n_experts=int(config["n_experts"]),
        expert_hidden=int(config["expert_hidden"]),
        dropout=float(config["dropout"]),
    )


def make_model(config: dict[str, Any], device: torch.device) -> TinyMoETransformer:
    model = TinyMoETransformer(model_config_from_dict(config))
    return model.to(device)


def ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def route_filename(seed: int, train_k: int, infer_k: int, step: int) -> str:
    return f"routes/seed{seed}_train{train_k}_infer{infer_k}_step{step}.npz"


def steps_for_train_k(config: dict[str, Any], train_k: int, mode: str) -> int:
    if mode == "smoke":
        return int(config["smoke_steps"])
    per_k = config.get("train_steps_by_k", {})
    return int(per_k.get(str(train_k), config["steps"]))


def checkpoints_for_train_k(
    config: dict[str, Any],
    train_k: int,
    steps: int,
    mode: str,
) -> list[int]:
    if mode == "smoke":
        return [0, steps]
    per_k = config.get("checkpoints_by_k", {})
    raw = per_k.get(str(train_k), config["checkpoints"])
    values = sorted({int(step) for step in raw if int(step) <= steps} | {0, steps})
    return values


def pair_specs(values: list[int]) -> list[tuple[int, int]]:
    ordered = sorted({int(value) for value in values})
    return [(small, large) for i, small in enumerate(ordered) for large in ordered[i + 1 :]]


def evaluate_and_collect(
    model: TinyMoETransformer,
    probe: Corpus,
    task_to_id: dict[str, int],
    config: dict[str, Any],
    run_dir: Path,
    seed: int,
    train_k: int,
    infer_k: int,
    step: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    eval_batch_size = int(config["eval_batch_size"])
    total_loss = 0.0
    total_tokens = 0
    arrays: dict[str, list[np.ndarray]] = {
        "sample_index": [],
        "task_id": [],
        "layer_id": [],
        "token_pos": [],
        "gate_logits": [],
        "selected_ids": [],
        "selected_weights": [],
    }

    with torch.no_grad():
        for start in range(0, probe.tokens.shape[0], eval_batch_size):
            end = min(start + eval_batch_size, probe.tokens.shape[0])
            batch = probe.tokens[start:end].to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits, aux_loss, routes = model(inputs, top_k=infer_k, collect_routes=True)
            loss = ce_loss(logits, targets)
            total_loss += float(loss.detach().cpu()) * targets.numel()
            total_tokens += int(targets.numel())

            batch_size, seq_len = inputs.shape
            sample_index = np.repeat(np.arange(start, end, dtype=np.int32), seq_len)
            task_id = np.repeat(
                np.array([task_to_id[t] for t in probe.task_types[start:end]], dtype=np.int16),
                seq_len,
            )
            token_pos = np.tile(np.arange(seq_len, dtype=np.int16), batch_size)

            for layer_id, route in enumerate(routes):
                n_records = batch_size * seq_len
                arrays["sample_index"].append(sample_index)
                arrays["task_id"].append(task_id)
                arrays["layer_id"].append(np.full(n_records, layer_id, dtype=np.int16))
                arrays["token_pos"].append(token_pos)
                arrays["gate_logits"].append(
                    route["gate_logits"].detach().cpu().numpy().reshape(n_records, -1).astype(np.float16)
                )
                arrays["selected_ids"].append(
                    route["selected_ids"].detach().cpu().numpy().reshape(n_records, infer_k).astype(np.int16)
                )
                arrays["selected_weights"].append(
                    route["selected_weights"].detach().cpu().numpy().reshape(n_records, infer_k).astype(np.float16)
                )

    route_path = run_dir / route_filename(seed, train_k, infer_k, step)
    route_path.parent.mkdir(parents=True, exist_ok=True)
    merged = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    np.savez_compressed(
        route_path,
        **merged,
        task_names=np.array(CATEGORIES),
    )
    freq = expert_frequency(merged["selected_ids"], int(config["n_experts"]))
    return {
        "seed": seed,
        "train_k": train_k,
        "inference_k": infer_k,
        "checkpoint_step": step,
        "route_file": route_path.relative_to(run_dir).as_posix(),
        "num_records": int(merged["selected_ids"].shape[0]),
        "validation_loss": total_loss / max(1, total_tokens),
        "max_expert_share": freq["max_share"],
        "expert_entropy": freq["entropy"],
        "expert_normalized_entropy": freq["normalized_entropy"],
    }


def append_manifest(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    path = run_dir / "manifest.csv"
    exists = path.exists()
    fieldnames = [
        "seed",
        "train_k",
        "inference_k",
        "checkpoint_step",
        "route_file",
        "num_records",
        "validation_loss",
        "max_expert_share",
        "expert_entropy",
        "expert_normalized_entropy",
    ]
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def train_one(
    config: dict[str, Any],
    seed: int,
    train_k: int,
    steps: int,
    checkpoints: list[int],
    initial_state: dict[str, torch.Tensor],
    train_corpus: Corpus,
    probe_corpus: Corpus,
    task_to_id: dict[str, int],
    run_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    model = make_model(config, device)
    model.load_state_dict({key: value.to(device) for key, value in initial_state.items()})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    checkpoint_set = set(checkpoints)
    manifest_rows: list[dict[str, Any]] = []
    log_path = run_dir / "train_log.csv"
    log_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as log_handle:
        log_writer = csv.DictWriter(
            log_handle,
            fieldnames=["seed", "train_k", "step", "loss", "ce_loss", "aux_loss"],
        )
        if not log_exists:
            log_writer.writeheader()

        def collect(step: int) -> None:
            for infer_k in config["inference_ks"]:
                manifest_rows.append(
                    evaluate_and_collect(
                        model=model,
                        probe=probe_corpus,
                        task_to_id=task_to_id,
                        config=config,
                        run_dir=run_dir,
                        seed=seed,
                        train_k=train_k,
                        infer_k=int(infer_k),
                        step=step,
                        device=device,
                    )
                )

        if 0 in checkpoint_set:
            collect(0)
            append_manifest(run_dir, manifest_rows)
            manifest_rows.clear()

        order_seed = int(config["data_order_seed"]) + seed
        batches = batch_indices(
            num_items=train_corpus.tokens.shape[0],
            batch_size=int(config["batch_size"]),
            steps=steps,
            seed=order_seed,
        )
        for step, indices in enumerate(batches, start=1):
            model.train()
            batch = train_corpus.tokens[indices].to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits, aux_loss, _ = model(inputs, top_k=train_k, collect_routes=False)
            base_loss = ce_loss(logits, targets)
            loss = base_loss + float(config["router_aux_loss_coef"]) * aux_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step == 1 or step % 50 == 0 or step in checkpoint_set:
                log_writer.writerow(
                    {
                        "seed": seed,
                        "train_k": train_k,
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "ce_loss": float(base_loss.detach().cpu()),
                        "aux_loss": float(aux_loss.detach().cpu()),
                    }
                )
                log_handle.flush()

            if step in checkpoint_set:
                collect(step)
                append_manifest(run_dir, manifest_rows)
                manifest_rows.clear()

    return manifest_rows


def read_manifest(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "manifest.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_route(run_dir: Path, row: dict[str, str]) -> dict[str, np.ndarray]:
    data = np.load(run_dir / row["route_file"])
    return {
        "selected_ids": data["selected_ids"],
        "gate_logits": data["gate_logits"],
    }


def write_metric_row(writer: csv.DictWriter, **kwargs: Any) -> None:
    row = {
        "metric_type": kwargs.get("metric_type", ""),
        "metric_name": kwargs.get("metric_name", ""),
        "seed": kwargs.get("seed", ""),
        "checkpoint_step": kwargs.get("checkpoint_step", ""),
        "train_k": kwargs.get("train_k", ""),
        "inference_k": kwargs.get("inference_k", ""),
        "train_k_b": kwargs.get("train_k_b", ""),
        "inference_k_b": kwargs.get("inference_k_b", ""),
        "pair": kwargs.get("pair", ""),
        "value": kwargs.get("value", ""),
        "extra": kwargs.get("extra", ""),
    }
    writer.writerow(row)


def analyze(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifest = read_manifest(run_dir)
    index = {
        (
            int(row["seed"]),
            int(row["train_k"]),
            int(row["inference_k"]),
            int(row["checkpoint_step"]),
        ): row
        for row in manifest
    }
    route_cache: dict[tuple[int, int, int, int], dict[str, np.ndarray]] = {}
    route_cache_order: list[tuple[int, int, int, int]] = []
    max_cached_routes = int(config.get("analysis_route_cache_size", 24))

    def route(key: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
        if key not in route_cache:
            route_cache[key] = load_route(run_dir, index[key])
            route_cache_order.append(key)
            while len(route_cache_order) > max_cached_routes:
                old_key = route_cache_order.pop(0)
                route_cache.pop(old_key, None)
        return route_cache[key]

    metrics_path = run_dir / "metrics.csv"
    fieldnames = [
        "metric_type",
        "metric_name",
        "seed",
        "checkpoint_step",
        "train_k",
        "inference_k",
        "train_k_b",
        "inference_k_b",
        "pair",
        "value",
        "extra",
    ]
    final_step = max(int(row["checkpoint_step"]) for row in manifest)
    final_step_by_train: dict[tuple[int, int], int] = {}
    for row in manifest:
        seed = int(row["seed"])
        train_k = int(row["train_k"])
        infer_k = int(row["inference_k"])
        step = int(row["checkpoint_step"])
        if train_k == infer_k:
            final_step_by_train[(seed, train_k)] = max(
                final_step_by_train.get((seed, train_k), -1),
                step,
            )
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in manifest:
            write_metric_row(
                writer,
                metric_type="expert_frequency",
                metric_name="max_expert_share",
                seed=row["seed"],
                checkpoint_step=row["checkpoint_step"],
                train_k=row["train_k"],
                inference_k=row["inference_k"],
                value=row["max_expert_share"],
            )
            write_metric_row(
                writer,
                metric_type="loss",
                metric_name="validation_loss",
                seed=row["seed"],
                checkpoint_step=row["checkpoint_step"],
                train_k=row["train_k"],
                inference_k=row["inference_k"],
                value=row["validation_loss"],
            )

        train_pairs = pair_specs([int(value) for value in config["train_ks"]])
        inference_pairs = pair_specs([int(value) for value in config["inference_ks"]])
        max_inference_k = max(int(value) for value in config["inference_ks"])
        for seed in config["seeds"]:
            seed_steps = sorted(
                {
                    int(row["checkpoint_step"])
                    for row in manifest
                    if int(row["seed"]) == int(seed)
                }
            )
            for step in seed_steps:
                for train_k in config["train_ks"]:
                    reference_key = (seed, train_k, max_inference_k, step)
                    if reference_key not in index:
                        continue
                    reference = route(reference_key)
                    for small, large in inference_pairs:
                        small_ids = topk_ids_from_logits(reference["gate_logits"], small)
                        large_ids = topk_ids_from_logits(reference["gate_logits"], large)
                        for name, value in metric_rows_for_pair(
                            "logit_cutoff_sanity",
                            small_ids,
                            large_ids,
                            reference["gate_logits"],
                            reference["gate_logits"],
                            int(config["n_experts"]),
                        ):
                            write_metric_row(
                                writer,
                                metric_type="logit_cutoff_sanity",
                                metric_name=name,
                                seed=seed,
                                checkpoint_step=step,
                                train_k=train_k,
                                inference_k=small,
                                inference_k_b=large,
                                pair=f"{small}-{large}",
                                value=value,
                                extra="derived_from_same_gate_logits",
                            )

                for train_k in config["train_ks"]:
                    for small, large in inference_pairs:
                        key_a = (seed, train_k, small, step)
                        key_b = (seed, train_k, large, step)
                        if key_a not in index or key_b not in index:
                            continue
                        a = route(key_a)
                        b = route(key_b)
                        for name, value in metric_rows_for_pair(
                            "same_router_cutoff",
                            a["selected_ids"],
                            b["selected_ids"],
                            a["gate_logits"],
                            b["gate_logits"],
                            int(config["n_experts"]),
                        ):
                            write_metric_row(
                                writer,
                                metric_type="same_weight_infer_path",
                                metric_name=name,
                                seed=seed,
                                checkpoint_step=step,
                                train_k=train_k,
                                inference_k=small,
                                inference_k_b=large,
                                pair=f"{small}-{large}",
                                value=value,
                            )

                for small, large in train_pairs:
                    key_a = (seed, small, small, step)
                    key_b = (seed, large, large, step)
                    if key_a not in index or key_b not in index:
                        continue
                    a = route(key_a)
                    b = route(key_b)
                    for name, value in metric_rows_for_pair(
                        "matched_train_k_pair",
                        a["selected_ids"],
                        b["selected_ids"],
                        a["gate_logits"],
                        b["gate_logits"],
                        int(config["n_experts"]),
                    ):
                        write_metric_row(
                            writer,
                            metric_type="matched_train_k_pair",
                            metric_name=name,
                            seed=seed,
                            checkpoint_step=step,
                            train_k=small,
                            inference_k=small,
                            train_k_b=large,
                            inference_k_b=large,
                            pair=f"{small}-{large}",
                            value=value,
                        )
                        if step == 0:
                            write_metric_row(
                                writer,
                                metric_type="step0_matched_path",
                                metric_name=name,
                                seed=seed,
                                checkpoint_step=step,
                                train_k=small,
                                inference_k=small,
                                train_k_b=large,
                                inference_k_b=large,
                                pair=f"{small}-{large}",
                                value=value,
                            )

                if step == 0:
                    for infer_k in config["inference_ks"]:
                        for left, right in train_pairs:
                            key_a = (seed, left, infer_k, step)
                            key_b = (seed, right, infer_k, step)
                            if key_a not in index or key_b not in index:
                                continue
                            a = route(key_a)
                            b = route(key_b)
                            for name, value in metric_rows_for_pair(
                                "step0_same_w0_same_infer",
                                a["selected_ids"],
                                b["selected_ids"],
                                a["gate_logits"],
                                b["gate_logits"],
                                int(config["n_experts"]),
                            ):
                                write_metric_row(
                                    writer,
                                    metric_type="step0_same_w0_same_infer",
                                    metric_name=name,
                                    seed=seed,
                                    checkpoint_step=step,
                                    train_k=left,
                                    inference_k=infer_k,
                                    train_k_b=right,
                                    inference_k_b=infer_k,
                                    pair=f"{left}-{right}",
                                    value=value,
                                )

                for train_k in config["train_ks"]:
                    matched = index.get((seed, train_k, train_k, step))
                    if matched is None:
                        continue
                    matched_loss = float(matched["validation_loss"])
                    for infer_k in config["inference_ks"]:
                        if infer_k == train_k:
                            continue
                        other = index.get((seed, train_k, infer_k, step))
                        if other is None:
                            continue
                        value = float(other["validation_loss"]) - matched_loss
                        write_metric_row(
                            writer,
                            metric_type="mismatch_cost",
                            metric_name="validation_loss_delta",
                            seed=seed,
                            checkpoint_step=step,
                            train_k=train_k,
                            inference_k=infer_k,
                            value=value,
                        )

            for small, large in train_pairs:
                step_a = final_step_by_train.get((int(seed), small))
                step_b = final_step_by_train.get((int(seed), large))
                if step_a is None or step_b is None:
                    continue
                key_a = (int(seed), small, small, step_a)
                key_b = (int(seed), large, large, step_b)
                if key_a not in index or key_b not in index:
                    continue
                a = route(key_a)
                b = route(key_b)
                for name, value in metric_rows_for_pair(
                    "matched_train_k_final_pair",
                    a["selected_ids"],
                    b["selected_ids"],
                    a["gate_logits"],
                    b["gate_logits"],
                    int(config["n_experts"]),
                ):
                    write_metric_row(
                        writer,
                        metric_type="matched_train_k_final_pair",
                        metric_name=name,
                        seed=seed,
                        checkpoint_step=f"{step_a}/{step_b}",
                        train_k=small,
                        inference_k=small,
                        train_k_b=large,
                        inference_k_b=large,
                        pair=f"{small}-{large}",
                        value=value,
                        extra="own_final_steps",
                    )

    summary = summarize(run_dir, config, manifest, metrics_path, final_step)
    write_plots(run_dir, metrics_path)
    return summary


def _read_metric_rows(metrics_path: Path) -> list[dict[str, str]]:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(
    run_dir: Path,
    config: dict[str, Any],
    manifest: list[dict[str, str]],
    metrics_path: Path,
    final_step: int,
) -> dict[str, Any]:
    rows = _read_metric_rows(metrics_path)
    if config.get("run_mode") == "smoke":
        expected_runs = 1
    else:
        expected_runs = len(config["seeds"]) * len(config["train_ks"])
    final_step_by_train: dict[tuple[int, int], int] = {}
    for row in manifest:
        seed = int(row["seed"])
        train_k = int(row["train_k"])
        infer_k = int(row["inference_k"])
        step = int(row["checkpoint_step"])
        if train_k == infer_k:
            final_step_by_train[(seed, train_k)] = max(
                final_step_by_train.get((seed, train_k), -1),
                step,
            )
    completed_runs = len(
        {
            (int(row["seed"]), int(row["train_k"]))
            for row in manifest
            if int(row["inference_k"]) == int(row["train_k"])
            and int(row["checkpoint_step"])
            == final_step_by_train.get((int(row["seed"]), int(row["train_k"])))
        }
    )

    logit_cutoff_nested = [
        float(row["value"])
        for row in rows
        if row["metric_type"] == "logit_cutoff_sanity" and row["metric_name"] == "nestedness"
    ]
    step0_nested = [
        float(row["value"])
        for row in rows
        if row["metric_type"] == "step0_same_w0_same_infer" and row["metric_name"] == "nestedness"
    ]
    same_weight_path_nested = [
        float(row["value"])
        for row in rows
        if row["metric_type"] == "same_weight_infer_path" and row["metric_name"] == "nestedness"
    ]
    collapse_rows = []
    for row in manifest:
        seed = int(row["seed"])
        train_k = int(row["train_k"])
        infer_k = int(row["inference_k"])
        step = int(row["checkpoint_step"])
        if train_k != infer_k:
            continue
        if step != final_step_by_train.get((seed, train_k)):
            continue
        if float(row["max_expert_share"]) > float(config["collapse_threshold"]):
            collapse_rows.append(row)
    final_pair_rows = [
        row
        for row in rows
        if row["metric_type"] == "matched_train_k_final_pair"
        and row["metric_name"] in {"nestedness", "top1_agreement", "spearman"}
    ]
    final_mismatch_rows = []
    for row in rows:
        if row["metric_type"] != "mismatch_cost" or row["metric_name"] != "validation_loss_delta":
            continue
        seed = int(row["seed"])
        train_k = int(row["train_k"])
        expected_step = final_step_by_train.get((seed, train_k))
        if expected_step is not None and int(row["checkpoint_step"]) == expected_step:
            final_mismatch_rows.append(row)

    pair_values: dict[tuple[str, str], list[float]] = {}
    for row in final_pair_rows:
        pair_values.setdefault((row["pair"], row["metric_name"]), []).append(float(row["value"]))

    mismatch_values: dict[tuple[str, str], list[float]] = {}
    for row in final_mismatch_rows:
        mismatch_values.setdefault((row["train_k"], row["inference_k"]), []).append(float(row["value"]))

    candidate_lines = []
    for (pair, metric), values in sorted(pair_values.items()):
        if metric not in {"nestedness", "top1_agreement"}:
            continue
        below = sum(1 for value in values if value < 0.95)
        min_support = max(2, len(values) // 2 + 1)
        candidate_lines.append((pair, metric, below, len(values), below >= min_support, min_support))

    title = str(config.get("experiment_name", "scratch_pilot")).replace("_", " ").title()
    lines = [
        f"# {title} Summary",
        "",
        f"Run directory: `{run_dir}`",
        f"Expected completed runs: {expected_runs}",
        f"Completed final matched runs: {completed_runs}",
        f"Final checkpoint step: {final_step}",
        "",
        "## Gates",
        "",
        f"- logit cutoff sanity nestedness min: {min(logit_cutoff_nested) if logit_cutoff_nested else 'NA'}",
        f"- step-0 same-W0 same-infer nestedness min: {min(step0_nested) if step0_nested else 'NA'}",
        f"- same-weight different-infer-path nestedness min: {min(same_weight_path_nested) if same_weight_path_nested else 'NA'}",
        f"- collapse threshold: {config['collapse_threshold']}",
        f"- collapsed final matched runs: {len(collapse_rows)}",
        "",
        "## Final Matched Train-k Pair Metrics",
        "",
        "| pair | metric | mean | values |",
        "|---|---:|---:|---|",
    ]
    for (pair, metric), values in sorted(pair_values.items()):
        mean_value = sum(values) / max(1, len(values))
        value_text = ", ".join(f"{value:.4f}" for value in values)
        lines.append(f"| {pair} | {metric} | {mean_value:.4f} | {value_text} |")

    lines.extend(["", "## Final Mismatch Cost", "", "| train_k | inference_k | mean_delta_loss | values |", "|---:|---:|---:|---|"])
    for (train_k, inference_k), values in sorted(mismatch_values.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        mean_value = sum(values) / max(1, len(values))
        value_text = ", ".join(f"{value:.4f}" for value in values)
        lines.append(f"| {train_k} | {inference_k} | {mean_value:.4f} | {value_text} |")

    lines.extend(["", "## Candidate Direction Check", "", "| pair | metric | seeds_below_0.95 | candidate_effect |", "|---|---|---:|---|"])
    for pair, metric, below, total, is_candidate, min_support in candidate_lines:
        lines.append(f"| {pair} | {metric} | {below}/{total} | {str(is_candidate).lower()} |")

    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
        "Use `candidate effect` only when a strict seed majority moves in the same direction.",
        f"For this run, the minimum seed support is {candidate_lines[0][5] if candidate_lines else 'NA'}.",
        "The same-weight different-infer-path row is not a sanity gate; it can fall below 1 because earlier MoE layers change hidden states when inference_k changes.",
            "If a gate fails, treat the run as an execution/measurement failure before making research claims.",
        ]
    )
    if collapse_rows:
        lines.extend(["", "## Collapsed Runs", ""])
        for row in collapse_rows:
            lines.append(
                f"- seed={row['seed']} train_k={row['train_k']} "
                f"max_expert_share={float(row['max_expert_share']):.4f}"
            )

    summary_path = run_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "completed_runs": completed_runs,
        "expected_runs": expected_runs,
        "logit_cutoff_nestedness_min": min(logit_cutoff_nested) if logit_cutoff_nested else None,
        "step0_nestedness_min": min(step0_nested) if step0_nested else None,
        "collapsed_runs": len(collapse_rows),
    }


def write_plots(run_dir: Path, metrics_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    rows = _read_metric_rows(metrics_path)
    if not any(row["checkpoint_step"].isdigit() for row in rows):
        return
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    def parse_pair(pair: str) -> tuple[int, int]:
        left, right = pair.split("-")
        return int(left), int(right)

    def draw_heatmap(
        filename: str,
        title: str,
        matrix: np.ndarray,
        xlabels: list[str],
        ylabels: list[str],
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = "viridis",
    ) -> None:
        if matrix.size == 0:
            return
        height = max(4.0, min(12.0, len(ylabels) * 0.45 + 2.0))
        width = max(5.0, min(12.0, len(xlabels) * 0.55 + 2.0))
        masked = np.ma.masked_invalid(matrix)
        fig, ax = plt.subplots(figsize=(width, height))
        image = ax.imshow(masked, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(title)
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                value = matrix[y, x]
                if not np.isfinite(value):
                    continue
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=150)
        plt.close(fig)

    for metric_name in ["nestedness", "top1_agreement", "spearman"]:
        selected = [
            row
            for row in rows
            if row["metric_type"] == "matched_train_k_final_pair"
            and row["metric_name"] == metric_name
        ]
        if not selected:
            continue
        if len(selected) > 80:
            grouped: dict[str, list[float]] = {}
            for row in selected:
                grouped.setdefault(row["pair"], []).append(float(row["value"]))
            labels = list(sorted(grouped))
            values = [float(np.mean(grouped[label])) for label in labels]
            title = f"Final matched mean {metric_name}"
        else:
            labels = [f"s{row['seed']} {row['pair']}" for row in selected]
            values = [float(row["value"]) for row in selected]
            title = f"Final matched {metric_name}"
        plt.figure(figsize=(max(6, min(24, len(labels) * 0.7)), 4))
        plt.bar(labels, values)
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45, ha="right")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(plot_dir / f"final_matched_{metric_name}.png", dpi=150)
        plt.close()

        ks = sorted({value for row in selected for value in parse_pair(row["pair"])})
        if ks:
            grouped: dict[tuple[int, int], list[float]] = {}
            for row in selected:
                grouped.setdefault(parse_pair(row["pair"]), []).append(float(row["value"]))
            index_by_k = {value: idx for idx, value in enumerate(ks)}
            matrix = np.full((len(ks), len(ks)), np.nan, dtype=np.float64)
            for (small, large), values_for_pair in grouped.items():
                matrix[index_by_k[small], index_by_k[large]] = float(np.mean(values_for_pair))
            draw_heatmap(
                filename=f"final_matched_{metric_name}_heatmap.png",
                title=f"Final matched mean {metric_name}",
                matrix=matrix,
                xlabels=[str(k) for k in ks],
                ylabels=[str(k) for k in ks],
                vmin=-1.0 if metric_name == "spearman" else 0.0,
                vmax=1.0,
                cmap="magma" if metric_name == "spearman" else "viridis",
            )

    mismatch_rows = [
        row
        for row in rows
        if row["metric_type"] == "mismatch_cost"
        and row["metric_name"] == "validation_loss_delta"
        and str(row["checkpoint_step"]).isdigit()
    ]
    final_step_by_train: dict[tuple[int, int], int] = {}
    for row in mismatch_rows:
        key = (int(row["seed"]), int(row["train_k"]))
        final_step_by_train[key] = max(final_step_by_train.get(key, -1), int(row["checkpoint_step"]))
    final_mismatch = [
        row
        for row in mismatch_rows
        if int(row["checkpoint_step"]) == final_step_by_train.get((int(row["seed"]), int(row["train_k"])), -1)
    ]
    if final_mismatch:
        train_ks = sorted({int(row["train_k"]) for row in final_mismatch})
        inference_ks = sorted({int(row["inference_k"]) for row in final_mismatch})
        grouped_delta: dict[tuple[int, int], list[float]] = {}
        for row in final_mismatch:
            grouped_delta.setdefault((int(row["train_k"]), int(row["inference_k"])), []).append(float(row["value"]))
        y_index = {value: idx for idx, value in enumerate(train_ks)}
        x_index = {value: idx for idx, value in enumerate(inference_ks)}
        matrix = np.full((len(train_ks), len(inference_ks)), np.nan, dtype=np.float64)
        for (train_k, inference_k), values_for_cell in grouped_delta.items():
            matrix[y_index[train_k], x_index[inference_k]] = float(np.mean(values_for_cell))
        for train_k in train_ks:
            if train_k in x_index:
                matrix[y_index[train_k], x_index[train_k]] = 0.0
        max_abs = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
        draw_heatmap(
            filename="final_mismatch_delta_loss_heatmap.png",
            title="Final mean validation loss delta",
            matrix=matrix,
            xlabels=[str(k) for k in inference_ks],
            ylabels=[str(k) for k in train_ks],
            vmin=-max_abs,
            vmax=max_abs,
            cmap="coolwarm",
        )


def run_all(config: dict[str, Any], run_dir: Path, mode: str, device: torch.device) -> None:
    config = dict(config)
    config["run_mode"] = mode
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "routes").mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_env(run_dir)

    if mode == "smoke":
        seeds = [int(config["seeds"][0])]
        train_ks = [int(config["train_ks"][0])]
        train_samples = int(config["smoke_train_samples_per_category"])
        probe_samples = int(config["smoke_probe_samples_per_category"])
    else:
        seeds = [int(x) for x in config["seeds"]]
        train_ks = [int(x) for x in config["train_ks"]]
        train_samples = int(config["train_samples_per_category"])
        probe_samples = int(config["probe_samples_per_category"])

    train_corpus = build_corpus(
        samples_per_category=train_samples,
        seq_len=int(config["seq_len"]),
        seed=int(config["data_seed"]),
    )
    if config.get("probe_jsonl"):
        probe_corpus = build_jsonl_corpus(Path(config["probe_jsonl"]), seq_len=int(config["seq_len"]))
    else:
        probe_corpus = build_corpus(
            samples_per_category=probe_samples,
            seq_len=int(config["seq_len"]),
            seed=int(config["data_seed"]) + 99_999,
        )
    task_to_id = {task: idx for idx, task in enumerate(CATEGORIES)}

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for seed in seeds:
        torch.manual_seed(seed)
        base_model = make_model(config, device=torch.device("cpu"))
        initial_state = {key: value.detach().cpu().clone() for key, value in base_model.state_dict().items()}
        torch.save(initial_state, checkpoint_dir / f"W0_seed{seed}.pt")
        del base_model
        for train_k in train_ks:
            steps = steps_for_train_k(config, train_k, mode)
            checkpoints = checkpoints_for_train_k(config, train_k, steps, mode)
            print(f"[run] seed={seed} train_k={train_k} steps={steps}", flush=True)
            train_one(
                config=config,
                seed=seed,
                train_k=train_k,
                steps=steps,
                checkpoints=checkpoints,
                initial_state=initial_state,
                train_corpus=train_corpus,
                probe_corpus=probe_corpus,
                task_to_id=task_to_id,
                run_dir=run_dir,
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    analyze(run_dir, config)


def apply_fallback(base_config: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    config = dict(base_config)
    config.update(fallback)
    if int(config["d_model"]) % int(config["n_heads"]) != 0:
        config["n_heads"] = 4 if int(config["d_model"]) % 4 == 0 else 3
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiny scratch MoE top-k pilot.")
    parser.add_argument("--config", type=Path, default=Path("configs/scratch_pilot.json"))
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/2020110906_matryo_topk"))
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    base_config = load_config(args.config)
    if base_config.get("probe_jsonl"):
        probe_path = Path(str(base_config["probe_jsonl"]))
        if not probe_path.is_absolute():
            cwd_probe_path = Path.cwd() / probe_path
            config_probe_path = args.config.parent / probe_path
            base_config["probe_jsonl"] = str(
                (cwd_probe_path if cwd_probe_path.exists() else config_probe_path).resolve()
            )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    timestamp = args.timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    fallbacks = base_config.get("oom_fallbacks", [{"batch_size": base_config["batch_size"], "d_model": base_config["d_model"]}])
    last_error = None
    for idx, fallback in enumerate(fallbacks):
        config = apply_fallback(base_config, fallback)
        run_name = args.run_name or str(config.get("experiment_name", "scratch_pilot"))
        run_dir = args.output_root / "runs" / run_name / f"{timestamp}_{args.mode}_fb{idx}_b{config['batch_size']}_d{config['d_model']}"
        try:
            run_all(config=config, run_dir=run_dir, mode=args.mode, device=device)
            print(f"[done] {run_dir}", flush=True)
            return
        except RuntimeError as exc:
            last_error = exc
            message = str(exc).lower()
            (run_dir / "failure.txt").parent.mkdir(parents=True, exist_ok=True)
            (run_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            if "out of memory" not in message and "cuda" not in message:
                raise
            print(f"[fallback] failed with {exc}; trying next fallback", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"all fallbacks failed; last error: {last_error}")


if __name__ == "__main__":
    main()
