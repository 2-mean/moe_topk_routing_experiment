"""B5: Training variability null — same-k different-seed routing alignment.

For each k, compare model(seed=i, k) vs model(seed=j, k) across all i<j pairs.
Then compare against across-k alignment (same seed, different k) from metrics.csv.

If across_k_alignment < within_k_alignment, k differences exceed training noise.
That calibrates Claim A: "training-time k changes routing structure."
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _binary_expert_matrix(selected: np.ndarray, n_experts: int) -> np.ndarray:
    """Convert selected expert indices [N, k] to binary matrix [N, E]. Vectorized."""
    N = selected.shape[0]
    mat = np.zeros((N, n_experts), dtype=np.uint8)
    row_idx = np.repeat(np.arange(N), selected.shape[1])
    mat[row_idx, selected.reshape(-1)] = 1
    return mat


def top1_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a[:, 0] == b[:, 0]))


def nestedness(small: np.ndarray, large: np.ndarray, n_experts: int = 32) -> float:
    """Vectorized nestedness: fraction of small experts also in large."""
    bm_s = _binary_expert_matrix(small, n_experts)
    bm_l = _binary_expert_matrix(large, n_experts)
    intersection = (bm_s & bm_l).sum(axis=1).astype(np.float32)
    small_size = small.shape[1]
    return float(np.mean(intersection / small_size))


def spearman_from_logits(la: np.ndarray, lb: np.ndarray) -> float:
    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(-v.astype(np.float32), axis=1)
        r = np.empty_like(order, dtype=np.float32)
        rows = np.arange(v.shape[0])[:, None]
        r[rows, order] = np.arange(v.shape[1], dtype=np.float32)
        return r
    ra, rb = ranks(la), ranks(lb)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra * ra).sum(axis=1) * (rb * rb).sum(axis=1))
    valid = den > 0
    if not np.any(valid):
        return float("nan")
    return float(np.mean(num[valid] / den[valid]))


def hypergeometric_calibrated_overlap(
    selected_a: np.ndarray,
    selected_b: np.ndarray,
    n_experts: int,
) -> float:
    """HCO: cardinality-corrected overlap. 0=random, 1=perfect nesting.
    
    Vectorized: precompute CDF table, then map intersection sizes.
    """
    ka = selected_a.shape[1]
    kb = selected_b.shape[1]
    small, large = (selected_a, selected_b) if ka <= kb else (selected_b, selected_a)
    ks, kl = min(ka, kb), max(ka, kb)
    E = n_experts

    from math import comb

    total = comb(E, ks)
    x_min = max(0, ks + kl - E)
    x_max = min(ks, kl)

    # Build CDF table once
    cdf: dict[int, float] = {}
    pmf: dict[int, float] = {}
    for x in range(x_min, x_max + 1):
        p = comb(kl, x) * comb(E - kl, ks - x) / total
        pmf[x] = p
    acc = 0.0
    for x in range(x_min, x_max + 1):
        cdf[x] = acc + 0.5 * pmf[x]
        acc += pmf[x]

    q_max = cdf.get(x_max, 0.0)
    if q_max <= 0.5:
        return float("nan")

    # Compute intersection sizes vectorized
    bm_s = _binary_expert_matrix(small, E)
    bm_l = _binary_expert_matrix(large, E)
    intersect_sizes = (bm_s & bm_l).sum(axis=1)  # [N]

    # Map intersection sizes to q scores
    cdf_arr = np.array([cdf.get(int(i), 0.0) for i in intersect_sizes], dtype=np.float64)
    scores = (cdf_arr - 0.5) / (q_max - 0.5)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_route(path: Path, infer_k: int) -> dict[str, np.ndarray]:
    data = np.load(path)
    n_layers = len(set(data["layer_id"].tolist()))
    return {
        "layer_id": data["layer_id"].astype(np.int16),
        "gate_logits": data["gate_logits"].astype(np.float32),
        "selected_ids": data["selected_ids"].astype(np.int16),
    }


def final_step(config: dict) -> dict[int, int]:
    steps_by_k = config.get("train_steps_by_k", {})
    result = {}
    for k in config.get("train_ks", []):
        result[int(k)] = int(steps_by_k.get(str(k), config.get("steps", 1500)))
    return result


# ---------------------------------------------------------------------------
# Core: within-k alignment
# ---------------------------------------------------------------------------

def compute_within_k_alignment(
    run_dir: Path,
    config: dict,
    n_experts: int,
) -> list[dict[str, Any]]:
    seeds = [int(s) for s in config["seeds"]]
    train_ks = [int(k) for k in config["train_ks"]]
    steps = final_step(config)
    routes_dir = run_dir / "routes"

    rows = []
    for k in train_ks:
        step = steps[k]
        for si, sj in combinations(seeds, 2):
            path_i = routes_dir / f"seed{si}_train{k}_infer{k}_step{step}.npz"
            path_j = routes_dir / f"seed{sj}_train{k}_infer{k}_step{step}.npz"
            if not path_i.exists() or not path_j.exists():
                continue
            ri = load_route(path_i, k)
            rj = load_route(path_j, k)
            t1 = top1_agreement(ri["selected_ids"], rj["selected_ids"])
            ns = nestedness(ri["selected_ids"], rj["selected_ids"], n_experts)
            sp = spearman_from_logits(ri["gate_logits"], rj["gate_logits"])
            hco = hypergeometric_calibrated_overlap(ri["selected_ids"], rj["selected_ids"], n_experts)
            rows.append({
                "comparison": "within_k",
                "k": k,
                "seed_i": si,
                "seed_j": sj,
                "top1_agreement": round(t1, 6),
                "nestedness": round(ns, 6),
                "spearman": round(sp, 6),
                "hco": round(hco, 6),
            })
    return rows


# ---------------------------------------------------------------------------
# Core: across-k alignment (from metrics.csv)
# ---------------------------------------------------------------------------

def extract_across_k_alignment(metrics_path: Path) -> list[dict[str, Any]]:
    """Pull matched_train_k_final_pair rows (same seed, different k)."""
    rows = []
    with metrics_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["metric_type"] != "matched_train_k_final_pair":
                continue
            if row["metric_name"] not in {"top1_agreement", "nestedness", "spearman"}:
                continue
            rows.append({
                "comparison": "across_k",
                "seed": int(row["seed"]),
                "train_k_a": int(row["train_k"]),
                "train_k_b": int(row["train_k_b"]),
                "pair": row["pair"],
                "metric_name": row["metric_name"],
                "value": float(row["value"]),
            })
    return rows


def extract_matched_loss(metrics_path: Path) -> list[dict[str, Any]]:
    """Extract validation_loss where train_k == inference_k (matched)."""
    rows = []
    with metrics_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["metric_type"] != "loss":
                continue
            if row["metric_name"] != "validation_loss":
                continue
            if row["train_k"] != row["inference_k"]:
                continue
            rows.append({
                "seed": int(row["seed"]),
                "train_k": int(row["train_k"]),
                "step": int(row["checkpoint_step"]),
                "loss": float(row["value"]),
            })
    return rows


def extract_mismatch_cost(metrics_path: Path) -> list[dict[str, Any]]:
    """Extract mismatch_cost rows for baseline analysis."""
    rows = []
    with metrics_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["metric_type"] != "mismatch_cost":
                continue
            if row["metric_name"] != "validation_loss_delta":
                continue
            rows.append({
                "seed": int(row["seed"]),
                "train_k": int(row["train_k"]),
                "infer_k": int(row["inference_k"]),
                "step": int(row["checkpoint_step"]),
                "delta": float(row["value"]),
            })
    return rows


# ---------------------------------------------------------------------------
# Analysis: B5 summary
# ---------------------------------------------------------------------------

def summarize_b5(
    within_rows: list[dict],
    across_rows: list[dict],
) -> list[dict[str, Any]]:
    """Compare within-k vs nearest across-k alignment per k."""
    # Within-k: aggregate per (k, metric)
    within_by_k: dict[tuple[int, str], list[float]] = defaultdict(list)
    for r in within_rows:
        for m in ("top1_agreement", "nestedness", "spearman", "hco"):
            within_by_k[(r["k"], m)].append(r[m])

    # Across-k: aggregate per (pair, metric)
    across_by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in across_rows:
        across_by_pair[(r["pair"], r["metric_name"])].append(r["value"])

    # Build comparison table: for each k, compare within vs closest across-k pair
    train_ks = sorted({r["k"] for r in within_rows})
    results = []
    for k in train_ks:
        # Nearest across-k pairs involving this k
        neighbor_pairs = [
            (f"{k}-{k+1}", f"{k}-{k+1}") if k < 8 else None,
            (f"{k-1}-{k}", f"{k-1}-{k}") if k > 1 else None,
        ]
        neighbor_pairs = [p[0] for p in neighbor_pairs if p is not None]

        for metric in ("top1_agreement", "nestedness", "spearman"):
            wvals = within_by_k.get((k, metric), [])
            if not wvals:
                continue
            w_mean = float(np.mean(wvals))
            w_std = float(np.std(wvals, ddof=1)) if len(wvals) > 1 else 0.0

            # Nearest across-k
            near_vals = []
            for p in neighbor_pairs:
                near_vals.extend(across_by_pair.get((p, metric), []))
            near_mean = float(np.mean(near_vals)) if near_vals else float("nan")

            # All across-k involving this k (to show max range)
            all_across = []
            for pair, m in across_by_pair:
                if m != metric:
                    continue
                parts = pair.split("-")
                if len(parts) == 2 and (int(parts[0]) == k or int(parts[1]) == k):
                    all_across.extend(across_by_pair[(pair, m)])
            across_mean = float(np.mean(all_across)) if all_across else float("nan")
            across_min = float(np.min(all_across)) if all_across else float("nan")

            ratio = w_mean / across_mean if across_mean > 0 else float("nan")

            results.append({
                "k": k,
                "metric": metric,
                "within_k_mean": round(w_mean, 5),
                "within_k_std": round(w_std, 5),
                "within_k_n_pairs": len(wvals),
                "nearest_across_k_mean": round(near_mean, 5),
                "all_across_k_mean": round(across_mean, 5),
                "all_across_k_min": round(across_min, 5),
                "within_to_across_ratio": round(ratio, 4),
                "k_effect_exceeds_noise": ratio > 1.0 if not math.isnan(ratio) else None,
            })
    return results


# ---------------------------------------------------------------------------
# Analysis: Random baselines for routing metrics
# ---------------------------------------------------------------------------

def compute_random_baselines(
    within_rows: list[dict],
    across_rows: list[dict],
    n_experts: int,
) -> list[dict[str, Any]]:
    """For each (a, b) pair: random baseline vs observed."""
    from math import comb

    def jaccard_random(a: int, b: int, E: int) -> float:
        exp_intersect = a * b / E
        return exp_intersect / (a + b - exp_intersect)

    # Observed across-k values
    obs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in across_rows:
        obs[(r["pair"], r["metric_name"])].append(r["value"])

    results = []
    seen_pairs: set[str] = set()
    for r in across_rows:
        pair = r["pair"]
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        a, b = (int(x) for x in pair.split("-"))

        random_top1 = 1.0 / n_experts
        random_nestedness = b / n_experts  # E[|S_a ∩ S_b| / a] = b/E
        random_jaccard = jaccard_random(a, b, n_experts)
        random_spearman = 0.0

        obs_top1 = float(np.mean(obs.get((pair, "top1_agreement"), [float("nan")])))
        obs_nested = float(np.mean(obs.get((pair, "nestedness"), [float("nan")])))
        obs_spearman = float(np.mean(obs.get((pair, "spearman"), [float("nan")])))

        results.append({
            "pair": pair,
            "a": a, "b": b,
            "random_top1": round(random_top1, 5),
            "random_nestedness": round(random_nestedness, 5),
            "random_jaccard": round(random_jaccard, 5),
            "random_spearman": round(random_spearman, 5),
            "obs_top1": round(obs_top1, 5),
            "obs_nestedness": round(obs_nested, 5),
            "obs_spearman": round(obs_spearman, 5),
            "top1_vs_random_ratio": round(obs_top1 / random_top1, 2) if not math.isnan(obs_top1) else None,
            "nestedness_excess": round(obs_nested - random_nestedness, 5) if not math.isnan(obs_nested) else None,
        })
    results.sort(key=lambda r: (r["a"], r["b"]))
    return results


# ---------------------------------------------------------------------------
# Analysis: Matched loss table
# ---------------------------------------------------------------------------

def summarize_matched_loss(loss_rows: list[dict]) -> list[dict[str, Any]]:
    by_k: dict[int, list[float]] = defaultdict(list)
    # Use final step only
    step_by_k: dict[int, int] = {}
    for r in loss_rows:
        if r["train_k"] not in step_by_k or r["step"] > step_by_k[r["train_k"]]:
            step_by_k[r["train_k"]] = r["step"]
    for r in loss_rows:
        if r["step"] == step_by_k.get(r["train_k"]):
            by_k[r["train_k"]].append(r["loss"])

    results = []
    for k in sorted(by_k):
        vals = by_k[k]
        results.append({
            "train_k": k,
            "matched_loss_mean": round(float(np.mean(vals)), 5),
            "matched_loss_std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 5),
            "matched_loss_min": round(float(np.min(vals)), 5),
            "matched_loss_max": round(float(np.max(vals)), 5),
            "n_seeds": len(vals),
        })
    return results


# ---------------------------------------------------------------------------
# Analysis: Mismatch asymmetry with CI and linear extrapolation
# ---------------------------------------------------------------------------

def summarize_asymmetry_with_ci(
    mismatch_rows: list[dict],
    gap1_baseline: float | None = None,
) -> list[dict[str, Any]]:
    # Final step only
    step_by_k: dict[int, int] = {}
    for r in mismatch_rows:
        k = r["train_k"]
        if k not in step_by_k or r["step"] > step_by_k[k]:
            step_by_k[k] = r["step"]

    # Build (seed, train_k, infer_k) -> delta at final step
    delta_map: dict[tuple[int, int, int], float] = {}
    for r in mismatch_rows:
        if r["step"] == step_by_k.get(r["train_k"]):
            delta_map[(r["seed"], r["train_k"], r["infer_k"])] = r["delta"]

    seeds = sorted({r["seed"] for r in mismatch_rows})
    train_ks = sorted({r["train_k"] for r in mismatch_rows})

    # Per-gap, per-seed asymmetry
    gap_seed_asym: dict[int, list[float]] = defaultdict(list)
    gap_seed_lohi: dict[int, list[float]] = defaultdict(list)
    gap_seed_hilo: dict[int, list[float]] = defaultdict(list)

    for a in train_ks:
        for b in train_ks:
            if b <= a:
                continue
            gap = b - a
            for seed in seeds:
                lo_hi = delta_map.get((seed, a, b))
                hi_lo = delta_map.get((seed, b, a))
                if lo_hi is None or hi_lo is None:
                    continue
                gap_seed_lohi[gap].append(lo_hi)
                gap_seed_hilo[gap].append(hi_lo)
                gap_seed_asym[gap].append(abs(hi_lo - lo_hi))

    # Also per-gap, averaging over seeds properly
    results = []
    for gap in sorted(gap_seed_asym):
        asym_vals = gap_seed_asym[gap]
        lohi_vals = gap_seed_lohi[gap]
        hilo_vals = gap_seed_hilo[gap]
        n = len(asym_vals)

        def ci95(vals: list[float]) -> tuple[float, float]:
            if len(vals) < 2:
                return float("nan"), float("nan")
            mean = np.mean(vals)
            sem = np.std(vals, ddof=1) / math.sqrt(len(vals))
            # t-critical for 95% CI, df=n-1
            from scipy import stats as scipy_stats
            t = scipy_stats.t.ppf(0.975, df=len(vals) - 1)
            return float(mean - t * sem), float(mean + t * sem)

        asym_mean = float(np.mean(asym_vals))
        asym_ci_lo, asym_ci_hi = ci95(asym_vals)

        # Linear extrapolation baseline from gap=1
        gap1_asym = float(np.mean(gap_seed_asym.get(1, [0.0])))
        linear_pred = gap * gap1_asym
        ratio = asym_mean / linear_pred if linear_pred > 0 else float("nan")

        # Sign consistency: fraction of pairs where hi_lo > lo_hi
        sign_consistent = sum(1 for h, l in zip(hilo_vals, lohi_vals) if h > l)
        sign_frac = sign_consistent / n if n > 0 else float("nan")

        results.append({
            "gap": gap,
            "n_observations": n,
            "lohi_mean": round(float(np.mean(lohi_vals)), 5),
            "lohi_std": round(float(np.std(lohi_vals, ddof=1)) if n > 1 else 0.0, 5),
            "hilo_mean": round(float(np.mean(hilo_vals)), 5),
            "hilo_std": round(float(np.std(hilo_vals, ddof=1)) if n > 1 else 0.0, 5),
            "asymmetry_mean": round(asym_mean, 5),
            "asymmetry_std": round(float(np.std(asym_vals, ddof=1)) if n > 1 else 0.0, 5),
            "asymmetry_ci95_lo": round(asym_ci_lo, 5) if not math.isnan(asym_ci_lo) else None,
            "asymmetry_ci95_hi": round(asym_ci_hi, 5) if not math.isnan(asym_ci_hi) else None,
            "linear_pred_from_gap1": round(linear_pred, 5),
            "ratio_vs_linear": round(ratio, 3) if not math.isnan(ratio) else None,
            "hilo_gt_lohi_fraction": round(sign_frac, 3),
            "hilo_gt_lohi_unanimous": sign_frac == 1.0,
        })
    return results


# ---------------------------------------------------------------------------
# Analysis: Noise floor for small deltas
# ---------------------------------------------------------------------------

def compute_noise_floor(mismatch_rows: list[dict]) -> list[dict[str, Any]]:
    # Final step only
    step_by_k: dict[int, int] = {}
    for r in mismatch_rows:
        k = r["train_k"]
        if k not in step_by_k or r["step"] > step_by_k[k]:
            step_by_k[k] = r["step"]

    # Group by (train_k, infer_k)
    by_pair: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in mismatch_rows:
        if r["step"] == step_by_k.get(r["train_k"]):
            by_pair[(r["train_k"], r["infer_k"])].append(r["delta"])

    results = []
    for (tk, ik), vals in sorted(by_pair.items()):
        gap = abs(ik - tk)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        n_pos = sum(1 for v in vals if v > 0)
        n_neg = sum(1 for v in vals if v < 0)
        snr = abs(mean) / std if std > 0 else float("inf")
        results.append({
            "train_k": tk,
            "infer_k": ik,
            "gap": gap,
            "direction": "lo_to_hi" if ik > tk else "hi_to_lo",
            "mean_delta": round(mean, 6),
            "std_delta": round(std, 6),
            "min_delta": round(float(np.min(vals)), 6),
            "max_delta": round(float(np.max(vals)), 6),
            "n_seeds": len(vals),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "sign_ratio_pos": round(n_pos / len(vals), 3),
            "snr": round(snr, 3),
            "strong_signal": snr >= 2.0 and (n_pos == len(vals) or n_neg == len(vals)),
        })
    return results


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def write_csv_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def print_section(title: str, rows: list[dict], cols: list[str]) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")
    header = " | ".join(f"{c[:14]:>14}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = " | ".join(f"{str(r.get(c, ''))[:14]:>14}" for c in cols)
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-experts", type=int, default=32)
    p.add_argument("--skip-b5", action="store_true", help="skip route-file loading (fast)")
    args = p.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics_path = run_dir / "metrics.csv"

    print(f"[baseline] run_dir = {run_dir}")
    print(f"[baseline] n_experts = {args.n_experts}")
    print(f"[baseline] seeds = {config['seeds']}")
    print(f"[baseline] train_ks = {config['train_ks']}")

    # ---- B5: within-k alignment ----
    if not args.skip_b5:
        print("\n[B5] Computing within-k different-seed alignment...")
        within_rows = compute_within_k_alignment(run_dir, config, args.n_experts)
        write_csv_rows(
            out_dir / "b5_within_k_raw.csv",
            within_rows,
            ["comparison", "k", "seed_i", "seed_j", "top1_agreement", "nestedness", "spearman", "hco"],
        )
        print(f"  wrote {len(within_rows)} rows")
    else:
        # Load previously computed
        within_path = out_dir / "b5_within_k_raw.csv"
        within_rows = []
        if within_path.exists():
            with within_path.open(newline="", encoding="utf-8") as f:
                within_rows = [
                    {**r, "k": int(r["k"]), "top1_agreement": float(r["top1_agreement"]),
                     "nestedness": float(r["nestedness"]), "spearman": float(r["spearman"]),
                     "hco": float(r["hco"])}
                    for r in csv.DictReader(f)
                ]

    # ---- Across-k from metrics ----
    print("[across-k] Extracting from metrics.csv...")
    across_rows = extract_across_k_alignment(metrics_path)

    # ---- B5 summary ----
    if within_rows:
        b5_summary = summarize_b5(within_rows, across_rows)
        write_csv_rows(
            out_dir / "b5_summary.csv",
            b5_summary,
            ["k", "metric", "within_k_mean", "within_k_std", "within_k_n_pairs",
             "nearest_across_k_mean", "all_across_k_mean", "all_across_k_min",
             "within_to_across_ratio", "k_effect_exceeds_noise"],
        )
        print_section(
            "B5: Within-k (training noise) vs Across-k alignment",
            [r for r in b5_summary if r["metric"] == "top1_agreement"],
            ["k", "metric", "within_k_mean", "all_across_k_mean", "within_to_across_ratio", "k_effect_exceeds_noise"],
        )

    # ---- B1: Random baselines ----
    print("\n[B1] Computing random baselines...")
    random_bl = compute_random_baselines(across_rows, across_rows, args.n_experts)
    write_csv_rows(
        out_dir / "b1_random_baselines.csv",
        random_bl,
        ["pair", "a", "b", "random_top1", "random_nestedness", "random_jaccard",
         "obs_top1", "obs_nestedness", "top1_vs_random_ratio", "nestedness_excess"],
    )

    # ---- B8: Matched loss table ----
    print("\n[B8] Extracting matched loss table...")
    loss_rows = extract_matched_loss(metrics_path)
    matched_loss = summarize_matched_loss(loss_rows)
    write_csv_rows(
        out_dir / "b8_matched_loss.csv",
        matched_loss,
        ["train_k", "matched_loss_mean", "matched_loss_std", "matched_loss_min", "matched_loss_max", "n_seeds"],
    )
    print_section(
        "B8: Matched loss by k (baseline for mismatch cost interpretation)",
        matched_loss,
        ["train_k", "matched_loss_mean", "matched_loss_std"],
    )

    # ---- B9: Asymmetry with CI and linear extrapolation ----
    print("\n[B9] Computing asymmetry statistics with CI and linear extrapolation...")
    mismatch_rows = extract_mismatch_cost(metrics_path)
    asym_stats = summarize_asymmetry_with_ci(mismatch_rows)
    write_csv_rows(
        out_dir / "b9_asymmetry_stats.csv",
        asym_stats,
        ["gap", "n_observations", "lohi_mean", "lohi_std", "hilo_mean", "hilo_std",
         "asymmetry_mean", "asymmetry_std", "asymmetry_ci95_lo", "asymmetry_ci95_hi",
         "linear_pred_from_gap1", "ratio_vs_linear", "hilo_gt_lohi_fraction", "hilo_gt_lohi_unanimous"],
    )
    print_section(
        "B9: Asymmetry by k-gap with 95% CI vs linear extrapolation",
        asym_stats,
        ["gap", "asymmetry_mean", "asymmetry_ci95_lo", "asymmetry_ci95_hi",
         "linear_pred_from_gap1", "ratio_vs_linear", "hilo_gt_lohi_unanimous"],
    )

    # ---- Noise floor ----
    print("\n[Noise] Computing noise floor for small mismatch deltas...")
    noise_rows = compute_noise_floor(mismatch_rows)
    write_csv_rows(
        out_dir / "noise_floor.csv",
        noise_rows,
        ["train_k", "infer_k", "gap", "direction", "mean_delta", "std_delta",
         "min_delta", "max_delta", "n_seeds", "n_positive", "n_negative",
         "sign_ratio_pos", "snr", "strong_signal"],
    )
    adj = [r for r in noise_rows if r["gap"] == 1]
    print_section(
        "Noise floor: Adjacent-k (gap=1) mismatch delta (signal-to-noise)",
        adj,
        ["train_k", "infer_k", "direction", "mean_delta", "std_delta", "snr", "strong_signal"],
    )

    # ---- Summary JSON ----
    summary = {
        "run": str(run_dir),
        "n_experts": args.n_experts,
        "b5_computed": not args.skip_b5,
        "b5_n_rows": len(within_rows),
        "b9_gap7_ratio_vs_linear": next(
            (r["ratio_vs_linear"] for r in asym_stats if r["gap"] == 7), None
        ),
        "b9_gap7_hilo_unanimous": next(
            (r["hilo_gt_lohi_unanimous"] for r in asym_stats if r["gap"] == 7), None
        ),
        "noise_floor_gap1_snr_mean": float(np.mean([r["snr"] for r in adj if not math.isinf(r["snr"])])) if adj else None,
    }
    (out_dir / "baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] wrote all baseline results to {out_dir}")


if __name__ == "__main__":
    main()
