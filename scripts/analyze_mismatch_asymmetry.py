"""Analyze mismatch asymmetry and routing alignment as a function of k-gap."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent.parent / "results"

RUNS = {
    "fixed_3seed": "sparse32_kgrid_fixed_step_3seed_summary",
    "mechanism_3seed": "sparse32_kgrid_mechanism_3seed_summary",
    "fixed_8seed": "sparse32_kgrid_fixed_step_8seed_summary",
    "mechanism_8seed": "sparse32_kgrid_same_compute_8seed_summary",
}


def parse_mismatch(path: Path) -> list[tuple[int, int, float]]:
    text = path.read_text(encoding="utf-8")
    rows = []
    in_mm = False
    for line in text.splitlines():
        if line.strip() == "| train_k | inference_k | mean_delta_loss | values |":
            in_mm = True
            continue
        if in_mm:
            if not line.startswith("|"):
                break
            if line.startswith("|---"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 3:
                try:
                    tk, ik, delta = int(parts[0]), int(parts[1]), float(parts[2])
                    rows.append((tk, ik, delta))
                except ValueError:
                    pass
    return rows


def parse_pair_metrics(path: Path) -> list[tuple[int, int, str, float]]:
    text = path.read_text(encoding="utf-8")
    rows = []
    in_t = False
    for line in text.splitlines():
        if line.strip() == "| pair | metric | mean | values |":
            in_t = True
            continue
        if in_t:
            if not line.startswith("|"):
                break
            if line.startswith("|---"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 3:
                try:
                    pair, metric, mean = parts[0], parts[1], float(parts[2])
                    a, b = (int(x) for x in pair.split("-"))
                    rows.append((a, b, metric, mean))
                except ValueError:
                    pass
    return rows


def analyze_asymmetry(label: str, dirname: str) -> dict:
    """Analyze mismatch asymmetry.

    For each k-gap, computes:
    - mean_lohi_delta: mean delta when train_k < infer_k (low-k model, high-k inference)
    - mean_hilo_delta: mean delta when train_k > infer_k (high-k model, low-k inference)
    - mean_asymmetry: mean |hilo - lohi| per pair
    """
    path = BASE / dirname / "summary.md"
    if not path.exists():
        return {}
    mm_rows = parse_mismatch(path)
    delta_map = {(tk, ik): d for tk, ik, d in mm_rows}

    by_gap_lohi: dict[int, list[float]] = defaultdict(list)  # low→high
    by_gap_hilo: dict[int, list[float]] = defaultdict(list)  # high→low
    by_gap_asym: dict[int, list[float]] = defaultdict(list)

    for tk, ik, d in mm_rows:
        gap = ik - tk  # positive = low→high direction
        if gap <= 0:
            continue
        rev = delta_map.get((ik, tk))
        if rev is not None:
            by_gap_lohi[gap].append(d)
            by_gap_hilo[gap].append(rev)
            by_gap_asym[gap].append(abs(rev - d))

    gap_stats = {}
    for gap in sorted(by_gap_lohi):
        lo_hi = by_gap_lohi[gap]
        hi_lo = by_gap_hilo[gap]
        asym = by_gap_asym[gap]
        gap_stats[gap] = {
            "mean_lohi_delta": round(sum(lo_hi) / len(lo_hi), 5),
            "mean_hilo_delta": round(sum(hi_lo) / len(hi_lo), 5),
            "mean_asymmetry": round(sum(asym) / len(asym), 5),
            "n_pairs": len(lo_hi),
        }
    return gap_stats


def analyze_alignment(label: str, dirname: str) -> dict:
    path = BASE / dirname / "summary.md"
    if not path.exists():
        return {}
    pm_rows = parse_pair_metrics(path)

    by_gap_t: dict[int, list[float]] = defaultdict(list)
    by_gap_n: dict[int, list[float]] = defaultdict(list)
    by_gap_s: dict[int, list[float]] = defaultdict(list)

    for a, b, metric, mean in pm_rows:
        gap = b - a
        if gap <= 0:
            continue
        if metric == "top1_agreement":
            by_gap_t[gap].append(mean)
        elif metric == "nestedness":
            by_gap_n[gap].append(mean)
        elif metric == "spearman":
            by_gap_s[gap].append(mean)

    result = {}
    for gap in sorted(by_gap_t):
        result[gap] = {
            "mean_top1_agreement": round(sum(by_gap_t[gap]) / len(by_gap_t[gap]), 5),
            "mean_nestedness": round(sum(by_gap_n.get(gap, [0])) / max(1, len(by_gap_n.get(gap, [0]))), 5),
            "mean_spearman": round(sum(by_gap_s.get(gap, [0])) / max(1, len(by_gap_s.get(gap, [0]))), 5),
            "n_pairs": len(by_gap_t[gap]),
        }
    return result


def main() -> None:
    print("=" * 70)
    print("MISMATCH ASYMMETRY BY K-GAP (higher k→lower k is the costly direction)")
    print("=" * 70)

    for label, dirname in RUNS.items():
        stats = analyze_asymmetry(label, dirname)
        if not stats:
            continue
            print(f"\n--- {label} ---")
        print(f"{'gap':>4} | {'lo→hi (cost)':>13} | {'hi→lo (cost)':>13} | {'asymmetry':>10} | n_pairs")
        print("-" * 62)
        for gap, s in stats.items():
            print(
                f"{gap:>4} | {s['mean_lohi_delta']:>13.4f} | {s['mean_hilo_delta']:>13.4f} | "
                f"{s['mean_asymmetry']:>10.4f} | {s['n_pairs']}"
            )

    print("\n\n" + "=" * 70)
    print("ROUTING ALIGNMENT BY K-GAP")
    print("=" * 70)

    for label, dirname in RUNS.items():
        stats = analyze_alignment(label, dirname)
        if not stats:
            continue
        print(f"\n--- {label} ---")
        print(f"{'gap':>4} | {'top1_agree':>11} | {'nestedness':>11} | {'spearman':>10} | n_pairs")
        print("-" * 60)
        for gap, s in stats.items():
            print(
                f"{gap:>4} | {s['mean_top1_agreement']:>11.4f} | {s['mean_nestedness']:>11.4f} | "
                f"{s['mean_spearman']:>10.4f} | {s['n_pairs']}"
            )

    print("\n\n" + "=" * 70)
    print("FIXED-STEP vs MECHANISM COMPARISON (8-seed, key pairs)")
    print("=" * 70)
    key_pairs = [(1, 2), (1, 4), (1, 8), (2, 4), (4, 8), (7, 8)]
    fs_mm = {(tk, ik): d for tk, ik, d in parse_mismatch(BASE / "sparse32_kgrid_fixed_step_8seed_summary" / "summary.md")}
    mc_mm = {(tk, ik): d for tk, ik, d in parse_mismatch(BASE / "sparse32_kgrid_same_compute_8seed_summary" / "summary.md")}
    fs_pm = {(a, b, m): v for a, b, m, v in parse_pair_metrics(BASE / "sparse32_kgrid_fixed_step_8seed_summary" / "summary.md")}
    mc_pm = {(a, b, m): v for a, b, m, v in parse_pair_metrics(BASE / "sparse32_kgrid_same_compute_8seed_summary" / "summary.md")}

    print(f"\n{'pair':>6} | {'fs_top1':>8} | {'mc_top1':>8} | {'fs_δ(a→b)':>10} | {'mc_δ(a→b)':>10} | {'fs_δ(b→a)':>10} | {'mc_δ(b→a)':>10}")
    print("-" * 78)
    for a, b in key_pairs:
        ft = fs_pm.get((a, b, "top1_agreement"), float("nan"))
        mt = mc_pm.get((a, b, "top1_agreement"), float("nan"))
        fab = fs_mm.get((a, b), float("nan"))
        mab = mc_mm.get((a, b), float("nan"))
        fba = fs_mm.get((b, a), float("nan"))
        mba = mc_mm.get((b, a), float("nan"))
        print(f"{a}-{b:>2} | {ft:>8.4f} | {mt:>8.4f} | {fab:>10.4f} | {mab:>10.4f} | {fba:>10.4f} | {mba:>10.4f}")

    out_path = Path(__file__).parent.parent / "results" / "mismatch_asymmetry_analysis.json"
    output = {}
    for label, dirname in RUNS.items():
        output[label] = {
            "asymmetry_by_gap": analyze_asymmetry(label, dirname),
            "alignment_by_gap": analyze_alignment(label, dirname),
        }
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote analysis JSON to {out_path}")


if __name__ == "__main__":
    main()
