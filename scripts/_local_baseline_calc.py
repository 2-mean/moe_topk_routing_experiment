"""Local baseline computations from summary data (no route files needed)."""
from __future__ import annotations
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t as t_dist

FS = Path("results/sparse32_kgrid_fixed_step_8seed_summary")
MC = Path("results/sparse32_kgrid_same_compute_8seed_summary")
E = 32

# ---------------------------------------------------------------------------
# Parse summary.md
# ---------------------------------------------------------------------------
def parse_summary(path: Path):
    text = (path / "summary.md").read_text(encoding="utf-8")
    pair_rows, mm_rows = [], []
    in_pair = in_mm = False
    for line in text.splitlines():
        if "| pair | metric | mean | values |" in line:
            in_pair, in_mm = True, False; continue
        if "| train_k | inference_k | mean_delta_loss | values |" in line:
            in_mm, in_pair = True, False; continue
        if in_pair:
            if not line.startswith("|"): in_pair = False; continue
            if line.startswith("|---"): continue
            p = [x.strip() for x in line.strip("|").split("|")]
            if len(p) >= 4:
                try:
                    pair, metric, mean_ = p[0], p[1], float(p[2])
                    vals = [float(x) for x in p[3].split(",")]
                    a, b = (int(x) for x in pair.split("-"))
                    pair_rows.append({"pair": pair, "a": a, "b": b, "metric": metric, "mean": mean_, "values": vals})
                except Exception:
                    pass
        elif in_mm:
            if not line.startswith("|"): in_mm = False; continue
            if line.startswith("|---"): continue
            p = [x.strip() for x in line.strip("|").split("|")]
            if len(p) >= 4:
                try:
                    tk, ik, mean_ = int(p[0]), int(p[1]), float(p[2])
                    vals = [float(x) for x in p[3].split(",")]
                    mm_rows.append({"train_k": tk, "infer_k": ik, "mean": mean_, "values": vals})
                except Exception:
                    pass
    return pair_rows, mm_rows


def load_metrics(path: Path):
    with (path / "metrics.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# B1: Random baselines
# ---------------------------------------------------------------------------
def print_random_baselines(pair_rows, label):
    print(f"\n{'='*76}")
    print(f"B1: RANDOM BASELINES vs OBSERVED  [{label}]")
    print(f"{'='*76}")
    fmt = "{:>6} | {:>9} | {:>9} | {:>8} | {:>10} | {:>9} | {:>9} | {:>9}"
    print(fmt.format("pair", "rand_top1", "obs_top1", "ratio_t1", "rand_nest", "obs_nest", "nest_exc", "nest/rand"))
    print("-" * 80)
    top1 = {r["pair"]: r for r in pair_rows if r["metric"] == "top1_agreement"}
    nest = {r["pair"]: r for r in pair_rows if r["metric"] == "nestedness"}
    for pair in sorted(top1.keys(), key=lambda x: (int(x.split("-")[0]), int(x.split("-")[1]))):
        a, b = (int(x) for x in pair.split("-"))
        rand_t1 = 1 / E
        rand_n = b / E
        ot1 = top1[pair]["mean"]
        on = nest[pair]["mean"] if pair in nest else float("nan")
        print(fmt.format(
            pair, f"{rand_t1:.4f}", f"{ot1:.4f}", f"{ot1/rand_t1:.2f}x",
            f"{rand_n:.4f}", f"{on:.4f}",
            f"{on-rand_n:.4f}", f"{on/rand_n:.2f}x",
        ))


# ---------------------------------------------------------------------------
# B8: Matched loss table
# ---------------------------------------------------------------------------
def get_matched_loss(path: Path):
    metrics = load_metrics(path)
    step_by_k: dict[int, int] = {}
    for r in metrics:
        if r["metric_type"] != "loss" or r["metric_name"] != "validation_loss": continue
        if r["train_k"] != r["inference_k"]: continue
        k, s = int(r["train_k"]), int(r["checkpoint_step"])
        if k not in step_by_k or s > step_by_k[k]:
            step_by_k[k] = s
    by_k: dict[int, list[float]] = defaultdict(list)
    for r in metrics:
        if r["metric_type"] != "loss" or r["metric_name"] != "validation_loss": continue
        if r["train_k"] != r["inference_k"]: continue
        k, s = int(r["train_k"]), int(r["checkpoint_step"])
        if s == step_by_k.get(k):
            by_k[k].append(float(r["value"]))
    return by_k


def print_matched_loss(fs_loss, mc_loss):
    print(f"\n{'='*76}")
    print("B8: MATCHED LOSS TABLE  [final-step, matched train_k = infer_k]")
    print(f"{'='*76}")
    fmt = "{:>3} | {:>9} | {:>8} | {:>9} | {:>8} | {:>9}"
    print(fmt.format("k", "fs_mean", "fs_std", "mc_mean", "mc_std", "diff(fs-mc)"))
    print("-" * 56)
    for k in sorted(fs_loss.keys()):
        fv, mv = fs_loss[k], mc_loss.get(k, [])
        fm, fs_ = np.mean(fv), (np.std(fv, ddof=1) if len(fv) > 1 else 0.0)
        mm, ms_ = (np.mean(mv), np.std(mv, ddof=1) if len(mv) > 1 else 0.0) if mv else (float("nan"), float("nan"))
        print(fmt.format(k, f"{fm:.5f}", f"{fs_:.5f}", f"{mm:.5f}", f"{ms_:.5f}", f"{fm-mm:+.5f}"))


# ---------------------------------------------------------------------------
# B9: Asymmetry with CI and linear extrapolation
# ---------------------------------------------------------------------------
def ci95_t(vals):
    n = len(vals)
    if n < 2:
        return float("nan"), float("nan")
    mu, se = np.mean(vals), np.std(vals, ddof=1) / math.sqrt(n)
    tc = t_dist.ppf(0.975, df=n - 1)
    return float(mu - tc * se), float(mu + tc * se)


def per_seed_asym(mm_rows):
    delta: dict[tuple[int, int, int], float] = {}
    for r in mm_rows:
        for si, v in enumerate(r["values"]):
            delta[(r["train_k"], r["infer_k"], si)] = v
    n_seeds = max(len(r["values"]) for r in mm_rows)
    ks = sorted({r["train_k"] for r in mm_rows})
    gap_asym: dict[int, list[float]] = defaultdict(list)
    gap_lohi: dict[int, list[float]] = defaultdict(list)
    gap_hilo: dict[int, list[float]] = defaultdict(list)
    for a in ks:
        for b in ks:
            if b <= a: continue
            gap = b - a
            for seed in range(n_seeds):
                lo = delta.get((a, b, seed))
                hi = delta.get((b, a, seed))
                if lo is None or hi is None: continue
                gap_lohi[gap].append(lo)
                gap_hilo[gap].append(hi)
                gap_asym[gap].append(abs(hi - lo))
    return gap_asym, gap_lohi, gap_hilo


def print_asymmetry_ci(mm_rows, label):
    print(f"\n--- {label} ---")
    gap_asym, gap_lohi, gap_hilo = per_seed_asym(mm_rows)
    A1 = float(np.mean(gap_asym[1]))
    fmt = "{:>4} | {:>9} | {:>8} | {:>8} | {:>11} | {:>7} | {:>9} | {:>5}"
    print(fmt.format("gap", "obs_mean", "ci95_lo", "ci95_hi", "linear_pred", "ratio", "hilo>lohi", "n"))
    print("-" * 80)
    for gap in sorted(gap_asym):
        vals = gap_asym[gap]
        lo, hi = ci95_t(vals)
        lp = gap * A1
        ratio = np.mean(vals) / lp if lp > 0 else float("nan")
        n_hgt = sum(1 for h, l in zip(gap_hilo[gap], gap_lohi[gap]) if h > l)
        n_tot = len(vals)
        # Does CI exclude linear prediction?
        above_linear = "YES" if (not math.isnan(lo) and lo > lp) else "NO"
        print(fmt.format(
            gap, f"{np.mean(vals):.4f}",
            f"{lo:.4f}" if not math.isnan(lo) else "nan",
            f"{hi:.4f}" if not math.isnan(hi) else "nan",
            f"{lp:.4f}", f"{ratio:.2f}x",
            f"{n_hgt}/{n_tot}", str(n_tot),
        ))
        if gap == 7:
            print(f"       ^ CI excludes linear prediction: {above_linear}  "
                  f"(CI lower bound {lo:.4f} vs linear {lp:.4f})")


# ---------------------------------------------------------------------------
# Noise floor: adjacent-k
# ---------------------------------------------------------------------------
def print_noise_floor(mm_rows, label):
    print(f"\n--- {label} ---")
    adj = [(r["train_k"], r["infer_k"], r["values"]) for r in mm_rows if abs(r["infer_k"] - r["train_k"]) == 1]
    fmt = "{:>7} | {:>9} | {:>8} | {:>6} | {:>5} | {:>5} | {:>5} | {:>8}"
    print(fmt.format("tk->ik", "mean", "std", "snr", "n_pos", "n_neg", "n_tot", "signal?"))
    print("-" * 68)
    for tk, ik, vals in sorted(adj):
        m, s = np.mean(vals), (np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        snr = abs(m) / s if s > 0 else 999.0
        np_ = sum(1 for v in vals if v > 0)
        nn_ = sum(1 for v in vals if v < 0)
        strong = "STRONG" if (snr >= 2.0 and (np_ == len(vals) or nn_ == len(vals))) else "noise"
        print(fmt.format(f"{tk}->{ik}", f"{m:.5f}", f"{s:.5f}", f"{snr:.2f}",
                         str(np_), str(nn_), str(len(vals)), strong))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fs_pairs, fs_mm = parse_summary(FS)
    mc_pairs, mc_mm = parse_summary(MC)

    print_random_baselines(fs_pairs, "fixed_step_8seed")
    print_random_baselines(mc_pairs, "same_compute_8seed")

    fs_loss = get_matched_loss(FS)
    mc_loss = get_matched_loss(MC)
    print_matched_loss(fs_loss, mc_loss)

    print(f"\n{'='*76}")
    print("B9: ASYMMETRY with 95% CI vs LINEAR EXTRAPOLATION  (per-seed)")
    print(f"{'='*76}")
    print_asymmetry_ci(fs_mm, "fixed_step_8seed")
    print_asymmetry_ci(mc_mm, "same_compute_8seed")

    print(f"\n{'='*76}")
    print("NOISE FLOOR: ADJACENT-K (gap=1) mismatch delta")
    print(f"{'='*76}")
    print_noise_floor(fs_mm, "fixed_step_8seed")
    print_noise_floor(mc_mm, "same_compute_8seed")
