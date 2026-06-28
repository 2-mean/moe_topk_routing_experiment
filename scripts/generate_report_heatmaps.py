"""Generate missing heatmaps for the full report.

Produces:
  results/report_figures/
    01_mismatch_delta_comparison.png     fixed vs same-compute mismatch delta side-by-side
    02_asymmetry_direction.png           asymmetry = delta(hi->lo) - delta(lo->hi)
    03_asymmetry_gap_ci.png              asymmetry by k-gap with 95% CI + linear baseline
    04_noise_floor_direction.png         adjacent-k SNR by direction
    05_top1_agreement_comparison.png     top1 fixed vs same-compute side-by-side
    06_matched_loss_comparison.png       matched loss by k, two budgets
    07_nestedness_comparison.png       nestedness (overlap recall) fixed vs same-compute
    08_spearman_comparison.png         spearman rank correlation fixed vs same-compute
    09_nestedness_excess_comparison.png  obs nestedness − random baseline (max k / E)
    presentation/                      larger single-budget heatmaps for slides
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path("results/report_figures")
PRES = OUT / "presentation"
OUT.mkdir(parents=True, exist_ok=True)
PRES.mkdir(parents=True, exist_ok=True)

FS = Path("results/sparse32_kgrid_fixed_step_8seed_summary")
MC = Path("results/sparse32_kgrid_same_compute_8seed_summary")


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


fs_pairs, fs_mm = parse_summary(FS)
mc_pairs, mc_mm = parse_summary(MC)
KS = list(range(1, 9))
E = 32

STYLE = {
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}
plt.rcParams.update(STYLE)


def annotate_cells(ax, mat, fmt=".3f", fontsize=7, threshold=None):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            color = "white" if (threshold is not None and abs(v) > threshold) else "black"
            ax.text(j, i, f"{v:{fmt}}", ha="center", va="center", fontsize=fontsize, color=color)


def build_pair_matrix(pair_rows, metric: str, diagonal: float = 1.0, symmetric: bool = True):
    mat = np.full((8, 8), np.nan)
    for r in pair_rows:
        if r["metric"] != metric:
            continue
        a, b = r["a"] - 1, r["b"] - 1
        mat[a, b] = r["mean"]
        if symmetric:
            mat[b, a] = r["mean"]
    for i in range(8):
        mat[i, i] = diagonal
    return mat


def build_nestedness_excess_matrix(pair_rows):
    mat = np.full((8, 8), np.nan)
    for r in pair_rows:
        if r["metric"] != "nestedness":
            continue
        a, b = r["a"], r["b"]
        rand = max(a, b) / E
        excess = r["mean"] - rand
        ai, bi = a - 1, b - 1
        mat[ai, bi] = excess
        mat[bi, ai] = excess
    for i in range(8):
        mat[i, i] = 0.0
    return mat


def save_single_heatmap(mat, title, out_path, vmin, vmax, cmap, cbar_label, fmt=".3f", dpi=180):
    fig, ax = plt.subplots(figsize=(8.5, 7.2), constrained_layout=True)
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"k={k}" for k in KS], rotation=45, ha="right")
    ax.set_yticks(range(8))
    ax.set_yticklabels([f"k={k}" for k in KS])
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("train_k (model B)", fontsize=10)
    ax.set_ylabel("train_k (model A)", fontsize=10)
    thr = (vmax - vmin) * 0.55 if vmax != vmin else None
    annotate_cells(ax, mat, fmt=fmt, fontsize=9, threshold=thr)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_presentation_pair(mats_labels, stem, vmin, vmax, cmap, cbar_label, fmt=".3f"):
    for mat, label in mats_labels:
        save_single_heatmap(
            mat,
            f"{stem} — {label} (8-seed mean)",
            PRES / f"{stem}_{label.lower().replace('-', '_')}.png",
            vmin, vmax, cmap, cbar_label, fmt=fmt,
        )


# ---------------------------------------------------------------------------
# Helper: build 8x8 mismatch delta matrix
# ---------------------------------------------------------------------------
def build_mismatch_matrix(mm_rows):
    mat = np.full((8, 8), np.nan)
    for r in mm_rows:
        tk, ik = r["train_k"] - 1, r["infer_k"] - 1
        mat[tk, ik] = r["mean"]
    for i in range(8):
        mat[i, i] = 0.0  # matched = 0 by definition
    return mat


# ---------------------------------------------------------------------------
# 01: Mismatch delta side-by-side
# ---------------------------------------------------------------------------
fs_mat = build_mismatch_matrix(fs_mm)
mc_mat = build_mismatch_matrix(mc_mm)
vmax = max(np.nanmax(np.abs(fs_mat)), np.nanmax(np.abs(mc_mat)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
for ax, mat, label in zip(axes, [fs_mat, mc_mat], ["Fixed-step", "Same-compute"]):
    im = ax.imshow(mat, vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(8)); ax.set_xticklabels([f"k_infer={k}" for k in KS], rotation=45, ha="right")
    ax.set_yticks(range(8)); ax.set_yticklabels([f"k_train={k}" for k in KS])
    ax.set_title(f"Mismatch delta loss — {label} 8-seed\n(train_k row, infer_k col; diagonal=0 by definition)")
    ax.set_xlabel("Inference k"); ax.set_ylabel("Train k")
    annotate_cells(ax, mat, fmt=".3f", fontsize=6.5, threshold=vmax * 0.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="loss(train,infer) − loss(train,train)")
fig.suptitle("Figure 1: Mismatch loss delta heatmap (mean over 8 seeds)\nPositive = inference mismatch increases loss; asymmetry visible: hi→lo >> lo→hi", fontsize=10)
fig.savefig(OUT / "01_mismatch_delta_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 01")
save_presentation_pair(
    [(fs_mat, "Fixed-step"), (mc_mat, "Same-compute")],
    "01_mismatch_delta", -vmax, vmax, "RdBu_r",
    "loss(train,infer) − loss(train,train)", fmt=".3f",
)


# ---------------------------------------------------------------------------
# 02: Asymmetry direction heatmap  |delta(hi→lo) - delta(lo→hi)|
# ---------------------------------------------------------------------------
def build_asymmetry_matrix(mm_rows):
    delta_map = {(r["train_k"], r["infer_k"]): r["mean"] for r in mm_rows}
    mat = np.full((8, 8), np.nan)
    for a in KS:
        for b in KS:
            if a == b:
                mat[a - 1, b - 1] = 0.0
                continue
            # signed asymmetry: positive = hi→lo more costly
            hi_lo = delta_map.get((max(a, b), min(a, b)), np.nan)
            lo_hi = delta_map.get((min(a, b), max(a, b)), np.nan)
            if not (np.isnan(hi_lo) or np.isnan(lo_hi)):
                if a > b:  # upper triangle: hi=row, lo=col
                    mat[a - 1, b - 1] = hi_lo - lo_hi  # signed: positive = hi→lo worse
                else:
                    mat[a - 1, b - 1] = lo_hi - hi_lo  # lower triangle: negated
    return mat


fs_asym = build_asymmetry_matrix(fs_mm)
mc_asym = build_asymmetry_matrix(mc_mm)
vmax_a = max(np.nanmax(np.abs(fs_asym)), np.nanmax(np.abs(mc_asym)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
for ax, mat, label in zip(axes, [fs_asym, mc_asym], ["Fixed-step", "Same-compute"]):
    im = ax.imshow(mat, vmin=-vmax_a, vmax=vmax_a, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(8)); ax.set_xticklabels([f"k={k}" for k in KS], rotation=45, ha="right")
    ax.set_yticks(range(8)); ax.set_yticklabels([f"k={k}" for k in KS])
    ax.set_title(f"Asymmetry: delta(hi→lo) − delta(lo→hi) — {label}")
    ax.set_xlabel("k_b"); ax.set_ylabel("k_a")
    annotate_cells(ax, mat, fmt=".2f", fontsize=6.5, threshold=vmax_a * 0.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="signed asymmetry")
    # mark diagonal
    for i in range(8):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="gray", lw=0.5))
fig.suptitle("Figure 2: Directional asymmetry heatmap\nRed (upper triangle): hi→lo more costly. Blue: lo→hi more costly.", fontsize=10)
fig.savefig(OUT / "02_asymmetry_direction.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 02")
save_presentation_pair(
    [(fs_asym, "Fixed-step"), (mc_asym, "Same-compute")],
    "02_asymmetry", -vmax_a, vmax_a, "RdBu_r",
    "delta(hi→lo) − delta(lo→hi)", fmt=".2f",
)


# ---------------------------------------------------------------------------
# 03: Asymmetry by k-gap with 95% CI and linear extrapolation
# ---------------------------------------------------------------------------
from collections import defaultdict
import math
from scipy.stats import t as t_dist

def per_seed_asym_by_gap(mm_rows):
    delta = {(r["train_k"], r["infer_k"], si): v
             for r in mm_rows for si, v in enumerate(r["values"])}
    n_seeds = max(len(r["values"]) for r in mm_rows)
    ks = sorted({r["train_k"] for r in mm_rows})
    gap_asym = defaultdict(list)
    gap_lohi = defaultdict(list)
    gap_hilo = defaultdict(list)
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

def ci95(vals):
    n = len(vals)
    if n < 2: return float("nan"), float("nan")
    mu, se = np.mean(vals), np.std(vals, ddof=1) / math.sqrt(n)
    tc = t_dist.ppf(0.975, df=n - 1)
    return float(mu - tc * se), float(mu + tc * se)

fs_ga, fs_gl, fs_gh = per_seed_asym_by_gap(fs_mm)
mc_ga, mc_gl, mc_gh = per_seed_asym_by_gap(mc_mm)

gaps = sorted(fs_ga.keys())
A1_fs = np.mean(fs_ga[1])
A1_mc = np.mean(mc_ga[1])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
for ax, ga, gl, gh, A1, label, color in zip(
    axes,
    [fs_ga, mc_ga], [fs_gl, mc_gl], [fs_gh, mc_gh],
    [A1_fs, A1_mc],
    ["Fixed-step", "Same-compute"],
    ["#2166ac", "#d6604d"],
):
    means = [np.mean(ga[g]) for g in gaps]
    ci_lo = [ci95(ga[g])[0] for g in gaps]
    ci_hi = [ci95(ga[g])[1] for g in gaps]
    linear = [g * A1 for g in gaps]
    lohi_means = [np.mean(gl[g]) for g in gaps]
    hilo_means = [np.mean(gh[g]) for g in gaps]

    x = np.array(gaps)
    ax.bar(x - 0.2, [abs(l) for l in lohi_means], 0.35, label="lo→hi |delta|", color="#92c5de", alpha=0.8)
    ax.bar(x + 0.2, hilo_means, 0.35, label="hi→lo delta", color="#d73027", alpha=0.8)
    ax.errorbar(x, means, yerr=[np.array(means) - np.array(ci_lo), np.array(ci_hi) - np.array(means)],
                fmt="o-", color=color, linewidth=2, markersize=5, label="|Asymmetry| (mean ± 95% CI)")
    ax.plot(x, linear, "--", color="gray", linewidth=1.5, label=f"Linear extrap from gap=1 ({A1:.3f}×gap)")
    ax.fill_between(x, ci_lo, ci_hi, alpha=0.15, color=color)
    # mark gap=7 separately
    g7_idx = gaps.index(7)
    ax.annotate(f"CI lo={ci_lo[g7_idx]:.3f}\nlinear={linear[g7_idx]:.3f}",
                xy=(7, means[g7_idx]), xytext=(6.3, means[g7_idx] * 0.85),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("k-gap"); ax.set_ylabel("Loss delta")
    ax.set_title(f"Asymmetry by k-gap — {label} 8-seed\nCI lower bound > linear extrapolation at gap=7")
    ax.set_xticks(gaps)
    ax.legend(fontsize=7, loc="upper left")

fig.suptitle("Figure 3: Mismatch asymmetry by k-gap with 95% seed-level CI vs linear extrapolation from gap=1", fontsize=10)
fig.savefig(OUT / "03_asymmetry_gap_ci.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 03")


# ---------------------------------------------------------------------------
# 04: Noise floor — adjacent-k SNR by direction
# ---------------------------------------------------------------------------
def noise_floor_data(mm_rows):
    rows = []
    for r in mm_rows:
        tk, ik = r["train_k"], r["infer_k"]
        if abs(ik - tk) != 1: continue
        vals = r["values"]
        m = np.mean(vals); s = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        snr = abs(m) / s if s > 0 else 99.0
        direction = "hi→lo" if tk > ik else "lo→hi"
        rows.append({"pair": f"{tk}→{ik}", "mean": m, "std": s, "snr": snr, "direction": direction})
    return rows

fs_nf = noise_floor_data(fs_mm)
mc_nf = noise_floor_data(mc_mm)

fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
for row_ax, nf_data, label in zip(axes, [fs_nf, mc_nf], ["Fixed-step", "Same-compute"]):
    pairs = [r["pair"] for r in nf_data]
    means = [r["mean"] for r in nf_data]
    stds = [r["std"] for r in nf_data]
    snrs = [r["snr"] for r in nf_data]
    dirs = [r["direction"] for r in nf_data]
    colors = ["#d73027" if d == "hi→lo" else "#4575b4" for d in dirs]

    ax_delta = row_ax[0]
    ax_snr = row_ax[1]

    # Mean delta with std bars
    bars = ax_delta.bar(pairs, means, color=colors, alpha=0.8)
    ax_delta.errorbar(range(len(pairs)), means, yerr=stds, fmt="none", color="black", capsize=3, linewidth=1)
    ax_delta.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_delta.set_xticklabels(pairs, rotation=45, ha="right", fontsize=7)
    ax_delta.set_ylabel("mean_delta_loss")
    ax_delta.set_title(f"{label}: adjacent-k mean delta ± std")
    red_patch = mpatches.Patch(color="#d73027", label="hi→lo")
    blue_patch = mpatches.Patch(color="#4575b4", label="lo→hi")
    ax_delta.legend(handles=[red_patch, blue_patch], fontsize=7)

    # SNR
    snr_colors = ["#d73027" if s >= 2.0 else "#cccccc" for s, d in zip(snrs, dirs)]
    ax_snr.bar(pairs, snrs, color=snr_colors, alpha=0.9)
    ax_snr.axhline(2.0, color="black", linestyle="--", linewidth=1.2, label="SNR=2 threshold")
    ax_snr.set_xticklabels(pairs, rotation=45, ha="right", fontsize=7)
    ax_snr.set_ylabel("SNR = |mean| / std")
    ax_snr.set_title(f"{label}: signal-to-noise ratio (SNR≥2 + all same sign = STRONG)")
    ax_snr.legend(fontsize=7)

fig.suptitle("Figure 4: Noise floor — adjacent-k (gap=1) mismatch\nRed bars: hi→lo (strong signal); gray/blue: lo→hi (noise level)", fontsize=10)
fig.savefig(OUT / "04_noise_floor_direction.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 04")


# ---------------------------------------------------------------------------
# 05: Top1 agreement — matched train-k pairs, fixed vs same-compute
# ---------------------------------------------------------------------------
def build_top1_matrix(pair_rows):
    return build_pair_matrix(pair_rows, "top1_agreement", diagonal=1.0)

fs_top1 = build_top1_matrix(fs_pairs)
mc_top1 = build_top1_matrix(mc_pairs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
for ax, mat, label in zip(axes, [fs_top1, mc_top1], ["Fixed-step", "Same-compute"]):
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(8)); ax.set_xticklabels([f"k={k}" for k in KS], rotation=45, ha="right")
    ax.set_yticks(range(8)); ax.set_yticklabels([f"k={k}" for k in KS])
    ax.set_title(f"Top-1 Agreement (matched final, same seed) — {label}\nRandom baseline = 1/32 = 0.031")
    ax.set_xlabel("train_k"); ax.set_ylabel("train_k")
    annotate_cells(ax, mat, fmt=".3f", fontsize=6.5, threshold=0.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="top1 agreement")
    # draw random baseline line annotation
    ax.text(7.6, 0, f"rand\n{1/E:.3f}", fontsize=6.5, color="gray", va="top")
fig.suptitle("Figure 5: Top-1 expert agreement between same-seed different-k models\n(Diagonal=1; random=0.031; oracle cutoff=1.0)", fontsize=10)
fig.savefig(OUT / "05_top1_agreement_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 05")
save_presentation_pair(
    [(fs_top1, "Fixed-step"), (mc_top1, "Same-compute")],
    "05_top1_agreement", 0, 1, "Blues", "top1 agreement", fmt=".3f",
)


# ---------------------------------------------------------------------------
# 06: Matched loss by k, two budgets
# ---------------------------------------------------------------------------
import csv
from collections import defaultdict as ddict

def get_matched_loss(summary_path):
    import csv
    metrics = list(csv.DictReader((summary_path / "metrics.csv").open(encoding="utf-8")))
    step_by_k = {}
    for r in metrics:
        if r["metric_type"] != "loss" or r["metric_name"] != "validation_loss": continue
        if r["train_k"] != r["inference_k"]: continue
        k, s = int(r["train_k"]), int(r["checkpoint_step"])
        if k not in step_by_k or s > step_by_k[k]: step_by_k[k] = s
    by_k = ddict(list)
    for r in metrics:
        if r["metric_type"] != "loss" or r["metric_name"] != "validation_loss": continue
        if r["train_k"] != r["inference_k"]: continue
        k, s = int(r["train_k"]), int(r["checkpoint_step"])
        if s == step_by_k.get(k): by_k[k].append(float(r["value"]))
    return {k: v for k, v in by_k.items()}

fs_loss = get_matched_loss(FS)
mc_loss = get_matched_loss(MC)

fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
ks_arr = np.array(KS)
fs_means = [np.mean(fs_loss.get(k, [np.nan])) for k in KS]
fs_stds = [np.std(fs_loss.get(k, [np.nan]), ddof=1) if len(fs_loss.get(k, [])) > 1 else 0 for k in KS]
mc_means = [np.mean(mc_loss.get(k, [np.nan])) for k in KS]
mc_stds = [np.std(mc_loss.get(k, [np.nan]), ddof=1) if len(mc_loss.get(k, [])) > 1 else 0 for k in KS]

ax.errorbar(ks_arr - 0.1, fs_means, yerr=fs_stds, fmt="o-", color="#2166ac", linewidth=2,
            markersize=6, capsize=4, label="Fixed-step (1500 steps all k)")
ax.errorbar(ks_arr + 0.1, mc_means, yerr=mc_stds, fmt="s-", color="#d6604d", linewidth=2,
            markersize=6, capsize=4, label="Same-compute (k×steps matched)")
ax.axvline(4, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="k=4 (common baseline)")
ax.set_xlabel("train_k = infer_k"); ax.set_ylabel("Validation loss")
ax.set_xticks(KS)
ax.set_title("Figure 6: Matched inference loss by k (mean ± std, 8 seeds)\n"
             "Fixed: higher k → lower loss. Same-compute: k=1 lowest (6000 update bias). k=4 = shared baseline.")
ax.legend(fontsize=8)
fig.savefig(OUT / "06_matched_loss_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 06")


# ---------------------------------------------------------------------------
# 07: Nestedness heatmap
# ---------------------------------------------------------------------------
fs_nest = build_pair_matrix(fs_pairs, "nestedness", diagonal=1.0)
mc_nest = build_pair_matrix(mc_pairs, "nestedness", diagonal=1.0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
for ax, mat, label in zip(axes, [fs_nest, mc_nest], ["Fixed-step", "Same-compute"]):
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(8)); ax.set_xticklabels([f"k={k}" for k in KS], rotation=45, ha="right")
    ax.set_yticks(range(8)); ax.set_yticklabels([f"k={k}" for k in KS])
    ax.set_title(f"Nestedness (overlap recall) — {label}\nRandom baseline = max(k_a,k_b)/32")
    ax.set_xlabel("train_k"); ax.set_ylabel("train_k")
    annotate_cells(ax, mat, fmt=".3f", fontsize=6.5, threshold=0.55)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="nestedness")
fig.suptitle("Figure 7: Smaller-k expert overlap recall within larger-k set (same-seed, different-k)", fontsize=10)
fig.savefig(OUT / "07_nestedness_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 07")
save_presentation_pair(
    [(fs_nest, "Fixed-step"), (mc_nest, "Same-compute")],
    "07_nestedness", 0, 1, "YlGnBu", "nestedness", fmt=".3f",
)


# ---------------------------------------------------------------------------
# 08: Spearman heatmap
# ---------------------------------------------------------------------------
fs_spear = build_pair_matrix(fs_pairs, "spearman", diagonal=1.0)
mc_spear = build_pair_matrix(mc_pairs, "spearman", diagonal=1.0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
for ax, mat, label in zip(axes, [fs_spear, mc_spear], ["Fixed-step", "Same-compute"]):
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="PuRd", aspect="auto")
    ax.set_xticks(range(8)); ax.set_xticklabels([f"k={k}" for k in KS], rotation=45, ha="right")
    ax.set_yticks(range(8)); ax.set_yticklabels([f"k={k}" for k in KS])
    ax.set_title(f"Spearman (gate logit ranking) — {label}\nRandom baseline = 0")
    ax.set_xlabel("train_k"); ax.set_ylabel("train_k")
    annotate_cells(ax, mat, fmt=".3f", fontsize=6.5, threshold=0.55)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="spearman")
fig.suptitle("Figure 8: Router logit ranking correlation between same-seed different-k models", fontsize=10)
fig.savefig(OUT / "08_spearman_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 08")
save_presentation_pair(
    [(fs_spear, "Fixed-step"), (mc_spear, "Same-compute")],
    "08_spearman", 0, 1, "PuRd", "spearman", fmt=".3f",
)


# ---------------------------------------------------------------------------
# 09: Nestedness excess (obs − max k / E)
# ---------------------------------------------------------------------------
fs_nex = build_nestedness_excess_matrix(fs_pairs)
mc_nex = build_nestedness_excess_matrix(mc_pairs)
nex_vmax = max(np.nanmax(fs_nex), np.nanmax(mc_nex))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
for ax, mat, label in zip(axes, [fs_nex, mc_nex], ["Fixed-step", "Same-compute"]):
    im = ax.imshow(mat, vmin=0, vmax=nex_vmax, cmap="Oranges", aspect="auto")
    ax.set_xticks(range(8)); ax.set_xticklabels([f"k={k}" for k in KS], rotation=45, ha="right")
    ax.set_yticks(range(8)); ax.set_yticklabels([f"k={k}" for k in KS])
    ax.set_title(f"Nestedness excess (obs − max k/32) — {label}\nStructure beyond cardinality")
    ax.set_xlabel("train_k"); ax.set_ylabel("train_k")
    annotate_cells(ax, mat, fmt=".3f", fontsize=6.5, threshold=nex_vmax * 0.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="excess over random")
fig.suptitle("Figure 9: Nestedness excess above uniform-random overlap baseline", fontsize=10)
fig.savefig(OUT / "09_nestedness_excess_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote 09")
save_presentation_pair(
    [(fs_nex, "Fixed-step"), (mc_nex, "Same-compute")],
    "09_nestedness_excess", 0, nex_vmax, "Oranges", "nestedness excess", fmt=".3f",
)


print(f"\nAll figures saved to {OUT}/")
print(f"Presentation-sized single-budget heatmaps saved to {PRES}/")
