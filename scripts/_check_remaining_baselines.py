"""Check remaining baseline gaps: expert frequency, routing-mismatch correlation, entropy."""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

FS = Path("results/sparse32_kgrid_fixed_step_8seed_summary")
MC = Path("results/sparse32_kgrid_same_compute_8seed_summary")


def sep(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# 1. Max expert share by k (uniformity check)
sep("Expert frequency: max_expert_share by k (matched inference)")
for label, path in [("fixed_step_8seed", FS), ("same_compute_8seed", MC)]:
    # Use metrics.csv: expert_frequency rows at matched inference
    rows = list(csv.DictReader((path / "metrics.csv").open(encoding="utf-8")))
    by_k: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r["metric_type"] == "expert_frequency" and r["metric_name"] == "max_expert_share":
            if r["train_k"] == r["inference_k"]:
                by_k[int(r["train_k"])].append(float(r["value"]))
    print(f"\n  {label}")
    print(f"  {'k':>3} | {'mean_share':>10} | {'std':>8} | {'n'}")
    for k in sorted(by_k):
        v = by_k[k]
        m = np.mean(v)
        s = np.std(v, ddof=1) if len(v) > 1 else 0.0
        note = " ← concentrated" if m > 0.15 else ""
        print(f"  {k:>3} | {m:>10.4f} | {s:>8.4f} | {len(v)}{note}")

# 2. Expert normalized entropy by k
sep("Expert normalized entropy by k (1=uniform, 0=collapsed)")
for label, path in [("fixed_step_8seed", FS), ("same_compute_8seed", MC)]:
    rows = list(csv.DictReader((path / "metrics.csv").open(encoding="utf-8")))
    by_k: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r["metric_type"] == "expert_frequency" and r["metric_name"] == "expert_normalized_entropy":
            if r["train_k"] == r["inference_k"]:
                by_k[int(r["train_k"])].append(float(r["value"]))
    print(f"\n  {label}")
    for k in sorted(by_k):
        v = by_k[k]
        m = np.mean(v)
        note = " ← near-uniform" if m > 0.9 else " ← concentrated"
        print(f"  k={k}: mean_entropy={m:.4f}{note}")

# 3. Routing-mismatch correlation (from diagnostics)
sep("Routing-mismatch correlation (metric_correlations.csv)")
for label, path in [("fixed_step_8seed", FS), ("same_compute_8seed", MC)]:
    diag = path / "diagnostics" / "metric_correlations.csv"
    if not diag.exists():
        print(f"  {label}: not found")
        continue
    rows = list(csv.DictReader(diag.open(encoding="utf-8")))
    cand = [
        (r["pair"], r["metric_name"], float(r["spearman"]), int(r["n"]))
        for r in rows
        if r["target"] == "mean_abs_delta_loss"
        and r["n"] not in ("0", "")
        and r["metric_name"] in {"nestedness", "top1_agreement", "spearman"}
        and r["spearman"] not in ("nan", "")
    ]
    cand.sort(key=lambda x: -abs(x[2]))
    print(f"\n  {label} - top correlations with |mismatch delta|:")
    print(f"  {'pair':>6} | {'metric':>16} | {'spearman':>9} | n")
    for pair, metric, sp, n in cand[:10]:
        star = " *" if abs(sp) >= 0.5 else ""
        print(f"  {pair:>6} | {metric:>16} | {sp:>+9.3f} | {n}{star}")

# 4. B5 implication: expert frequency vs routing alignment
sep("B5 interpretation: why does within-k alignment converge to ~random?")
print("Within-k alignment ~= 1/E = 0.031 (random).")
print("Expert entropy near 1.0 (uniform) means many experts used.")
print("Each model learns a different subset -> within-k alignment ~= random.")
print("GOOD for Claim A: same-k gives diverse routing, across-k shows systematic structure above it.")

# 5. Step 0 same-W0 analysis
sep("Step-0 oracle: same W_0 different k routing (already in metrics)")
for label, path in [("fixed_step_8seed", FS), ("same_compute_8seed", MC)]:
    metrics_path = path / "metrics.csv"
    if not metrics_path.exists():
        continue
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    step0 = [float(r["value"]) for r in rows
              if r["metric_type"] == "step0_same_w0_same_infer"
              and r["metric_name"] == "nestedness"]
    step0_top1 = [float(r["value"]) for r in rows
                  if r["metric_type"] == "step0_same_w0_same_infer"
                  and r["metric_name"] == "top1_agreement"]
    if step0:
        print(f"  {label}: step-0 nestedness mean={np.mean(step0):.4f}, "
              f"top1={np.mean(step0_top1):.4f}, n={len(step0)}")

# 6. Missing baselines assessment
sep("REMAINING BASELINE GAPS: priority assessment")
gaps = [
    ("B3 Frequency-empirical null",
     "Can within-k ≈ random already serve as indirect evidence? YES partially.",
     "B5 shows within-k ≈ 1/E = random baseline. Entropy near 1.0 (uniform) "
     "explains WHY: diverse expert usage means cross-seed correlation is near-zero. "
     "Formal B3 (off-diagonal FAO) would strengthen Claim C further but B5 partially covers it.",
     "LOW-MEDIUM (B5 partially covers)"),
    ("B6 Step-matched within-k",
     "How much does same-k routing drift during training?",
     "Would need step-0 vs final alignment for same (k, seed). "
     "Relevant for same-compute: k=1 has 6000 steps vs k=8 at 750 steps. "
     "Already verifiable from existing checkpoint data if we compare "
     "step-0 vs step-final within same (seed, k).",
     "MEDIUM (affects same-compute Claim A interpretation)"),
    ("Routing-mismatch mechanism",
     "Does routing divergence PREDICT mismatch cost?",
     "metric_correlations.csv has Pearson/Spearman but n=3-9 seeds only. "
     "Underpowered for strong claim. Current correlations are descriptive only.",
     "LOW (underpowered, don't overclaim)"),
    ("Cross-budget same-k alignment",
     "Does fixed-step vs same-compute produce same routing at k=4?",
     "k=4 at 1500 steps is identical in both settings. "
     "If we run within-k across the two budget runs, "
     "they should show high alignment. Nice sanity check.",
     "LOW (nice to have, not essential)"),
]

for name, question, detail, priority in gaps:
    print(f"\n  [{priority}] {name}")
    print(f"  Q: {question}")
    print(f"  → {detail}")
