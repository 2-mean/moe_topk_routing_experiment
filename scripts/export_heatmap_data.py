"""Export 8x8 heatmap matrices as JSON for HTML presentation (stdlib only)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FS = ROOT / "results/sparse32_kgrid_fixed_step_8seed_summary/summary.md"
MC = ROOT / "results/sparse32_kgrid_same_compute_8seed_summary/summary.md"
OUT = ROOT / "results/report_figures/heatmap_data.json"
E = 32
KS = list(range(1, 9))


def parse_summary(path: Path):
    text = path.read_text(encoding="utf-8")
    pair_rows, mm_rows = [], []
    in_pair = in_mm = False
    for line in text.splitlines():
        if "| pair | metric | mean | values |" in line:
            in_pair, in_mm = True, False
            continue
        if "| train_k | inference_k | mean_delta_loss | values |" in line:
            in_mm, in_pair = True, False
            continue
        if in_pair:
            if not line.startswith("|"):
                in_pair = False
                continue
            if line.startswith("|---"):
                continue
            p = [x.strip() for x in line.strip("|").split("|")]
            if len(p) >= 4:
                try:
                    pair, metric, mean_ = p[0], p[1], float(p[2])
                    a, b = (int(x) for x in pair.split("-"))
                    pair_rows.append({"pair": pair, "a": a, "b": b, "metric": metric, "mean": mean_})
                except Exception:
                    pass
        elif in_mm:
            if not line.startswith("|"):
                in_mm = False
                continue
            if line.startswith("|---"):
                continue
            p = [x.strip() for x in line.strip("|").split("|")]
            if len(p) >= 4:
                try:
                    tk, ik, mean_ = int(p[0]), int(p[1]), float(p[2])
                    mm_rows.append({"train_k": tk, "infer_k": ik, "mean": mean_})
                except Exception:
                    pass
    return pair_rows, mm_rows


def pair_matrix(pair_rows, metric, diagonal=1.0, symmetric=True):
    mat = [[None] * 8 for _ in range(8)]
    for r in pair_rows:
        if r["metric"] != metric:
            continue
        a, b = r["a"] - 1, r["b"] - 1
        mat[a][b] = round(r["mean"], 4)
        if symmetric:
            mat[b][a] = round(r["mean"], 4)
    for i in range(8):
        mat[i][i] = diagonal
    return mat


def mismatch_matrix(mm_rows):
    mat = [[None] * 8 for _ in range(8)]
    for r in mm_rows:
        mat[r["train_k"] - 1][r["infer_k"] - 1] = round(r["mean"], 4)
    for i in range(8):
        mat[i][i] = 0.0
    return mat


def asymmetry_matrix(mm_rows):
    delta = {(r["train_k"], r["infer_k"]): r["mean"] for r in mm_rows}
    mat = [[None] * 8 for _ in range(8)]
    for a in KS:
        for b in KS:
            if a == b:
                mat[a - 1][b - 1] = 0.0
                continue
            hi_lo = delta.get((max(a, b), min(a, b)))
            lo_hi = delta.get((min(a, b), max(a, b)))
            if hi_lo is None or lo_hi is None:
                continue
            if a > b:
                mat[a - 1][b - 1] = round(hi_lo - lo_hi, 4)
            else:
                mat[a - 1][b - 1] = round(lo_hi - hi_lo, 4)
    return mat


def nestedness_excess(pair_rows):
    mat = [[None] * 8 for _ in range(8)]
    for r in pair_rows:
        if r["metric"] != "nestedness":
            continue
        a, b = r["a"], r["b"]
        excess = round(r["mean"] - max(a, b) / E, 4)
        mat[a - 1][b - 1] = excess
        mat[b - 1][a - 1] = excess
    for i in range(8):
        mat[i][i] = 0.0
    return mat


def budget_block(path: Path):
    pairs, mm = parse_summary(path)
    return {
        "top1_agreement": pair_matrix(pairs, "top1_agreement"),
        "nestedness": pair_matrix(pairs, "nestedness"),
        "spearman": pair_matrix(pairs, "spearman"),
        "nestedness_excess": nestedness_excess(pairs),
        "mismatch_delta": mismatch_matrix(mm),
        "asymmetry": asymmetry_matrix(mm),
    }


payload = {
    "labels": [f"k={k}" for k in KS],
    "random_baselines": {"top1": round(1 / E, 4), "spearman": 0.0},
    "fixed_step": budget_block(FS),
    "same_compute": budget_block(MC),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"wrote {OUT}")
