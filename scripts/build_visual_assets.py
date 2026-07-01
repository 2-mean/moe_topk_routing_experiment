"""Regenerate all visual PNG assets (matplotlib figures + manifest).

Canonical output: results/report_figures/
  01–10 comparison PNGs + presentation/ singles + numeric JSON

Also exports heatmap_data.json for optional tooling.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results/report_figures"
MANIFEST = FIG / "manifest.json"

EXPECTED = [
    "01_mismatch_delta_comparison.png",
    "02_asymmetry_direction.png",
    "03_asymmetry_gap_ci.png",
    "04_noise_floor_direction.png",
    "05_top1_agreement_comparison.png",
    "06_matched_loss_comparison.png",
    "07_nestedness_comparison.png",
    "08_spearman_comparison.png",
    "09_nestedness_excess_comparison.png",
    "10_pct_change_comparison.png",
]

NUMERIC = ["heatmap_data.json", "pct_change_matrices.json"]


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/export_heatmap_data.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/generate_report_heatmaps.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/build_presentation_html.py")], check=True)

    files = []
    for name in EXPECTED:
        p = FIG / name
        if p.exists():
            files.append(name)
    pres = sorted(p.name for p in (FIG / "presentation").glob("*.png")) if (FIG / "presentation").exists() else []
    numeric = [name for name in NUMERIC if (FIG / name).exists()]

    manifest = {
        "generated": str(date.today()),
        "generator": "scripts/generate_report_heatmaps.py (matplotlib)",
        "directory": "results/report_figures/",
        "figures": files,
        "numeric": numeric,
        "presentation": [f"presentation/{n}" for n in pres],
        "regenerate": "python scripts/build_visual_assets.py",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST} ({len(files)} comparison + {len(pres)} presentation PNGs)")


if __name__ == "__main__":
    main()
