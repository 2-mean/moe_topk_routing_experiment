"""Deprecated: use scripts/build_visual_assets.py or scripts/generate_report_heatmaps.py."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/generate_report_heatmaps.py")], check=True)
