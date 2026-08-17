"""Regenerate every report figure and the report HTML, in order.

Run from the repo root AFTER scripts/report/run_models.py. Produces
runs/matched_filter_results.html (self-contained, images inlined as base64).
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
STEPS = [
    "fig_main.py",  # radar/lateral/levels/pinna TF/geo + final_summary.json
    "fig_coupling.py",  # coupling comb C(f) vs bare-source controls
    "fig_fr_overlay.py",  # per-probe FR overlays (level+shape / shape-only)
    "fig_local_uniformity.py",  # plane-fit gradient-vs-residual 2-axis plot
    "make_report.py",  # assemble the HTML
]

for step in STEPS:
    print(f"== {step}")
    subprocess.run([sys.executable, str(HERE / step)], check=True)
