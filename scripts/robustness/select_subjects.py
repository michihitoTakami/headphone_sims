"""Pick a shape-diverse subject panel from runs/subjects_verify.json (issue #3).

Greedy max-min (farthest-point) selection in normalized feature space
(canal recession, concha depth, pinna extents) over the subjects that passed
verification. pp1 is always included (the report baseline) and seeds the
selection. Writes runs/subjects_selected.json with the panel, largest-spread
first.

Usage (repo root):  uv run python scripts/robustness/select_subjects.py [n_total]
"""

import json
import sys

import numpy as np

FEATURES = ["recession_mm", "concha_depth_mm", "extent_y_mm", "extent_x_mm"]


def main() -> None:
    n_total = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    with open("runs/subjects_verify.json", encoding="utf-8") as fh:
        results = [r for r in json.load(fh)["results"] if r["ok"]]
    subjects = [r["subject"] for r in results]
    feats = np.array([[r[f] for f in FEATURES] for r in results])
    z = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-12)

    chosen = [subjects.index(1)] if 1 in subjects else [0]
    while len(chosen) < min(n_total, len(subjects)):
        d = np.min(
            np.linalg.norm(z[:, None, :] - z[None, chosen, :], axis=2), axis=1
        )
        d[chosen] = -1.0
        chosen.append(int(np.argmax(d)))

    panel = [subjects[i] for i in chosen]
    print("panel:", panel)
    for i in chosen:
        r = results[subjects.index(subjects[i])]
        print(
            f"  pp{r['subject']:<3d} recession {r['recession_mm']:5.1f}  "
            f"concha {r['concha_depth_mm']:5.1f}  extent {r['extent_x_mm']:.0f}x{r['extent_y_mm']:.0f}"
        )
    with open("runs/subjects_selected.json", "w", encoding="utf-8") as fh:
        json.dump({"subjects": [s for s in panel if s != 1], "panel_with_pp1": panel}, fh)
    print("saved runs/subjects_selected.json (run_subjects.py reads 'subjects')")


if __name__ == "__main__":
    main()
