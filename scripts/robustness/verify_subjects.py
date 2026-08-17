"""Pipeline-robustness check for HUTUBS subjects beyond pp1 (issue #3, step 2).

For every ``data/hutubs/pp{N}_3DheadMesh.ply`` found (or an explicit list via
argv), verify that the pp1-tuned pipeline holds:

1. canal detection — the interaural-axis convention must land within ~2 mm of
   the axis (HUTUBS alignment guarantee);
2. probe selection — protrusion mask + concha disc + driver line-of-sight must
   yield a healthy probe count on the pinna proper;
3. scene assembly — a full ``build_scene`` with the Z1R v2 config must not
   raise (no pinna/part intersection, probes survive solid filtering).

Also extracts per-subject pinna shape features (canal recession behind the
tip, probe-cloud extents) used to pick a shape-diverse subject panel, and
saves a per-subject probe-layout figure for the report.

Run from the repo root:  uv run python scripts/robustness/verify_subjects.py
"""

import dataclasses
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK HK"
import numpy as np
import trimesh

from headphone_sims.experiments.config import load_config
from headphone_sims.geometry.pinna import find_ear_canal_entrance
from headphone_sims.geometry.scene import build_scene

BASE_CFG = "configs/hutubs_70mm_z1r_v2.yaml"


def check_subject(subject: int) -> dict:
    mesh_path = f"data/hutubs/pp{subject}_3DheadMesh.ply"
    head = trimesh.load(mesh_path)
    canal = find_ear_canal_entrance(head, side="left", lateral_axis=1)
    axis_dist = float(np.hypot(canal[0], canal[2]))  # distance from the y axis

    cfg = load_config(BASE_CFG).scene
    cfg = dataclasses.replace(
        cfg, pinna=dataclasses.replace(cfg.pinna, mesh_path=mesh_path), snapshot_every=0
    )
    built = build_scene(cfg, device="cpu")
    probes = built.probe_positions
    canal_pos = np.asarray(built.canal_position)
    rel = probes - canal_pos

    # Shape features in scene coords (driver side = -z): recession = how far
    # the canal sits behind the pinna's most protruding point; extents = probe
    # cloud span along the pinna (y) and across it (x).
    z_tip = float(probes[:, 2].min())
    recession = float(canal_pos[2] - z_tip)
    extent_y = float(np.ptp(rel[:, 1]))
    extent_x = float(np.ptp(rel[:, 0]))
    concha = np.linalg.norm(rel[:, :2], axis=1) < 15e-3
    concha_depth = float(canal_pos[2] - probes[concha, 2].min()) if concha.any() else 0.0

    return {
        "subject": subject,
        "canal_axis_dist_mm": axis_dist * 1e3,
        "canal_mesh_mm": [float(v) * 1e3 for v in canal],
        "n_probes": len(probes),
        "n_concha_probes": int(concha.sum()),
        "recession_mm": recession * 1e3,
        "concha_depth_mm": concha_depth * 1e3,
        "extent_y_mm": extent_y * 1e3,
        "extent_x_mm": extent_x * 1e3,
        "_rel_probes": rel.tolist(),
    }


def main() -> None:
    if len(sys.argv) > 1:
        subjects = [int(a) for a in sys.argv[1:]]
    else:
        subjects = sorted(
            int(m.group(1))
            for p in Path("data/hutubs").glob("pp*_3DheadMesh.ply")
            if (m := re.match(r"pp(\d+)_", p.name))
        )
    results, failures = [], []
    for s in subjects:
        try:
            r = check_subject(s)
            ok = r["canal_axis_dist_mm"] < 2.0 and r["n_probes"] >= 150
            r["ok"] = bool(ok)
            results.append(r)
            print(
                f"pp{s:<3d} axis {r['canal_axis_dist_mm']:.2f}mm probes {r['n_probes']:3d} "
                f"(concha {r['n_concha_probes']}) recession {r['recession_mm']:.1f}mm "
                f"extent {r['extent_x_mm']:.0f}x{r['extent_y_mm']:.0f}mm "
                f"{'OK' if ok else 'SUSPECT'}"
            )
        except Exception as e:
            failures.append({"subject": s, "error": str(e)})
            print(f"pp{s:<3d} FAILED: {e}")

    n = len(results)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, r in enumerate(results):
        ax = axes[i // cols][i % cols]
        ax.set_visible(True)
        rel = np.asarray(r["_rel_probes"])
        depth = (rel[:, 2] - rel[:, 2].min()) * 1e3
        sc = ax.scatter(rel[:, 0] * 1e3, rel[:, 1] * 1e3, c=depth, s=7, cmap="viridis")
        ax.plot(0, 0, "r+", ms=10, mew=2)
        ax.set_aspect("equal")
        ax.set_title(
            f"pp{r['subject']}  {r['n_probes']}点 奥行{r['recession_mm']:.0f}mm",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)
    fig.colorbar(sc, ax=axes, label="先端からの奥行き (mm)", shrink=0.6)
    fig.suptitle("被験者別プローブ配置(外耳道基準、+ = 外耳道入口)", fontsize=12)
    fig.savefig("runs/subjects_probe_layouts.png", dpi=110, bbox_inches="tight")
    print("saved runs/subjects_probe_layouts.png")

    for r in results:
        del r["_rel_probes"]
    out = {"results": results, "failures": failures}
    with open("runs/subjects_verify.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("saved runs/subjects_verify.json")


if __name__ == "__main__":
    main()
