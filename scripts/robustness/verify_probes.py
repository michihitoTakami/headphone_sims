"""Probe-placement audit of the existing panel runs (issue #6, task 4).

For each panel subject x model, rebuilds the structured scene geometry and
checks the run's STORED probe positions (the ones the published numbers were
computed on) against:

- the legacy pressure-stencil air check (what the runs were vetted with),
- the stencil-aware check including the three staggered velocity stencils
  (a velocity corner on a rigid-masked face records a forced zero and biases
  intensity/incidence/diffuseness),
- achieved clearance to the nearest solid voxel center vs. the intended
  probe_offset.

The canal reference probe is reported separately: every TF/comb result hangs
off it, so a flagged canal probe would matter far more than a flagged
periphery probe.

Outputs: runs/probe_verification.json + console table.
Usage (repo root): uv run python scripts/robustness/verify_probes.py
"""

import dataclasses
import json
import time

import numpy as np
from common import MODEL_CONFIGS, structured_paths

from headphone_sims.experiments.config import load_config
from headphone_sims.geometry.scene import (
    build_scene,
    probe_solid_clearance,
    probe_stencil_air_mask,
)

MODELS = ["z1r", "lcd", "dx", "dca2"]


def legacy_pressure_mask(probes, solid, grid) -> np.ndarray:
    """The pre-issue-#6 check: 8 pressure-stencil corners air, clamped."""
    solid_np = solid.numpy()
    shape = np.asarray(grid.shape)
    keep = np.ones(len(probes), dtype=bool)
    for i, pos in enumerate(probes):
        base = np.floor(pos / grid.dx - 0.5).astype(int)
        for corner in range(8):
            idx = base + np.array([(corner >> 2) & 1, (corner >> 1) & 1, corner & 1])
            idx = np.clip(idx, 0, shape - 1)
            if solid_np[idx[0], idx[1], idx[2]]:
                keep[i] = False
                break
    return keep


def main() -> None:
    with open("runs/subjects_selected.json", encoding="utf-8") as fh:
        subjects = json.load(fh)["panel_with_pp1"]

    report: dict = {"subjects": subjects, "runs": {}}
    for subject in subjects:
        for model in MODELS:
            _inc, pin_path = structured_paths(subject, model)
            d = np.load(pin_path)
            pos = d["positions"]
            ref = int(d["reference_index"])

            t0 = time.time()
            cfg = load_config(MODEL_CONFIGS[model]).scene
            if subject != 1:
                cfg = dataclasses.replace(
                    cfg,
                    pinna=dataclasses.replace(
                        cfg.pinna, mesh_path=f"data/hutubs/pp{subject}_3DheadMesh.ply"
                    ),
                )
            cfg = dataclasses.replace(cfg, snapshot_every=0)
            built = build_scene(cfg, device="cpu")

            legacy_ok = legacy_pressure_mask(pos, built.solid, built.grid)
            stencil_ok = probe_stencil_air_mask(pos, built.solid, built.grid)
            clearance = probe_solid_clearance(pos, built.solid, built.grid)
            finite = clearance[np.isfinite(clearance)]
            entry = {
                "n_probes": len(pos),
                "n_legacy_fail": int((~legacy_ok).sum()),
                "n_stencil_fail": int((~stencil_ok).sum()),
                "canal_ref_ok": bool(stencil_ok[ref]),
                "canal_ref_clearance_mm": float(clearance[ref] * 1e3),
                "clearance_min_mm": float(finite.min() * 1e3) if len(finite) else None,
                "clearance_p05_mm": (
                    float(np.percentile(finite, 5) * 1e3) if len(finite) else None
                ),
                "n_below_offset_minus_dx": int(
                    (clearance < cfg.probe_offset - built.grid.dx).sum()
                ),
                "stencil_fail_indices": [int(i) for i in np.flatnonzero(~stencil_ok)],
            }
            report["runs"][f"pp{subject}_{model}"] = entry
            print(
                f"pp{subject:<3d} {model:4s} n={entry['n_probes']:3d} "
                f"legacy_fail={entry['n_legacy_fail']:2d} "
                f"stencil_fail={entry['n_stencil_fail']:2d} "
                f"canal_ok={entry['canal_ref_ok']} "
                f"clear_min={entry['clearance_min_mm']:.2f}mm "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    n_fail = sum(e["n_stencil_fail"] for e in report["runs"].values())
    n_tot = sum(e["n_probes"] for e in report["runs"].values())
    canal_bad = [k for k, e in report["runs"].items() if not e["canal_ref_ok"]]
    report["total"] = {
        "n_probes": n_tot,
        "n_stencil_fail": n_fail,
        "stencil_fail_fraction": n_fail / n_tot if n_tot else 0.0,
        "canal_ref_flagged": canal_bad,
    }
    with open("runs/probe_verification.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(
        f"total: {n_fail}/{n_tot} probes fail the stencil check "
        f"({100 * n_fail / max(n_tot, 1):.1f}%); canal refs flagged: {canal_bad or 'none'}"
    )
    print("saved runs/probe_verification.json")


if __name__ == "__main__":
    main()
