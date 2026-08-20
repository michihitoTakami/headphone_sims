"""Grid-convergence runs at dx = 0.5 / 0.4 / 0.3 mm (issue #6, task 3).

The AMTS insert's thinnest feature (between-row wall, ~0.57 mm) is ~1.1
cells at the production dx = 0.5 mm; the published spatial-spectral results
must be shown stable under refinement before being read as physics. pp1,
the model with the finest geometry (dca2) plus one reference (z1r):

- dx = 0.5 mm reuses the report set (batch3_* / bare_*),
- dx = 0.4 / 0.3 mm run structured incident/pinna (with v) and bare
  incident/pinna here (bare pairs are needed so C(f) sees the same numerical
  dispersion/staircase in numerator and denominator).

Voxelized filter porosity (what the grid actually realizes) is recorded per
dx into runs/convergence_porosity.json — a direct staircasing indicator.

GPU cost on an RTX 5060 Ti: dx 0.4 ~ 2-3 min/run, dx 0.3 ~ 8-10 min/run
(~1.5 h total, ~7 GB VRAM peak at dx 0.3).

Outputs: runs/conv_{model}_dx{um}_{phase}.npz (+ _bare_), convergence_porosity.json
Usage (repo root): uv run python scripts/robustness/run_convergence.py
"""

import dataclasses
import json
import os
import time

import numpy as np

from headphone_sims.experiments.config import load_config
from headphone_sims.geometry.scene import build_scene

MODELS = [
    ("z1r", "configs/hutubs_70mm_z1r_v2.yaml"),
    ("dca2", "configs/hutubs_dca_amts_real.yaml"),
]
DXS = [0.4e-3, 0.3e-3]  # 0.5e-3 = the existing report set


def run_one(scene_cfg, out: str, incident: bool, save_v: bool) -> list[float]:
    if os.path.exists(out):
        print("skip", out, flush=True)
        return []
    t0 = time.time()
    built = build_scene(scene_cfg, incident_only=incident)
    res = built.simulation.run()
    arrays = dict(
        p=res.p.astype(np.float32),
        positions=res.positions,
        dt=res.dt,
        dx=res.dx,
        driver_center=np.asarray(built.driver_center),
        canal=np.asarray(built.canal_position),
        reference_index=built.reference_index,
    )
    if save_v:
        arrays["v"] = res.v.astype(np.float32)
    np.savez_compressed(out, **arrays)
    print(
        f"done {out} ({built.grid.n_cells / 1e6:.0f}M cells, {time.time() - t0:.0f}s)",
        flush=True,
    )
    return built.porosities


def main() -> None:
    porosity: dict = {}
    for key, cfg_path in MODELS:
        base = load_config(cfg_path).scene
        base = dataclasses.replace(base, snapshot_every=0)
        # dx = 0.5 porosity for reference (geometry build only, no run needed:
        # the incident run below is skipped when the npz exists, so grab it
        # from a cheap CPU geometry build).
        built = build_scene(dataclasses.replace(base, record_ms=0.01), device="cpu")
        porosity[f"{key}_dx500"] = built.porosities
        for dx in DXS:
            um = round(dx * 1e6)
            cfg = dataclasses.replace(base, dx=dx)
            for phase, incident in [("incident", True), ("pinna", False)]:
                por = run_one(
                    cfg, f"runs/conv_{key}_dx{um}_{phase}.npz", incident, save_v=True
                )
                if por:
                    porosity[f"{key}_dx{um}"] = por
            bare = dataclasses.replace(cfg, filters=(), baffle=False)
            for phase, incident in [("incident", True), ("pinna", False)]:
                run_one(
                    bare, f"runs/conv_{key}_dx{um}_bare_{phase}.npz", incident, save_v=False
                )
    with open("runs/convergence_porosity.json", "w", encoding="utf-8") as fh:
        json.dump(porosity, fh, indent=1)
    print("saved runs/convergence_porosity.json")


if __name__ == "__main__":
    main()
