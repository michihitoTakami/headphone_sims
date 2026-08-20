"""Sponge/domain-margin sensitivity runs (issue #6, task 5).

The default absorber is a 30-cell graded sponge validated to -40 dB on a
coarse bare grid only; record lengths (2-2.5 ms) contain several boundary
round trips, so fine spectral structure (the C(f) comb) could in principle
carry boundary reflections. Two stress conditions on pp1, for the model with
the finest spectral claims (dca2) plus one reference (z1r):

- sp45:  sponge 30 -> 45 cells, margins unchanged
- sp60m: sponge 30 -> 60 cells AND margins x1.5 (lateral 15->22.5 mm,
         axial 20->30 mm)

Baseline = the report set (batch3_* / bare_*). Each condition runs
structured incident/pinna (with v) and bare incident/pinna so C(f) and
post-pinna metrics are computed exactly as in the report.

Outputs: runs/bs_{model}_{cond}_{phase}.npz, runs/bs_{model}_{cond}_bare_{phase}.npz
Usage (repo root): uv run python scripts/robustness/run_boundary.py
"""

import dataclasses
import os
import time

import numpy as np

from headphone_sims.experiments.config import load_config
from headphone_sims.geometry.scene import build_scene

MODELS = [
    ("z1r", "configs/hutubs_70mm_z1r_v2.yaml"),
    ("dca2", "configs/hutubs_dca_amts_real.yaml"),
]
CONDITIONS = [
    ("sp45", dict(sponge_thickness=45)),
    ("sp60m", dict(sponge_thickness=60, lateral_margin=22.5e-3, axial_margin=30e-3)),
    # Same-code control at the production absorber: distinguishes "the code
    # changed" from "the boundary changed" when comparing against batch3_*.
    ("sp30re", dict()),
    # C-PML at the same 30-cell depth (~-114 dB floor): the arbiter — its
    # C(f) is boundary-clean to far below the comb structure being measured.
    ("cpml", dict(absorber="cpml")),
    # Voxelization-phase probes at the SAME absorber: a half-cell lateral
    # lattice shift of the whole assembly isolates staircase-realization
    # sensitivity (the confound that made the naive sp45/sp60m comparison
    # blow up for models whose published alignment was half-integer).
    # Published lattice phases (old ceil-artifact sizing): z1r half-integer
    # (150.5 cells), dca2 integer (152.0). Under the fixed sizing:
    # - voxa (margin 15.25 mm): z1r 150.5 = published phase (control);
    #   dca2 152.5 = flipped (measures dca2's voxel-phase sensitivity).
    # - voxb (margin 15.0 mm):  z1r 150.0 = flipped (measures z1r's);
    #   dca2 152.0 = published phase (control).
    ("voxa", dict(lateral_margin=15.25e-3)),
    ("voxb", dict()),
]


def run_one(scene_cfg, out: str, incident: bool, save_v: bool) -> None:
    if os.path.exists(out):
        print("skip", out, flush=True)
        return
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
    print(f"done {out} ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    for key, cfg_path in MODELS:
        base = load_config(cfg_path).scene
        base = dataclasses.replace(base, snapshot_every=0)
        for cond, overrides in CONDITIONS:
            cfg = dataclasses.replace(base, **overrides)
            for phase, incident in [("incident", True), ("pinna", False)]:
                run_one(cfg, f"runs/bs_{key}_{cond}_{phase}.npz", incident, save_v=True)
            bare = dataclasses.replace(cfg, filters=(), baffle=False)
            for phase, incident in [("incident", True), ("pinna", False)]:
                run_one(
                    bare, f"runs/bs_{key}_{cond}_bare_{phase}.npz", incident, save_v=False
                )


if __name__ == "__main__":
    main()
