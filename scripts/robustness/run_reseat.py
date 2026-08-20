"""Re-seat sensitivity runs on pp1, 4 models (issue #3, auxiliary experiment).

Conditions perturb the v2 placement in two axes:

- axial (pad compression / loose fit): distance -1 mm ("axm1") and +2 mm
  ("axp2"). -2 mm would touch the front structure of the closest models, so
  -1 mm is the physical inward limit.
- vertical (worn high/low): pinna offset_y -2 mm ("vym2") and +2 mm ("vyp2")
  — the AMTS slope axis runs along y, so this also translates DCA's
  position-dependent transmission map across the pinna.

The unperturbed baseline is the report set (batch3_* / bare_*). Each
condition gets structured incident/pinna (with v) and bare incident/pinna,
so C(f) and post-pinna metrics are computed exactly as in the report.

Outputs: runs/rs_{model}_{cond}_{phase}.npz and runs/rs_{model}_{cond}_bare_{phase}.npz

Usage (repo root):  uv run python scripts/robustness/run_reseat.py
"""

import dataclasses
import os
import time

import numpy as np

from headphone_sims.experiments.config import load_config
from headphone_sims.geometry.scene import build_scene

MODELS = [
    ("z1r", "configs/hutubs_70mm_z1r_v2.yaml"),
    ("lcd", "configs/hutubs_90mm_planar_v2.yaml"),
    ("dx", "configs/hutubs_40mm_dome_v2.yaml"),
    ("dca2", "configs/hutubs_dca_amts_real.yaml"),
]
CONDITIONS = [
    ("axm1", dict(distance=-1e-3)),
    ("axp2", dict(distance=+2e-3)),
    ("vym2", dict(offset_y=-2e-3)),
    ("vyp2", dict(offset_y=+2e-3)),
]


def perturb(scene_cfg, deltas):
    if "distance" in deltas:
        scene_cfg = dataclasses.replace(
            scene_cfg, distance=scene_cfg.distance + deltas["distance"]
        )
    if "offset_y" in deltas:
        scene_cfg = dataclasses.replace(
            scene_cfg,
            pinna=dataclasses.replace(scene_cfg.pinna, offset_y=deltas["offset_y"]),
        )
    return scene_cfg


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
        for cond, deltas in CONDITIONS:
            cfg = perturb(base, deltas)
            for phase, incident in [("incident", True), ("pinna", False)]:
                run_one(cfg, f"runs/rs_{key}_{cond}_{phase}.npz", incident, save_v=True)
            bare = dataclasses.replace(cfg, filters=(), baffle=False)
            for phase, incident in [("incident", True), ("pinna", False)]:
                run_one(
                    bare, f"runs/rs_{key}_{cond}_bare_{phase}.npz", incident, save_v=False
                )


if __name__ == "__main__":
    main()
