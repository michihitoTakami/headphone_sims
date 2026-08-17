"""All simulation runs needed by the 4-model comparison report.

Run from the repo root (GPU recommended; ~1-2 min per run). Skips any run
whose output already exists — delete the .npz to force a re-run.

Outputs (runs/, gitignored):
- batch3_{z1r,lcd,dx,dca2}_{incident,pinna}.npz  — structured models
  (incident = pinna solid removed; saves v for incidence/diffuseness metrics)
- bare_{z1r,lcd,dx,dca}_{incident,pinna}.npz     — bare-source controls
  (no filters, no baffle; pressure only) for coupling isolation C(f)
"""

import dataclasses
import os

import numpy as np

from headphone_sims.experiments.config import load_config
from headphone_sims.geometry.scene import build_scene

STRUCTURED = [
    ("z1r", "configs/hutubs_70mm_z1r_v2.yaml"),
    ("lcd", "configs/hutubs_90mm_planar_v2.yaml"),
    ("dx", "configs/hutubs_40mm_dome_v2.yaml"),
    ("dca2", "configs/hutubs_dca_amts_real.yaml"),
]
# Bare pairs depend only on the driver, so dca (idealized) and dca2 (real
# plug map) share the same bare control.
BARE = [
    ("z1r", "configs/hutubs_70mm_z1r_v2.yaml"),
    ("lcd", "configs/hutubs_90mm_planar_v2.yaml"),
    ("dx", "configs/hutubs_40mm_dome_v2.yaml"),
    ("dca", "configs/hutubs_dca_amts_real.yaml"),
]

for key, cfg_path in STRUCTURED:
    cfg = load_config(cfg_path).scene
    for phase, incident in [("incident", True), ("pinna", False)]:
        out = f"runs/batch3_{key}_{phase}.npz"
        if os.path.exists(out):
            print("skip", out)
            continue
        built = build_scene(cfg, incident_only=incident)
        res = built.simulation.run()
        np.savez_compressed(
            out, p=res.p.astype(np.float32), v=res.v.astype(np.float32),
            positions=res.positions, dt=res.dt,
            driver_center=np.asarray(built.driver_center),
            canal=np.asarray(built.canal_position),
            reference_index=built.reference_index,
        )
        print("done", out)

for key, cfg_path in BARE:
    base = load_config(cfg_path).scene
    scene = dataclasses.replace(base, filters=(), baffle=False, snapshot_every=0)
    for phase, incident in [("incident", True), ("pinna", False)]:
        out = f"runs/bare_{key}_{phase}.npz"
        if os.path.exists(out):
            print("skip", out)
            continue
        built = build_scene(scene, incident_only=incident)
        res = built.simulation.run()
        np.savez_compressed(
            out, p=res.p.astype(np.float32), positions=res.positions, dt=res.dt,
            driver_center=np.asarray(built.driver_center),
            reference_index=built.reference_index,
        )
        print("done", out)
