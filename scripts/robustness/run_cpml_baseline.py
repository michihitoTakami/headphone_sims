"""C-PML re-baseline of the 4-model comb comparison (issue #6, task 5 follow-up).

The boundary study showed the production sponge biases the canal C(f): dca2's
comb swing reads 11.9 dB under the sponge but 15.9 dB under the boundary-clean
C-PML (z1r: 15.9 -> 17.4 dB) — enough to matter for the published
"shallowest comb" ranking. This runs the remaining two models (lcd, dx) with
absorber="cpml" so all four have a boundary-clean comb measurement
(z1r/dca2 cpml pairs already exist as bs_{model}_cpml_*).

Outputs: runs/bs_{lcd,dx}_cpml_{phase}.npz (+ _bare_)
Usage (repo root): uv run python scripts/robustness/run_cpml_baseline.py
"""

import dataclasses
import os
import time

import numpy as np

from headphone_sims.experiments.config import load_config
from headphone_sims.geometry.scene import build_scene

MODELS = [
    ("lcd", "configs/hutubs_90mm_planar_v2.yaml"),
    ("dx", "configs/hutubs_40mm_dome_v2.yaml"),
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
        cfg = dataclasses.replace(base, absorber="cpml", snapshot_every=0)
        for phase, incident in [("incident", True), ("pinna", False)]:
            run_one(cfg, f"runs/bs_{key}_cpml_{phase}.npz", incident, save_v=True)
        bare = dataclasses.replace(cfg, filters=(), baffle=False)
        for phase, incident in [("incident", True), ("pinna", False)]:
            run_one(bare, f"runs/bs_{key}_cpml_bare_{phase}.npz", incident, save_v=False)


if __name__ == "__main__":
    main()
