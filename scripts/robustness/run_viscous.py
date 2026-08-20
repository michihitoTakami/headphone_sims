"""DCA-AMTS re-evaluation with viscous bore losses (issue #6 / B1).

The rigid lossless model was known to understate AMTS absorption (no
thermoviscous neck losses). This runs the dca2 (real plug map) structured
pair with Maa/Crandall viscous bore losses enabled on BOTH front-stack
filters (magnet slots a=1.25 mm, AMTS tubes a=1.1 mm; sigma ~ 1e3 1/s at the
7 kHz band center). The bare-source control is loss-free by construction
(filters removed), so the existing bare_dca_* pair stays the C(f) reference.

Outputs: runs/vs_dca2_{incident,pinna}.npz
Usage (repo root): uv run python scripts/robustness/run_viscous.py
"""

import dataclasses
import os
import time

import numpy as np

from headphone_sims.experiments.config import load_config
from headphone_sims.geometry.scene import build_scene


def main() -> None:
    base = load_config("configs/hutubs_dca_amts_real.yaml").scene
    filters = tuple(
        dataclasses.replace(f, viscous_losses=True) for f in base.filters
    )
    cfg = dataclasses.replace(base, filters=filters, snapshot_every=0)
    for phase, incident in [("incident", True), ("pinna", False)]:
        out = f"runs/vs_dca2_{phase}.npz"
        if os.path.exists(out):
            print("skip", out, flush=True)
            continue
        t0 = time.time()
        built = build_scene(cfg, incident_only=incident)
        res = built.simulation.run()
        np.savez_compressed(
            out,
            p=res.p.astype(np.float32),
            v=res.v.astype(np.float32),
            positions=res.positions,
            dt=res.dt,
            dx=res.dx,
            driver_center=np.asarray(built.driver_center),
            canal=np.asarray(built.canal_position),
            reference_index=built.reference_index,
        )
        print(f"done {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
