"""Multi-subject runs for the individual-robustness study (issue #3, step 3).

For each subject in the panel and each of the 4 models (Z1R / LCD / DX /
DCA-AMTS real plug map), runs the same 4 simulations as the pp1 report set:

- structured incident / pinna  (runs/ms_pp{N}_{model}_{phase}.npz, with v)
- bare-source incident / pinna (runs/ms_pp{N}_bare_{model}_{phase}.npz)

Additionally runs a TRANSPARENT-DRIVER reference pair per subject x model
(runs/ms_pp{N}_tr_{model}_{phase}.npz): the model's own aperture emits the
same waveform from the same position, but as an additive monopole sheet with
no driver body, no baffle, no structure — the pinna reflection passes back
through freely. Its pinna TF contains the PINNA-ONLY notches that this
model's illumination geometry would produce; the preservation analyses
(fig_preservation.py, fig_notch_preservation.py) measure how much of that
the real product delivers.

pp1's structured/bare runs are NOT run here — batch3_* / bare_* outputs are
reused by the analysis (the transparent pair IS run for pp1). Skips any
existing output; delete the .npz to force a re-run.

Usage (repo root):
    uv run python scripts/robustness/run_subjects.py 2 10 58 ...
    uv run python scripts/robustness/run_subjects.py   # panel from runs/subjects_selected.json
"""

import dataclasses
import json
import os
import sys
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
        driver_center=np.asarray(built.driver_center),
        canal=np.asarray(built.canal_position),
        reference_index=built.reference_index,
    )
    if save_v:
        arrays["v"] = res.v.astype(np.float32)
    np.savez_compressed(out, **arrays)
    print(f"done {out} ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    if len(sys.argv) > 1:
        subjects = [int(a) for a in sys.argv[1:]]
    else:
        with open("runs/subjects_selected.json", encoding="utf-8") as fh:
            subjects = json.load(fh)["subjects"]
    print("subjects:", subjects, flush=True)
    for subject in [1, *subjects]:
        mesh_path = f"data/hutubs/pp{subject}_3DheadMesh.ply"
        for key, cfg_path in MODELS:
            base = load_config(cfg_path).scene
            tr = dataclasses.replace(
                base,
                pinna=dataclasses.replace(base.pinna, mesh_path=mesh_path),
                filters=(),
                baffle=False,
                transparent_driver=True,
                snapshot_every=0,
            )
            for phase, incident in [("incident", True), ("pinna", False)]:
                run_one(
                    tr, f"runs/ms_pp{subject}_tr_{key}_{phase}.npz", incident, save_v=False
                )
    for subject in subjects:
        mesh_path = f"data/hutubs/pp{subject}_3DheadMesh.ply"
        for key, cfg_path in MODELS:
            base = load_config(cfg_path).scene
            base = dataclasses.replace(
                base,
                pinna=dataclasses.replace(base.pinna, mesh_path=mesh_path),
                snapshot_every=0,
            )
            for phase, incident in [("incident", True), ("pinna", False)]:
                run_one(
                    base, f"runs/ms_pp{subject}_{key}_{phase}.npz", incident, save_v=True
                )
            bare = dataclasses.replace(base, filters=(), baffle=False)
            for phase, incident in [("incident", True), ("pinna", False)]:
                run_one(
                    bare,
                    f"runs/ms_pp{subject}_bare_{key}_{phase}.npz",
                    incident,
                    save_v=False,
                )


if __name__ == "__main__":
    main()
