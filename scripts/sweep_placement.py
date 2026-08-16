"""Placement sweep: 3 styles x distances x tilts, single (with-filter) runs."""

import dataclasses
import json
import traceback
from pathlib import Path

from headphone_sims.experiments.config import load_config
from headphone_sims.experiments.runner import run_experiment

STYLES = {
    "40mm": "configs/hutubs_40mm_grille_optimized.yaml",
    "z1r": "configs/hutubs_70mm_z1r_fib.yaml",
    "lcd": "configs/hutubs_90mm_planar_fazor.yaml",
}
DISTANCES = [12e-3, 15e-3, 20e-3, 25e-3]
TILTS = [0.0, -7.5, -15.0]

out_root = Path("runs/sweep_placement")
summary = []
for style, cfg_path in STYLES.items():
    base = load_config(cfg_path)
    for dist in DISTANCES:
        for tilt in TILTS:
            name = f"{style}_d{dist * 1e3:.0f}_t{abs(tilt):.1f}"
            run_dir = out_root / name
            if (run_dir / "metrics.json").exists():
                print(f"skip {name} (exists)")
            else:
                scene = dataclasses.replace(
                    base.scene,
                    distance=dist,
                    driver=dataclasses.replace(base.scene.driver, tilt_deg=tilt),
                    snapshot_every=0,
                )
                config = dataclasses.replace(
                    base, name=name, scene=scene, compare_without_filters=False
                )
                try:
                    run_experiment(config, run_dir=run_dir)
                except Exception:
                    print(f"FAILED {name}")
                    traceback.print_exc()
                    continue
            m = json.loads((run_dir / "metrics.json").read_text())["metrics"]
            summary.append({"style": style, "distance_mm": dist * 1e3, "tilt": tilt, **m})

(out_root / "summary.json").write_text(json.dumps(summary, indent=1))
print(f"done: {len(summary)} runs -> {out_root / 'summary.json'}")
