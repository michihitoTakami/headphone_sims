"""Grid-convergence analysis (issue #6, task 3). Run after run_convergence.py.

Per model (z1r reference, dca2 = finest geometry) across dx 0.5/0.4/0.3 mm:

(a) canal C(f) comb: swing / deepest notch / notch frequency per dx,
(b) 8k / 10k illumination-map NN correlation between consecutive dx,
(c) core post-pinna metrics incl. the geometry-free spectral-shape spread,
(d) voxelized porosity per dx (from run_convergence.py).

Convergence read-out: the 0.4->0.3 change of each indicator should be
smaller than the 0.5->0.4 change, and the headline ordering (DCA's
spectral-shape spread >> the reference's) must hold at every dx — otherwise
the published spatial-spectral claims are resolution artifacts.

Outputs: runs/convergence.png, runs/convergence_summary.json
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK HK"
import numpy as np
from common import (
    aperture_for,
    band_map,
    bare_paths,
    comb_stats,
    coupling_db,
    load_result,
    map_corr,
    structured_paths,
)

from headphone_sims.analysis.metrics import compute_metrics

MODELS = {"z1r": ("MDR-Z1R型", "#C05B21"), "dca2": ("DCA型(AMTS実測)", "#7A4B94")}
DX_UM = [500, 400, 300]
METRIC_KEYS = [
    "similarity_mean",
    "level_spread_db",
    "arrival_spread_ms",
    "spectral_shape_spread_db_rms",
]


def conv_paths(model: str, dx_um: int, structured: bool) -> tuple[str, str]:
    if dx_um == 500:
        return structured_paths(1, model) if structured else bare_paths(1, model)
    mid = "" if structured else "bare_"
    return (
        f"runs/conv_{model}_dx{dx_um}_{mid}incident.npz",
        f"runs/conv_{model}_dx{dx_um}_{mid}pinna.npz",
    )


def core_metrics(pin_path: str, model: str) -> dict:
    res, d = load_result(pin_path)
    return compute_metrics(
        res,
        tuple(d["driver_center"]),
        int(d["reference_index"]),
        band=(5000.0, 10000.0),
        aperture=aperture_for(model, d["driver_center"]),
    ).summary()


def main() -> None:
    porosity = {}
    if os.path.exists("runs/convergence_porosity.json"):
        with open("runs/convergence_porosity.json", encoding="utf-8") as fh:
            porosity = json.load(fh)

    summary: dict = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (model, (label, color)) in zip(axes, MODELS.items(), strict=True):
        entry: dict = {"dx": {}}
        combs, maps = {}, {}
        styles = {500: ("-", 1.8), 400: ("--", 1.3), 300: (":", 1.3)}
        for um in DX_UM:
            sp = conv_paths(model, um, True)
            bp = conv_paths(model, um, False)
            f, c = coupling_db(sp, bp)
            combs[um] = (f, c)
            maps[um] = {
                fk: band_map(sp[0], fk, model) for fk in (8000.0, 10000.0)
            }
            entry["dx"][um] = {
                **comb_stats(f, c),
                "metrics": core_metrics(sp[1], model),
                "porosity": porosity.get(f"{model}_dx{um}"),
            }
            ls, lw = styles[um]
            ax.plot(f / 1e3, c, ls=ls, lw=lw, color=color, label=f"dx {um / 1000:.1f} mm")
        # map correlations between consecutive resolutions
        for a, b in ((500, 400), (400, 300)):
            for fk in (8000.0, 10000.0):
                xy_a, l_a = maps[a][fk]
                xy_b, l_b = maps[b][fk]
                entry[f"map{int(fk / 1000)}k_corr_{a}_{b}"] = map_corr(xy_a, l_a, xy_b, l_b)
        # convergence ratios: |change 0.4->0.3| / |change 0.5->0.4|
        entry["deltas"] = {}
        for key in METRIC_KEYS:
            v = [entry["dx"][um]["metrics"][key] for um in DX_UM]
            d1, d2 = abs(v[1] - v[0]), abs(v[2] - v[1])
            entry["deltas"][key] = {
                "d_500_400": float(v[1] - v[0]),
                "d_400_300": float(v[2] - v[1]),
                "contracting": bool(d2 <= d1) if d1 > 0 else True,
            }
        for key in ("swing_db", "notch_db"):
            v = [entry["dx"][um][key] for um in DX_UM]
            entry["deltas"][key] = {
                "d_500_400": float(v[1] - v[0]),
                "d_400_300": float(v[2] - v[1]),
                "contracting": bool(abs(v[2] - v[1]) <= abs(v[1] - v[0]))
                if abs(v[1] - v[0]) > 0
                else True,
            }
        v = [entry["dx"][um]["notch_hz"] for um in DX_UM]
        entry["deltas"]["notch_shift_oct"] = {
            "d_500_400": float(np.log2(v[1] / v[0])),
            "d_400_300": float(np.log2(v[2] / v[1])),
        }
        summary[model] = entry
        ax.axvspan(5, 10, color="#888", alpha=0.08)
        ax.set_xscale("log")
        ax.set_xticks([2, 4, 8, 16])
        ax.set_xticklabels(["2", "4", "8", "16"])
        ax.set_xlim(2, 16)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)
        ax.set_xlabel("周波数 (kHz)")
        ax.set_ylabel("C(f) (dB)")
        ax.set_title(f"{label}: 格子解像度とC(f)", fontsize=11)

        print(f"== {model}")
        for um in DX_UM:
            e = entry["dx"][um]
            m = e["metrics"]
            print(
                f"  dx{um} swing {e['swing_db']:5.1f} dB notch {e['notch_db']:6.1f} dB "
                f"@ {e['notch_hz'] / 1e3:5.2f} kHz  core sim {m['similarity_mean']:.3f} "
                f"shapeσ {m['spectral_shape_spread_db_rms']:.2f} dB",
                flush=True,
            )
        for a, b in ((500, 400), (400, 300)):
            print(
                f"  map corr {a}->{b}: 8k {entry[f'map8k_corr_{a}_{b}']:.3f} "
                f"10k {entry[f'map10k_corr_{a}_{b}']:.3f}"
            )

    # headline ordering check: DCA's shape spread stays well above the reference
    shape = {
        m: [summary[m]["dx"][um]["metrics"]["spectral_shape_spread_db_rms"] for um in DX_UM]
        for m in MODELS
    }
    summary["ordering_dca_gt_ref_all_dx"] = bool(
        all(d > z for d, z in zip(shape["dca2"], shape["z1r"], strict=True))
    )
    print(f"DCA shape-σ > Z1R at every dx: {summary['ordering_dca_gt_ref_all_dx']}")

    fig.suptitle("格子収束: dx = 0.5 / 0.4 / 0.3 mm (pp1)", fontsize=12.5)
    fig.savefig("runs/convergence.png", dpi=115, bbox_inches="tight")
    with open("runs/convergence_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("saved runs/convergence.png, runs/convergence_summary.json")


if __name__ == "__main__":
    main()
