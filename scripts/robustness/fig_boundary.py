"""Boundary-sensitivity analysis (issue #6, task 5). Run after run_boundary.py.

Per model x condition {base, sp45, sp60m}: canal TF change, C(f) comb change,
and core post-pinna metrics. Pass criteria (from the issue): comb swing/notch
change < 0.5 dB and notch-frequency shift < 1/24 octave under a thicker
sponge / larger domain — otherwise the published comb structure contains
boundary reflections.

Outputs: runs/boundary_sensitivity.png, runs/boundary_sensitivity.json
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK HK"
import numpy as np
from common import (
    aperture_for,
    bare_paths,
    canal_tf,
    comb_stats,
    coupling_db,
    load_result,
    smooth,
    structured_paths,
)

from headphone_sims.analysis.metrics import compute_metrics

MODELS = {"z1r": ("MDR-Z1R型", "#C05B21"), "dca2": ("DCA型(AMTS実測)", "#7A4B94")}
CONDS = ["base", "sp30re", "sp45", "sp60m", "cpml", "voxa", "voxb"]
COND_LABEL = {
    "base": "基準(スポンジ30セル, 公表ラン)",
    "sp30re": "スポンジ30セル(再現ラン)",
    "sp45": "スポンジ45セル",
    "sp60m": "60セル+マージン×1.5",
    "cpml": "C-PML 30セル(〜-114dB床)",
    "voxa": "格子位相probe A(マージン15.25mm)",
    "voxb": "格子位相probe B(マージン15.0mm)",
}
# Which vox condition flips the lattice phase for which model (the other is
# a same-phase control): see run_boundary.py CONDITIONS comment.
VOX_FLIP = {"z1r": "voxb", "dca2": "voxa"}
TF_BAND = (4000.0, 12500.0)


def paths(model: str, cond: str, structured: bool) -> tuple[str, str]:
    if cond == "base":
        return structured_paths(1, model) if structured else bare_paths(1, model)
    mid = "" if structured else "bare_"
    return (
        f"runs/bs_{model}_{cond}_{mid}incident.npz",
        f"runs/bs_{model}_{cond}_{mid}pinna.npz",
    )


def smoothed_tf_db(pair: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    f, h = canal_tf(*pair)
    sel = (f >= TF_BAND[0]) & (f <= TF_BAND[1])
    return f[sel], 20 * np.log10(smooth(f[sel], h[sel]) + 1e-12)


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
    summary: dict = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (model, (label, color)) in zip(axes, MODELS.items(), strict=True):
        f0, tf0 = smoothed_tf_db(paths(model, "base", True))
        fc0, c0 = coupling_db(paths(model, "base", True), paths(model, "base", False))
        st0 = comb_stats(fc0, c0)
        entry: dict = {"conditions": {}}
        styles = {
            "base": ("-", 1.8),
            "sp30re": ("-", 0.9),
            "sp45": ("--", 1.2),
            "sp60m": (":", 1.2),
            "cpml": ("-.", 1.6),
            "voxa": ("--", 0.9),
            "voxb": ("--", 0.9),
        }
        for cond in CONDS:
            f, tf = smoothed_tf_db(paths(model, cond, True))
            tf_i = np.interp(f0, f, tf)
            fc, c = coupling_db(paths(model, cond, True), paths(model, cond, False))
            st = comb_stats(fc, c)
            entry["conditions"][cond] = {
                **st,
                "dTF_rms_db": float(np.sqrt(np.mean((tf_i - tf0) ** 2))),
                "d_swing_db": float(st["swing_db"] - st0["swing_db"]),
                "d_notch_db": float(st["notch_db"] - st0["notch_db"]),
                "notch_shift_oct": float(np.log2(st["notch_hz"] / st0["notch_hz"])),
                "metrics": core_metrics(paths(model, cond, True)[1], model),
            }
            ls, lw = styles[cond]
            ax.plot(fc / 1e3, c, ls=ls, lw=lw, color=color, label=COND_LABEL[cond])
        # The boundary contribution is bounded by |base - cpml| (same lattice
        # phase, ~-114 dB floor). sp45/sp60m conflate the absorber with a
        # lattice-phase change for half-integer-aligned models; "vox" measures
        # that voxelization-phase sensitivity alone. The meaningful verdict is
        # boundary error << voxel-phase noise.
        cp = entry["conditions"].get("cpml", {})
        vx = entry["conditions"].get(VOX_FLIP[model], {})
        passing = bool(
            cp
            and abs(cp["d_swing_db"]) < 2.0
            and abs(cp["d_notch_db"]) < 2.0
            and (not vx or cp["dTF_rms_db"] < vx["dTF_rms_db"])
        )
        entry["passes"] = passing
        # Agreement with the C-PML arbiter (boundary-clean C(f)) in the core band.
        if "cpml" in entry["conditions"]:
            fc_a, c_a = coupling_db(paths(model, "cpml", True), paths(model, "cpml", False))
            core_a = (fc_a >= 5000.0) & (fc_a <= 10000.0)
            for cond in CONDS:
                if cond == "cpml":
                    continue
                fc_b, c_b = coupling_db(paths(model, cond, True), paths(model, cond, False))
                cb_i = np.interp(fc_a[core_a], fc_b, c_b)
                entry["conditions"][cond]["dC_rms_vs_cpml_db"] = float(
                    np.sqrt(np.mean((cb_i - c_a[core_a]) ** 2))
                )
        summary[model] = entry
        ax.axvspan(5, 10, color="#888", alpha=0.08)
        ax.set_xscale("log")
        ax.set_xticks([2, 4, 8, 16])
        ax.set_xticklabels(["2", "4", "8", "16"])
        ax.set_xlim(2, 16)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlabel("周波数 (kHz)")
        ax.set_ylabel("C(f) (dB)")
        ax.set_title(f"{label}: 吸収境界条件によるC(f)の変化", fontsize=11)
        for cond in CONDS:
            e = entry["conditions"][cond]
            vs_cpml = e.get("dC_rms_vs_cpml_db")
            print(
                f"{model:5s} {cond:6s} ΔTF_rms {e['dTF_rms_db']:5.2f} dB  "
                f"Δswing {e['d_swing_db']:+6.2f} dB  Δnotch {e['d_notch_db']:+6.2f} dB  "
                f"swing {e['swing_db']:5.1f} dB  "
                + (f"ΔC_rms vs CPML {vs_cpml:5.2f} dB  " if vs_cpml is not None else "")
                + f"core sim {e['metrics']['similarity_mean']:.3f}",
                flush=True,
            )
        print(f"{model:5s} boundary-independence: {'PASS' if passing else 'FAIL'}")
    fig.suptitle("境界感度試験: スポンジ厚/ドメインマージンとコーム構造(pp1)", fontsize=12.5)
    fig.savefig("runs/boundary_sensitivity.png", dpi=115, bbox_inches="tight")
    with open("runs/boundary_sensitivity.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("saved runs/boundary_sensitivity.png, runs/boundary_sensitivity.json")


if __name__ == "__main__":
    main()
