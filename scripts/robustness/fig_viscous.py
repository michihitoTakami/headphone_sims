"""Rigid vs viscous DCA-AMTS comparison (issue #6 / B1). After run_viscous.py.

Compares the dca2 structured pair with and without Maa/Crandall bore losses
against the shared loss-free bare control: C(f) comb (swing / deepest notch),
canal TF, and core post-pinna metrics. Expectation: viscous losses damp the
quarter-wave plug resonances -> milder comb extremes and softened
band-to-place structure; since the model is viscous-only (no thermal loss),
the true part lies between the rigid and viscous curves, closer to viscous.

Outputs: runs/viscous_comparison.png, runs/viscous_summary.json
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK HK"
from common import (
    aperture_for,
    bare_paths,
    comb_stats,
    coupling_db,
    load_result,
    structured_paths,
)

from headphone_sims.analysis.metrics import compute_metrics

VS_PAIR = ("runs/vs_dca2_incident.npz", "runs/vs_dca2_pinna.npz")


def core_metrics(pin_path: str) -> dict:
    res, d = load_result(pin_path)
    return compute_metrics(
        res,
        tuple(d["driver_center"]),
        int(d["reference_index"]),
        band=(5000.0, 10000.0),
        aperture=aperture_for("dca2", d["driver_center"]),
    ).summary()


def main() -> None:
    bare = bare_paths(1, "dca2")
    cases = {
        "rigid": (structured_paths(1, "dca2"), "#7A4B94", "-"),
        "viscous": (VS_PAIR, "#2E7D51", "--"),
    }
    summary: dict = {}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name, (pair, color, ls) in cases.items():
        f, c = coupling_db(pair, bare)
        st = comb_stats(f, c)
        summary[name] = {**st, "metrics": core_metrics(pair[1])}
        ax.plot(f / 1e3, c, color=color, ls=ls, lw=1.7, label=f"DCA {name}")
        m = summary[name]["metrics"]
        print(
            f"{name:8s} swing {st['swing_db']:5.1f} dB  notch {st['notch_db']:6.1f} dB "
            f"@ {st['notch_hz'] / 1e3:5.2f} kHz  core sim {m['similarity_mean']:.3f} "
            f"shapeσ {m['spectral_shape_spread_db_rms']:.2f} dB",
            flush=True,
        )
    summary["delta"] = {
        "swing_db": summary["viscous"]["swing_db"] - summary["rigid"]["swing_db"],
        "notch_db": summary["viscous"]["notch_db"] - summary["rigid"]["notch_db"],
        "shape_spread_db": (
            summary["viscous"]["metrics"]["spectral_shape_spread_db_rms"]
            - summary["rigid"]["metrics"]["spectral_shape_spread_db_rms"]
        ),
    }
    ax.axvspan(5, 10, color="#888", alpha=0.08)
    ax.set_xscale("log")
    ax.set_xticks([2, 4, 8, 16])
    ax.set_xticklabels(["2", "4", "8", "16"])
    ax.set_xlim(2, 16)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlabel("周波数 (kHz)")
    ax.set_ylabel("C(f) (dB)")
    ax.set_title("DCA型(AMTS実測): 剛体 vs 粘性ボア損失(Maa/Crandall)", fontsize=11.5)
    fig.savefig("runs/viscous_comparison.png", dpi=115, bbox_inches="tight")
    with open("runs/viscous_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("saved runs/viscous_comparison.png, runs/viscous_summary.json")


if __name__ == "__main__":
    main()
