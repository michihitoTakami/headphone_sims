"""Re-seat sensitivity analysis, 4 models on pp1 (issue #3, auxiliary).

Per model x condition {base, ax-1mm, ax+2mm, vy-2mm, vy+2mm}:
- C(f) comb: deepest-notch frequency shift and RMS spectral change vs base
  in the 5-10 kHz core band -> df/dmm per axis (axial predicts ~200-450 Hz/mm
  from round-trip path geometry; AMTS map translation adds ~130-200 Hz/mm
  along y).
- post-pinna core metrics (similarity / level spread / arrival) per condition
  -> per-model variation score (hypothesis: DCA largest, comb shift + AMTS
  transmission-map translation compounding).
- incident 10 kHz illumination-map correlation base<->shifted (probes matched
  canal-relative): AMTS map translation shows up as decorrelation under vy.

Outputs: runs/reseat.png, runs/reseat_summary.json
Run from the repo root after run_reseat.py.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK HK"
import numpy as np
from common import (
    SPATIAL_CORE,
    aperture_for,
    band_map,
    comb_stats,
    coupling_db,
    load_result,
    map_corr,
    reseat_paths,
)

from headphone_sims.analysis.metrics import compute_metrics

MODELS = {
    "z1r": ("MDR-Z1R型", "#C05B21"),
    "lcd": ("LCD型", "#46688A"),
    "dx": ("DX10000CL型", "#2E7D51"),
    "dca2": ("DCA型(AMTS実測)", "#7A4B94"),
}
CONDS = ["base", "axm1", "axp2", "vym2", "vyp2"]
COND_MM = {"base": 0.0, "axm1": -1.0, "axp2": +2.0, "vym2": -2.0, "vyp2": +2.0}
COND_LABEL = {
    "base": "基準", "axm1": "軸 -1mm", "axp2": "軸 +2mm",
    "vym2": "上下 -2mm", "vyp2": "上下 +2mm",
}


def core_metrics(pin_path: str, model: str) -> dict:
    res, d = load_result(pin_path)
    return compute_metrics(
        res, tuple(d["driver_center"]), int(d["reference_index"]),
        band=(5000.0, 10000.0),
        aperture=aperture_for(model, d["driver_center"]),
    ).summary()


def comb_shift_hz(f0, c0, f1, c1, band=SPATIAL_CORE, max_shift_oct=0.35) -> float:
    """Whole-comb frequency shift (Hz at the band center) via log-f xcorr.

    Deepest-notch tracking hops between comb teeth under mm-scale re-seat, so
    the translation of the whole C(f) pattern is estimated instead: both
    curves are resampled on a uniform log2-frequency grid and the lag
    maximizing their correlation gives the shift in octaves.
    """
    n = 512
    lg = np.linspace(np.log2(band[0]), np.log2(band[1]), n)
    a = np.interp(lg, np.log2(f0), c0)
    b = np.interp(lg, np.log2(f1), c1)
    a -= a.mean()
    b -= b.mean()
    step = lg[1] - lg[0]
    max_lag = int(max_shift_oct / step)
    best_lag, best = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[: n - lag], b[lag:]
        else:
            x, y = a[-lag:], b[: n + lag]
        r = float(np.corrcoef(x, y)[0, 1])
        if r > best:
            best, best_lag = r, lag
    f_center = float(np.sqrt(band[0] * band[1]))
    return f_center * (2.0 ** (best_lag * step) - 1.0)


def main() -> None:
    summary: dict = {}
    for model, (label, color) in MODELS.items():
        combs, mets = {}, {}
        for cond in CONDS:
            sp = reseat_paths(model, cond, structured=True)
            bp = reseat_paths(model, cond, structured=False)
            freqs, c = coupling_db(sp, bp)
            combs[cond] = (freqs, c)
            mets[cond] = core_metrics(sp[1], model)
        f0, c0 = combs["base"]
        core = (f0 >= SPATIAL_CORE[0]) & (f0 <= SPATIAL_CORE[1])
        entry: dict = {"conditions": {}}
        for cond in CONDS:
            f, c = combs[cond]
            ci = np.interp(f0, f, c)
            st = comb_stats(f, c)
            st["dC_rms_db"] = float(np.sqrt(np.mean((ci[core] - c0[core]) ** 2)))
            st["notch_shift_hz"] = st["notch_hz"] - comb_stats(f0, c0)["notch_hz"]
            # whole-comb translation (deepest-notch tracking tooth-hops)
            st["comb_shift_hz"] = 0.0 if cond == "base" else comb_shift_hz(f0, c0, f, c)
            st["metrics"] = mets[cond]
            entry["conditions"][cond] = st
        # df/dmm per axis from the whole-comb shift (through-origin fit)
        ax_pts = [("axm1", -1.0), ("axp2", 2.0)]
        vy_pts = [("vym2", -2.0), ("vyp2", 2.0)]
        for name, pts in (("axial", ax_pts), ("vertical", vy_pts)):
            mm = np.array([m for _, m in pts])
            sh = np.array([entry["conditions"][c]["comb_shift_hz"] for c, _ in pts])
            entry[f"{name}_shift_hz_per_mm"] = float((sh @ mm) / (mm @ mm))
            dc = np.array([entry["conditions"][c]["dC_rms_db"] for c, _ in pts])
            entry[f"{name}_dC_rms_db_per_mm"] = float(np.mean(dc / np.abs(mm)))
        # per-model variation score: worst-case core-similarity drop + RMS comb change
        sims = np.array([mets[c]["similarity_mean"] for c in CONDS])
        entry["similarity_range"] = float(sims.max() - sims.min())
        entry["dC_rms_db_max"] = float(
            max(entry["conditions"][c]["dC_rms_db"] for c in CONDS if c != "base")
        )
        # illumination-map translation (10k): correlation base vs vy shifts
        xy0, l0 = band_map(reseat_paths(model, "base", True)[0], 10000.0, model)
        for cond in ("vym2", "vyp2", "axp2"):
            xy1, l1 = band_map(reseat_paths(model, cond, True)[0], 10000.0, model)
            entry["conditions"][cond]["map10k_corr"] = map_corr(xy0, l0, xy1, l1)
        summary[model] = entry
        print(
            f"{model:5s} comb df/dmm ax {entry['axial_shift_hz_per_mm']:+.0f} "
            f"vy {entry['vertical_shift_hz_per_mm']:+.0f} Hz/mm  "
            f"ΔC_rms max {entry['dC_rms_db_max']:.2f} dB  "
            f"sim range {entry['similarity_range']:.3f}",
            flush=True,
        )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    # (a) C(f) overlay for the most-shifted condition per model
    ax = axes[0][0]
    for model, (label, color) in MODELS.items():
        sp = reseat_paths(model, "base", True)
        bp = reseat_paths(model, "base", False)
        f, c = coupling_db(sp, bp)
        ax.plot(f / 1e3, c, color=color, lw=1.6, label=label)
        f2, c2 = coupling_db(
            reseat_paths(model, "axp2", True), reseat_paths(model, "axp2", False)
        )
        ax.plot(f2 / 1e3, c2, color=color, lw=1.1, ls="--", alpha=0.65)
    ax.axvspan(5, 10, color="#888", alpha=0.08)
    ax.set_xscale("log"); ax.set_xticks([2, 4, 8, 16]); ax.set_xticklabels(["2", "4", "8", "16"])
    ax.set_xlim(2, 16); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xlabel("周波数 (kHz)"); ax.set_ylabel("C(f) (dB)")
    ax.set_title("C(f): 実線=基準、破線=軸+2mm", fontsize=11)
    # (b) whole-comb translation per condition
    ax = axes[0][1]
    for model, (label, color) in MODELS.items():
        y = [summary[model]["conditions"][c]["comb_shift_hz"] for c in CONDS]
        ax.plot(range(len(CONDS)), y, "o-", color=color, lw=1.6, label=label)
    ax.axhline(0, color="#999", lw=0.7)
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([COND_LABEL[c] for c in CONDS], fontsize=8.5)
    ax.grid(alpha=0.3); ax.set_ylabel("コーム全体の移動 (Hz, log-f相互相関)")
    ax.set_title("コームパターンの平行移動(装着ズレ)", fontsize=11)
    # (c) RMS comb change per condition
    ax = axes[1][0]
    width = 0.19
    conds = [c for c in CONDS if c != "base"]
    for i, (model, (label, color)) in enumerate(MODELS.items()):
        y = [summary[model]["conditions"][c]["dC_rms_db"] for c in conds]
        ax.bar(np.arange(len(conds)) + (i - 1.5) * width, y, width, color=color, label=label)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([COND_LABEL[c] for c in conds], fontsize=8.5)
    ax.grid(alpha=0.3, axis="y"); ax.set_ylabel("ΔC RMS (dB, 5-10 kHz)")
    ax.set_title("コームのスペクトル変化量", fontsize=11)
    # (d) core similarity per condition
    ax = axes[1][1]
    for model, (label, color) in MODELS.items():
        y = [summary[model]["conditions"][c]["metrics"]["similarity_mean"] for c in CONDS]
        ax.plot(range(len(CONDS)), y, "o-", color=color, lw=1.6, label=label)
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([COND_LABEL[c] for c in CONDS], fontsize=8.5)
    ax.grid(alpha=0.3); ax.set_ylabel("核心帯 波形一致(振幅込み)")
    ax.set_title("ポスト・ピンナ指標の変動", fontsize=11)
    fig.suptitle("再装着シミュレーション: 駆動距離±/上下±の摂動(pp1、4モデル)", fontsize=13)
    fig.savefig("runs/reseat.png", dpi=115, bbox_inches="tight")
    with open("runs/reseat_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("saved runs/reseat.png, runs/reseat_summary.json")


if __name__ == "__main__":
    main()
