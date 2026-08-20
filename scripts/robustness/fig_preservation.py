"""Individual-pinna preservation vs the transparent-driver reference
(issue #3 follow-up).

Reference TF_tr: the model's OWN aperture emitting the same waveform from the
same position as an additive monopole sheet — no driver body, no baffle, no
structure (ms_pp{N}_tr_{model}_*). Its pinna TF contains the pinna-only
response that this model's illumination geometry would produce; aperture,
distance, and waveform match the structured run by construction, so nothing
is confounded. Two more rungs decompose where the response is lost:

- P_tot  = corr( TF_struct, TF_tr )   the full product vs pinna-only
- P_body = corr( TF_bare,   TF_tr )   driver-body re-scattering alone
  (bare = same driver as a solid/hard source, no structure)
- P_str  = corr( TF_struct, TF_bare ) front structure alone

Signature versions subtract the per-model panel mean (s_i = TF_i - mean_j
TF_j), keeping only what makes THIS ear different — the common concha
resonance inflates raw correlations but carries no individual information:

- P_sig_tot = corr( s_struct_i, s_tr_i )    headline: does the product
  deliver the individual response the transparent driver would?
- P_sig_str = corr( s_struct_i, s_bare_i )  structure-only signature term

All TFs canal-probe, 1/24-oct-smoothed dB on a common log-f grid. Bands:
4-12.5 kHz (pinna-notch band, primary) and the 5-10 kHz core.

Outputs: runs/preservation.png, runs/preservation_summary.json
Run from the repo root after run_subjects.py. Optional argv: model keys.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK HK"
import numpy as np
from common import bare_paths, canal_tf, smooth, structured_paths, transparent_paths

MODELS = {
    "z1r": ("MDR-Z1R型", "#C05B21"),
    "lcd": ("LCD型", "#46688A"),
    "dx": ("DX10000CL型", "#2E7D51"),
    "dca2": ("DCA型(AMTS)", "#7A4B94"),
}
BANDS = {"pinna": (4000.0, 12500.0), "core": (5000.0, 10000.0)}
N_GRID = 192


def tf_db_on(paths: tuple[str, str], grid: np.ndarray) -> np.ndarray:
    f, h = canal_tf(*paths)
    sel = (f >= grid[0] * 0.9) & (f <= grid[-1] * 1.1)
    sm = smooth(f[sel], h[sel])
    return np.interp(grid, f[sel], 20 * np.log10(sm + 1e-12))


def corr_rows(a: np.ndarray, b: np.ndarray) -> list[float]:
    return [float(np.corrcoef(a[i], b[i])[0, 1]) for i in range(a.shape[0])]


def stats(vals: list[float]) -> dict:
    return {"values": vals, "mean": float(np.mean(vals)), "std": float(np.std(vals))}


def main() -> None:
    models = {k: v for k, v in MODELS.items() if not sys.argv[1:] or k in sys.argv[1:]}
    with open("runs/subjects_selected.json", encoding="utf-8") as fh:
        subjects = json.load(fh)["panel_with_pp1"]

    summary: dict = {"subjects": subjects, "models": {}}
    sig_examples: dict = {}  # model -> (grid, s_tr, s_struct)
    for band_name, (f_lo, f_hi) in BANDS.items():
        grid = np.geomspace(f_lo, f_hi, N_GRID)
        for model in models:
            tt = np.stack([tf_db_on(transparent_paths(s, model), grid) for s in subjects])
            tb = np.stack([tf_db_on(bare_paths(s, model), grid) for s in subjects])
            ts = np.stack([tf_db_on(structured_paths(s, model), grid) for s in subjects])
            s_tr = tt - tt.mean(axis=0)
            s_bare = tb - tb.mean(axis=0)
            s_struct = ts - ts.mean(axis=0)
            entry = {
                "P_tot": stats(corr_rows(ts, tt)),
                "P_body": stats(corr_rows(tb, tt)),
                "P_str": stats(corr_rows(ts, tb)),
                "P_sig_tot": stats(corr_rows(s_struct, s_tr)),
                "P_sig_str": stats(corr_rows(s_struct, s_bare)),
            }
            summary["models"].setdefault(model, {})[band_name] = entry
            if band_name == "pinna":
                sig_examples[model] = (grid, s_tr, s_struct)

    for model in models:
        p = summary["models"][model]["pinna"]
        print(
            f"{model:5s} P_body {p['P_body']['mean']:.3f}  P_str {p['P_str']['mean']:.3f}  "
            f"P_tot {p['P_tot']['mean']:.3f}  sig_str {p['P_sig_str']['mean']:.3f}  "
            f"sig_tot {p['P_sig_tot']['mean']:.3f}±{p['P_sig_tot']['std']:.3f}",
            flush=True,
        )

    ranked = sorted(models, key=lambda m: summary["models"][m]["pinna"]["P_sig_tot"]["mean"])
    lo_m, hi_m = ranked[0], ranked[-1]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    # (a) ladder: mean P_body / P_str / P_tot per model
    ax = axes[0][0]
    comp = [("P_body", "ドライバ実体のみ", 0.5), ("P_str", "構造のみ", 0.75), ("P_tot", "製品全体", 1.0)]
    width = 0.26
    for j, (key, klabel, shade) in enumerate(comp):
        for i, (model, (label, color)) in enumerate(models.items()):
            p = summary["models"][model]["pinna"][key]
            ax.bar(i + (j - 1) * width, p["mean"], width, yerr=p["std"], capsize=3,
                   color=color, alpha=shade, edgecolor="none",
                   label=klabel if i == 0 else None)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([v[0] for v in models.values()], fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylabel("相関係数(平均±σ)")
    ax.set_title("保存の分解(基準=透明ドライバ、開口・距離一致)", fontsize=11)
    ax.legend(fontsize=8)
    # (b) headline: signature preservation vs the transparent reference
    ax = axes[0][1]
    for i, (model, (label, color)) in enumerate(models.items()):
        vals = np.array(summary["models"][model]["pinna"]["P_sig_tot"]["values"])
        ax.scatter([i] * len(vals), vals, color=color, s=30, alpha=0.85, zorder=3)
        ax.errorbar(i, vals.mean(), yerr=vals.std(), color=color, fmt="_",
                    ms=26, capsize=6, lw=2, zorder=2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([v[0] for v in models.values()], fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylabel("P_sig_tot")
    ax.set_title("個人署名の保存 vs 透明ドライバ(点=被験者)", fontsize=11)
    # (c)(d) example signatures: transparent (dashed) vs structured (solid)
    for ax, model, tag in [(axes[1][0], hi_m, "最良"), (axes[1][1], lo_m, "最悪")]:
        grid, s_tr, s_struct = sig_examples[model]
        shades = plt.cm.cividis(np.linspace(0.05, 0.9, len(subjects)))
        for i, s in enumerate(subjects):
            ax.plot(grid / 1e3, s_tr[i], color=shades[i], lw=1.0, ls="--", alpha=0.7)
            ax.plot(grid / 1e3, s_struct[i], color=shades[i], lw=1.4,
                    label=f"pp{s}" if model == hi_m else None)
        ax.set_xscale("log")
        ax.set_xticks([4, 6, 8, 10, 12])
        ax.set_xticklabels(["4", "6", "8", "10", "12"])
        ax.grid(alpha=0.3)
        ax.set_xlabel("周波数 (kHz)")
        ax.set_ylabel("個人署名 (dB, パネル平均比)")
        p = summary["models"][model]["pinna"]["P_sig_tot"]
        ax.set_title(
            f"{tag}: {models[model][0]}(破線=透明基準の署名、実線=製品、P_sig_tot {p['mean']:.2f})",
            fontsize=11,
        )
    axes[1][0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "個人ピンナ応答の保存 — 透明ドライバ(同一開口・実体なし)基準(外耳道、8被験者)",
        fontsize=13,
    )
    fig.savefig("runs/preservation.png", dpi=115, bbox_inches="tight")
    with open("runs/preservation_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("saved runs/preservation.png, runs/preservation_summary.json")


if __name__ == "__main__":
    main()
