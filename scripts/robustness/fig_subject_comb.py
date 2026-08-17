"""Inter-subject coupling-comb variance, 4 models (issue #3, step 4a/4b).

Per subject x model: C(f) (report protocol), comb swing / deepest notch in
the 5-10 kHz core band, and the collision test between the hardware comb
notches and the subject's own pinna notches (from the bare-source pinna TF):
a hardware notch within 1/12 octave of a pinna notch reinforces/erases an
individual cue ("collision"); one falling in a gap imprints a false cue.

Reads runs/subjects_selected.json (panel_with_pp1). Outputs:
- runs/subject_comb.png            C(f) overlays per model + notch scatter
- runs/subject_comb_summary.json   per-model per-subject stats + variance

Run from the repo root after run_subjects.py.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK HK"
import numpy as np
from common import (
    SPATIAL_CORE,
    bare_paths,
    canal_tf,
    comb_stats,
    coupling_db,
    notch_freqs,
    octave_distance,
    smooth,
    structured_paths,
)

MODELS = {
    "z1r": ("MDR-Z1R型(開口71%)", "#C05B21"),
    "lcd": ("LCD型(開口32%)", "#46688A"),
    "dx": ("DX10000CL型(開口52%)", "#2E7D51"),
    "dca2": ("DCA型(AMTS実測)", "#7A4B94"),
}
COLLISION_OCT = 1 / 12  # ~ +/-6% in frequency


def main() -> None:
    with open("runs/subjects_selected.json", encoding="utf-8") as fh:
        subjects = json.load(fh)["panel_with_pp1"]

    summary: dict = {}
    fig = plt.figure(figsize=(13.5, 12.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.85], hspace=0.42, wspace=0.22)
    model_axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(len(MODELS))]
    ax_notch = fig.add_subplot(gs[2, :])
    shades = plt.cm.cividis(np.linspace(0.05, 0.9, len(subjects)))

    for mi, (model, (label, color)) in enumerate(MODELS.items()):
        ax = model_axes[mi]
        rows = []
        for si, subject in enumerate(subjects):
            freqs, c = coupling_db(
                structured_paths(subject, model), bare_paths(subject, model)
            )
            st = comb_stats(freqs, c)
            hw_notches = notch_freqs(freqs, c, SPATIAL_CORE)
            # subject's own pinna notches: bare-source pinna TF, 4-12.5 kHz
            fb, hb = canal_tf(*bare_paths(subject, model))
            selb = (fb >= 1000) & (fb <= 16000)
            tf_db = 20 * np.log10(smooth(fb[selb], hb[selb]) + 1e-12)
            pinna_notches = notch_freqs(fb[selb], tf_db, (4000.0, 12500.0))
            dists = [octave_distance(f, pinna_notches) for f in hw_notches]
            st.update(
                subject=subject,
                hw_notches_hz=[float(f) for f in hw_notches],
                pinna_notches_hz=[float(f) for f in pinna_notches],
                n_collisions=int(sum(d <= COLLISION_OCT for d in dists)),
                n_false_cues=int(sum(d > COLLISION_OCT for d in dists)),
            )
            rows.append(st)
            ax.plot(freqs / 1e3, c, color=shades[si], lw=1.2,
                    label=f"pp{subject}" if mi == 0 else None)
            ax_notch.scatter(
                [mi + (si - len(subjects) / 2) * 0.055] * len(hw_notches),
                np.asarray(hw_notches) / 1e3,
                color=color, s=26, alpha=0.85, edgecolors="none",
            )
        swings = np.array([r["swing_db"] for r in rows])
        notches = np.array([r["notch_hz"] for r in rows])
        depths = np.array([r["notch_db"] for r in rows])
        summary[model] = {
            "per_subject": rows,
            "swing_mean_db": float(swings.mean()),
            "swing_std_db": float(swings.std()),
            "notch_hz_min": float(notches.min()),
            "notch_hz_max": float(notches.max()),
            "notch_depth_mean_db": float(depths.mean()),
            "collisions_total": int(sum(r["n_collisions"] for r in rows)),
            "false_cues_total": int(sum(r["n_false_cues"] for r in rows)),
        }
        ax.axvspan(5, 10, color="#888", alpha=0.08)
        ax.axhline(0, color="#999", lw=0.7)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xticklabels(["1", "2", "4", "8", "16"])
        ax.grid(alpha=0.3)
        ax.set_xlabel("周波数 (kHz)")
        ax.set_ylabel("C(f) (dB)")
        ax.set_title(
            f"{label}: 振れ幅 {swings.mean():.1f}±{swings.std():.1f} dB", fontsize=11
        )
        print(
            f"{model:4s} swing {swings.mean():.1f}±{swings.std():.1f} dB  "
            f"notch {notches.min()/1e3:.1f}-{notches.max()/1e3:.1f} kHz  "
            f"collisions {summary[model]['collisions_total']}"
            f"/{summary[model]['collisions_total'] + summary[model]['false_cues_total']}"
        )

    # Ear-derived comb-depth ranking: per-subject swing across models, with
    # pinna shape features and the intrinsic (transparent-TF) main-notch depth.
    from common import transparent_paths
    from scipy.signal import find_peaks
    from scipy.stats import pearsonr, spearmanr

    verify = {
        r["subject"]: r
        for r in json.load(open("runs/subjects_verify.json"))["results"]
    }
    grid = np.geomspace(4000, 12500, 384)

    def tf_db(paths):
        f, h = canal_tf(*paths)
        sel = (f >= 3600) & (f <= 13800)
        return np.interp(grid, f[sel], 20 * np.log10(smooth(f[sel], h[sel]) + 1e-12))

    ranking = []
    for s in subjects:
        rows = [
            next(x for x in summary[m]["per_subject"] if x["subject"] == s)
            for m in MODELS
        ]
        proms = []
        for m in MODELS:
            tf = tf_db(transparent_paths(s, m))
            idx, pr = find_peaks(-tf, prominence=2.0)
            proms.append(float(max(pr["prominences"])) if len(idx) else 0.0)
        ranking.append({
            "subject": s,
            "swing_mean_db": float(np.mean([r["swing_db"] for r in rows])),
            "swing_max_db": float(np.max([r["swing_db"] for r in rows])),
            "notch_min_db": float(np.min([r["notch_db"] for r in rows])),
            "own_notch_db": float(np.mean(proms)),
            "recession_mm": verify[s]["recession_mm"],
            "concha_depth_mm": verify[s]["concha_depth_mm"],
        })
    ranking.sort(key=lambda r: -r["swing_mean_db"])
    x_sw = [r["swing_mean_db"] for r in ranking]
    corrs = {}
    for feat in ("recession_mm", "concha_depth_mm", "own_notch_db"):
        xf = [r[feat] for r in ranking]
        corrs[feat] = {
            "pearson": float(pearsonr(xf, x_sw)[0]),
            "spearman": float(spearmanr(xf, x_sw)[0]),
        }
    summary["ear_ranking"] = {"ranking": ranking, "swing_corr": corrs}
    for r in ranking:
        print(
            f"ear pp{r['subject']:<3d} swing {r['swing_mean_db']:.1f}dB "
            f"(max {r['swing_max_db']:.1f}) concha {r['concha_depth_mm']:.1f}mm"
        )
    print("swing corr:", {k: round(v["spearman"], 2) for k, v in corrs.items()})

    ax_notch.set_xticks(range(len(MODELS)))
    ax_notch.set_xticklabels([v[0].split("(")[0] for v in MODELS.values()], fontsize=9)
    ax_notch.set_ylabel("ハードウェアノッチ周波数 (kHz)")
    ax_notch.set_title("コームノッチの被験者間ばらつき(5-10 kHz核心帯)", fontsize=11)
    ax_notch.grid(alpha=0.3, axis="y")
    model_axes[0].legend(fontsize=7.5, ncol=2)
    fig.suptitle(
        f"カップリングコーム C(f) の個人間分散({len(subjects)}被験者、外耳道プローブ)",
        fontsize=13,
    )
    fig.savefig("runs/subject_comb.png", dpi=115, bbox_inches="tight")
    with open("runs/subject_comb_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("saved runs/subject_comb.png, runs/subject_comb_summary.json")


if __name__ == "__main__":
    main()
