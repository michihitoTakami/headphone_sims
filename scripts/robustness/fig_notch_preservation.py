"""Notch-level preservation vs the transparent-driver reference.

Correlation (fig_preservation.py) scores the whole curve; spatial cues ride
on discrete features — above all the pinna NOTCHES. This checks them
directly, per subject x model, canal probe, 4-12.5 kHz, 1/24-oct-smoothed:

- reference notches: local minima of TF_tr (transparent driver — the
  pinna-only notches this model's illumination geometry produces) with
  prominence >= PROM_DB; typically the subject's main notch (~8-10 kHz)
- PRESERVED if the product TF has a notch within TOL_OCT (1/12 octave);
  preserved notches also report frequency shift (cents) and depth change
  (dB), with a "preserved but attenuated" flag when >6 dB shallower
- LOST notches are classified by what the coupling comb C(f) (structured TF
  / bare TF) does at the reference frequency: a comb PEAK within TOL_OCT
  fills the pinna notch in ("comb-peak fill-in"); a comb NOTCH there would
  be a collision; otherwise "other". Preserved notches also record whether
  a comb notch overlaps (alignment deepens the notch rather than killing it)
- spurious rate: product notches with no reference counterpart — cues the
  hardware injected (mostly the comb)

Counts per subject are small (~1 notch each — the main pinna notch, by
design), so per-model rates aggregate matched/total over the panel;
per-subject ratios are shown as points.

Outputs: runs/notch_preservation.png, runs/notch_preservation_summary.json
Run from the repo root after run_subjects.py (needs ms_pp{N}_tr_*).
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK HK"
import numpy as np
from common import (
    bare_paths,
    canal_tf,
    coupling_db,
    notch_freqs,
    smooth,
    structured_paths,
    transparent_paths,
)
from scipy.signal import find_peaks

MODELS = {
    "z1r": ("MDR-Z1R型", "#C05B21"),
    "lcd": ("LCD型", "#46688A"),
    "dx": ("DX10000CL型", "#2E7D51"),
    "dca2": ("DCA型(AMTS)", "#7A4B94"),
}
BAND = (4000.0, 12500.0)
N_GRID = 384
PROM_DB = 4.0
TOL_OCT = 1 / 12
ATTEN_DB = 6.0


def tf_db_on(paths: tuple[str, str], grid: np.ndarray) -> np.ndarray:
    f, h = canal_tf(*paths)
    sel = (f >= grid[0] * 0.9) & (f <= grid[-1] * 1.1)
    sm = smooth(f[sel], h[sel])
    return np.interp(grid, f[sel], 20 * np.log10(sm + 1e-12))


def notches(grid: np.ndarray, tf_db: np.ndarray) -> list[tuple[float, float]]:
    idx, props = find_peaks(-tf_db, prominence=PROM_DB)
    return [(float(grid[i]), float(p)) for i, p in zip(idx, props["prominences"])]


def match(ref: list, cand: list) -> list[tuple[tuple, tuple | None]]:
    """Greedy nearest-in-octave matching of reference notches to candidates."""
    out = []
    used: set[int] = set()
    for rf, rp in ref:
        best_j, best_d = None, TOL_OCT
        for j, (cf, _cp) in enumerate(cand):
            if j in used:
                continue
            d = abs(np.log2(cf / rf))
            if d <= best_d:
                best_j, best_d = j, d
        if best_j is None:
            out.append(((rf, rp), None))
        else:
            used.add(best_j)
            out.append(((rf, rp), cand[best_j]))
    return out


def main() -> None:
    with open("runs/subjects_selected.json", encoding="utf-8") as fh:
        subjects = json.load(fh)["panel_with_pp1"]
    grid = np.geomspace(BAND[0], BAND[1], N_GRID)

    summary: dict = {"subjects": subjects, "tol_oct": TOL_OCT, "prominence_db": PROM_DB,
                     "attenuated_db": ATTEN_DB, "models": {}}
    pairs_all: dict = {}
    for model in MODELS:
        n_ref = kept = kept_atten = kept_comb_notch = 0
        lost_peak = lost_notch = lost_other = 0
        shifts, ddepths, per_subj, pairs = [], [], [], []
        n_spur = n_prod = 0
        ref_deepest, ref_totals = [], []
        for s in subjects:
            tf_ref = tf_db_on(transparent_paths(s, model), grid)
            ref_n = notches(grid, tf_ref)
            # pinna-derived comb depth of THIS model's illumination: deepest
            # reference notch (>=2 dB detection so shallow ears still count)
            idx_all, pr_all = find_peaks(-tf_ref, prominence=2.0)
            proms = pr_all["prominences"] if len(idx_all) else np.array([0.0])
            ref_deepest.append(float(proms.max()))
            ref_totals.append(float(proms[proms >= PROM_DB].sum()))
            struct_n = notches(grid, tf_db_on(structured_paths(s, model), grid))
            fc, cdb = coupling_db(
                structured_paths(s, model), bare_paths(s, model)
            )
            hw_notch = notch_freqs(fc, cdb, BAND)
            hw_peak = notch_freqs(fc, -cdb, BAND)  # C(f) maxima

            def near(rf: float, freqs: np.ndarray) -> bool:
                return any(abs(np.log2(f / rf)) <= TOL_OCT for f in freqs)

            n_ref += len(ref_n)
            m_tot = match(ref_n, struct_n)
            hit = sum(1 for _, c in m_tot if c is not None)
            kept += hit
            if ref_n:
                per_subj.append(hit / len(ref_n))
            for (rf, rp), c in m_tot:
                if c is not None:
                    cf, cp = c
                    shifts.append(1200.0 * np.log2(cf / rf))
                    ddepths.append(cp - rp)
                    pairs.append((rf, cf))
                    if cp - rp <= -ATTEN_DB:
                        kept_atten += 1
                    if near(rf, hw_notch):
                        kept_comb_notch += 1
                elif near(rf, hw_peak):
                    lost_peak += 1
                elif near(rf, hw_notch):
                    lost_notch += 1
                else:
                    lost_other += 1
            m_rev = match(struct_n, ref_n)
            n_spur += sum(1 for _, c in m_rev if c is None)
            n_prod += len(struct_n)
        pairs_all[model] = pairs
        entry = {
            "n_reference_notches": n_ref,
            "pinna_comb_depth_mean_db": float(np.mean(ref_deepest)),
            "pinna_comb_depth_std_db": float(np.std(ref_deepest)),
            "pinna_comb_total_mean_db": float(np.mean(ref_totals)),
            "recall": kept / max(n_ref, 1),
            "preserved_attenuated": kept_atten,
            "preserved_with_comb_notch": kept_comb_notch,
            "lost_comb_peak": lost_peak,
            "lost_comb_notch": lost_notch,
            "lost_other": lost_other,
            "per_subject_recall": per_subj,
            "freq_shift_cents_median": float(np.median(np.abs(shifts))) if shifts else None,
            "depth_change_db_mean": float(np.mean(ddepths)) if ddepths else None,
            "spurious_rate": n_spur / max(n_prod, 1),
            "n_product_notches": n_prod,
        }
        summary["models"][model] = entry
        print(
            f"{model:5s} ref {n_ref}  recall {entry['recall']:.2f} "
            f"(減衰 {kept_atten}, コームノッチ重なり {kept_comb_notch})  "
            f"消失: コーム山埋没 {lost_peak} / コームノッチ {lost_notch} / 他 {lost_other}  "
            f"|Δf| {entry['freq_shift_cents_median'] or 0:.0f}c "
            f"Δ深さ {entry['depth_change_db_mean'] or 0:+.1f}dB  "
            f"偽ノッチ {entry['spurious_rate']:.2f} ({n_prod}本中)",
            flush=True,
        )

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    # (a) stacked outcome bars per model
    ax = axes[0]
    for i, (model, (label, color)) in enumerate(MODELS.items()):
        e = summary["models"][model]
        n = max(e["n_reference_notches"], 1)
        ok = (e["recall"] * n - e["preserved_attenuated"]) / n
        att = e["preserved_attenuated"] / n
        lp = e["lost_comb_peak"] / n
        ln_ = e["lost_comb_notch"] / n
        lo = e["lost_other"] / n
        ax.bar(i, ok, 0.6, color=color,
               label="保存" if i == 0 else None)
        ax.bar(i, att, 0.6, bottom=ok, color=color, alpha=0.45,
               label="保存(6dB超減衰)" if i == 0 else None)
        ax.bar(i, lp, 0.6, bottom=ok + att, color="#B3352B", alpha=0.85,
               label="消失: コーム山で埋没" if i == 0 else None)
        ax.bar(i, ln_, 0.6, bottom=ok + att + lp, color="#E08A2E", alpha=0.85,
               label="消失: コームノッチ重なり" if i == 0 else None)
        ax.bar(i, lo, 0.6, bottom=ok + att + lp + ln_, color="#9AA3AB",
               label="消失: その他" if i == 0 else None)
        ps = e["per_subject_recall"]
        ax.scatter([i] * len(ps), ps, color="#222", s=14, zorder=3)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([v[0] for v in MODELS.values()], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylabel("基準ノッチの割合")
    ax.set_title(f"個人ノッチの運命(±1/12oct、プロミネンス≥{PROM_DB:.0f}dB)", fontsize=10.5)
    ax.legend(fontsize=7.5)
    # (b) preserved-notch scatter
    ax = axes[1]
    lo_f, hi_f = BAND[0] / 1e3, BAND[1] / 1e3
    ax.plot([lo_f, hi_f], [lo_f, hi_f], color="#999", lw=0.8)
    rng = np.random.default_rng(0)
    for model, (label, color) in MODELS.items():
        p = np.array(pairs_all[model]) / 1e3
        if len(p):
            jit = 1.0 + rng.uniform(-0.006, 0.006, size=p.shape)
            ax.scatter(p[:, 0] * jit[:, 0], p[:, 1] * jit[:, 1], color=color, s=26,
                       alpha=0.8, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ticks = [4, 6, 8, 10, 12]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(map(str, ticks)); ax.set_yticklabels(map(str, ticks))
    ax.grid(alpha=0.3)
    ax.set_xlabel("透明基準のノッチ (kHz)")
    ax.set_ylabel("製品のノッチ (kHz)")
    ax.set_title("保存されたノッチの対応(対角=無変形、微小ジッタ付き)", fontsize=10.5)
    ax.legend(fontsize=7.5)
    # (c) spurious rates
    ax = axes[2]
    for i, (model, (label, color)) in enumerate(MODELS.items()):
        e = summary["models"][model]
        ax.bar(i, e["spurious_rate"], 0.55, color=color)
        ax.text(i, e["spurious_rate"] + 0.02, f"{e['n_product_notches']}本", ha="center",
                fontsize=8, color="#555")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([v[0] for v in MODELS.values()], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylabel("偽ノッチ率")
    ax.set_title("製品ノッチのうち基準に無い割合(=注入されたキュー)", fontsize=10.5)
    fig.suptitle(
        "ノッチ単位の個人キュー保存 — 透明ドライバ基準(外耳道、8被験者、4〜12.5kHz)",
        fontsize=12.5,
    )
    fig.savefig("runs/notch_preservation.png", dpi=115, bbox_inches="tight")
    with open("runs/notch_preservation_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("saved runs/notch_preservation.png, runs/notch_preservation_summary.json")


if __name__ == "__main__":
    main()
