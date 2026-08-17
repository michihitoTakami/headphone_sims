"""Inter-subject variance of post-pinna metrics -> robustness ranking
(issue #3, steps 4c/5).

Per subject x model: compute_metrics on the structured pinna run (report
protocol: 1-12.5 kHz standard band and the 5-10 kHz spatial-cue core).
Inter-subject spread (std over subjects) of each aggregate = how much the
design's delivered spatial-cue quality depends on whose ear it meets.

Hypothesis test (step 5): designs with a smaller coupling-comb swing should
show smaller inter-subject spread — reads the per-model swing from
runs/subject_comb_summary.json (run fig_subject_comb.py first).

Outputs: runs/subject_variance.png, runs/subject_variance_summary.json
Run from the repo root after run_subjects.py.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK HK"
import numpy as np
from common import structured_paths

from headphone_sims.analysis.metrics import compute_metrics
from headphone_sims.fdtd.simulation import SimulationResult

MODELS = {
    "z1r": ("MDR-Z1R型", "#C05B21"),
    "lcd": ("LCD型", "#46688A"),
    "dx": ("DX10000CL型", "#2E7D51"),
    "dca2": ("DCA型(AMTS)", "#7A4B94"),
}
KEYS = [
    ("similarity_mean", "波形一致(振幅込み)", +1),
    ("level_spread_db", "レベルばらつき σ (dB)", -1),
    ("arrival_spread_ms", "到達時間ばらつき (ms)", -1),
    ("incidence_spread_deg", "入射角ばらつき (deg)", -1),
]


def metrics_for(subject: int, model: str) -> tuple[dict, dict]:
    _inc_path, pin_path = structured_paths(subject, model)
    d = np.load(pin_path)
    res = SimulationResult(
        dt=float(d["dt"]), dx=0.5e-3, positions=d["positions"],
        p=d["p"].astype(float), v=d["v"].astype(float),
        source_waveform=np.zeros(d["p"].shape[0]),
    )
    args = (tuple(d["driver_center"]), int(d["reference_index"]))
    std = compute_metrics(res, *args).summary()
    core = compute_metrics(res, *args, band=(5000.0, 10000.0)).summary()
    return std, core


def main() -> None:
    with open("runs/subjects_selected.json", encoding="utf-8") as fh:
        subjects = json.load(fh)["panel_with_pp1"]

    per = {m: {"std": [], "core": []} for m in MODELS}
    for model in MODELS:
        for subject in subjects:
            std, core = metrics_for(subject, model)
            std["subject"] = core["subject"] = subject
            per[model]["std"].append(std)
            per[model]["core"].append(core)
            print(
                f"pp{subject:<3d} {model:4s} core sim {core['similarity_mean']:.3f} "
                f"lvlσ {core['level_spread_db']:.2f} arr {core['arrival_spread_ms']*1e3:.0f}µs",
                flush=True,
            )

    summary: dict = {"subjects": subjects, "models": {}}
    for model in MODELS:
        entry = {}
        for band in ("std", "core"):
            rows = per[model][band]
            entry[band] = {
                k: {
                    "values": [float(r[k]) for r in rows],
                    "mean": float(np.mean([r[k] for r in rows])),
                    "std": float(np.std([r[k] for r in rows])),
                }
                for k, _, _ in KEYS
            }
        summary["models"][model] = entry

    # hypothesis: comb swing vs inter-subject spread of core similarity
    try:
        with open("runs/subject_comb_summary.json", encoding="utf-8") as fh:
            comb = json.load(fh)
        swings = {m: comb[m]["swing_mean_db"] for m in MODELS}
    except FileNotFoundError:
        swings = {}
    spreads = {
        m: summary["models"][m]["core"]["similarity_mean"]["std"] for m in MODELS
    }
    summary["hypothesis"] = {"swing_mean_db": swings, "core_similarity_std": spreads}

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.2))
    x = np.arange(len(MODELS))
    for ax, (key, label, _) in zip(axes[:3], KEYS[:3], strict=False):
        for i, (model, (mlabel, color)) in enumerate(MODELS.items()):
            vals = np.array(summary["models"][model]["core"][key]["values"])
            if key == "arrival_spread_ms":
                vals = vals * 1e3
            ax.scatter([i] * len(vals), vals, color=color, s=30, alpha=0.8, zorder=3)
            ax.errorbar(
                i, vals.mean(), yerr=vals.std(), color=color, fmt="_", ms=26,
                capsize=6, lw=2, zorder=2,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([v[0] for v in MODELS.values()], fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.set_title(
            label.replace("(ms)", "(µs)") if key == "arrival_spread_ms" else label,
            fontsize=10.5,
        )
    ax = axes[3]
    for model, (mlabel, color) in MODELS.items():
        if swings:
            ax.scatter(
                swings[model], spreads[model], color=color, s=90, label=mlabel, zorder=3
            )
    ax.set_xlabel("コーム振れ幅 平均 (dB)")
    ax.set_ylabel("核心帯 波形一致の被験者間σ")
    ax.set_title("仮説: コームが強い設計ほど個人差に敏感", fontsize=10.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)
    fig.suptitle(
        f"ポスト・ピンナ指標(5-10 kHz核心帯)の被験者間分散({len(subjects)}被験者、点=被験者)",
        fontsize=12.5,
    )
    fig.savefig("runs/subject_variance.png", dpi=115, bbox_inches="tight")
    with open("runs/subject_variance_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("saved runs/subject_variance.png, runs/subject_variance_summary.json")


if __name__ == "__main__":
    main()
