"""Main figures for the 4-model comparison report (radar, lateral,
levels, pinna TFs, geometry views, final_summary.json). Run from repo root."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK HK"
import numpy as np

from headphone_sims.analysis import signals
from headphone_sims.analysis.metrics import compute_metrics
from headphone_sims.experiments.config import load_config
from headphone_sims.fdtd.simulation import SimulationResult
from headphone_sims.geometry.scene import driver_aperture

C = 343.0
MODELS = {
    "Z1R": dict(label="MDR-Z1R型", short="Z1R", color="#C05B21",
                inc="runs/batch3_z1r_incident.npz", pin="runs/batch3_z1r_pinna.npz"),
    "LCD": dict(label="LCD型", short="LCD", color="#46688A",
                inc="runs/batch3_lcd_incident.npz", pin="runs/batch3_lcd_pinna.npz"),
    "DX": dict(label="DX10000CL型", short="DX10000CL", color="#2E7D51",
               inc="runs/batch3_dx_incident.npz", pin="runs/batch3_dx_pinna.npz"),
    "DCA": dict(label="DCA型(AMTS実測)", short="DCA(AMTS)", color="#7A4B94",
                inc="runs/batch3_dca2_incident.npz", pin="runs/batch3_dca2_pinna.npz"),
}
CFGS = {"Z1R": "configs/hutubs_70mm_z1r_v2.yaml",
        "LCD": "configs/hutubs_90mm_planar_v2.yaml",
        "DX": "configs/hutubs_40mm_dome_v2.yaml",
        "DCA": "configs/hutubs_dca_amts_real.yaml"}

def aperture_of(key, driver_center):
    """Near-field metric geometry: the model's radiating footprint."""
    drv = load_config(CFGS[key]).scene.driver
    return driver_aperture(drv, tuple(float(c) for c in driver_center))

def load_result(path, meta_path=None):
    d = np.load(path)
    meta = np.load(meta_path) if meta_path else d
    dx = float(d["dx"]) if "dx" in d.files else 0.5e-3  # legacy files: 0.5mm default
    res = SimulationResult(dt=float(d["dt"]), dx=dx, positions=d["positions"],
                           p=d["p"].astype(float), v=d["v"].astype(float),
                           source_waveform=np.zeros(d["p"].shape[0]))
    return res, meta

metrics_inc, metrics_pin, laterals, apertures = {}, {}, {}, {}
metrics_inc_core, metrics_pin_core = {}, {}
for key, m in MODELS.items():
    res, meta = load_result(m["inc"])
    args = (tuple(meta["driver_center"]), int(meta["reference_index"]))
    ap = apertures[key] = aperture_of(key, meta["driver_center"])
    metrics_inc[key] = compute_metrics(res, *args, window_pre=0.15e-3, window_post=0.45e-3,
                                       aperture=ap)
    metrics_inc_core[key] = compute_metrics(res, *args, window_pre=0.15e-3,
                                            window_post=0.45e-3, band=(5000.0, 10000.0),
                                            aperture=ap)
    res_p, _ = load_result(m["pin"], m["inc"])
    metrics_pin[key] = compute_metrics(res_p, *args, aperture=ap)
    metrics_pin_core[key] = compute_metrics(res_p, *args, band=(5000.0, 10000.0), aperture=ap)
    dc = meta["driver_center"]
    laterals[key] = np.linalg.norm(meta["positions"][:, :2] - dc[:2], axis=1) * 1e3

# --- fig 1: radar (incident, normalized best=outer) ---
RAD = [("波形一致(振幅込み)", "similarity_mean", +1), ("入射角の揃い", "incidence_spread_deg", -1),
       ("方向純度(1-拡散度)", "diffuseness_mean", -1), ("時間整列", "arrival_spread_ms", -1),
       ("レベル均一性", "level_spread_db", -1)]
vals = {k: [metrics_inc[k].summary()[m] for _, m, _ in RAD] for k in MODELS}
norm = np.zeros((len(MODELS), len(RAD)))
for j, (_, mk, sign) in enumerate(RAD):
    col = np.array([vals[k][j] for k in MODELS]) * sign
    lo, hi = col.min(), col.max()
    norm[:, j] = 0.25 + 0.75 * (col - lo) / (hi - lo + 1e-12)
angles = np.linspace(0, 2*np.pi, len(RAD), endpoint=False).tolist()
fig, ax = plt.subplots(figsize=(6.2, 5.6), subplot_kw={"projection": "polar"})
for i, (key, m) in enumerate(MODELS.items()):
    v = norm[i].tolist() + [norm[i][0]]
    ax.plot(angles + [angles[0]], v, color=m["color"], lw=2, label=m["short"])
    ax.fill(angles + [angles[0]], v, color=m["color"], alpha=0.12)
ax.set_xticks(angles); ax.set_xticklabels([r[0] for r in RAD], fontsize=9)
ax.set_yticklabels([]); ax.set_ylim(0, 1.05)
ax.set_title("入射波面品質の総合バランス(外側ほど良い、モデル間相対)", fontsize=11, pad=18)
ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), fontsize=9)
fig.savefig("runs/final_radar.png", dpi=120, bbox_inches="tight")

# --- fig 2: lateral similarity profile ---
fig, ax = plt.subplots(figsize=(7.2, 4.2))
bins = [(0,10),(10,20),(20,30),(30,45)]
x = [5, 15, 25, 37.5]
for key, m in MODELS.items():
    lat = laterals[key]; sim = metrics_inc[key].similarity
    means = [sim[(lat>=lo)&(lat<hi)].mean() for lo,hi in bins]
    mins = [sim[(lat>=lo)&(lat<hi)].min() for lo,hi in bins]
    ax.plot(x, means, "o-", color=m["color"], lw=2, label=f"{m['short']} 平均")
    ax.plot(x, mins, "o--", color=m["color"], lw=1, alpha=0.6, label=f"{m['short']} 最悪")
ax.set_xlabel("ドライバ軸からの横距離 (mm)"); ax.set_ylabel("入射波形類似度")
ax.set_title("入射波形の均一性: ピンナ中心部から周縁部まで", fontsize=11)
ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=3)
fig.savefig("runs/final_lateral.png", dpi=120, bbox_inches="tight")

# --- fig 3: level maps 3x4 ---
centers = signals.third_octave_centers(1000, 12500)
fig, axes = plt.subplots(len(MODELS), 4, figsize=(17, 12.5 / 3 * len(MODELS)))
for row, (key, mm) in enumerate(MODELS.items()):
    d = np.load(mm["inc"])
    p, dt, pos, dc = d["p"].astype(float), float(d["dt"]), d["positions"], d["driver_center"]
    p = signals.bandpass_zero_phase(p, dt, 1000.0, 12500.0, axis=0)
    t = np.arange(p.shape[0]) * dt
    r = apertures[key].nearest_distance(pos)  # near-field first-arrival reference
    masks = np.stack([(t >= ri/C - 0.15e-3) & (t <= ri/C + 0.45e-3) for ri in r], axis=1)
    p_win = p * masks
    rms = np.sqrt((p_win**2).sum(axis=0))
    mags = signals.band_magnitudes(p_win, dt, centers, axis=0)
    panels = [("広帯域 1-12.5k", 20*np.log10(rms / rms.mean()))]
    for f_want in (5000., 8000., 10000.):
        i = int(np.argmin(np.abs(centers - f_want)))
        panels.append((f"{centers[i]/1000:.0f} kHz帯", 20*np.log10(mags[i] / mags[i].mean())))
    for col, (title, v) in enumerate(panels):
        ax = axes[row][col]
        sc = ax.scatter(pos[:,0]*1e3, pos[:,1]*1e3, c=v, s=24, cmap="RdBu_r", vmin=-9, vmax=9)
        ax.set_aspect("equal"); ax.set_title(f"{title}  σ={v.std():.1f}dB", fontsize=10)
        if col == 0:
            ax.set_ylabel(mm["short"], fontsize=11)
fig.colorbar(sc, ax=axes, label="レベル (dB, 各モデル平均比)", shrink=0.5)
fig.suptitle("入射音量分布 1-12.5kHz帯域制限(ピンナ位置プローブ・各モデル最小距離配置)", fontsize=13)
fig.savefig("runs/final_levels.png", dpi=110, bbox_inches="tight")

# --- fig 4: pinna transfer functions ---
def smooth(freqs, mag, frac=24):
    out = np.empty_like(mag)
    for i, f0 in enumerate(freqs):
        sel = (freqs >= f0*2**(-1/(2*frac))) & (freqs <= f0*2**(1/(2*frac)))
        out[i] = mag[sel].mean()
    return out
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for key, mm in MODELS.items():
    inc, pin = np.load(mm["inc"]), np.load(mm["pin"])
    dt = float(inc["dt"]); ref = int(inc["reference_index"])
    freqs = np.fft.rfftfreq(inc["p"].shape[0], dt)
    sel = (freqs >= 1000) & (freqs <= 20000)
    H = np.fft.rfft(pin["p"][:, ref].astype(float)) / (np.fft.rfft(inc["p"][:, ref].astype(float)) + 1e-20)
    axes[0].plot(freqs[sel]/1e3, 20*np.log10(smooth(freqs[sel], np.abs(H)[sel])), color=mm["color"], lw=1.7, label=mm["short"])
    near = np.linalg.norm(inc["positions"] - inc["canal"], axis=1) < 10e-3
    Hn = np.fft.rfft(pin["p"][:, near].astype(float), axis=0) / (np.fft.rfft(inc["p"][:, near].astype(float), axis=0) + 1e-20)
    mag = np.sqrt((np.abs(Hn)**2).mean(axis=1))
    axes[1].plot(freqs[sel]/1e3, 20*np.log10(smooth(freqs[sel], mag[sel])), color=mm["color"], lw=1.7)
for ax, t_ in zip(axes, ["外耳道入口プローブ", "外耳道周辺10mmの平均"]):
    ax.set_xscale("log"); ax.set_xticks([1,2,4,8,16]); ax.set_xticklabels(["1","2","4","8","16"])
    ax.axhline(0, color="#999", lw=0.7); ax.grid(alpha=0.3)
    ax.set_xlabel("周波数 (kHz)"); ax.set_ylabel("ピンナ伝達関数 (dB)"); ax.set_title(t_, fontsize=11)
axes[0].legend(fontsize=9)
fig.suptitle("ピンナ伝達関数 = ピンナ有り / 入射波面(コンカ共鳴とノッチ構造)", fontsize=12)
fig.savefig("runs/final_pinna_tf.png", dpi=115, bbox_inches="tight")

# --- scene geometry views for the 3 models ---
from headphone_sims.geometry.scene import build_scene
from headphone_sims.viz.scene_view import save_scene_views

for key, cfg_path in CFGS.items():
    cfg = load_config(cfg_path).scene
    built = build_scene(cfg, device="cpu")
    save_scene_views(built.solid, built.grid.dx, built.driver_center,
                     built.probe_positions, f"runs/final_geo_{key}.png")

# --- summary tables json ---
out = {"incident": {k: metrics_inc[k].summary() for k in MODELS},
       "pinna": {k: metrics_pin[k].summary() for k in MODELS},
       "incident_core": {k: metrics_inc_core[k].summary() for k in MODELS},
       "pinna_core": {k: metrics_pin_core[k].summary() for k in MODELS}}
json.dump(out, open("runs/final_summary.json", "w"), indent=1)
for phase in ("incident", "pinna"):
    print(f"== {phase}")
    for k in MODELS:
        s = out[phase][k]
        print(f"  {k:4s} sim(amp) {s['similarity_mean']:.3f}/{s['similarity_min']:.3f} "
              f"shape {s['shape_similarity_mean']:.3f} lvl σ{s['level_spread_db']:.1f}/min{s['level_min_db']:.1f}dB "
              f"incid {s['incidence_spread_deg']:5.1f} diff {s['diffuseness_mean']:.3f} "
              f"arr {s['arrival_spread_ms']*1e3:4.0f}µs")
print("figures saved")
