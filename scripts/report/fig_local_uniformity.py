"""Local-uniformity decomposition: plane-fit gradient vs residual, 4 models.

Run from the repo root; writes runs/local_uniformity.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK HK"
import numpy as np
from scipy.spatial import cKDTree

from headphone_sims.analysis import signals
from headphone_sims.experiments.config import load_config
from headphone_sims.geometry.scene import driver_aperture

C = 343.0
MODELS = {"Z1R": ("MDR-Z1R型", "#C05B21", "runs/batch3_z1r_incident.npz"),
          "LCD": ("LCD型", "#46688A", "runs/batch3_lcd_incident.npz"),
          "DX": ("DX10000CL型", "#2E7D51", "runs/batch3_dx_incident.npz"),
          "DCA": ("DCA型(AMTS実測)", "#7A4B94", "runs/batch3_dca2_incident.npz")}
CFGS = {"Z1R": "configs/hutubs_70mm_z1r_v2.yaml",
        "LCD": "configs/hutubs_90mm_planar_v2.yaml",
        "DX": "configs/hutubs_40mm_dome_v2.yaml",
        "DCA": "configs/hutubs_dca_amts_real.yaml"}


def smooth(freqs, mag, frac=6):
    out = np.empty_like(mag)
    for i, f0 in enumerate(freqs):
        sel = (freqs >= f0*2**(-1/(2*frac))) & (freqs <= f0*2**(1/(2*frac)))
        out[i] = mag[sel].mean(axis=0)
    return out


def plane_stats(pos, vals, tree, radius=7e-3, min_nb=5):
    vals2 = vals if vals.ndim == 2 else vals[None, :]
    grads, resids = [], []
    for i in range(pos.shape[0]):
        nb = tree.query_ball_point(pos[i], radius)
        if len(nb) < min_nb + 1:
            continue
        X = pos[nb] - pos[i]
        A = np.hstack([X, np.ones((len(nb), 1))])
        g_here, r_here = [], []
        for row in vals2:
            coef, *_ = np.linalg.lstsq(A, row[nb], rcond=None)
            pred = A @ coef
            g_here.append(np.linalg.norm(coef[:3]) * 1e-3)
            r_here.append(np.sqrt(((row[nb] - pred) ** 2).mean()))
        grads.append(np.mean(g_here))
        resids.append(np.mean(r_here))
    return np.array(grads), np.array(resids)


stats = {}
for key, (label, color, path) in MODELS.items():
    d = np.load(path)
    p, dt, pos, dc = d["p"].astype(float), float(d["dt"]), d["positions"], d["driver_center"]
    t = np.arange(p.shape[0]) * dt
    ap = driver_aperture(load_config(CFGS[key]).scene.driver, tuple(float(c) for c in dc))
    r = ap.nearest_distance(pos)  # near-field first-arrival reference
    masks = np.stack([(t >= ri/C-0.15e-3) & (t <= ri/C+0.45e-3) for ri in r], axis=1)
    pb = signals.bandpass_zero_phase(p, dt, 1000.0, 12500.0, axis=0) * masks
    L = 20*np.log10(np.sqrt((pb**2).sum(axis=0)) + 1e-15)
    P = np.abs(np.fft.rfft(p * masks, axis=0))
    freqs = np.fft.rfftfreq(p.shape[0], dt)
    sel = (freqs >= 5000) & (freqs <= 10000)
    Ls = 20*np.log10(smooth(freqs[sel], P[sel]) + 1e-15)
    Ls = Ls - Ls.mean(axis=0, keepdims=True)
    tree = cKDTree(pos)
    gl, rl = plane_stats(pos, L, tree)
    gs, rs = plane_stats(pos, Ls[::4], tree)
    stats[key] = dict(label=label, color=color,
                      lvl=(gl.mean(), rl.mean()), shp=(gs.mean(), rs.mean()))
    print(f"{key:4s} lvl grad {gl.mean():.3f} dB/mm resid {rl.mean():.2f} | "
          f"shape grad {gs.mean():.3f} resid {rs.mean():.2f}")

OFFSETS = {("lvl", "DX"): (10, -16), ("lvl", "DCA"): (10, 8)}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
for ax, kind, title in [(axes[0], "lvl", "広帯域レベル(1-12.5kHz)"),
                        (axes[1], "shp", "スペクトル形状(5-10kHz)")]:
    for key, st in stats.items():
        x, y = st[kind]
        ax.scatter([x], [y], s=180, color=st["color"], zorder=3)
        off = OFFSETS.get((kind, key), (10, 6))
        ax.annotate(st["label"], (x, y), textcoords="offset points", xytext=off, fontsize=10)
    ax.set_xlabel("局所勾配の平均 (dB/mm) — 系統的な差分の量")
    ax.set_ylabel("平面フィット残差の平均 (dB) — 局所的な乱れ")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
fig.suptitle("入射波面の局所均一性: 近傍7mmの平面フィットによる「系統勾配」と「乱れ」の分解", fontsize=12)
fig.tight_layout()
fig.savefig("runs/local_uniformity.png", dpi=115, bbox_inches="tight")
print("saved runs/local_uniformity.png")
