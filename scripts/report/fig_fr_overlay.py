"""Per-probe incident FR overlays (level+shape / shape-only), 4 models.

Run from the repo root; writes runs/probe_fr_overlay.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK HK"
import numpy as np

C = 343.0
MODELS = {"Z1R": ("MDR-Z1R型", "runs/batch3_z1r_incident.npz"),
          "LCD": ("LCD型", "runs/batch3_lcd_incident.npz"),
          "DX": ("DX10000CL型", "runs/batch3_dx_incident.npz"),
          "DCA": ("DCA型(AMTS実測)", "runs/batch3_dca2_incident.npz")}


def smooth(freqs, mag, frac=6):
    out = np.empty_like(mag)
    for i, f0 in enumerate(freqs):
        sel = (freqs >= f0*2**(-1/(2*frac))) & (freqs <= f0*2**(1/(2*frac)))
        out[i] = mag[sel].mean(axis=0)
    return out


fig, axes = plt.subplots(2, 4, figsize=(18, 7.6), sharex=True, sharey="row")
for col, (key, (label, path)) in enumerate(MODELS.items()):
    d = np.load(path)
    p, dt, pos, dc = d["p"].astype(float), float(d["dt"]), d["positions"], d["driver_center"]
    t = np.arange(p.shape[0]) * dt
    r = np.linalg.norm(pos - dc, axis=1)
    masks = np.stack([(t >= ri/C - 0.15e-3) & (t <= ri/C + 0.45e-3) for ri in r], axis=1)
    P = np.abs(np.fft.rfft((p * masks), axis=0))
    freqs = np.fft.rfftfreq(p.shape[0], dt)
    sel = (freqs >= 1000) & (freqs <= 14000)
    S = smooth(freqs[sel], P[sel])
    L = 20*np.log10(S + 1e-15)
    dev = L - L.mean(axis=1, keepdims=True)
    shape = dev - dev.mean(axis=0, keepdims=True)
    rlat = np.linalg.norm(pos[:, :2] - dc[:2], axis=1)
    order = np.argsort(rlat)
    span = float(rlat.max() - rlat.min()) + 1e-9
    colors = plt.cm.viridis((rlat - rlat.min()) / span)
    for row, V in [(0, dev), (1, shape)]:
        ax = axes[row][col]
        for i in order:
            ax.plot(freqs[sel]/1e3, V[:, i], color=colors[i], lw=0.6, alpha=0.35)
        ax.axhline(0, color="#444", lw=0.6)
        ax.set_xscale("log"); ax.set_xticks([1, 2, 4, 8, 12])
        ax.set_xticklabels(["1", "2", "4", "8", "12"])
        ax.axvspan(5, 10, color="#888", alpha=0.10)
        ax.grid(alpha=0.25)
        sig = V.std(axis=1)
        band = (freqs[sel] >= 5000) & (freqs[sel] <= 10000)
        ax.text(0.03, 0.05, f"5-10kHz σ平均 {sig[band].mean():.1f} dB",
                transform=ax.transAxes, fontsize=10, fontweight="bold")
    axes[0][col].set_title(label, fontsize=12)
axes[0][0].set_ylabel("レベル偏差 (dB)\n(モデル平均スペクトル比)", fontsize=10)
axes[1][0].set_ylabel("形状偏差 (dB)\n(各プローブの平均レベル除去後)", fontsize=10)
axes[0][0].set_ylim(-16, 10); axes[1][0].set_ylim(-12, 10)
for ax in axes[1]:
    ax.set_xlabel("周波数 (kHz)")
sm = plt.cm.ScalarMappable(cmap="viridis")
cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.01)
cb.set_label("駆動軸からの横距離(暗=中心、明=周縁)")
fig.suptitle("全プローブの入射周波数応答の重ね描き — 上段: レベル込み偏差 / 下段: スペクトル形状のみ(灰帯=空間キュー核心帯)",
             fontsize=13)
fig.savefig("runs/probe_fr_overlay.png", dpi=110, bbox_inches="tight")
print("saved runs/probe_fr_overlay.png")
