"""Coupling comb C(f) = structured canal TF / bare-source canal TF, 4 models."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK HK"
import numpy as np

MODELS = {
    "Z1R": ("MDR-Z1R型(開口71%)", "#C05B21"),
    "LCD": ("LCD型(開口32%)", "#46688A"),
    "DX": ("DX10000CL型(開口52%)", "#2E7D51"),
    "DCA": ("DCA型(AMTS実測)", "#7A4B94"),
}


def smooth(freqs, mag, frac=24):
    out = np.empty_like(mag)
    for i, f0 in enumerate(freqs):
        sel = (freqs >= f0 * 2 ** (-1 / (2 * frac))) & (freqs <= f0 * 2 ** (1 / (2 * frac)))
        out[i] = mag[sel].mean()
    return out


RUNKEY = {"Z1R": "z1r", "LCD": "lcd", "DX": "dx", "DCA": "dca2"}
BAREKEY = {"Z1R": "z1r", "LCD": "lcd", "DX": "dx", "DCA": "dca"}

def canal_tf(prefix, key):
    k = (RUNKEY if prefix == "batch3" else BAREKEY)[key]
    inc = np.load(f"runs/{prefix}_{k}_incident.npz")
    pin = np.load(f"runs/{prefix}_{k}_pinna.npz")
    dt = float(inc["dt"])
    ref = int(inc["reference_index"])
    n = min(inc["p"].shape[0], pin["p"].shape[0])
    freqs = np.fft.rfftfreq(n, dt)
    H = np.fft.rfft(pin["p"][:n, ref].astype(float)) / (
        np.fft.rfft(inc["p"][:n, ref].astype(float)) + 1e-20
    )
    return freqs, np.abs(H)


fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
stats = {}
for key, (label, color) in MODELS.items():
    fs, Hs = canal_tf("batch3", key)
    fb, Hb = canal_tf("bare", key)
    n = min(len(fs), len(fb))
    # resample bare onto structured frequency grid if lengths differ
    Hb_i = np.interp(fs, fb, Hb)
    sel = (fs >= 1000) & (fs <= 16000)
    C = 20 * np.log10(smooth(fs[sel], Hs[sel]) + 1e-12) - 20 * np.log10(
        smooth(fs[sel], Hb_i[sel]) + 1e-12
    )
    axes[0].plot(fs[sel] / 1e3, C, color=color, lw=1.7, label=label)
    axes[1].plot(fs[sel] / 1e3, 20 * np.log10(smooth(fs[sel], Hs[sel]) + 1e-12),
                 color=color, lw=1.7, label=label)
    core = (fs[sel] >= 5000) & (fs[sel] <= 10000)
    swing = C[core].max() - C[core].min()
    jmin = np.argmin(C[core])
    stats[key] = (swing, C[core][jmin], fs[sel][core][jmin])
    print(f"{key:4s} 5-10kHz swing {swing:.1f} dB, min {C[core][jmin]:+.1f} dB @ "
          f"{fs[sel][core][jmin]/1e3:.1f} kHz")

for ax, title, ylab in zip(
    axes,
    ["カップリング成分 C(f) = 構造ありTF / 裸ソースTF(外耳道)",
     "ピンナ伝達関数(構造あり、参考)"],
    ["C(f) (dB)", "TF (dB)"],
):
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.axhline(0, color="#999", lw=0.7)
    ax.axvspan(5, 10, color="#888", alpha=0.08)
    ax.grid(alpha=0.3)
    ax.set_xlabel("周波数 (kHz)")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=11)
axes[0].legend(fontsize=8.5)
fig.suptitle("前面構造⇄ピンナのカップリング反射(灰帯 = 空間キュー核心帯 5〜10kHz)", fontsize=12)
fig.savefig("runs/coupling_comb.png", dpi=115, bbox_inches="tight")
print("saved runs/coupling_comb.png")
