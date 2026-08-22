"""Shared loading/analysis helpers for the issue-#3 robustness scripts.

File conventions:
- pp1 reuses the report set: runs/batch3_{model}_{phase}.npz (model key
  "dca2" for the real plug map) and runs/bare_{model}_{phase}.npz (bare key
  "dca").
- other subjects: runs/ms_pp{N}_{model}_{phase}.npz and
  runs/ms_pp{N}_bare_{model}_{phase}.npz.
- re-seat conditions: runs/rs_{model}_{cond}_{phase}.npz and
  runs/rs_{model}_{cond}_bare_{phase}.npz.

C(f) protocol (identical to scripts/report/fig_coupling.py): canal-probe
transfer function TF = FFT(pinna)/FFT(incident) per run pair, 1/24-octave
smoothed; C = TF_structured / TF_bare in dB.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from headphone_sims.analysis import signals
from headphone_sims.analysis.metrics import ApertureSpec
from headphone_sims.fdtd.simulation import SimulationResult

SPATIAL_CORE = (5000.0, 10000.0)
C_SOUND = 343.0

MODEL_CONFIGS = {
    "z1r": "configs/hutubs_70mm_z1r_v2.yaml",
    "lcd": "configs/hutubs_90mm_planar_v2.yaml",
    "dx": "configs/hutubs_40mm_dome_v2.yaml",
    "dca": "configs/hutubs_dca_amts.yaml",
    "dca2": "configs/hutubs_dca_amts_real.yaml",
}


def load_result(path: str) -> tuple[SimulationResult, dict]:
    """SimulationResult rebuilt from a batch .npz plus the raw arrays.

    dx comes from the file when present; legacy files (recorded before dx was
    saved) were all produced at the 0.5 mm default. Runs saved without v get
    zeros (pressure-only metrics stay valid; intensity metrics do not).
    """
    d = np.load(path)
    dx = float(d["dx"]) if "dx" in d.files else 0.5e-3
    p = d["p"].astype(float)
    v = d["v"].astype(float) if "v" in d.files else np.zeros((3, *p.shape))
    res = SimulationResult(
        dt=float(d["dt"]),
        dx=dx,
        positions=d["positions"],
        p=p,
        v=v,
        source_waveform=np.zeros(p.shape[0]),
    )
    return res, d


def aperture_for(model: str, driver_center: np.ndarray) -> ApertureSpec:
    """The model's radiating-aperture footprint (near-field metric geometry)."""
    from headphone_sims.experiments.config import load_config
    from headphone_sims.geometry.scene import driver_aperture

    drv = load_config(MODEL_CONFIGS[model]).scene.driver
    center = (float(driver_center[0]), float(driver_center[1]), float(driver_center[2]))
    return driver_aperture(drv, center)


def band_map(
    inc_path: str, f_want: float, model: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(canal-relative xy, band level dB re mean) on an incident run.

    The direct window uses the nearest-aperture-point distance when the
    ``model`` key is given (near-field correct), else the driver-center ray.
    """
    d = np.load(inc_path)
    p, dt = d["p"].astype(float), float(d["dt"])
    pos, dc, canal = d["positions"], d["driver_center"], d["canal"]
    p = signals.bandpass_zero_phase(p, dt, 1000.0, 12500.0, axis=0)
    t = np.arange(p.shape[0]) * dt
    if model is not None:
        r = aperture_for(model, dc).nearest_distance(pos)
    else:
        r = np.linalg.norm(pos - dc, axis=1)
    masks = np.stack(
        [(t >= ri / C_SOUND - 0.15e-3) & (t <= ri / C_SOUND + 0.45e-3) for ri in r], axis=1
    )
    centers = signals.third_octave_centers(1000, 12500)
    i = int(np.argmin(np.abs(centers - f_want)))
    mags = signals.band_magnitudes(p * masks, dt, centers[i : i + 1], axis=0)[0]
    lvl = 20 * np.log10(mags / mags.mean())
    return pos[:, :2] - canal[:2], lvl


def map_corr(a_xy, a_lvl, b_xy, b_lvl) -> float:
    """Correlation of two probe maps, nearest-neighbor matched (<=1.5mm)."""
    d = np.linalg.norm(a_xy[:, None, :] - b_xy[None, :, :], axis=2)
    j = d.argmin(axis=1)
    ok = d[np.arange(len(a_xy)), j] < 1.5e-3
    return float(np.corrcoef(a_lvl[ok], b_lvl[j[ok]])[0, 1])


def smooth(freqs, mag, frac=24):
    out = np.empty_like(mag)
    for i, f0 in enumerate(freqs):
        sel = (freqs >= f0 * 2 ** (-1 / (2 * frac))) & (freqs <= f0 * 2 ** (1 / (2 * frac)))
        out[i] = mag[sel].mean()
    return out


def structured_paths(subject: int, model: str) -> tuple[str, str]:
    if subject == 1:
        return (f"runs/batch3_{model}_incident.npz", f"runs/batch3_{model}_pinna.npz")
    return (
        f"runs/ms_pp{subject}_{model}_incident.npz",
        f"runs/ms_pp{subject}_{model}_pinna.npz",
    )


def bare_paths(subject: int, model: str) -> tuple[str, str]:
    if subject == 1:
        bare = "dca" if model == "dca2" else model
        return (f"runs/bare_{bare}_incident.npz", f"runs/bare_{bare}_pinna.npz")
    return (
        f"runs/ms_pp{subject}_bare_{model}_incident.npz",
        f"runs/ms_pp{subject}_bare_{model}_pinna.npz",
    )


def transparent_paths(subject: int, model: str) -> tuple[str, str]:
    """Transparent-driver reference pair: the model's aperture as an additive
    monopole sheet — no driver body, no baffle, no structure."""
    return (
        f"runs/ms_pp{subject}_tr_{model}_incident.npz",
        f"runs/ms_pp{subject}_tr_{model}_pinna.npz",
    )


def reseat_paths(model: str, cond: str, structured: bool) -> tuple[str, str]:
    if cond == "base":
        return (
            structured_paths(1, model) if structured else bare_paths(1, model)
        )
    mid = "" if structured else "bare_"
    return (
        f"runs/rs_{model}_{cond}_{mid}incident.npz",
        f"runs/rs_{model}_{cond}_{mid}pinna.npz",
    )


def canal_tf(inc_path: str, pin_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Canal-probe transfer function |FFT(pinna)/FFT(incident)| (unsmoothed)."""
    inc = np.load(inc_path)
    pin = np.load(pin_path)
    dt = float(inc["dt"])
    ref = int(inc["reference_index"])
    n = min(inc["p"].shape[0], pin["p"].shape[0])
    freqs = np.fft.rfftfreq(n, dt)
    h = np.fft.rfft(pin["p"][:n, ref].astype(float)) / (
        np.fft.rfft(inc["p"][:n, ref].astype(float)) + 1e-20
    )
    return freqs, np.abs(h)


def coupling_db(
    struct_pair: tuple[str, str],
    bare_pair: tuple[str, str],
    f_lo: float = 1000.0,
    f_hi: float = 16000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """1/24-oct-smoothed C(f) in dB on the structured run's frequency grid."""
    fs, hs = canal_tf(*struct_pair)
    fb, hb = canal_tf(*bare_pair)
    hb_i = np.interp(fs, fb, hb)
    sel = (fs >= f_lo) & (fs <= f_hi)
    c = 20 * np.log10(smooth(fs[sel], hs[sel]) + 1e-12) - 20 * np.log10(
        smooth(fs[sel], hb_i[sel]) + 1e-12
    )
    return fs[sel], c


def comb_stats(freqs: np.ndarray, c_db: np.ndarray, band=SPATIAL_CORE) -> dict:
    """Swing / deepest notch of C(f) within ``band``."""
    core = (freqs >= band[0]) & (freqs <= band[1])
    fc, cc = freqs[core], c_db[core]
    j = int(np.argmin(cc))
    return {
        "swing_db": float(cc.max() - cc.min()),
        "notch_db": float(cc[j]),
        "notch_hz": float(fc[j]),
    }


def notch_freqs(
    freqs: np.ndarray,
    mag_db: np.ndarray,
    band: tuple[float, float],
    prominence: float = 3.0,
) -> np.ndarray:
    """Local-minimum frequencies of a smoothed dB curve within ``band``."""
    sel = (freqs >= band[0]) & (freqs <= band[1])
    idx, _ = find_peaks(-mag_db[sel], prominence=prominence)
    return freqs[sel][idx]


def octave_distance(f_a: float, f_b: np.ndarray) -> float:
    """Distance from f_a to the nearest of f_b, in octaves (inf if empty)."""
    if len(f_b) == 0:
        return float("inf")
    return float(np.min(np.abs(np.log2(f_b / f_a))))
