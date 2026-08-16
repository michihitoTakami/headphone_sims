"""Signal utilities: envelopes, windows, fractional-octave bands, deconvolution."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import hilbert

FloatArray = npt.NDArray[np.float64]


def envelope(x: FloatArray, axis: int = 0) -> FloatArray:
    """Hilbert magnitude envelope."""
    return np.asarray(np.abs(hilbert(x, axis=axis)), dtype=np.float64)


def third_octave_centers(f_min: float = 500.0, f_max: float = 20_000.0) -> FloatArray:
    """IEC base-2 third-octave center frequencies within [f_min, f_max]."""
    n = np.arange(-30, 31)
    centers = 1000.0 * 2.0 ** (n / 3.0)
    return centers[(centers >= f_min) & (centers <= f_max)]


def band_magnitudes(
    x: FloatArray,
    dt: float,
    centers: FloatArray,
    axis: int = 0,
) -> FloatArray:
    """RMS spectral magnitude of each third-octave band, shape (n_bands, ...)."""
    x = np.moveaxis(x, axis, 0)
    spec = np.abs(np.fft.rfft(x, axis=0))
    freqs = np.fft.rfftfreq(x.shape[0], dt)
    out = np.zeros((len(centers), *x.shape[1:]))
    for i, fc in enumerate(centers):
        lo, hi = fc * 2.0 ** (-1.0 / 6.0), fc * 2.0 ** (1.0 / 6.0)
        sel = (freqs >= lo) & (freqs < hi)
        if not sel.any():
            sel = np.zeros_like(freqs, dtype=bool)
            sel[np.argmin(np.abs(freqs - fc))] = True
        out[i] = np.sqrt(np.mean(spec[sel] ** 2, axis=0))
    return out


def deconvolve(
    recorded: FloatArray,
    source: FloatArray,
    dt: float,
    regularization: float = 1e-3,
    axis: int = 0,
) -> FloatArray:
    """Impulse response by regularized spectral division (Tikhonov).

    H = R * conj(S) / (|S|^2 + eps * max|S|^2). Frequencies where the source
    has no energy are suppressed rather than amplified.
    """
    recorded = np.moveaxis(recorded, axis, 0)
    n = recorded.shape[0]
    r_spec = np.fft.rfft(recorded, n=n, axis=0)
    s_spec = np.fft.rfft(source, n=n)
    denom = np.abs(s_spec) ** 2 + regularization * float(np.max(np.abs(s_spec) ** 2))
    h_spec = r_spec * (np.conj(s_spec) / denom).reshape((-1,) + (1,) * (recorded.ndim - 1))
    h = np.fft.irfft(h_spec, n=n, axis=0)
    return np.moveaxis(h, 0, axis)


def time_window(
    n_steps: int,
    dt: float,
    t_center: float,
    pre: float,
    post: float,
) -> npt.NDArray[np.bool_]:
    """Boolean mask selecting samples in [t_center - pre, t_center + post]."""
    t = np.arange(n_steps) * dt
    mask: npt.NDArray[np.bool_] = (t >= t_center - pre) & (t <= t_center + post)
    return mask


def normalized_max_crosscorr(x: FloatArray, y: FloatArray) -> tuple[float, int]:
    """Max of the normalized cross-correlation and its lag (samples).

    SHAPE ONLY: 1.0 means identical waveforms up to a pure delay and ANY
    scale — a probe receiving almost no sound still scores high if the tiny
    waveform has the right shape. Use :func:`amplitude_aware_match` when the
    sound-pressure level matters (it almost always does).
    """
    ex = float(np.sqrt(np.sum(x**2)))
    ey = float(np.sqrt(np.sum(y**2)))
    if ex == 0.0 or ey == 0.0:
        return 0.0, 0
    xc = np.correlate(x, y, mode="full") / (ex * ey)
    lag = int(np.argmax(xc)) - (len(y) - 1)
    return float(xc.max()), lag


def amplitude_aware_match(x: FloatArray, y: FloatArray) -> float:
    """Waveform match including amplitude: 2*max_tau(x*y_tau) / (|x|^2+|y|^2).

    1.0 only if x equals y up to a pure delay INCLUDING scale; x = a*y gives
    2a/(1+a^2) (e.g. -6 dB -> 0.80, -10 dB -> 0.58). Delay-invariant,
    amplitude-sensitive.
    """
    ex = float(np.sum(x**2))
    ey = float(np.sum(y**2))
    if ex == 0.0 or ey == 0.0:
        return 0.0
    xc = np.correlate(x, y, mode="full")
    return float(2.0 * xc.max() / (ex + ey))


def onset_time(envelope_win: FloatArray, dt: float, threshold: float = 0.5) -> float:
    """First time the (windowed) envelope exceeds ``threshold`` of its max.

    Robust leading-edge arrival estimate — unlike the envelope peak, it is
    stable for weak or multi-bump arrivals.
    """
    peak = float(envelope_win.max())
    if peak <= 0.0:
        return float("nan")
    idx = int(np.argmax(envelope_win >= threshold * peak))
    return idx * dt
