"""Pinna wavefront metrics.

All metrics operate on a :class:`headphone_sims.fdtd.simulation.SimulationResult`
plus scene metadata (driver center, reference probe). The direct-arrival time
window per probe is [t_geo - pre, t_geo + post] with t_geo the geometric
propagation time from the driver center.

Metrics (per probe unless noted):
- similarity: AMPLITUDE-AWARE waveform match vs. the reference probe
  (2*max_xcorr/(Ex+Ey); 1.0 = same waveform AND same level up to a delay;
  a probe at -10 dB scores ~0.58 even with a perfect shape)
- shape_similarity: the legacy shape-only normalized cross-correlation
- level_re_ref_db: broadband direct-window RMS level relative to the
  reference probe
- incidence_deg / incidence_deviation_deg: direction of the time-integrated
  acoustic intensity, and its angle to the geometric driver->probe ray
- arrival_spread_ms (scalar): std over probes of (envelope ONSET time -
  t_geo); onset = first crossing of 50% of the window's envelope peak
- spectral_deviation_db (n_bands, n_probes): band magnitude relative to the
  1/r-scaled reference probe
- diffuseness: 1 - |integral of I dt| / integral of |I| dt over the window
  (0 = coherent plane-like transport, 1 = fully diffuse)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from headphone_sims.analysis import signals
from headphone_sims.fdtd.simulation import SimulationResult

FloatArray = npt.NDArray[np.float64]

SOUND_SPEED = 343.0


@dataclass
class PinnaMetrics:
    probe_positions: FloatArray  # (n, 3)
    similarity: FloatArray  # (n,) amplitude-aware match vs reference
    shape_similarity: FloatArray  # (n,) legacy shape-only xcorr
    level_re_ref_db: FloatArray  # (n,) broadband level re the reference probe
    incidence_deviation_deg: FloatArray  # (n,)
    incidence_spread_deg: float  # dispersion of intensity directions across probes
    # Onset time minus geometric time; contains a pulse-shape constant offset
    # (onset leads the envelope peak), so the SPREAD is the meaningful number.
    arrival_error_ms: FloatArray
    arrival_spread_ms: float
    band_centers: FloatArray  # (n_bands,)
    spectral_deviation_db: FloatArray  # (n_bands, n)
    diffuseness: FloatArray  # (n,)

    def summary(self) -> dict[str, float]:
        return {
            "similarity_mean": float(np.mean(self.similarity)),
            "similarity_min": float(np.min(self.similarity)),
            "shape_similarity_mean": float(np.mean(self.shape_similarity)),
            "level_spread_db": float(np.std(self.level_re_ref_db)),
            "level_min_db": float(np.min(self.level_re_ref_db)),
            "incidence_deviation_deg_mean": float(np.mean(self.incidence_deviation_deg)),
            "incidence_deviation_deg_max": float(np.max(self.incidence_deviation_deg)),
            "incidence_spread_deg": self.incidence_spread_deg,
            "arrival_spread_ms": self.arrival_spread_ms,
            "spectral_deviation_db_rms": float(np.sqrt(np.mean(self.spectral_deviation_db**2))),
            "diffuseness_mean": float(np.mean(self.diffuseness)),
            "diffuseness_max": float(np.max(self.diffuseness)),
        }


def _direct_window_masks(
    result: SimulationResult,
    driver_center: FloatArray,
    pre: float,
    post: float,
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    # t_geo is measured from the driver rim-plane center; a dome diaphragm
    # radiates from up to dome_depth closer (<= ~0.02 ms early), well inside
    # `pre`, so the window still captures the direct sound.
    r = np.linalg.norm(result.positions - driver_center, axis=1)
    t_geo = r / SOUND_SPEED
    n_steps = result.p.shape[0]
    masks = np.stack(
        [signals.time_window(n_steps, result.dt, tg, pre, post) for tg in t_geo], axis=1
    )
    return t_geo, masks


def compute_metrics(
    result: SimulationResult,
    driver_center: tuple[float, float, float],
    reference_index: int,
    window_pre: float = 0.2e-3,
    window_post: float = 0.5e-3,
    band_min: float = 1000.0,
    band_max: float = 20_000.0,
) -> PinnaMetrics:
    center = np.asarray(driver_center, dtype=np.float64)
    n_probes = result.p.shape[1]
    t_geo, masks = _direct_window_masks(result, center, window_pre, window_post)

    p_win = result.p * masks  # (n_steps, n)
    v_win = result.v * masks[None]  # (3, n_steps, n)

    # (a) waveform match vs. reference probe: amplitude-aware (primary) and
    # shape-only (secondary), plus explicit level.
    ref = p_win[:, reference_index]
    similarity = np.zeros(n_probes)
    shape_similarity = np.zeros(n_probes)
    for i in range(n_probes):
        similarity[i] = signals.amplitude_aware_match(p_win[:, i], ref)
        shape_similarity[i], _ = signals.normalized_max_crosscorr(p_win[:, i], ref)
    rms = np.sqrt((p_win**2).sum(axis=0))
    ref_rms = max(rms[reference_index], 1e-30)
    level_re_ref_db = 20.0 * np.log10(np.maximum(rms, 1e-30) / ref_rms)

    # (b) incidence direction from time-integrated intensity I = p*v.
    intensity = np.einsum("tn,ctn->cn", p_win, v_win) * result.dt  # (3, n)
    i_norm = np.linalg.norm(intensity, axis=0)
    i_norm = np.where(i_norm == 0.0, 1.0, i_norm)
    i_dir = intensity / i_norm
    ray = result.positions - center
    ray_dir = (ray / np.linalg.norm(ray, axis=1, keepdims=True)).T  # (3, n)
    cos_dev = np.clip(np.sum(i_dir * ray_dir, axis=0), -1.0, 1.0)
    incidence_deviation = np.degrees(np.arccos(cos_dev))
    mean_dir = i_dir.mean(axis=1)
    mean_norm = float(np.linalg.norm(mean_dir))
    mean_dir = mean_dir / mean_norm if mean_norm > 0.0 else np.array([0.0, 0.0, 1.0])
    cos_spread = np.clip(np.sum(i_dir * mean_dir[:, None], axis=0), -1.0, 1.0)
    incidence_spread = float(np.degrees(np.std(np.arccos(cos_spread))))

    # (c) arrival-time error and spread (robust onset, not peak).
    env = signals.envelope(p_win, axis=0)
    t_on = np.array([signals.onset_time(env[:, i], result.dt) for i in range(n_probes)])
    arrival_error = (t_on - t_geo) * 1e3  # ms
    arrival_error = np.nan_to_num(arrival_error, nan=0.0)
    arrival_spread = float(np.std(arrival_error))

    # (d) spectral deviation vs. 1/r-scaled reference.
    centers = signals.third_octave_centers(band_min, band_max)
    mags = signals.band_magnitudes(p_win, result.dt, centers, axis=0)  # (bands, n)
    r = np.linalg.norm(ray, axis=1)
    expected = mags[:, reference_index : reference_index + 1] * (r[reference_index] / r)
    with np.errstate(divide="ignore", invalid="ignore"):
        deviation_db = 20.0 * np.log10(mags / expected)
    deviation_db = np.nan_to_num(deviation_db, nan=0.0, posinf=0.0, neginf=-80.0)

    # (e) diffuseness.
    inst = np.abs(np.einsum("tn,ctn->ctn", p_win, v_win))
    total_flux = np.sqrt((inst**2).sum(axis=0)).sum(axis=0) * result.dt
    total_flux = np.where(total_flux == 0.0, 1.0, total_flux)
    diffuseness = 1.0 - np.linalg.norm(intensity, axis=0) / total_flux

    return PinnaMetrics(
        probe_positions=result.positions,
        similarity=similarity,
        shape_similarity=shape_similarity,
        level_re_ref_db=level_re_ref_db,
        incidence_deviation_deg=incidence_deviation,
        incidence_spread_deg=incidence_spread,
        arrival_error_ms=arrival_error,
        arrival_spread_ms=arrival_spread,
        band_centers=centers,
        spectral_deviation_db=deviation_db,
        diffuseness=np.clip(diffuseness, 0.0, 1.0),
    )


@dataclass
class PairedComparison:
    """With-filter vs. no-filter comparison on identical probe layouts."""

    residual_energy_db: FloatArray  # (n,) energy of (with - scaled/aligned without)
    band_difference_db: FloatArray  # (n_bands, n)
    band_centers: FloatArray

    def summary(self) -> dict[str, float]:
        return {
            "residual_energy_db_mean": float(np.mean(self.residual_energy_db)),
            "residual_energy_db_max": float(np.max(self.residual_energy_db)),
            "band_difference_db_rms": float(np.sqrt(np.mean(self.band_difference_db**2))),
        }


def compare_runs(
    with_filter: SimulationResult,
    without_filter: SimulationResult,
    band_min: float = 1000.0,
    band_max: float = 20_000.0,
) -> PairedComparison:
    if with_filter.p.shape != without_filter.p.shape:
        raise ValueError("paired runs must share n_steps and probe layout")
    a = with_filter.p
    b = without_filter.p
    # Per-probe least-squares scale of the reference onto the filtered run.
    denom = np.sum(b * b, axis=0)
    denom = np.where(denom == 0.0, 1.0, denom)
    scale = np.sum(a * b, axis=0) / denom
    residual = a - scale * b
    e_res = np.sum(residual**2, axis=0)
    e_ref = np.sum(b**2, axis=0) * scale**2
    e_ref = np.where(e_ref == 0.0, 1.0, e_ref)
    residual_db = 10.0 * np.log10(np.maximum(e_res / e_ref, 1e-12))

    centers = signals.third_octave_centers(band_min, band_max)
    mag_a = signals.band_magnitudes(a, with_filter.dt, centers, axis=0)
    mag_b = signals.band_magnitudes(b, without_filter.dt, centers, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        band_diff = 20.0 * np.log10(mag_a / mag_b)
    band_diff = np.nan_to_num(band_diff, nan=0.0, posinf=0.0, neginf=-80.0)
    return PairedComparison(
        residual_energy_db=residual_db,
        band_difference_db=band_diff,
        band_centers=centers,
    )
