"""Pinna wavefront metrics.

All metrics operate on a :class:`headphone_sims.fdtd.simulation.SimulationResult`
plus scene metadata (driver center, reference probe). The direct-arrival time
window per probe is [t_geo - pre, t_geo + post]; with an :class:`ApertureSpec`
given, t_geo is the propagation time from the NEAREST point of the radiating
aperture (correct in the near field of a large source), otherwise the legacy
driver-center distance.

Near-field caveat (issue #6): every probe here sits at 9-40 mm from apertures
40-90 mm across, so point-source geometry (a single driver-center ray, 1/r
level decay) does not describe the field. Metrics that depend on it are kept
as DIAGNOSTICS and marked below; the geometry-free spectral-shape metrics are
the primary spatial-spectral quantities.

Metrics (per probe unless noted):
- similarity: AMPLITUDE-AWARE waveform match vs. the reference probe
  (2*max_xcorr/(Ex+Ey); 1.0 = same waveform AND same level up to a delay;
  a probe at -10 dB scores ~0.58 even with a perfect shape)
- shape_similarity: the legacy shape-only normalized cross-correlation
- level_re_ref_db: broadband direct-window RMS level relative to the
  reference probe (raw ratio, no geometric correction)
- incidence_deviation_deg [diagnostic]: angle of the time-integrated
  intensity to the geometric driver-CENTER ray — a point-source reference
  that large apertures do not obey; incidence_axial_deviation_deg gives the
  plane-wave-limit alternative (angle to the aperture normal)
- arrival_spread_ms (scalar) [diagnostic when no aperture is given]: std
  over probes of (envelope ONSET time - t_geo); onset = first crossing of
  50% of the window's envelope peak
- spectral_deviation_db (n_bands, n_probes) [legacy diagnostic]: band
  magnitude relative to the 1/r-scaled reference probe; 1/r is a
  point-source law and over/under-corrects in the near field — prefer
  spectral_shape_spread_db
- spectral_shape_db / spectral_shape_spread_db: per-probe band levels with
  the probe's own broadband mean removed (H_shape_i(f) = L_i(f) -
  mean_f L_i(f)) and their per-band std across probes — geometry-free, the
  near-field-safe spatial-spectral stability measure
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


@dataclass(frozen=True)
class ApertureSpec:
    """Radiating-aperture footprint on the source plane.

    With the aperture size comparable to (or larger than) the source-ear
    distance, the first direct arrival at a probe comes from the NEAREST
    point of the aperture, not its center, and the expected incidence tends
    toward the aperture normal (plane-wave limit). Passing this to
    :func:`compute_metrics` makes the direct-window/arrival references
    nearest-point based instead of center-ray based.

    Rect apertures follow the scene convention (see
    ``fdtd.sources.RectangularPistonSource``): the in-plane height axis is
    the projection of scene +y onto the aperture plane.
    """

    center: tuple[float, float, float]
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    radius: float | None = None  # disc aperture
    width: float | None = None  # rect aperture (with height)
    height: float | None = None

    def unit_normal(self) -> FloatArray:
        n = np.asarray(self.normal, dtype=np.float64)
        return n / np.linalg.norm(n)

    def nearest_distance(self, positions: FloatArray) -> FloatArray:
        """Distance from each position (n, 3) to the closest aperture point."""
        n = self.unit_normal()
        rel = np.asarray(positions, dtype=np.float64) - np.asarray(self.center)
        axial = rel @ n
        lat = rel - axial[:, None] * n[None, :]
        if self.width is not None and self.height is not None:
            e_h = np.array([0.0, 1.0, 0.0]) - n[1] * n
            if np.linalg.norm(e_h) < 1e-9:
                e_h = np.array([1.0, 0.0, 0.0]) - n[0] * n
            e_h /= np.linalg.norm(e_h)
            e_w = np.cross(n, e_h)
            d_w = np.maximum(np.abs(lat @ e_w) - self.width / 2.0, 0.0)
            d_h = np.maximum(np.abs(lat @ e_h) - self.height / 2.0, 0.0)
            lat_out_sq = d_w**2 + d_h**2
        else:
            d_r = np.maximum(
                np.linalg.norm(lat, axis=1) - (self.radius if self.radius is not None else 0.0),
                0.0,
            )
            lat_out_sq = d_r**2
        return np.asarray(np.sqrt(axial**2 + lat_out_sq))


@dataclass
class PinnaMetrics:
    probe_positions: FloatArray  # (n, 3)
    similarity: FloatArray  # (n,) amplitude-aware match vs reference
    shape_similarity: FloatArray  # (n,) legacy shape-only xcorr
    level_re_ref_db: FloatArray  # (n,) broadband level re the reference probe
    # Diagnostic: deviation from the driver-CENTER ray (point-source geometry).
    incidence_deviation_deg: FloatArray  # (n,)
    # Deviation from the aperture normal — the plane-wave-limit reference for
    # apertures comparable to the source-ear distance.
    incidence_axial_deviation_deg: FloatArray  # (n,)
    incidence_spread_deg: float  # dispersion of intensity directions across probes
    # Onset time minus geometric time; contains a pulse-shape constant offset
    # (onset leads the envelope peak), so the SPREAD is the meaningful number.
    arrival_error_ms: FloatArray
    arrival_spread_ms: float
    band_centers: FloatArray  # (n_bands,)
    # Legacy diagnostic: band level re the 1/r-scaled reference probe. The 1/r
    # law is point-source geometry — invalid in the near field of a large
    # aperture; use spectral_shape_* for spatial-spectral stability.
    spectral_deviation_db: FloatArray  # (n_bands, n)
    # Geometry-free spectral shape: per-probe band level minus the probe's own
    # broadband mean (no distance model), and its per-band std across probes.
    spectral_shape_db: FloatArray  # (n_bands, n)
    spectral_shape_spread_db: FloatArray  # (n_bands,)
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
            "incidence_axial_deviation_deg_mean": float(
                np.mean(self.incidence_axial_deviation_deg)
            ),
            "incidence_spread_deg": self.incidence_spread_deg,
            "arrival_spread_ms": self.arrival_spread_ms,
            "spectral_deviation_db_rms": float(np.sqrt(np.mean(self.spectral_deviation_db**2))),
            "spectral_shape_spread_db_rms": float(
                np.sqrt(np.mean(self.spectral_shape_spread_db**2))
            ),
            "spectral_shape_spread_db_max": float(np.max(self.spectral_shape_spread_db)),
            "diffuseness_mean": float(np.mean(self.diffuseness)),
            "diffuseness_max": float(np.max(self.diffuseness)),
        }


def direct_window_masks(
    result: SimulationResult,
    driver_center: FloatArray,
    pre: float,
    post: float,
    aperture: ApertureSpec | None = None,
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Per-probe direct-arrival window masks (n_steps, n) and t_geo (n,).

    With ``aperture`` given, t_geo uses the nearest-point distance over the
    radiating footprint — the correct first-arrival reference when the
    aperture size is comparable to the probe distance. Without it, the legacy
    driver-center distance (a dome diaphragm radiates from up to dome_depth
    closer, <= ~0.02 ms early, well inside ``pre``).
    """
    if aperture is not None:
        r = aperture.nearest_distance(result.positions)
    else:
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
    band_max: float = 12_500.0,
    band: tuple[float, float] | None = (1000.0, 12_500.0),
    aperture: ApertureSpec | None = None,
) -> PinnaMetrics:
    """``band`` zero-phase band-limits p and v before all metrics — the
    default 1-12.5 kHz drops the >14 kHz region, which contributes little to
    spatial hearing (pinna cues live mainly in ~4-12 kHz). Pass None for the
    legacy full-bandwidth behavior, or e.g. (5000, 10000) to focus on the
    core spatial-cue band.

    ``aperture`` switches the direct-window/arrival geometry to nearest-point
    distances over the radiating footprint and supplies the aperture normal
    for incidence_axial_deviation_deg — pass it whenever the aperture size is
    not small against the probe distance (all headphone scenes here).
    """
    center = np.asarray(driver_center, dtype=np.float64)
    n_probes = result.p.shape[1]
    t_geo, masks = direct_window_masks(result, center, window_pre, window_post, aperture)

    p_all = result.p
    # Leapfrog stagger: the recorded v lags p by dt/2 (sampled after the
    # velocity half-step). Midpoint-average v onto p's time samples so the
    # intensity p*v carries no systematic phase error (~1.4 deg at 10 kHz).
    v_all = result.v.copy()
    v_all[:, :-1, :] = 0.5 * (v_all[:, :-1, :] + v_all[:, 1:, :])
    if band is not None:
        p_all = signals.bandpass_zero_phase(p_all, result.dt, band[0], band[1], axis=0)
        v_all = signals.bandpass_zero_phase(v_all, result.dt, band[0], band[1], axis=1)
    p_win = p_all * masks  # (n_steps, n)
    v_win = v_all * masks[None]  # (3, n_steps, n)

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
    axis_dir = aperture.unit_normal() if aperture is not None else np.array([0.0, 0.0, 1.0])
    cos_axial = np.clip(i_dir.T @ axis_dir, -1.0, 1.0)
    incidence_axial_deviation = np.degrees(np.arccos(cos_axial))
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

    # (d) legacy diagnostic: spectral deviation vs. 1/r-scaled reference —
    # point-source geometry, kept for continuity with earlier runs only.
    centers = signals.third_octave_centers(band_min, band_max)
    mags = signals.band_magnitudes(p_win, result.dt, centers, axis=0)  # (bands, n)
    r = np.linalg.norm(ray, axis=1)
    expected = mags[:, reference_index : reference_index + 1] * (r[reference_index] / r)
    with np.errstate(divide="ignore", invalid="ignore"):
        deviation_db = 20.0 * np.log10(mags / expected)
    deviation_db = np.nan_to_num(deviation_db, nan=0.0, posinf=0.0, neginf=-80.0)

    # (d2) geometry-free spectral shape (issue #6): each probe's band levels
    # with its own broadband mean removed — no 1/r, no distance model. The
    # per-band std across probes measures spatio-spectral stability without
    # assuming any propagation law.
    band_db = 20.0 * np.log10(np.maximum(mags, 1e-30))
    shape_db = band_db - band_db.mean(axis=0, keepdims=True)
    shape_spread_db = shape_db.std(axis=1)

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
        incidence_axial_deviation_deg=incidence_axial_deviation,
        incidence_spread_deg=incidence_spread,
        arrival_error_ms=arrival_error,
        arrival_spread_ms=arrival_spread,
        band_centers=centers,
        spectral_deviation_db=deviation_db,
        spectral_shape_db=shape_db,
        spectral_shape_spread_db=shape_spread_db,
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
