"""Known-answer tests for pinna metrics on synthetic fields."""

import numpy as np
import pytest

from headphone_sims.analysis.metrics import compare_runs, compute_metrics
from headphone_sims.fdtd.simulation import SimulationResult

RHO_C = 1.204 * 343.0
C = 343.0


def _plane_wave_result(
    n_probes: int = 20, jitter: float = 0.0, seed: int = 0
) -> tuple[SimulationResult, tuple[float, float, float]]:
    """Plane wave traveling +z past probes on the z=60mm plane.

    The driver center is far below on the axis so the geometric rays are
    nearly parallel to +z; p and v are plane-wave consistent (v = p/(rho c) z).
    """
    rng = np.random.default_rng(seed)
    dt = 2e-6
    n_steps = 800
    driver_center = (0.0, 0.0, -0.3)  # far below: rays within ~2.5 deg of +z
    pos = np.zeros((n_probes, 3))
    pos[:, 0] = rng.uniform(-15e-3, 15e-3, n_probes)
    pos[:, 1] = rng.uniform(-15e-3, 15e-3, n_probes)
    pos[:, 2] = 60e-3

    t = np.arange(n_steps) * dt
    fc = 8000.0
    p = np.zeros((n_steps, n_probes))
    r = np.linalg.norm(pos - np.asarray(driver_center), axis=1)
    for i in range(n_probes):
        t0 = r[i] / C + jitter * rng.standard_normal()
        env = np.exp(-0.5 * ((t - t0) / 80e-6) ** 2)
        p[:, i] = env * np.sin(2 * np.pi * fc * (t - t0))
    v = np.zeros((3, n_steps, n_probes))
    v[2] = p / RHO_C
    wf = np.zeros(n_steps)
    wf[0] = 1.0
    result = SimulationResult(dt=dt, dx=0.5e-3, positions=pos, p=p, v=v, source_waveform=wf)
    return result, driver_center


def test_plane_wave_is_ideal() -> None:
    result, center = _plane_wave_result()
    m = compute_metrics(result, center, reference_index=0)
    assert m.similarity.min() > 0.99  # amplitude-aware: shape AND level match
    assert m.shape_similarity.min() > 0.99
    assert np.abs(m.level_re_ref_db).max() < 0.5
    assert m.incidence_deviation_deg.max() < 3.0
    assert m.incidence_spread_deg < 2.0
    # Onset leads the envelope peak by a pulse-shape constant (same for all
    # probes); the meaningful quantity is the spread.
    assert np.ptp(m.arrival_error_ms) < 0.02
    assert m.arrival_spread_ms < 0.02
    assert m.diffuseness.max() < 0.1
    # Plane wave has no 1/r decay; deviation vs the 1/r-scaled reference stays
    # small because probe distances differ by well under 1% at 0.3 m.
    assert np.abs(m.spectral_deviation_db).max() < 1.0


def test_incoherent_field_is_diffuse() -> None:
    result, center = _plane_wave_result()
    rng = np.random.default_rng(1)
    # Replace velocity with sign-flipping noise: p and v uncorrelated.
    result.v = rng.standard_normal(result.v.shape) * float(np.abs(result.p).max()) / RHO_C
    m = compute_metrics(result, center, reference_index=0)
    assert m.diffuseness.mean() > 0.7


def test_arrival_jitter_detected() -> None:
    result, center = _plane_wave_result(jitter=50e-6, seed=2)
    m = compute_metrics(result, center, reference_index=0)
    assert m.arrival_spread_ms > 0.02


def test_compare_runs_identical_is_silent() -> None:
    result, _ = _plane_wave_result()
    cmp = compare_runs(result, result)
    assert cmp.residual_energy_db.max() < -100.0
    assert np.abs(cmp.band_difference_db).max() < 1e-6


def test_compare_runs_detects_added_reflection() -> None:
    result, _ = _plane_wave_result()
    modified, _ = _plane_wave_result()
    echo = np.roll(modified.p, 150, axis=0) * 0.3
    modified.p = modified.p + echo
    cmp = compare_runs(modified, result)
    # 0.3 amplitude echo -> residual energy ~ -10.5 dB.
    assert -13.0 < cmp.residual_energy_db.mean() < -8.0


def test_amplitude_scaling_penalized() -> None:
    """A half-amplitude probe has perfect shape but similarity ~0.80."""
    result, center = _plane_wave_result()
    result.p[:, 5] *= 0.5
    result.v[:, :, 5] *= 0.5
    m = compute_metrics(result, center, reference_index=0)
    assert m.shape_similarity[5] > 0.99  # shape unchanged
    assert m.similarity[5] == pytest.approx(2 * 0.5 / (1 + 0.25), abs=0.02)  # 0.80
    assert m.level_re_ref_db[5] == pytest.approx(-6.0, abs=0.3)
    assert m.summary()["level_min_db"] < -5.5
