# Pinna wavefront metrics

Computed by `headphone_sims.analysis.metrics.compute_metrics` on receiver
signals sampled 1.5 mm off the pinna surface (or on a planar disc array when no
pinna is present). Each probe uses a direct-arrival window
`[t_geo - 0.2 ms, t_geo + 0.5 ms]`. With an `ApertureSpec` passed (the
recommended mode for every headphone scene), `t_geo` is the propagation time
from the **nearest point of the radiating aperture**; without one it falls back
to the legacy driver-center distance.

## Near-field caveat (issue #6)

All headphone scenes here place probes 9-40 mm from apertures 40-90 mm across —
deep in the source near field. Point-source geometry (a single driver-center
ray, `1/r` level decay) does not describe such a field, so metrics built on it
are kept as **diagnostics only** and are marked below. The geometry-free
spectral-shape metrics are the primary spatial-spectral quantities.

| metric | definition | ideal |
| --- | --- | --- |
| similarity | amplitude-aware waveform match `2·max_xcorr/(Ex+Ey)` of windowed p vs. the ear-canal reference probe (pure delay allowed) | 1.0 |
| shape_similarity | legacy shape-only normalized cross-correlation | 1.0 |
| level_re_ref_db | broadband window RMS re the reference probe (raw ratio, no geometric correction) | 0 dB |
| incidence_deviation_deg *(diagnostic)* | angle between the time-integrated intensity direction `∫ p v dt` and the geometric driver-**center**→probe ray — a point-source reference large apertures do not obey | 0 |
| incidence_axial_deviation_deg | angle between the intensity direction and the aperture normal (the plane-wave-limit reference for large apertures) | 0 |
| incidence_spread_deg | std of intensity directions across probes | small |
| arrival_error_ms / arrival_spread_ms | envelope **onset** time (first crossing of 50% of the window envelope peak) − t_geo; std across probes. Geometry-referenced, so pass an aperture; contains a pulse-shape constant offset — the spread is the meaningful number | 0 |
| spectral_deviation_db *(legacy diagnostic)* | 1/3-octave band magnitude vs. the **1/r-scaled** reference probe — invalid in the near field of a large aperture; kept for continuity | 0 dB |
| spectral_shape_db / spectral_shape_spread_db | per-probe 1/3-oct band level minus the probe's own broadband mean, `H_shape_i(f) = L_i(f) − mean_f L_i(f)`; spread = per-band std across probes. No distance model — the near-field-safe spatial-spectral stability measure | 0 dB |
| diffuseness | `1 − |∫I dt| / ∫|I| dt` (0 = coherent transport, 1 = diffuse) | 0 |

`ApertureSpec(center, normal, radius | width+height)` describes the radiating
footprint; `nearest_distance()` gives per-probe first-arrival distances (rect
in-plane height axis = projection of scene +y, matching
`RectangularPistonSource`). `BuiltScene.aperture` carries the scene's own spec.

## Paired filter comparison

`compare_runs(with_filter, without_filter)` on identical probe layouts:

- residual_energy_db: energy of `p_with − s·p_without` (per-probe least-squares
  scale s) relative to the scaled reference — how much of the field the filter
  *changed* (scattering, echoes), independent of overall attenuation.
- band_difference_db: 1/3-octave magnitude change caused by the filter.

Known-answer tests: an ideal plane wave scores ~1.0 similarity, <3 deg
incidence deviation, <0.1 diffuseness; sign-random velocity gives diffuseness
>0.7; a 0.3-amplitude echo yields residual energy ≈ −10.5 dB. Broadband gain
differences leave spectral_shape_spread_db unchanged while genuinely different
per-probe spectra raise it (see `tests/unit/test_metrics.py`).
