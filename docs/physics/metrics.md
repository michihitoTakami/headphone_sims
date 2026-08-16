# Pinna wavefront metrics

Computed by `headphone_sims.analysis.metrics.compute_metrics` on receiver
signals sampled 1.5 mm off the pinna surface (or on a planar disc array when no
pinna is present). Each probe uses a direct-arrival window
`[t_geo - 0.2 ms, t_geo + 0.5 ms]` with `t_geo = |probe - driver_center| / c`.

| metric | definition | ideal |
| --- | --- | --- |
| similarity | max normalized cross-correlation of windowed p vs. the ear-canal reference probe (pure delay allowed) | 1.0 |
| incidence_deviation_deg | angle between the time-integrated intensity direction `∫ p v dt` and the geometric driver→probe ray | 0 |
| incidence_spread_deg | std of intensity directions across probes | small |
| arrival_error_ms / arrival_spread_ms | Hilbert-envelope peak time − t_geo; std across probes | 0 |
| spectral_deviation_db | 1/3-octave band magnitude vs. the 1/r-scaled reference probe | 0 dB |
| diffuseness | `1 − |∫I dt| / ∫|I| dt` (0 = coherent transport, 1 = diffuse) | 0 |

## Paired filter comparison

`compare_runs(with_filter, without_filter)` on identical probe layouts:

- residual_energy_db: energy of `p_with − s·p_without` (per-probe least-squares
  scale s) relative to the scaled reference — how much of the field the filter
  *changed* (scattering, echoes), independent of overall attenuation.
- band_difference_db: 1/3-octave magnitude change caused by the filter.

Known-answer tests: an ideal plane wave scores ~1.0 similarity, <3 deg
incidence deviation, <0.1 diffuseness; sign-random velocity gives diffuseness
>0.7; a 0.3-amplitude echo yields residual energy ≈ −10.5 dB.
