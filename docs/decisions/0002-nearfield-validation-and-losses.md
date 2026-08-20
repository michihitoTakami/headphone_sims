# ADR 0002: Near-field analysis references, viscous bore losses, C-PML

Date: 2026-08. Status: adopted (issue #6).

## Context

Every headphone scene places probes 9-40 mm from apertures 40-90 mm across —
deep in the source near field — while several metrics assumed point-source
geometry (driver-center rays, 1/r level decay). Issue #6 also flagged
unverified numerics: no grid-convergence study (the AMTS between-row wall is
~1.1 cells at dx = 0.5 mm), a probe air check covering only the pressure
stencil, and a single coarse-grid sponge validation. Separately, the rigid
lossless solver was known to understate AMTS absorption (no thermoviscous
neck losses).

## Decisions

1. **Aperture-aware metric geometry.** `ApertureSpec` (disc / rect footprint)
   supplies nearest-point first-arrival distances for direct windows and
   arrival metrics, plus the aperture normal for an axial incidence
   deviation. Driver-center-ray and 1/r-based metrics remain available but
   are documented as diagnostics.
2. **Geometry-free spectral shape is the primary spatial-spectral metric.**
   `spectral_shape_db` (per-probe band level minus the probe's own broadband
   mean) and its per-probe spread replace 1/r-referenced spectral deviation
   for near-field claims.
3. **Stencil-aware probe acceptance.** Probes must have every corner of the
   pressure AND the three staggered velocity interpolation stencils in air
   (velocity faces adjacent to solid are rigid-masked zeros); out-of-grid
   stencils are rejected, not clamped. The paired-run probe override is
   verified rather than trusted. Audit of the published 8-subject panel:
   0/8213 probes failed the old pressure-only check, 163 (2.0%) fail the
   velocity-stencil check (biased intensity metrics only; canal TF/comb use
   pressure and were unaffected).
4. **Viscous bore losses (opt-in per filter).** Maa/Crandall tube resistance
   at a band-center frequency, applied as an equivalent-fluid momentum sink
   on bore air cells. Viscous-only lower bound; rigid remains the zero-loss
   bound. Validated against the plane-wave sigma-slab attenuation law (5%)
   and the formula's analytic limits.
5. **C-PML absorber (opt-in per scene).** Roden & Gedney recursive
   convolution, kappa = 1, R = 1e-6 grading. Measured ~-114 dB at 15 cells
   vs the sponge's -60 dB at 30 cells on the same setup. The sponge stays
   the default until the published run set is regenerated.

## Consequences

- Published pp1/panel numbers remain reproducible: legacy metrics are
  unchanged when no aperture is passed; new metrics are additive.
- Convergence (dx 0.5/0.4/0.3), boundary-sensitivity, and probe-audit
  results live in Issue #6 and `runs/*.json`; docs record only the adopted
  mechanisms above.
- Not modeled, still: thermal boundary-layer losses, air absorption
  (negligible over <50 mm paths below 20 kHz), finite surface impedance of
  skin, staircase surface-area correction (gated on the convergence result).
