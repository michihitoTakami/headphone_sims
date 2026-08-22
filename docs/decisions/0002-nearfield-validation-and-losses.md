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

6. **Deterministic domain sizing.** A float-noise ceil artifact
   (`ceil(300.0000000004) = 301`) could put the whole assembly half a cell
   off the voxel lattice depending on part dimensions; fixed with an epsilon
   guard. Scene-realization (voxelization-phase) noise on the canal C(f) at
   dx = 0.5 mm was measured at 1.4-1.9 dB RMS for a pure half-cell shift and
   up to ~6.5 dB RMS for a re-realization with domain resize — larger than
   the boundary error, and the noise floor for any tooth-level C(f) claim.

## Study outcomes (2026-08, details in Issue #6 / runs/*.json)

- Boundary: the production 30-cell sponge agrees with the C-PML arbiter at
  identical lattice phase within 1.1-1.7 dB RMS core C(f) for 3 of 4 models;
  the exception is DCA-AMTS, whose comb swing reads 4 dB too shallow under
  the sponge (11.9 dB published vs 15.9 dB boundary-clean). The pp1
  boundary-clean comb ranking becomes LCD 20.7 > Z1R 17.4 > DCA 15.9 >
  DX 13.6 dB — DCA loses the "shallowest comb" title to DX. Comb-dependent
  claims should be re-based on absorber="cpml".
- Grid convergence (dx 0.5/0.4/0.3, z1r + dca2): the headline spatial
  quantities are robust — incident illumination maps correlate >= 0.98
  between resolutions, voxelized porosity is dx-stable within 2%, and the
  spatio-spectral spread sigma(L10k - L8k) holds at ~4 dB (DCA) vs ~1 dB
  (reference) at every dx. NOT converged at dx = 0.5: canal-C(f) tooth
  structure (swing drifts 15.9->19.1 / 11.9->17.8 dB toward the
  boundary-clean values) and the absolute level of the amplitude-aware
  similarity aggregate (z1r core 0.49->0.74). Staircase-area correction
  (PFFDTD-style) gate is therefore TRIGGERED for canal-spectrum work —
  scheduled as a follow-up together with a CPML re-baseline of the run sets.
- Viscous bore losses on DCA at dx 0.5: comb swing 11.9 -> 11.0 dB, deepest
  notch -5.7 -> -5.0 dB — real but small softening; the rigid model's AMTS
  absorption underestimate is dominated by the boundary/resolution effects
  above, not by the missing viscosity.

## Consequences

- Published pp1/panel numbers remain reproducible: legacy metrics are
  unchanged when no aperture is passed; new metrics are additive.
- Not modeled, still: thermal boundary-layer losses, air absorption
  (negligible over <50 mm paths below 20 kHz), finite surface impedance of
  skin, staircase surface-area correction (gate triggered, see above).
