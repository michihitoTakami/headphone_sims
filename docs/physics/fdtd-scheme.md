# FDTD scheme

Linear acoustics in air (c = 343 m/s, rho = 1.204 kg/m3) on a uniform staggered
grid (Yee-like), leapfrog in time:

```
v_n <- v_n - (dt / (rho dx)) * dp/dn      on interior cell faces
p   <- p   - (rho c^2 dt / dx) * div(v)   at cell centers
```

- Pressure at cell centers `(nx, ny, nz)`; velocity components on interior
  faces (`vx: (nx-1, ny, nz)` etc.). Cell (i,j,k) center is at `((i+.5)dx, ...)`.
- Time step from the 3D CFL limit: `dt = S dx / (c sqrt(3))`, Courant S = 0.9.
- The leapfrog stores p and v half a step apart; the naive energy sum
  oscillates a few percent but is bounded (verified in tests).

## Rigid solids

Cell-based occupancy; a face is masked (v_n = 0) when either adjacent cell is
solid. This is exact and airtight for voxel geometry, including one-voxel-thick
plates and diagonally touching cells (verified by the sealed-shell test).
Implementation: masks are folded with damping into one multiplier tensor per
velocity component, applied once per step.

## Absorbing boundary

Two absorbers, selected by `SceneConfig.absorber`:

- `"sponge"` (default): graded sponge over the outer `thickness` cells
  (default 30): multiplicative damping `exp(-sigma dt)` with
  `sigma = sigma_max * depth^3`, sigma_max = 6e4/s. Measured reflection
  < -40 dB (validation test).
- `"cpml"`: convolutional PML (Roden & Gedney recursive convolution,
  kappa = 1) on the pressure-gradient and velocity-divergence terms inside
  the boundary slabs; polynomial sigma grading designed for R = 1e-6, linear
  alpha ramp for grazing/evanescent stabilization. Measured reflection
  ~ -114 dB at 15 cells (vs sponge -60 dB at 30 cells on the same setup) —
  use for late-window metrics, long records (fine frequency resolution), or
  boundary-sensitivity checks. Memory: psi variables only in the slabs.

## Viscous bore losses (optional, per filter)

`FilterSpec(viscous_losses=True)` applies Maa/Crandall's tube resistance
`Phi = (8 eta / a^2) sqrt(1 + s^2/32)` (evaluated at
`viscous_eval_frequency`, default 7 kHz) as an equivalent-fluid momentum sink
`sigma = Phi / rho` on the bore air cells (`plate_bore` / `rect_plate_bore`;
the AMTS pattern excludes the carved-away air above its sloped top).
Viscous-only, frequency-fixed: no thermal boundary-layer loss, no reactive
correction — a lower bound on real thermoviscous damping, with the rigid
default as the zero-loss bound. Validated: the sigma-slab plane-wave
attenuation matches `exp(-sigma L / 2c)` within 5% (validation test), and the
Maa formula is unit-tested against its Poiseuille and boundary-layer limits.

## Sources

- `PointSource`: soft (additive) monopole on pressure.
- `PistonSource`: normal-velocity injection over a (possibly annular, tilted)
  disc. Default `hard=True` overwrites the face values — the correct model for
  a piston set into a rigid wall. The scene builder places a solid wall whose
  front surface lies on the source plane, so the piston faces are the wall's
  boundary faces. Validated against the analytic baffled-piston directivity
  2 J1(ka sin t)/(ka sin t) within 0.10 absolute up to 40 degrees at 10 kHz.
  A *soft* velocity source in free air radiates a dipole-like cos(theta)
  pattern instead — that is physics, not a bug.
- `RectangularPistonSource`: same hard-source semantics over a width x height
  rectangle (planar-magnetic diaphragm); height runs along scene +y.
- `RigidBodySource`: a voxelized rigid body (e.g. a dome diaphragm) translating
  along a fixed axis d. For pure translation every surface point moves with
  the body, so the face carrying grid component `axis` is set to
  `U(t) * d[axis]` — a **uniform weight per component**; the shape enters only
  through which faces are selected (the body's open boundary faces in the full
  scene solid), never through per-face normal weights. The open boundary z-face
  count of a convex bump on a baffle equals its projected disc area, so the
  total volume velocity equals the flat piston's (`U * pi R^2`) — dome and flat
  runs are level-matched and agree at low frequency by construction. After the
  waveform the faces clamp to zero: the body remains a rigid scatterer.
  Direct-sound metrics windows use the rim-plane center; a dome arrives up to
  `dome_depth/c` (~0.02 ms) early, well inside the 0.2 ms pre-window.

## Injection ordering

Per step: `step_velocity` (grad + masks) → velocity sources → `step_pressure`
(div + damping) → pressure sources → receiver sampling. Hard velocity sources
must run after the masks, hence the two-half-step API.

## Receivers

Trilinear interpolation of p and each staggered v component onto probe
positions. The p/v half-step time offset (~0.4 us at dx = 0.5 mm) is
midpoint-corrected at analysis time (`compute_metrics` averages consecutive v
samples onto p's time grid before forming intensities). Probes are accepted
only when every corner of ALL FOUR interpolation stencils (p and the three
staggered v components) reads an air cell — a velocity corner on a
rigid-masked face records a forced zero and biases intensity metrics
(`probe_stencil_air_mask`); achieved surface clearance is reported per build.

## Default resolution

dx = 0.5 mm → 34 points per wavelength at 20 kHz (dispersion error < 0.1%);
resolves >= 1 mm filter holes with >= 2 cells. Fine mode dx = 0.3 mm for
convergence checks. Watch out: a plate whose mid-plane is exactly on a cell
boundary quantizes its thickness by up to one layer — keep part planes off the
lattice or check voxel counts (see `plate()` porosity return).
