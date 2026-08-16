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

Graded sponge over the outer `thickness` cells (default 30): multiplicative
damping `exp(-sigma dt)` with `sigma = sigma_max * depth^3`, sigma_max = 6e4/s.
Measured reflection < -40 dB (validation test). CPML is a possible later
upgrade if late-window metrics need a cleaner floor.

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

## Injection ordering

Per step: `step_velocity` (grad + masks) → velocity sources → `step_pressure`
(div + damping) → pressure sources → receiver sampling. Hard velocity sources
must run after the masks, hence the two-half-step API.

## Receivers

Trilinear interpolation of p and each staggered v component onto probe
positions; the p/v half-step time offset (~0.4 us at dx = 0.5 mm) is left
uncorrected (negligible in the audio band).

## Default resolution

dx = 0.5 mm → 34 points per wavelength at 20 kHz (dispersion error < 0.1%);
resolves >= 1 mm filter holes with >= 2 cells. Fine mode dx = 0.3 mm for
convergence checks. Watch out: a plate whose mid-plane is exactly on a cell
boundary quantizes its thickness by up to one layer — keep part planes off the
lattice or check voxel counts (see `plate()` porosity return).
