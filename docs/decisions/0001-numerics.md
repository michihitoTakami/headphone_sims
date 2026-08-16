# 0001 — Numerical scheme and parameters

Adopted 2026-08.

## Decision

Staggered-grid pressure–velocity leapfrog FDTD in PyTorch (eager), float32,
default dx = 0.5 mm, Courant 0.9, 30-cell cubic-graded sponge (sigma_max
6e4 1/s).

## Why

- Face-based velocity masks give exact, airtight rigid boundaries — essential
  for perforated plates 1–2 cells thick. Pseudospectral methods (k-Wave/j-Wave)
  handle binary rigid masks poorly (Gibbs), BEM (Mesh2HRTF) cannot practically
  mesh sub-mm perforations at 20 kHz.
- The velocity field is needed anyway for intensity-based incidence metrics.
- No existing open-source headphone/pinna time-domain simulator exists; the
  kernel is small (~200 lines) while the reusable knowledge (scheme, staircase
  caveats, validation battery) came from PFFDTD, deepwave, and k-Wave.

## Measured performance (RTX 5060 Ti 16 GB, torch 2.13 cu130)

- 39 M cells (360x360x300, production headphone scene at dx = 0.5 mm):
  ~20 ms/step, ~81 s per 4000-step (3 ms) run, ~2 GB VRAM.
- Unit-scale scenes run in seconds on CPU.

## Alternatives kept in reserve

- CPML (port from deepwave) if sponge floor (−40 dB) limits late-window metrics.
- torch.compile / CUDA graphs if kernel-launch overhead matters; eager is
  canonical (triton support on sm_120 was unverified at adoption time).
- PFFDTD-style staircase surface-area correction if hole-edge staircasing
  shifts filter resonances (check with dx = 0.3 mm convergence runs first).
