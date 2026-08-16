# headphone-sims

3D acoustic FDTD simulator (PyTorch/GPU) for analyzing how a headphone driver's
wavefront reaches the pinna — including diffraction through perforated front
filters (e.g., planar-magnetic grilles) and its impact on spatial cues.

## What it answers

- Does the whole pinna receive a nearly identical waveform and incidence angle?
- Does diffraction through complex driver-front filters degrade spatial information?
- What driver size, mounting angle, distance, and filter design deliver the most
  natural wavefront across the pinna?

## Quick start

```bash
uv sync --all-groups
uv run pytest                         # unit tests (CPU)
uv run pytest -m gpu                  # GPU validation tests
uv run headphone-sims fetch-data hutubs --subjects 1
uv run headphone-sims run configs/example_hex_filter.yaml
```

Simulation outputs land in `runs/<timestamp>_<name>/` with the resolved config,
recorded receiver signals, metrics JSON, and an HTML report.

## Method

Staggered-grid pressure–velocity leapfrog FDTD (Yee-like) on a uniform voxel
grid, default dx = 0.5 mm (34 points per wavelength at 20 kHz), rigid solids via
face velocity masks, absorbing sponge outer boundary. See `docs/` for the
physics, metric definitions, and adopted decisions.

## Geometry data

Pinna/head meshes are downloaded (not redistributed here) from:

- HUTUBS head meshes — Brinkmann et al. 2019, CC BY 4.0, DOI 10.14279/depositonce-8487
- Aachen high-resolution KEMAR scan — RWTH ITA, CC BY 4.0, DOI 10.18154/RWTH-2020-11307
- VIKING pinna scans — Spagnol et al. 2019, CC BY 4.0, Zenodo 4160401

## License

MIT
