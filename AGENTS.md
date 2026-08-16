# AGENTS.md

Shared entry point for coding agents. This file is a pointer, not the source
of truth — the canonical documentation index is `docs/README.md`.

## Project in one paragraph

3D acoustic FDTD simulator (PyTorch, GPU) of headphone driver → front filter →
pinna wave propagation. Goal: quantify wavefront uniformity and incidence
angles across the pinna and the diffraction impact of perforated driver-front
filters (planar-magnetic grilles). Core physics in `src/headphone_sims/fdtd/`,
geometry in `src/headphone_sims/geometry/`, metrics in
`src/headphone_sims/analysis/`.

## Ground rules

- `uv` manages the environment; `uv run pytest` must pass before any commit.
- ruff + mypy strict are enforced (`uv run ruff check`, `uv run mypy`).
- Validation physics tests are marked `slow`/`gpu`; run them when touching
  `fdtd/` or `geometry/`.
- Documentation policy: `docs/` = stable decisions and physics only.
