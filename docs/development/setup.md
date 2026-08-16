# Setup and workflow

## Environment

```bash
uv sync --all-groups
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

GPU: any CUDA device supported by the installed torch wheel (PyPI wheels bundle
the CUDA runtime; RTX 5060 Ti / sm_120 verified with torch 2.13 cu130).

## Tests

- `uv run pytest` — unit tests, CPU, seconds.
- `uv run pytest -m "slow or gpu"` — physics validation (free-field 1/r,
  sponge reflection, baffled-piston directivity, CPU/GPU parity). Run these
  whenever `fdtd/` or `geometry/` changes.

## Running experiments

```bash
uv run headphone-sims fetch-data hutubs --subjects 1
uv run headphone-sims run configs/example_hex_filter.yaml
uv run headphone-sims bench            # ms/step + VRAM on this machine
```

Outputs land in `runs/<timestamp>_<name>/` (gitignored): resolved config,
`signals.npz`, `metrics.json`, `report/report.html`.

## Quality gates

pre-commit: ruff (+format), YAML/JSON checks, large-file guard, detect-secrets.
pre-push: mypy strict (repo-wide) + unit tests. Never `--no-verify`.
