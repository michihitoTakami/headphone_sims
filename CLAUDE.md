# CLAUDE.md

Think in English, answer in Japanese.

This file is an entry point, not the source of truth — stable documentation
lives in `docs/`. Start with `docs/README.md`.

## Commands

```bash
uv sync --all-groups          # environment
uv run pytest                 # unit tests (CPU, fast)
uv run pytest -m "slow or gpu"  # validation runs (GPU if available)
uv run ruff check src tests && uv run ruff format src tests
uv run mypy                   # strict, repo-wide
uv run headphone-sims run configs/example_hex_filter.yaml
```

## Rules

- Never push to `main` directly; use a branch + PR.
- Never use `--no-verify`.
- `docs/` holds only stable, adopted content; investigation notes go to Issues/PRs.
- Keep downloaded meshes in `data/` (gitignored) and simulation outputs in `runs/`.
