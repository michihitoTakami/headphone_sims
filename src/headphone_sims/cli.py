"""Command-line interface: run experiments, fetch datasets, benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> int:
    from headphone_sims.experiments.config import load_config
    from headphone_sims.experiments.runner import run_experiment

    for config_path in args.config:
        config = load_config(config_path)
        run_experiment(config)
    return 0


def _cmd_fetch_data(args: argparse.Namespace) -> int:
    from headphone_sims.geometry import datasets

    data_dir = Path(args.data_dir)
    if args.dataset == "hutubs":
        for subject in args.subjects or [1]:
            path = datasets.fetch_hutubs_mesh(int(subject), data_dir)
            print(f"{path}  ({datasets.ATTRIBUTIONS['hutubs']})")
    elif args.dataset == "kemar":
        path = datasets.fetch_kemar(data_dir)
        print(f"{path}  ({datasets.ATTRIBUTIONS['kemar']})")
    elif args.dataset == "viking":
        for subject in args.subjects or ["A"]:
            path = datasets.fetch_viking_pinna(str(subject), data_dir)
            print(f"{path}  ({datasets.ATTRIBUTIONS['viking']})")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    import time

    import torch

    from headphone_sims.fdtd.boundaries import SpongeConfig, attach_damping
    from headphone_sims.fdtd.kernel import FdtdState, step
    from headphone_sims.grid import Grid

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    grid = Grid.create(tuple(args.shape), dx=args.dx * 1e-3)
    state = FdtdState.zeros(grid, device=device)
    attach_damping(state, sponge=SpongeConfig())
    for _ in range(5):
        step(state)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step(state)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / args.steps * 1e3
    print(f"grid {grid.shape} = {grid.n_cells / 1e6:.1f}M cells on {device}")
    print(f"{ms:.2f} ms/step -> {ms * 4000 / 1e3:.0f} s per 4000-step run")
    if device == "cuda":
        print(f"VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="headphone-sims", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run experiment YAML config(s)")
    p_run.add_argument("config", nargs="+", help="path(s) to experiment YAML")
    p_run.set_defaults(func=_cmd_run)

    p_fetch = sub.add_parser("fetch-data", help="download public pinna/head meshes")
    p_fetch.add_argument("dataset", choices=["hutubs", "kemar", "viking"])
    p_fetch.add_argument("--subjects", nargs="*", help="HUTUBS numbers / VIKING letters")
    p_fetch.add_argument("--data-dir", default="data")
    p_fetch.set_defaults(func=_cmd_fetch_data)

    p_bench = sub.add_parser("bench", help="benchmark the FDTD kernel")
    p_bench.add_argument("--shape", type=int, nargs=3, default=[360, 360, 300])
    p_bench.add_argument("--dx", type=float, default=0.5, help="grid spacing in mm")
    p_bench.add_argument("--steps", type=int, default=100)
    p_bench.add_argument("--device", default=None)
    p_bench.set_defaults(func=_cmd_bench)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
