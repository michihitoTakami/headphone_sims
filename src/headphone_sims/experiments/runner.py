"""Run experiments into reproducible run directories.

Layout: runs/<timestamp>_<name>/
    config.yaml     resolved configuration
    signals.npz     receiver p/v, probe positions, source waveform
    metrics.json    per-run summary (+ paired comparison when requested)
    report/         PNG figures + report.html
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from headphone_sims.analysis.metrics import PinnaMetrics, compare_runs, compute_metrics
from headphone_sims.experiments.config import ExperimentConfig, config_to_dict, save_config
from headphone_sims.fdtd.simulation import SimulationResult
from headphone_sims.geometry.scene import BuiltScene, build_scene
from headphone_sims.viz.report import write_report
from headphone_sims.viz.scene_view import save_scene_views
from headphone_sims.viz.slices import save_slice_animation
from headphone_sims.viz.surface import save_probe_heatmap


def _save_signals(path: Path, result: SimulationResult) -> None:
    np.savez_compressed(
        path,
        p=result.p.astype(np.float32),
        v=result.v.astype(np.float32),
        positions=result.positions,
        source_waveform=result.source_waveform,
        dt=result.dt,
        dx=result.dx,
    )


def _figures(
    run_dir: Path,
    built: BuiltScene,
    result: SimulationResult,
    metrics: PinnaMetrics,
) -> list[Path]:
    report_dir = run_dir / "report"
    figs = [
        save_scene_views(
            built.solid,
            result.dx,
            built.driver_center,
            metrics.probe_positions,
            report_dir / "scene_geometry.png",
        ),
        save_probe_heatmap(
            metrics.probe_positions,
            metrics.similarity,
            report_dir / "similarity.png",
            title="Waveform similarity vs. reference",
            label="max normalized xcorr",
            vmin=0.0,
            vmax=1.0,
        ),
        save_probe_heatmap(
            metrics.probe_positions,
            metrics.incidence_deviation_deg,
            report_dir / "incidence_deviation.png",
            title="Incidence angle deviation from geometric ray",
            label="degrees",
            cmap="magma",
        ),
        save_probe_heatmap(
            metrics.probe_positions,
            metrics.arrival_error_ms,
            report_dir / "arrival_error.png",
            title="Arrival-time error (envelope peak - r/c)",
            label="ms",
            cmap="coolwarm",
        ),
        save_probe_heatmap(
            metrics.probe_positions,
            metrics.diffuseness,
            report_dir / "diffuseness.png",
            title="Diffuseness (0 = coherent, 1 = diffuse)",
            label="psi",
            vmin=0.0,
            vmax=1.0,
            cmap="inferno",
        ),
    ]
    if result.snapshots:
        figs.append(
            save_slice_animation(result.snapshots, result.dt, result.dx, report_dir / "field.gif")
        )
    return figs


def run_experiment(config: ExperimentConfig, run_dir: Path | None = None) -> Path:
    """Execute one experiment (plus optional no-filter reference) and report."""
    if run_dir is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(config.output_dir) / f"{stamp}_{config.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")

    built = build_scene(config.scene, device=config.device)
    ppw = built.grid.points_per_wavelength(20_000.0)
    print(
        f"[{config.name}] grid {built.grid.shape} ({built.grid.n_cells / 1e6:.1f}M cells), "
        f"{ppw:.0f} ppw @20kHz, {built.simulation.n_steps} steps, "
        f"device={built.simulation.device.type}"
    )
    if built.porosities:
        print(f"[{config.name}] filter porosities: {[f'{x:.2f}' for x in built.porosities]}")
    result = built.simulation.run()
    _save_signals(run_dir / "signals.npz", result)

    metrics = compute_metrics(result, built.driver_center, built.reference_index)
    payload: dict[str, Any] = {"metrics": metrics.summary()}

    if config.compare_without_filters and config.scene.filters:
        ref_scene = dataclasses.replace(config.scene, filters=())
        ref_built = build_scene(
            ref_scene, device=config.device, probes_override=built.probe_positions
        )
        ref_result = ref_built.simulation.run()
        _save_signals(run_dir / "signals_nofilter.npz", ref_result)
        comparison = compare_runs(result, ref_result)
        payload["filter_comparison"] = comparison.summary()
        ref_metrics = compute_metrics(
            ref_result, ref_built.driver_center, ref_built.reference_index
        )
        payload["metrics_nofilter"] = ref_metrics.summary()

    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    figures = _figures(run_dir, built, result, metrics)
    summary: dict[str, float] = dict(payload["metrics"])
    if "filter_comparison" in payload:
        summary |= {f"filter_{k}": v for k, v in payload["filter_comparison"].items()}
    write_report(
        run_dir / "report" / "report.html",
        title=config.name,
        summary=summary,
        figure_paths=figures,
        config=config_to_dict(config),
    )
    print(f"[{config.name}] report: {run_dir / 'report' / 'report.html'}")
    return run_dir
