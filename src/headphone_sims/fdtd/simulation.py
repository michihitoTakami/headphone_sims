"""Simulation orchestration: assemble state, run the leapfrog loop, collect results."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch

from headphone_sims.fdtd import kernel
from headphone_sims.fdtd.boundaries import CpmlConfig, CpmlState, SpongeConfig, attach_damping
from headphone_sims.fdtd.kernel import FdtdState
from headphone_sims.fdtd.receivers import BakedReceivers, ReceiverArray
from headphone_sims.fdtd.sources import BakedSource, Source
from headphone_sims.grid import Grid

FloatArray = npt.NDArray[np.float64]

_DEFAULT_SPONGE = SpongeConfig()


@dataclass
class SnapshotConfig:
    """Capture a 2D pressure slice every ``every`` steps for visualization."""

    every: int = 20
    axis: int = 2  # slice normal
    index: int | None = None  # defaults to mid-plane


@dataclass
class SimulationResult:
    """Recorded receiver signals and metadata (all on CPU)."""

    dt: float
    dx: float
    positions: FloatArray  # (n_probes, 3)
    p: FloatArray  # (n_steps, n_probes)
    v: FloatArray  # (3, n_steps, n_probes)
    source_waveform: FloatArray  # (n_steps,) of the first source, for deconvolution
    snapshots: list[tuple[int, FloatArray]] = field(default_factory=list)

    @property
    def sample_rate(self) -> float:
        return 1.0 / self.dt

    @property
    def times(self) -> FloatArray:
        return np.arange(self.p.shape[0], dtype=np.float64) * self.dt


class Simulation:
    """One FDTD run: geometry + sources + receivers on a device."""

    def __init__(
        self,
        grid: Grid,
        sources: list[Source],
        receivers: ReceiverArray,
        n_steps: int,
        solid: torch.Tensor | None = None,
        sigma_material: torch.Tensor | None = None,
        sponge: SpongeConfig | None = _DEFAULT_SPONGE,
        cpml: CpmlConfig | None = None,
        device: torch.device | str | None = None,
        snapshot: SnapshotConfig | None = None,
    ) -> None:
        if not sources:
            raise ValueError("at least one source is required")
        if cpml is not None and sponge is not None and sponge is not _DEFAULT_SPONGE:
            raise ValueError("pass either sponge or cpml, not both")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.grid = grid
        self.n_steps = n_steps
        self.snapshot = snapshot

        self.state = FdtdState.zeros(grid, device=self.device)
        if cpml is not None:
            sponge = None
            self.state.cpml = CpmlState.build(grid, cpml, self.device)
        attach_damping(self.state, sponge=sponge, solid=solid, sigma_material=sigma_material)
        self.baked_sources: list[BakedSource] = [s.bake(grid, self.device) for s in sources]
        self.baked_receivers: BakedReceivers = receivers.bake(grid, n_steps, self.device)
        self._source_waveform = np.asarray(sources[0].waveform, dtype=np.float64)

    def run(self, progress_every: int = 0) -> SimulationResult:
        st = self.state
        v = (st.vx, st.vy, st.vz)
        snapshots: list[tuple[int, FloatArray]] = []
        snap = self.snapshot
        slicer: tuple[slice | int, ...] | None = None
        if snap is not None:
            index = snap.index if snap.index is not None else self.grid.shape[snap.axis] // 2
            sl: list[slice | int] = [slice(None)] * 3
            sl[snap.axis] = index
            slicer = tuple(sl)

        for it in range(self.n_steps):
            kernel.step_velocity(st)
            for src in self.baked_sources:
                src.inject_velocity(v, it)
            kernel.step_pressure(st)
            for src in self.baked_sources:
                src.inject_pressure(st.p, it)
            self.baked_receivers.sample(st.p, v, it)
            if slicer is not None and snap is not None and it % snap.every == 0:
                snapshots.append((it, st.p[slicer].detach().cpu().numpy().astype(np.float64)))
            if progress_every and it % progress_every == 0:
                peak = float(st.p.abs().max())
                print(f"step {it}/{self.n_steps}  |p|_max={peak:.3e}")

        br = self.baked_receivers
        wf = np.zeros(self.n_steps, dtype=np.float64)
        m = min(self.n_steps, self._source_waveform.shape[0])
        wf[:m] = self._source_waveform[:m]
        return SimulationResult(
            dt=self.grid.dt,
            dx=self.grid.dx,
            positions=br.positions,
            p=br.rec_p.cpu().numpy().astype(np.float64),
            v=np.stack([r.cpu().numpy().astype(np.float64) for r in br.rec_v]),
            source_waveform=wf,
            snapshots=snapshots,
        )
