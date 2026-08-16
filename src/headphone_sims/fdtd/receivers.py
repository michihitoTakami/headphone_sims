"""Virtual microphone arrays: trilinear field sampling at arbitrary points.

Pressure and the three staggered velocity components are each interpolated
from their own grid onto the probe positions. Note the leapfrog half-step
time offset between p and v (dt/2 ~ 0.4 us at default resolution); it is
negligible for the audio-band metrics and left uncorrected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from headphone_sims.grid import Grid

_OFFSET_P = (0.5, 0.5, 0.5)


def _stagger_offsets(axis: int) -> tuple[float, float, float]:
    off = [0.5, 0.5, 0.5]
    off[axis] = 1.0
    return (off[0], off[1], off[2])


def _trilinear_lists(
    positions: torch.Tensor,
    shape: tuple[int, int, int],
    offsets: tuple[float, float, float],
    dx: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat corner indices (n, 8) and weights (n, 8) for trilinear interpolation."""
    n = positions.shape[0]
    i0s: list[torch.Tensor] = []
    fracs: list[torch.Tensor] = []
    for a in range(3):
        u = positions[:, a] / dx - offsets[a]
        i0 = torch.clamp(u.floor().to(torch.int64), 0, shape[a] - 2)
        frac = torch.clamp(u - i0.to(u.dtype), 0.0, 1.0)
        i0s.append(i0)
        fracs.append(frac)

    idx = torch.zeros((n, 8), dtype=torch.int64)
    w = torch.ones((n, 8), dtype=torch.float64)
    corner = 0
    for cx in (0, 1):
        for cy in (0, 1):
            for cz in (0, 1):
                ii = i0s[0] + cx
                jj = i0s[1] + cy
                kk = i0s[2] + cz
                idx[:, corner] = (ii * shape[1] + jj) * shape[2] + kk
                wx = fracs[0] if cx else 1.0 - fracs[0]
                wy = fracs[1] if cy else 1.0 - fracs[1]
                wz = fracs[2] if cz else 1.0 - fracs[2]
                w[:, corner] = wx * wy * wz
                corner += 1
    return idx, w.to(torch.float32)


@dataclass
class BakedReceivers:
    """Precomputed sampling lists and per-step recording buffers."""

    positions: npt.NDArray[np.float64]  # (n, 3) meters
    idx_p: torch.Tensor
    w_p: torch.Tensor
    idx_v: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    w_v: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    rec_p: torch.Tensor  # (n_steps, n)
    rec_v: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def sample(self, p: torch.Tensor, v: tuple[torch.Tensor, ...], it: int) -> None:
        self.rec_p[it] = (p.view(-1)[self.idx_p] * self.w_p).sum(dim=1)
        for comp, idx, w, rec in zip(v, self.idx_v, self.w_v, self.rec_v, strict=True):
            rec[it] = (comp.view(-1)[idx] * w).sum(dim=1)


@dataclass(frozen=True)
class ReceiverArray:
    """Probe positions in meters, shape (n, 3)."""

    positions: npt.NDArray[np.float64]

    def bake(self, grid: Grid, n_steps: int, device: torch.device) -> BakedReceivers:
        pos = torch.as_tensor(np.asarray(self.positions, dtype=np.float64))
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"positions must have shape (n, 3), got {tuple(pos.shape)}")
        n = pos.shape[0]
        idx_p, w_p = _trilinear_lists(pos, grid.shape, _OFFSET_P, grid.dx)
        idx_v = []
        w_v = []
        for axis in range(3):
            idx, w = _trilinear_lists(
                pos, grid.velocity_shape(axis), _stagger_offsets(axis), grid.dx
            )
            idx_v.append(idx.to(device))
            w_v.append(w.to(device))
        return BakedReceivers(
            positions=np.asarray(self.positions, dtype=np.float64),
            idx_p=idx_p.to(device),
            w_p=w_p.to(device),
            idx_v=(idx_v[0], idx_v[1], idx_v[2]),
            w_v=(w_v[0], w_v[1], w_v[2]),
            rec_p=torch.zeros((n_steps, n), dtype=torch.float32, device=device),
            rec_v=tuple(  # type: ignore[arg-type]
                torch.zeros((n_steps, n), dtype=torch.float32, device=device) for _ in range(3)
            ),
        )
