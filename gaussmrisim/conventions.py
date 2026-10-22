"""Physical coordinates and Fourier conventions.

Positions are in mm, frequencies in cycles/mm, and volumes use [z, y, x] layout.
The transform convention is integral rho(r) exp(-i 2 pi k.r) dr.
Gaussian weights are integrated masses, invariant under rigid motion.
MRI-NUFFT coordinates t in [-0.5, 0.5) map to k = t / voxel_mm.
Large physical phases are reduced modulo one cycle in float64 before casting.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

TWO_PI_SQ = 19.739208802178716          # 2 * pi^2, the Gaussian k-space decay constant


@dataclass(frozen=True)
class FOV:
    """An axis-aligned field of view in millimetres, and the grid it implies."""

    fov_mm: tuple[float, float, float]
    matrix: tuple[int, int, int]
    center_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def voxel_mm(self) -> tuple[float, float, float]:
        return tuple(f / n for f, n in zip(self.fov_mm, self.matrix))

    @property
    def voxel_volume_mm3(self) -> float:
        v = self.voxel_mm
        return float(v[0] * v[1] * v[2])

    @property
    def k_nyquist(self) -> tuple[float, float, float]:
        """Highest representable frequency, cycles/mm, per axis."""
        return tuple(0.5 / v for v in self.voxel_mm)

    def grid_coords(self, device="cpu", dtype=torch.float32,
                    oversample: int = 1) -> Tensor:
        """Voxel-centre coordinates ``[P, 3]`` in mm, **matching a ``[nz, ny, nx]`` volume**.

        The ordering is z-slowest / x-fastest, i.e. row ``p`` of the result is the
        coordinate of ``volume.reshape(-1)[p]`` for a volume in the renderer's layout.
        This is not a free choice: arrays here are ``[nz, ny, nx]`` while :class:`FOV` is
        ``(x, y, z)``, and an x-slowest ordering silently pairs a volume with transposed
        coordinates. That cost a whole experiment -- a Shepp-Logan target built from
        x-slowest coordinates was fitted transposed, which the image metric could not see
        (the fit reproduced whatever it was given) and which showed up only as a k-space
        error of 0.88 against the analytic truth.

        ``oversample > 1`` refines the grid *inside the same FOV*, so a convergence study
        varies only the sampling.
        """
        axes = []
        for n, f, c in zip(self.matrix, self.fov_mm, self.center_mm):
            m = int(n) * int(oversample)
            d = f / m
            axes.append(torch.arange(m, device=device, dtype=dtype) * d - 0.5 * f + 0.5 * d + c)
        gz, gy, gx = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)

    def grid_shape(self, oversample: int = 1) -> tuple[int, int, int]:
        return tuple(int(n) * int(oversample) for n in self.matrix)

    def cartesian_k(self, device="cpu", dtype=torch.float32) -> Tensor:
        """The full Cartesian k-space grid matched to this FOV, ``[P, 3]`` cycles/mm."""
        axes = []
        for n, f in zip(self.matrix, self.fov_mm):
            axes.append((torch.arange(n, device=device, dtype=dtype) - n // 2) / f)
        g = torch.meshgrid(*axes, indexing="ij")
        return torch.stack([a.reshape(-1) for a in g], dim=-1)


def traj_to_k(traj: Tensor, voxel_mm: float | tuple[float, float, float]) -> Tensor:
    """mri-nufft normalized trajectory ``[S, 3] in [-0.5, 0.5)`` -> cycles/mm."""
    if isinstance(voxel_mm, (int, float)):
        return traj / float(voxel_mm)
    v = torch.as_tensor(voxel_mm, device=traj.device, dtype=traj.dtype)
    return traj / v


def cycles_mod_one(k: Tensor, mu: Tensor) -> Tensor:
    """``(k . mu) mod 1`` in cycles, computed in float64 -> ``[S, N]`` float32.

    See the module docstring: this is the only numerically safe way to form the
    Gaussian phase in physical units. The reduction is exact because
    ``exp(-i 2 pi x)`` has period 1 in ``x``, so discarding whole cycles discards
    nothing, and it happens while the value still has float64 precision.
    """
    out = torch.remainder(k.double() @ mu.double().T, 1.0)
    return out.to(k.dtype)
