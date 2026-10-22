"""Physical receive-coil fields, in millimetres. Ported from AGNC ``simulation/coils.py``.

Each element is a circular current loop tilted to face the FOV centre, and the complex
sensitivity follows reciprocity ``S = B_x - i B_y`` with ``B`` from Biot-Savart, the
loop discretized into straight segments (Guerquin-Kern 2012, Eq. 2).

The point of using these rather than a smooth analytic profile is that they are **not**
band-limited in the Fourier basis. Fitting them always leaves a residual, which is what
lets the simulation use a rich CSM and hand the reconstruction a different, poorer one --
without that gap there is a coil-sensitivity inverse crime hiding in the benchmark.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..conventions import FOV


def coil_centres(num_coils: int, radius_mm: float, seed: int = 0,
                 device="cpu") -> Tensor:
    """``[C, 3]`` element centres on a Fibonacci sphere of radius ``radius_mm``.

    Fibonacci rather than a lat/long grid: it spreads the elements near-uniformly for
    *any* count, so changing the channel count does not change the array's character.
    A small jitter breaks the exact symmetry, which otherwise makes several coils
    linearly dependent and the CSM fit needlessly ill-conditioned.
    """
    g = torch.Generator().manual_seed(int(seed))
    jit = 0.06 * radius_mm * (torch.rand(int(num_coils), 3, generator=g) - 0.5)
    ga = math.pi * (3.0 - math.sqrt(5.0))
    out = []
    for i in range(int(num_coils)):
        z = 1.0 - 2.0 * (i + 0.5) / float(num_coils)
        rho = math.sqrt(max(0.0, 1.0 - z * z))
        th = ga * i
        out.append([radius_mm * rho * math.cos(th) + float(jit[i, 0]),
                    radius_mm * rho * math.sin(th) + float(jit[i, 1]),
                    radius_mm * z + float(jit[i, 2])])
    return torch.tensor(out, device=device, dtype=torch.float32)


def biot_savart_maps(points: Tensor, num_coils: int = 15, array_radius_mm: float = 140.0,
                     loop_radius_mm: float = 55.0, n_segments: int = 128,
                     seed: int = 0, chunk: int = 65536) -> Tensor:
    """Complex sensitivities at ``points [P, 3]`` (mm) -> ``[C, P]``, RSS-normalized.

    Chunked over points because the intermediate is ``[P, n_segments, 3]``.
    """
    dev = points.device
    centres = coil_centres(num_coils, array_radius_mm, seed, device=dev)
    t = torch.linspace(0, 2 * math.pi, int(n_segments) + 1, device=dev)[:-1]
    out = torch.empty(int(num_coils), points.shape[0], device=dev, dtype=torch.complex64)

    for i in range(int(num_coils)):
        c = centres[i]
        normal = -c / c.norm()
        e1 = torch.cross(normal, torch.tensor([0.0, 0.0, 1.0], device=dev), dim=0)
        e1 = e1 / e1.norm().clamp_min(1e-8)
        e2 = torch.cross(normal, e1, dim=0)
        loop = c[None] + loop_radius_mm * (torch.cos(t)[:, None] * e1[None]
                                           + torch.sin(t)[:, None] * e2[None])
        dl = torch.roll(loop, -1, dims=0) - loop
        mid = 0.5 * (loop + torch.roll(loop, -1, dims=0))
        for a in range(0, points.shape[0], chunk):
            diff = points[a: a + chunk, None, :] - mid[None]
            d3 = diff.norm(dim=-1).clamp_min(1e-2).pow(3)
            b = (torch.cross(dl[None].expand_as(diff), diff, dim=-1) / d3[..., None]).sum(1)
            out[i, a: a + chunk] = (b[:, 0] - 1j * b[:, 1]).to(torch.complex64)
    return out / out.abs().square().sum(0).sqrt().max().clamp_min(1e-12)


def array_for_fov(fov: FOV, num_coils: int = 15, oversample: int = 1,
                  array_radius_mm: float | None = None, seed: int = 0,
                  device="cpu", **kw) -> tuple[Tensor, Tensor]:
    """``(points, maps)`` on the FOV grid, with the array sized to enclose it.

    Default radius is 1.1x the FOV corner: the array must be strictly outside the
    object, since the Biot-Savart field is near-singular on the wire.
    """
    pts = fov.grid_coords(device=device, oversample=oversample)
    if array_radius_mm is None:
        array_radius_mm = 1.1 * float(torch.tensor(fov.fov_mm).norm() / 2)
    return pts, biot_savart_maps(pts, num_coils, array_radius_mm, seed=seed, **kw)
