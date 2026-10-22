"""Independent raster references, and the metrics that compare against them.

The closed-form transform has no measurable approximation error, so verification is
one-directional: **the raster must converge to the closed form**, never the reverse.
A raster reference is therefore always reported with the sampling that produced it,
and a claim of agreement is only meaningful together with a convergence trend.

Why a point-sampled raster converges at all: sampling ``rho`` on a grid of spacing
``d`` and summing with the voxel volume is a Riemann sum whose error, by Poisson
summation, is the sum of spectral aliases ``rho_hat(k + j/d)``. For a Gaussian of
width ``sigma`` those decay like ``exp(-2 pi^2 sigma^2 / d^2)``, so at ``sigma = 4 d``
the aliasing is far below float32 -- which is what makes this a *strict* test rather
than a loose one.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..conventions import FOV
from ..representation.gaussian3d import GaussianMixture3D


def raster_dft(volume: Tensor, points: Tensor, k: Tensor, voxel_volume: float,
               chunk: int = 0) -> Tensor:
    """Direct (non-FFT) DFT of a sampled volume at arbitrary ``k [S, 3]``.

    Direct rather than FFT-plus-interpolation on purpose: interpolating an FFT would
    introduce exactly the gridding error this project exists to avoid, and would make
    the reference no longer independent of the thing being tested.
    """
    vals = volume.reshape(-1).to(torch.complex128)
    chunk = chunk or max(1, int(2e7 // max(k.shape[0], 1)))
    out = torch.zeros(k.shape[0], device=k.device, dtype=torch.complex128)
    for a in range(0, points.shape[0], chunk):
        ph = 2 * torch.pi * torch.remainder(
            k.double() @ points[a: a + chunk].double().T, 1.0)          # [S, p]
        out += torch.polar(torch.ones_like(ph), -ph) @ vals[a: a + chunk]
    return out * float(voxel_volume)


def raster_reference(mixture: GaussianMixture3D, fov: FOV, k: Tensor,
                     oversample: int = 4, coeff: Optional[Tensor] = None,
                     modulation: Optional[Tensor] = None,
                     point_chunk: int = 262144, cutoff_sigma: float = 4.0,
                     strict: bool = True) -> Tensor:
    """Sample the mixture on an oversampled grid, then DFT it at ``k``.

    ``modulation [P]`` multiplies the sampled object before the transform -- that is
    how a coil sensitivity enters the reference, evaluated pointwise in the image
    domain so that the multi-coil identity is tested against something that never
    used the identity.

    ``strict`` refuses to build a reference whose box does not contain the object out to
    ``cutoff_sigma``. A truncated box is a **domain** error, not a sampling error, so it
    does not shrink when you oversample -- which is exactly what makes it dangerous:
    it looks like a converged disagreement with the closed form and invites you to go
    hunting in the kernel. (Measured: a deformation that pushed primitives 4 mm past the
    edge produced a rock-steady 2.2e-3 at every oversampling factor, and 2.9e-7 once the
    box was widened.)
    """
    dev = k.device
    reach = float((mixture.mu.abs()
                   + float(cutoff_sigma) * mixture.sigma().max(-1, keepdim=True).values).max())
    half = 0.5 * min(fov.fov_mm) - max(abs(c) for c in fov.center_mm)
    if reach > half:
        msg = (f"object reaches {reach:.1f} mm but the reference box half-extent is "
               f"{half:.1f} mm: the raster would truncate it, giving a constant error "
               f"that oversampling cannot fix. Widen the FOV or pass strict=False.")
        if strict:
            raise ValueError(msg)
        import warnings
        warnings.warn(msg, stacklevel=2)
    total = 1
    for n in fov.grid_shape(oversample):
        total *= n
    dv = fov.voxel_volume_mm3 / (oversample ** 3)
    acc = torch.zeros(k.shape[0], device=dev, dtype=torch.complex128)
    pts_all = fov.grid_coords(device=dev, dtype=torch.float64, oversample=oversample)
    for a in range(0, total, point_chunk):
        pts = pts_all[a: a + point_chunk]
        vals = mixture.to(dtype=torch.float64).image(pts, coeff=coeff)
        if modulation is not None:
            vals = vals.to(torch.complex128) * modulation[a: a + point_chunk]
        acc += raster_dft(vals, pts, k, dv)
    return acc


def nrmse(a: Tensor, b: Tensor) -> float:
    """Complex relative L2 error of ``a`` against reference ``b``."""
    return float((a - b).abs().square().sum().sqrt() / b.abs().square().sum().sqrt())


def band_nrmse(a: Tensor, b: Tensor, k: Tensor, k_max: float) -> float:
    """NRMSE restricted to ``|k| <= k_max`` -- the acquisition band that matters.

    A phantom is only ever asked for frequencies inside the acquisition band, so a
    representation error outside it is not an error of the simulation. Reporting the
    unrestricted number instead makes every finite Gaussian mixture look worse than
    it is, and hides where it is actually wrong.
    """
    m = k.norm(dim=-1) <= float(k_max)
    return nrmse(a[m], b[m])


def shell_nrmse(pred, ref, k, r_max: float, shells: int = 16,
                fov=None) -> tuple:
    """Relative error resolved by ``|k|`` -> (per-shell NRMSE ``[S]``, shell centres).

    The global ``||err|| / ||ref||`` over a random k set is dominated by the smallest
    radius drawn, because a brain spectrum falls off far faster than the sampling
    density rises. Measured on one fixed phantom, that ratio moved 0.0263 -> 0.0522
    across four seeds at n=4096 and did **not** stabilize with more samples (2.7x
    spread at n=65536): it is governed by an extreme order statistic, not an average.

    Within a shell, ``|rho_hat|`` varies by far less, so a per-shell ratio is a
    well-conditioned quantity. Report the curve; if one number is needed, take the mean
    over shells, which weights the band uniformly instead of by signal power, and say
    which of the two is being quoted.
    """
    import torch

    r = k.norm(dim=-1)
    hi = float(r.max()) if fov is None else float(r_max) * float(fov.k_nyquist[0])
    edges = torch.linspace(0.0, hi, int(shells) + 1, device=k.device)
    out, mid = [], []
    for i in range(int(shells)):
        m = (r >= edges[i]) & (r < edges[i + 1] if i + 1 < int(shells) else r <= edges[i + 1])
        if int(m.sum()) < 8:
            continue
        p, q = pred[..., m], ref[..., m]
        out.append(float((p - q).abs().pow(2).sum().sqrt()
                         / q.abs().pow(2).sum().sqrt().clamp_min(1e-30)))
        mid.append(float(0.5 * (edges[i] + edges[i + 1])))
    return out, mid
