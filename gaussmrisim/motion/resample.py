"""Interpolating a sampled volume under a rigid pose -- the voxel phantom's only option.

This module exists to make the raster baseline in the motion study *fair*. Trilinear is
what most simulators actually do, but it is also the weakest interpolator anyone would
defend, so a comparison that only beats trilinear proves less than it looks. Cubic
B-spline (Unser's, the standard in medical image registration and the default in
``scipy.ndimage`` at ``order=3``) is the honest opponent.

Two conventions matter and both are easy to get wrong:

* **B-spline interpolation is not B-spline smoothing.** The cubic B-spline basis is not
  interpolating -- ``B3(0) = 2/3``, not 1 -- so convolving the samples directly with it
  blurs. The volume must first be run through the recursive **prefilter** that turns
  samples into spline *coefficients*; only then does the reconstruction pass through the
  original samples. Skipping the prefilter yields a resampler that loses ~a third of the
  contrast at every pose and would flatter our own method enormously.
* The 4-tap-per-axis (64 total) evaluation is done as **8 trilinear lookups** at shifted
  positions (Sigg & Hadwiger 2005). Weights ``w0..w3`` are paired as ``g0 = w0+w1`` and
  ``g1 = w2+w3``, sampled at ``i-1+w1/g0`` and ``i+1+w3/g1``; hardware trilinear then
  reproduces the cubic sum exactly, so this is an optimization, not an approximation.

``bspline_prefilter`` is verified by ``scripts/verify_resample.py``: at zero displacement
the resampler must return the input volume (the interpolation property), and against
``scipy.ndimage.map_coordinates(order=3)`` it must agree to float32 precision.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

# The single pole of the cubic B-spline direct filter, z1 = sqrt(3) - 2.
_POLE = math.sqrt(3.0) - 2.0
_GAIN = 6.0                                    # (1 - z)(1 - 1/z) for the cubic


def _prefilter_axis(c: Tensor, dim: int, pole: float = _POLE,
                    gain: float = _GAIN, tol: float = 1e-8) -> Tensor:
    """Unser's causal/anticausal recursion along one axis, in place on a copy."""
    c = c.movedim(dim, 0).contiguous() * gain
    n = c.shape[0]
    if n == 1:
        return (c / gain).movedim(0, dim)
    # Causal initialization: c[0] = sum_k c[k] z^k over a horizon where z^k < tol.
    horizon = min(n, int(math.ceil(math.log(tol) / math.log(abs(pole)))))
    zk = pole
    s = c[0].clone()
    for k in range(1, horizon):
        s = s + zk * c[k]
        zk *= pole
    c[0] = s
    for k in range(1, n):                       # causal
        c[k] = c[k] + pole * c[k - 1]
    c[n - 1] = (pole / (pole - 1.0)) * c[n - 1]  # anticausal initialization
    for k in range(n - 2, -1, -1):              # anticausal
        c[k] = pole * (c[k + 1] - c[k])
    return c.movedim(0, dim)


def bspline_prefilter(vol: Tensor) -> Tensor:
    """Samples -> cubic B-spline coefficients, separably along all three axes."""
    c = vol.to(torch.float32)
    for d in range(c.dim()):
        c = _prefilter_axis(c, d)
    return c


def _cubic_taps(x: Tensor):
    """Two shifted sample positions and their weights, per axis (Sigg & Hadwiger)."""
    i = torch.floor(x)
    a = x - i
    a2 = a * a
    a3 = a2 * a
    w0 = (1.0 - 3.0 * a + 3.0 * a2 - a3) / 6.0
    w1 = (4.0 - 6.0 * a2 + 3.0 * a3) / 6.0
    w2 = (1.0 + 3.0 * a + 3.0 * a2 - 3.0 * a3) / 6.0
    w3 = a3 / 6.0
    g0 = w0 + w1
    g1 = w2 + w3
    p0 = i - 1.0 + w1 / g0
    p1 = i + 1.0 + w3 / g1
    return (p0, p1), (g0, g1)


def resample_index(vol: Tensor, idx: Tensor, mode: str = "bspline",
                   coeffs: Tensor | None = None) -> Tensor:
    """Sample ``vol`` [nz, ny, nx] at fractional **index** coordinates ``idx[..., (x,y,z)]``.

    ``mode='trilinear'`` is a single ``grid_sample``; ``mode='bspline'`` is the cubic
    B-spline, which needs ``vol`` prefiltered -- pass ``coeffs`` to reuse it across poses.
    Out-of-volume samples are zero, matching the raster phantom's own zero padding.
    """
    nz, ny, nx = vol.shape
    n = torch.tensor([nx, ny, nz], device=vol.device, dtype=idx.dtype)

    def sample(src: Tensor, at: Tensor) -> Tensor:
        g = 2.0 * at / (n - 1.0) - 1.0                       # -> [-1, 1], (x, y, z)
        return torch.nn.functional.grid_sample(
            src[None, None], g[None], mode="bilinear", padding_mode="zeros",
            align_corners=True)[0, 0]

    if mode == "trilinear":
        return sample(vol, idx)
    if mode != "bspline":
        raise ValueError(f"unknown mode {mode!r}")
    c = bspline_prefilter(vol) if coeffs is None else coeffs
    (p0, p1), (g0, g1) = _cubic_taps(idx)
    out = torch.zeros(idx.shape[:-1], device=vol.device, dtype=vol.dtype)
    for bx in (0, 1):
        for by in (0, 1):
            for bz in (0, 1):
                at = torch.stack([p1[..., 0] if bx else p0[..., 0],
                                  p1[..., 1] if by else p0[..., 1],
                                  p1[..., 2] if bz else p0[..., 2]], dim=-1)
                w = ((g1[..., 0] if bx else g0[..., 0])
                     * (g1[..., 1] if by else g0[..., 1])
                     * (g1[..., 2] if bz else g0[..., 2]))
                out = out + w * sample(c, at)
    return out


def rotate_raster(vol: Tensor, fov, R: Tensor, t: Tensor, mode: str = "trilinear",
                  coeffs: Tensor | None = None) -> Tensor:
    """Move a sampled volume by ``r -> R r + t``, resampling with ``mode``.

    Output voxel at physical ``r`` takes the input at ``R^T (r - t)``. The grid is built
    in (x, y, z) because that is what ``grid_sample`` wants, while the volume is
    ``[nz, ny, nx]``.
    """
    dev = vol.device
    ax = [torch.linspace(-0.5 * f + 0.5 * v + c, 0.5 * f - 0.5 * v + c, m, device=dev)
          for f, v, m, c in zip(fov.fov_mm, fov.voxel_mm, fov.matrix, fov.center_mm)]
    gz, gy, gx = torch.meshgrid(ax[2], ax[1], ax[0], indexing="ij")
    r = torch.stack([gx, gy, gz], dim=-1)                    # [nz, ny, nx, 3] mm
    src = (r - t) @ R                                        # R^T (r - t)
    origin = torch.tensor([a[0] for a in ax], device=dev)
    step = torch.tensor(fov.voxel_mm, device=dev)
    return resample_index(vol, (src - origin) / step, mode=mode, coeffs=coeffs)
