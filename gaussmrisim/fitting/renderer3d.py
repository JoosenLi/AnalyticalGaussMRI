"""Tile-binned volumetric renderer: the image-domain half of the fitting loop.

Splits the same way the Fourier engine does. PyTorch owns the parameterization
(quaternion -> rotation -> covariance -> precision, log-scale, mass) so autograd
carries those Jacobians; CUDA sees a flat list of ``(mu, P, norm)`` and a
primitive-to-tile assignment. Nothing about the representation is duplicated in the
kernel, so the two cannot drift.

The binning itself is a sort, not a search: an axis-aligned box per primitive from its
cutoff ellipsoid, one ``(tile, primitive)`` key per overlap, sorted by tile, and a
prefix range per tile. That is the 3DGS tiling idea applied to a voxel grid rather than
an image plane; the closest published version is R2-Gaussian's CUDA voxelizer
(NeurIPS 2024), which is worth reading for its analysis of the integration bias in
vanilla 3DGS -- a bias that does not apply here only because our primitives are
densities integrated exactly, not alpha-composited.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

from ..fourier.cuda_ft import _SYM
from ..representation.gaussian3d import GaussianMixture3D

TILE = 8       # must match render3d_kernel.cu
#: lanes sharing one primitive in the backward kernel (1, 2, 4, 8, 16 or 32)
BACKWARD_GROUP = int(os.environ.get("GMRISIM_BWD_GROUP", "16"))


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    src = Path(__file__).resolve().parent / "cuda"
    return load(name="_gmrisim_render3d",
                sources=[str(src / "render3d_kernel.cu")],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=bool(int(os.environ.get("GMRISIM_VERBOSE_CUDA_BUILD", "0"))))


@dataclass
class Patch:
    """A rendering target: an axis-aligned voxel box in millimetres."""

    origin_mm: tuple           # coordinate of voxel (0, 0, 0)
    voxel_mm: tuple
    shape: tuple               # (nx, ny, nz)

    @property
    def tiles(self) -> tuple:
        return tuple((n + TILE - 1) // TILE for n in self.shape)

    def n_tiles(self) -> int:
        t = self.tiles
        return int(t[0] * t[1] * t[2])


def pack_render(mixture: GaussianMixture3D) -> tuple[Tensor, Tensor, Tensor]:
    """``(mu, prec6, norm)``: precision matrix and the peak height ``m / ((2pi)^{3/2}|Sigma|^{1/2})``.

    Inverting through the eigen-decomposition rather than ``linalg.inv`` keeps this
    exact and cheap: ``Sigma^{-1} = R diag(sigma^{-2}) R^T`` and
    ``|Sigma|^{1/2} = prod(sigma)``, both already available from the parameterization.
    """
    # effective_sigma folds in the isotropic band-limit filter, which commutes with the
    # rotation -- so the cheap eigen route stays exact and no inverse is ever needed
    R, s = mixture.rotation(), mixture.effective_sigma()
    norm = mixture.mass / ((2 * torch.pi) ** 1.5 * s.prod(dim=-1))
    prec6 = _sym6(R, s.square().reciprocal())
    return mixture.mu.contiguous(), prec6, norm


_I = [i for i, _ in _SYM]
_J = [j for _, j in _SYM]


def _sym6(R: Tensor, w: Tensor) -> Tensor:
    """The six entries of ``R diag(w) R^T`` -> ``[N, 6]`` in ``_SYM`` order.

    ``P_ij = sum_k R_ik R_jk w_k``, written elementwise. The same quantity as the batched
    matmul it replaces, but ``R @ diag @ R^T`` on ``[N, 3, 3]`` dispatches a full cuBLAS
    GEMM per product and was **19% of a fitting step's GPU time** (profiled on grey
    matter, N=100k, 34 Mvox; ``scripts/profile_fit_step.py``) for 3x3 work.
    """
    return (R[:, _I, :] * R[:, _J, :] * w[:, None, :]).sum(-1).contiguous()


def axis_widths(mixture: GaussianMixture3D) -> Tensor:
    """``sqrt(Sigma_jj)`` per axis ``[N, 3]`` -- the half-widths the AABB needs.

    Not the principal widths. For a rotated primitive the bounding box of the cutoff
    ellipsoid is set by the covariance *diagonal*, and using the principal widths
    instead under-covers it: the primitive is then simply absent from the tiles it
    reaches, which shows up as a small dense error everywhere (measured: 1e-3 relative
    at N=64) and as badly wrong scale/orientation gradients, because the clip is a
    discontinuity. The identity-rotation case stays exact, so a single-primitive test
    cannot catch this.
    """
    R = mixture.rotation()
    return torch.sqrt((R.square() * mixture.effective_sigma().square()[:, None, :]).sum(-1))


def bin_to_tiles(mu: Tensor, sigma: Tensor, patch: Patch, cutoff_sigma: float
                 ) -> tuple[Tensor, Tensor, Tensor]:
    """``(tile_start, tile_count, prim_idx)`` for the kernel.

    ``sigma`` must be :func:`axis_widths`, not the principal widths.
    """
    dev = mu.device
    o = torch.tensor(patch.origin_mm, device=dev, dtype=mu.dtype)
    d = torch.tensor(patch.voxel_mm, device=dev, dtype=mu.dtype)
    n = torch.tensor(patch.shape, device=dev, dtype=torch.long)
    tshape = torch.tensor(patch.tiles, device=dev, dtype=torch.long)

    half = float(cutoff_sigma) * sigma                                     # [N, 3] mm
    lo = torch.clamp(torch.floor((mu - half - o) / d / TILE).long(), min=0)
    hi = torch.minimum(torch.floor((mu + half - o) / d / TILE).long(), tshape - 1)
    span = (hi - lo + 1).clamp(min=0)
    count = span.prod(dim=-1)
    keep = count > 0
    if not bool(keep.any()):
        z = torch.zeros(patch.n_tiles(), device=dev, dtype=torch.int32)
        return z, z, torch.zeros(0, device=dev, dtype=torch.int32)

    idx = torch.nonzero(keep).flatten()
    lo, span, count = lo[keep], span[keep], count[keep]
    offs = torch.cat([torch.zeros(1, device=dev, dtype=torch.long), count.cumsum(0)])
    total = int(offs[-1])
    # expand each primitive's box into its (tile, primitive) pairs without a Python loop
    pos = torch.arange(total, device=dev) - offs[:-1].repeat_interleave(count)
    src = torch.arange(len(count), device=dev).repeat_interleave(count)
    sx = span[src, 0]
    ti = lo[src, 0] + pos % sx
    tj = lo[src, 1] + (pos // sx) % span[src, 1]
    tk = lo[src, 2] + pos // (sx * span[src, 1])
    tid = (tk * tshape[1] + tj) * tshape[0] + ti

    order = torch.argsort(tid)
    tid, prim = tid[order], idx[src][order].to(torch.int32)
    counts = torch.bincount(tid, minlength=patch.n_tiles()).to(torch.int32)
    starts = torch.cat([torch.zeros(1, device=dev, dtype=torch.int32),
                        counts.cumsum(0)[:-1].to(torch.int32)])
    return starts.contiguous(), counts.contiguous(), prim.contiguous()


class _Render(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, prec6, norm, starts, counts, prim, patch, cutoff):
        out = _ext().forward(mu.contiguous(), prec6.contiguous(), norm.contiguous(),
                             starts, counts, prim, list(map(float, patch.origin_mm)),
                             list(map(float, patch.voxel_mm)), list(map(int, patch.shape)),
                             float(cutoff))
        ctx.save_for_backward(mu, prec6, norm, starts, counts, prim)
        ctx.patch, ctx.cutoff = patch, cutoff
        return out

    @staticmethod
    def backward(ctx, grad_out):
        mu, prec6, norm, starts, counts, prim = ctx.saved_tensors
        g = _ext().backward(grad_out, mu, prec6, norm, starts, counts, prim,
                            list(map(float, ctx.patch.origin_mm)),
                            list(map(float, ctx.patch.voxel_mm)),
                            list(map(int, ctx.patch.shape)), float(ctx.cutoff),
                            BACKWARD_GROUP)
        return g[0], g[1], g[2], None, None, None, None, None


def render(mixture: GaussianMixture3D, patch: Patch, cutoff_sigma: float = 4.0,
           binning=None) -> Tensor:
    """Render ``rho`` into the patch -> ``[nz, ny, nx]``, differentiable.

    ``binning`` reuses a previous :func:`bin_to_tiles` result; rebuild it whenever
    ``mu`` or the scales have moved appreciably. It is *not* free to keep stale --
    a primitive that has drifted out of its tile list simply stops contributing there,
    which looks like a bad fit rather than like a bug.
    """
    mu, prec6, norm = pack_render(mixture)
    if binning is None:
        binning = bin_to_tiles(mu.detach(), axis_widths(mixture).detach(), patch, cutoff_sigma)
    return _Render.apply(mu, prec6, norm, *binning, patch, cutoff_sigma)
