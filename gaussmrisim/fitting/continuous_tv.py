"""Monte-Carlo total variation of a Gaussian mixture, for the image-domain fit.

Ported from the ICLR demo (``recon/continuous_tv.py``): an unbiased estimate of the
Huber-TV functional

    TV(rho) = (1/|B|) int_B sqrt(h^2 |grad rho(r)|^2 + delta^2) dr

at uniformly drawn points of the box ``B``, with the ANALYTIC gradient

    grad rho(r) = - sum_n c_n Sigma_n^-1 (r - mu_n) exp(-0.5 (r-mu_n)^T Sigma_n^-1 (r-mu_n)),
    c_n = m_n / ((2 pi)^{3/2} |Sigma_n|^{1/2})           (peak height of mass m_n)

-- no grid, no render cutoff, no finite-difference bias; its sampling variance is
absorbed by the optimizer like any minibatch. ``h`` (the voxel size) puts it in the
finite-difference convention (value change per voxel), so ``tv_weight`` means the
same thing at every resolution.

What it is for: the image loss only sees the mixture at voxel centres, so it is blind to
oscillation BETWEEN them -- ripples, and the narrow spikes whose spectral replicas were
measured to dominate the in-band k-space error (2026-10-15). TV charges every such
oscillation and charges a clean edge only its jump times area, independent of how sharp
the edge is. It changes the FIT (the object is still whatever mixture comes out, and its
k-space is evaluated exactly); report it with its weight.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

_SYM = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    src = Path(__file__).resolve().parent / "cuda"
    return load(name="_gmrisim_grad_rho", sources=[str(src / "grad_rho_kernel.cu")],
                extra_cuda_cflags=["-O3"],
                verbose=bool(int(os.environ.get("GMRISIM_VERBOSE_CUDA_BUILD", "0"))))


def geometry(mixture) -> tuple[Tensor, Tensor, Tensor]:
    """``(mu [N,3], K = Sigma^-1 packed [N,6], c peak [N])`` -- differentiable."""
    R = mixture.rotation()
    s = mixture.effective_sigma()
    K = R @ torch.diag_embed(s.pow(-2)) @ R.transpose(-1, -2)
    kp = torch.stack([K[:, i, j] for i, j in _SYM], -1)
    c = mixture.mass / ((2 * math.pi) ** 1.5 * s.prod(-1))
    return mixture.mu, kp, c


class _GradRho(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pts, mu, kp, c, chunk):
        ctx.save_for_backward(pts, mu, kp, c)
        return _ext().grad_rho_forward(pts.contiguous(), mu.contiguous(), kp.contiguous(),
                                       c.contiguous(), int(chunk))

    @staticmethod
    def backward(ctx, g):
        pts, mu, kp, c = ctx.saved_tensors
        gmu, gk, gc = _ext().grad_rho_backward(g.contiguous(), pts.contiguous(),
                                               mu.contiguous(), kp.contiguous(),
                                               c.contiguous())
        return None, gmu, gk, gc, None


def grad_rho(pts: Tensor, mu: Tensor, kp: Tensor, c: Tensor, chunk: int = 1024) -> Tensor:
    """``grad rho`` at ``pts [M,3]`` -> ``[M,3]`` (CUDA, float32, differentiable)."""
    return _GradRho.apply(pts, mu, kp, c, chunk)


def grad_rho_reference(pts: Tensor, mu: Tensor, kp: Tensor, c: Tensor) -> Tensor:
    """PyTorch reference of :func:`grad_rho` (any dtype; for the checks)."""
    K = torch.zeros(kp.shape[0], 3, 3, dtype=kp.dtype, device=kp.device)
    for idx, (i, j) in enumerate(_SYM):
        K[:, i, j] = kp[:, idx]
        K[:, j, i] = kp[:, idx]
    d = pts[:, None, :] - mu[None]                                    # [M,N,3]
    z = torch.einsum("nij,mnj->mni", K, d)
    b = torch.exp(-0.5 * (d * z).sum(-1))                            # [M,N]
    return -torch.einsum("mni,mn,n->mi", z, b, c)


def sampled_tv(mixture, lo: Tensor, hi: Tensor, n_samples: int, h: float,
               delta: float, generator: torch.Generator | None = None) -> Tensor:
    """``mean_q sqrt(h^2 |grad rho(r_q)|^2 + delta^2)``, ``r_q`` uniform in ``[lo, hi]`` (mm)."""
    pts = lo + (hi - lo) * torch.rand(int(n_samples), 3, device=lo.device,
                                      generator=generator, dtype=lo.dtype)
    g = grad_rho(pts, *geometry(mixture))
    return torch.sqrt(h * h * g.square().sum(-1) + delta * delta).mean()
