"""Closed-form continuous Fourier transform of the Gaussian mixture.

    rho_hat(k) = sum_n a_n m_n exp(-2 pi^2 k^T Sigma_n k) exp(-i 2 pi k . mu_n)

This module is the **accuracy reference** of the whole project. Rules:

* it is evaluated at arbitrary ``k``, never on a grid, and never through an NUFFT;
* it uses the *untruncated* Gaussian -- any image-domain cutoff used while fitting
  is a property of the fit, not of the object, and must not appear here;
* the phase is reduced modulo one cycle in float64 before casting
  (see :mod:`gaussmrisim.conventions`), so results are float32-clean even at
  100 um resolution over a 200 mm FOV.

The fused CUDA implementation uses the same transform and phase reduction.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..conventions import TWO_PI_SQ, cycles_mod_one
from ..representation.gaussian3d import GaussianMixture3D


def _quad_form(k: Tensor, mixture: GaussianMixture3D) -> Tensor:
    """``k^T Sigma_n k`` -> ``[S, N]``, via the principal axes (no 3x3 products).

    ``k^T R diag(sigma^2) R^T k = || sigma * (R^T k) ||^2``, so the cost is one
    ``[S, N, 3]`` rotation instead of an explicit covariance, and the same
    ``R^T k`` is what a fused kernel would compute.
    """
    R, s = mixture.rotation(), mixture.sigma()
    local = torch.einsum("sj,njk->snk", k, R)            # R^T k, [S, N, 3]
    q = (local * s[None]).square().sum(-1)
    if mixture.dilation_mm:                              # + s^2 |k|^2 from the filter
        q = q + (float(mixture.dilation_mm) ** 2) * k.square().sum(-1)[:, None]
    return q


def kspace(mixture: GaussianMixture3D, k: Tensor, coeff: Optional[Tensor] = None,
           chunk: int = 0, out_dtype=torch.complex64) -> Tensor:
    """Object k-space at ``k [S, 3]`` in cycles/mm -> complex ``[S]``.

    ``coeff [N]`` is the per-primitive contrast coefficient ``a_n(theta)`` from the
    physics layer; ``None`` means proton-density-like (all ones).
    """
    if k.ndim != 2 or k.shape[-1] != 3:
        raise ValueError(f"k must be [S, 3] cycles/mm, got {tuple(k.shape)}")
    n = mixture.n
    w = mixture.mass.to(out_dtype) if coeff is None else (mixture.mass * coeff).to(out_dtype)
    chunk = chunk or max(1, int(4e7 // max(n, 1)))
    out = torch.empty(k.shape[0], device=k.device, dtype=out_dtype)
    for a in range(0, k.shape[0], chunk):
        kk = k[a: a + chunk]
        env = torch.exp(-TWO_PI_SQ * _quad_form(kk, mixture))           # [s, N] real
        ph = 2 * torch.pi * cycles_mod_one(kk, mixture.mu)              # [s, N] rad
        basis = torch.polar(env, -ph).to(out_dtype)
        out[a: a + chunk] = basis @ w
    return out


def spectral_envelope(mixture: GaussianMixture3D, k_radius: Tensor,
                      coeff: Optional[Tensor] = None) -> Tensor:
    """Per-primitive upper bound on ``|rho_hat_n(k)|`` at radius ``|k| = r``.

    ``|rho_hat_n(k)| = |a_n m_n| exp(-2 pi^2 k^T Sigma_n k) <=
    |a_n m_n| exp(-2 pi^2 |k|^2 sigma_min,n^2)``, because the quadratic form is
    bounded below by the smallest principal width. This is what makes error-controlled
    culling possible for Gaussians and impossible for polyhedra: a facet contributes
    at every frequency with only a ``|k|^-2`` envelope, whereas a primitive whose
    smallest width is ``sigma`` is *provably* negligible beyond
    ``|k| > sqrt(ln(|a m| / eps) / (2 pi^2)) / sigma``.

    Returns ``[R, N]`` for ``k_radius [R]``.
    """
    m = mixture.mass.abs() if coeff is None else (mixture.mass * coeff).abs()
    smin = mixture.sigma().min(dim=-1).values                          # [N]
    return m[None, :] * torch.exp(-TWO_PI_SQ * (k_radius[:, None] * smin[None, :]).square())


def cull_radius(mixture: GaussianMixture3D, tol: float,
                coeff: Optional[Tensor] = None) -> Tensor:
    """Frequency radius beyond which each primitive is below ``tol`` in magnitude.

    Inverts :func:`spectral_envelope`. Primitives with ``|a m| <= tol`` never matter
    and get radius 0. Used to turn "skip negligible interactions" into a statement
    with a bound instead of an unexplained sigma cutoff.
    """
    m = (mixture.mass.abs() if coeff is None else (mixture.mass * coeff).abs())
    smin = mixture.sigma().min(dim=-1).values
    ratio = (m / float(tol)).clamp_min(1.0)
    return torch.sqrt(torch.log(ratio) / TWO_PI_SQ) / smin.clamp_min(1e-12)
