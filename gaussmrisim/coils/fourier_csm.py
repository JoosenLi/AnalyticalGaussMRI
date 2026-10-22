"""Analytical parallel MRI: Fourier-basis receive sensitivities.

    s_c(r) = sum_v beta_{c,v} exp(i 2 pi v . r)
    =>  F{s_c rho}(k) = sum_v beta_{c,v} rho_hat(k - v)

The identity is the Fourier modulation theorem and is **dimension independent** --
it holds in 3D exactly as in 2D. Ngo et al. (MRM 2016) state that "no solution
exists for 3D phantoms that results in a purely analytical expression"; that was
true of *polyhedral* phantoms in 2015 and was closed by BART's DRO work (Scholand
et al., ISMRM 2023) for Bezier geometry with precomputed 3D coil profiles. So the
identity is not a contribution here -- the contribution, if any, is what it costs.

Cost structure (this is the part that matters)
----------------------------------------------
The naive cost is ``O(N M V C)``. Two of those factors are avoidable:

* **C is free.** All coils share one frequency set, so evaluate the shifted object
  spectrum ``H[v, m] = rho_hat(k_m - v)`` once and mix coils with a ``[C, V] @ [V, M]``
  matmul. Never evaluate the mixture per coil.
* **V is compressible.** Sensitivities are smooth, so their spectra are concentrated;
  :func:`select_modes` picks the modes carrying a stated fraction of the joint
  across-coil energy, and the resulting CSM error is reported, not assumed.

What remains, ``O(N M V)``, is the real cost and is where the spectral cull applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..fourier.gaussian_ft import kspace
from ..representation.gaussian3d import GaussianMixture3D


@dataclass
class FourierCSM:
    """``C`` coils over a shared set of ``V`` spatial frequencies (cycles/mm)."""

    freqs: Tensor          # [V, 3] cycles/mm
    coeffs: Tensor         # [C, V] complex

    @property
    def n_coils(self) -> int:
        return int(self.coeffs.shape[0])

    @property
    def n_modes(self) -> int:
        return int(self.freqs.shape[0])

    def to(self, device) -> "FourierCSM":
        return FourierCSM(self.freqs.to(device), self.coeffs.to(device))

    def sensitivity(self, points: Tensor, chunk: int = 0) -> Tensor:
        """Evaluate ``s_c(r)`` at continuous ``points [P, 3]`` -> ``[C, P]`` complex."""
        chunk = chunk or max(1, int(4e7 // max(self.n_modes, 1)))
        out = torch.empty(self.n_coils, points.shape[0], device=points.device,
                          dtype=torch.complex64)
        for a in range(0, points.shape[0], chunk):
            ph = 2 * torch.pi * (points[a: a + chunk] @ self.freqs.T)
            out[:, a: a + chunk] = self.coeffs.to(torch.complex64) @ torch.polar(
                torch.ones_like(ph), ph).to(torch.complex64).T
        return out

    def select_modes(self, energy: float = 0.999) -> "FourierCSM":
        """Keep the joint top-K modes carrying ``energy`` of the across-coil power.

        Joint, not per coil: the coils share the frequency set, so dropping a mode
        that one coil needs costs that coil everything. Ranking by
        ``sum_c |beta_{c,v}|^2`` is the only selection consistent with sharing.
        """
        p = self.coeffs.abs().square().sum(0)
        order = torch.argsort(p, descending=True)
        c = torch.cumsum(p[order], 0) / p.sum().clamp_min(1e-30)
        keep = order[: int(torch.searchsorted(c, torch.tensor(float(energy), device=c.device)) + 1)]
        return FourierCSM(self.freqs[keep].contiguous(), self.coeffs[:, keep].contiguous())


def cartesian_mode_grid(n_side: int, fov_mm, device="cpu", dtype=torch.float32,
                        harmonic: float = 0.5) -> Tensor:
    """A centred low-frequency cube of ``n_side^3`` modes, spacing ``harmonic / FOV``.

    ``harmonic = 1`` puts the modes on true FOV harmonics, which makes the basis
    orthogonal over the FOV and the Gram matrix well conditioned. ``harmonic = 0.5``
    (the default, and AGNC's ``df = pi/2`` on its ``[-1, 1]`` box) halves the spacing:
    the lowest modes then vary over *twice* the FOV, which is what real sensitivities
    look like -- smooth, with no periodicity at the FOV edge. The price is a
    non-orthogonal basis that needs the ridge, and that price is why ``ridge`` is not
    optional in :func:`fit_fourier_csm`.
    """
    f = torch.as_tensor(fov_mm, device=device, dtype=dtype)
    j = (torch.arange(n_side, device=device, dtype=dtype) - (n_side - 1) / 2.0) * float(harmonic)
    g = torch.meshgrid(j / f[0], j / f[1], j / f[2], indexing="ij")
    return torch.stack([a.reshape(-1) for a in g], dim=-1).contiguous()


def fit_fourier_csm(maps: Tensor, points: Tensor, freqs: Tensor,
                    ridge: float = 1e-3, weights: Optional[Tensor] = None,
                    support_radius_mm: Optional[float] = None,
                    support_centre_mm=(0.0, 0.0, 0.0),
                    noise_std: float = 0.0, seed: int = 0) -> FourierCSM:
    """Least-squares fit of ``[C, P]`` complex maps sampled at ``points [P, 3]``.

    The Gram matrix is ill-conditioned whenever the modes are dense relative to the
    support (AGNC measured coefficients ~12x larger than the map itself, cancelling in
    the sum), so ``ridge`` is **relative to the Gram diagonal**, which is the number of
    sample points. An absolute ridge is meaningless and silently leaves the fit
    unregularized as the sample count grows.
    """
    ph = 2 * torch.pi * (points.double() @ freqs.double().T)
    A = torch.polar(torch.ones_like(ph), ph)                       # [P, V] complex128
    if weights is not None:
        A = A * weights.double().sqrt()[:, None]
        maps = maps * weights.sqrt()[None, :]
    G = A.conj().T @ A
    G = G + float(ridge) * torch.diagonal(G).abs().mean() * torch.eye(
        G.shape[0], device=G.device, dtype=G.dtype)
    rhs = A.conj().T @ maps.to(torch.complex128).T                 # [V, C]
    beta = torch.linalg.solve(G, rhs)
    return FourierCSM(freqs, beta.T.to(torch.complex64).contiguous())


def normalize_rss(csm: FourierCSM, points: Tensor) -> FourierCSM:
    """Scale coefficients so the root-sum-of-squares sensitivity peaks at 1 on ``points``.

    ``points`` must be the object support, not the whole box: a fit is only meaningful
    where it was fitted, and outside it the expansion can diverge -- normalizing by that
    spurious peak would shrink the entire map.
    """
    with torch.no_grad():
        peak = float(csm.sensitivity(points).abs().square().sum(0).sqrt().max().clamp_min(1e-12))
    return FourierCSM(csm.freqs, csm.coeffs / peak)


def coil_kspace(mixture: GaussianMixture3D, k: Tensor, csm: FourierCSM,
                coeff: Optional[Tensor] = None, mode_chunk: int = 0,
                k_chunk: int = 0) -> Tensor:
    """Multi-coil analytical k-space ``[C, S]``.

    ``mode_chunk`` bounds the peak memory of the ``[V, S]`` shifted spectrum, which is
    the largest intermediate in the whole simulator (``V = 1000``, ``S = 10^6`` is
    8 GB in complex64). Chunking over modes accumulates into the ``[C, S]`` output
    instead, so peak memory becomes ``O(C S + mode_chunk * S)``.
    """
    v, s = csm.n_modes, k.shape[0]
    beta = csm.coeffs.to(torch.complex64)
    mode_chunk = mode_chunk or v
    out = torch.zeros(csm.n_coils, s, device=k.device, dtype=torch.complex64)
    for a in range(0, v, mode_chunk):
        shifts = csm.freqs[a: a + mode_chunk]                       # [v', 3]
        kk = (k[None, :, :] - shifts[:, None, :]).reshape(-1, 3)    # [v'*S, 3]
        H = kspace(mixture, kk, coeff=coeff, chunk=k_chunk).reshape(shifts.shape[0], s)
        out += beta[:, a: a + mode_chunk] @ H
    return out
