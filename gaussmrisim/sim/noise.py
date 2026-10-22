"""Complex Gaussian noise at a target SNR, correlated across coils.

SNR is defined against the *ideal* object's mean squared signal, as in SNAKE, so that a
number is comparable between the two simulators and between sequences. Noise is added in
k-space because that is where a scanner adds it; adding it in the image domain and
transforming would correlate it across samples.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def coil_noise(shape, coil_cov: Tensor, gen: torch.Generator,
               device="cpu") -> Tensor:
    """Complex noise ``[..., C, M]`` with covariance ``coil_cov [C, C]`` across coils."""
    c = coil_cov.shape[0]
    L = torch.linalg.cholesky(coil_cov.to(torch.complex64)
                              + 1e-12 * torch.eye(c, device=coil_cov.device,
                                                  dtype=torch.complex64))
    n = torch.randn(*shape, dtype=torch.float32, device=device, generator=gen) \
        + 1j * torch.randn(*shape, dtype=torch.float32, device=device, generator=gen)
    n = n / (2.0 ** 0.5)                       # unit variance per complex sample
    return torch.einsum("ij,...jm->...im", L, n)


def cov_for_snr(signal_power: float, snr: float, n_coils: int,
                base: Optional[Tensor] = None, device="cpu") -> Tensor:
    """Scale a coil covariance so the resulting noise gives ``snr`` against the signal."""
    cov = torch.eye(int(n_coils), device=device) if base is None else base.to(device)
    return cov * (float(signal_power) / float(snr))
