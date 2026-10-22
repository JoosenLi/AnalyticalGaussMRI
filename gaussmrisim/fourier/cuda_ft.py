"""CUDA entry point for the analytical Fourier forward model.

Same mathematics as :mod:`gaussmrisim.fourier.gaussian_ft`, which stays the accuracy
reference: this module is only allowed to be faster, never different, and
``tests/verify_cuda.py`` asserts that on every call signature.

Chunking is over **k-space samples**, matching AGNC's convention. It is needed for a
different reason here: the output is only ``[C, M]`` (the ``[V, M]`` spectrum lives in
registers), so chunking bounds the output and the launch grid, not an intermediate.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from ..coils.fourier_csm import FourierCSM
from ..representation.gaussian3d import GaussianMixture3D

_SYM = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    src = Path(__file__).resolve().parent / "cuda"
    return load(name="_gmrisim_gaussian_ft3d",
                sources=[str(src / "gaussian_ft3d_kernel.cu")],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=bool(int(os.environ.get("GMRISIM_VERBOSE_CUDA_BUILD", "0"))))


def pack_cov6(mixture: GaussianMixture3D) -> Tensor:
    """``Sigma`` as ``[N, 6]`` (xx, xy, xz, yy, yz, zz), mm^2.

    Built in PyTorch rather than in the kernel so the quaternion normalization and the
    log-scale map stay in one place and are differentiable; the kernel then never has
    to know the parameterization. Same split AGNC uses for its tiled kernel.
    """
    cov = mixture.covariance()
    return torch.stack([cov[:, i, j] for i, j in _SYM], dim=-1).contiguous()


def kspace_cuda(mixture: GaussianMixture3D, k: Tensor,
                csm: Optional[FourierCSM] = None,
                coeff: Optional[Tensor] = None,
                k_chunk: int = 0,
                r2s_per_ms: Optional[Tensor] = None,
                dt_ms: Optional[Tensor] = None) -> Tensor:
    """Analytical k-space on the GPU -> ``[M]`` (no CSM) or ``[C, M]`` (with CSM).

    ``coeff [N]`` is the contrast coefficient ``a_n(theta)``; it multiplies the mass
    before the kernel sees it, so contrast costs nothing at evaluation time.

    ``r2s_per_ms [N]`` together with ``dt_ms [M]`` (time since the echo) adds the
    readout T2* decay ``exp(-dt_m * r2s_n)`` **inside** the kernel, folded into the
    exponential the Gaussian magnitude already computes -- so it costs no extra
    transcendental. This is only well defined because a primitive belongs to exactly
    one tissue, which is what per-tissue fitting buys; on a shared voxel grid the
    decay differs per tissue and the alternative is one full evaluation per tissue.

    It changes the object, so it is opt-in: passing neither argument reproduces the
    previous behaviour exactly.
    """
    if not k.is_cuda:
        raise ValueError("kspace_cuda needs CUDA tensors; never fall back to CPU")
    w = mixture.mass.to(torch.complex64) if coeff is None else (mixture.mass * coeff).to(torch.complex64)
    mu = mixture.mu.float().contiguous()
    cov6 = pack_cov6(mixture).float().contiguous()
    wr, wi = w.real.contiguous(), w.imag.contiguous()

    if csm is None:
        shifts = torch.zeros(1, 3, device=k.device, dtype=torch.float32)
        br = bi = None
        c = 1
    else:
        shifts = csm.freqs.float().contiguous()
        beta = csm.coeffs.to(torch.complex64)
        br, bi = beta.real.contiguous(), beta.imag.contiguous()
        c = csm.n_coils

    m = int(k.shape[0])
    k_chunk = k_chunk or m
    out = torch.empty(c, m, device=k.device, dtype=torch.complex64)
    for a in range(0, m, k_chunk):
        kk = k[a: a + k_chunk].float().contiguous()
        dt = None if dt_ms is None else dt_ms[a: a + k_chunk].float().contiguous()
        r2s = None if r2s_per_ms is None else r2s_per_ms.float().contiguous()
        re, im = _ext().forward(kk, shifts, mu, cov6, wr, wi, br, bi, r2s, dt)
        out[:, a: a + k_chunk] = torch.complex(re, im)
    return out[0] if csm is None else out
