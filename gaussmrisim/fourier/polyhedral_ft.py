"""Exact Fourier transform of closed polyhedral surfaces (Ngo et al., MRM 2016).

A Python/CUDA port of ``fastmtpolyhedralFTcellpoly.cpp`` from
``github.com/trimngo/polyphantom`` (MATLAB MEX, CPU, 2010), for triangle meshes. Same
convention as the rest of the project, ``F(k) = int rho(r) exp(-i 2 pi k.r) dr`` with k
in cycles/mm.

For a closed, outward surface with faces ``f`` (unit normal ``N_f``) and edges ``e``
(length ``L_e``, unit direction ``t_e``, in-plane outward normal ``n_e = t_e x N_f``,
midpoint ``c_e``), the indicator of its interior transforms as

    F(k) = -1/(2 pi |k|)^2 * sum_f S_f(k),
    S_f(k) = (k.N_f) / (|k|^2 - (k.N_f)^2) * sum_e L_e (k.n_e) sinc(L_e k.t_e) e^{-i2pi k.c_e}

with ``sinc(x) = sin(pi x)/(pi x)``; when ``k`` is parallel to ``N_f`` the face term is
``-i pi (k.N_f) 2A_f exp(-i 2 pi k.v0_f)``, and ``F(0)`` is the volume.

**Precision is not optional here.** The face terms are of order ``perimeter/|k|`` and
cancel down to ``|k|^2 V``; with ~1e6 faces and ``|k| = 1/FOV`` the cancellation is
~1e5, so float32 face terms cannot represent the low-frequency spectrum. The kernel is
built for both float64 and float32 and the float32 error is measured, never assumed.
Phases ``k.c_e`` are formed in float64 and reduced mod 1 cycle in both variants
(``conventions.cycles_mod_one``'s rule).
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from ..datasets.polyhedral import Mesh

#: per-face record: N(3) | for e in 0..2: L n_e (3), L t_e (3), c_e (3) | v0 (3) | 2A | value
REC = 3 + 27 + 3 + 1 + 1


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    src = Path(__file__).resolve().parent / "cuda"
    return load(name="_gmrisim_polyhedral_ft",
                sources=[str(src / "polyhedral_ft_kernel.cu")],
                extra_cuda_cflags=["-O3"],          # no fast-math: precision is the point
                verbose=bool(int(os.environ.get("GMRISIM_VERBOSE_CUDA_BUILD", "0"))))


def pack(meshes: Sequence[Mesh], values: Sequence[float], device="cuda") -> Tensor:
    """All faces of all surfaces -> ``[F, REC]`` float64, value folded in per face."""
    rows = []
    for m, val in zip(meshes, values):
        t = m.tri().to(device=device, dtype=torch.float64)          # [F, 3, 3]
        cr = torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=-1)
        two_a = cr.norm(dim=-1)
        N = cr / two_a[:, None]
        cols = [N]
        for e in range(3):
            a, b = t[:, e], t[:, (e + 1) % 3]
            Lt = b - a                                               # L * t
            Ln = torch.cross(Lt, N, dim=-1)                          # L * n (|t x N| = 1)
            cols += [Ln, Lt, 0.5 * (a + b)]
        cols += [t[:, 0], two_a[:, None],
                 torch.full_like(two_a[:, None], float(val))]
        rows.append(torch.cat(cols, dim=1))
    return torch.cat(rows).contiguous()


def volume(meshes: Sequence[Mesh], values: Sequence[float]) -> float:
    """``F(0) = sum_c value_c * V_c``."""
    return float(sum(v * m.signed_volume() for m, v in zip(meshes, values)))


def polyhedral_ft(faces: Tensor, k: Tensor, precision: str = "double",
                  k_chunk: int = 65536) -> Tensor:
    """CUDA evaluation -> complex128 ``[M]``. ``faces`` from :func:`pack`.

    ``precision``: "double" (fp64 throughout, the reference) or "f32" (fp32 with a
    compensated phase, the variant comparable to the Gaussian kernel; face-k pairs
    within ~6 degrees of parallel fall back to fp64, see the kernel).
    """
    ext = _ext()
    k64 = k.to(device=faces.device, dtype=torch.float64).contiguous()
    out = torch.empty(k64.shape[0], dtype=torch.complex128, device=faces.device)
    if precision == "f32":
        # all-float32: records and k cast once, phase by compensated fp32 dot product
        f, fa, kk = ext.forward_f32, faces.float().contiguous(), k64.float().contiguous()
    else:
        if precision != "double":
            raise ValueError(f"precision must be 'double' or 'f32', got {precision!r}")
        f, fa, kk = ext.forward_double, faces, k64
    for a in range(0, k64.shape[0], k_chunk):
        out[a:a + k_chunk] = torch.view_as_complex(f(fa, kk[a:a + k_chunk]))
    return out


def polyhedral_ft_reference(faces: Tensor, k: Tensor, face_chunk: int = 4096) -> Tensor:
    """Plain PyTorch float64, the accuracy reference for the kernel. ``[M]`` complex128.

    Excludes k = 0 and exactly-parallel faces (the kernel handles both); used only on
    random k where neither occurs.
    """
    k = k.to(faces.device, torch.float64)
    kk = (k * k).sum(-1)
    acc = torch.zeros(k.shape[0], dtype=torch.complex128, device=faces.device)
    for a in range(0, faces.shape[0], face_chunk):
        r = faces[a:a + face_chunk]
        N = r[:, 0:3]
        kN = k @ N.T                                                  # [M, Fc]
        s = torch.zeros_like(kN, dtype=torch.complex128)
        for e in range(3):
            Ln, Lt, c = (r[:, 3 + 9 * e:6 + 9 * e], r[:, 6 + 9 * e:9 + 9 * e],
                         r[:, 9 + 9 * e:12 + 9 * e])
            kn, kt, kc = k @ Ln.T, k @ Lt.T, torch.remainder(k @ c.T, 1.0)
            s = s + kn * torch.sinc(kt) * torch.exp(-2j * math.pi * kc)
        acc += (r[:, -1] * kN / (kk[:, None] - kN * kN) * s).sum(-1)
    return -acc / (2 * math.pi) ** 2 / kk
