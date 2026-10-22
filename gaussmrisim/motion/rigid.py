"""Interpolation-free rigid motion, applied per k-space sample.

For a pose ``(R, t)`` acting on the object, ``rho_t(r) = rho(R^T (r - t))`` and

    rho_hat_t(k) = exp(-i 2 pi k . t) * rho_hat(R^T k).                      (*)

Two consequences worth stating separately, because they are the whole point:

1. **No resampling anywhere.** Raster simulators must interpolate the volume into
   each pose, and that interpolation error is indistinguishable from the motion
   artefact being studied -- which is exactly the problem APHABAMAS was written to
   measure. Here the pose enters as a rotation of the query frequency and a phase.
2. **Per-sample motion is free.** Identity (*) touches only ``k``, not the
   primitives, so every one of the ``M`` samples may have its own pose at no extra
   cost -- no re-packing, no re-sort, no per-state k-space. Transforming the
   primitives instead (:meth:`GaussianMixture3D.rigid_transform`) is equally exact
   but costs ``O(N)`` per pose, so it is only preferable when poses are few.

The two paths are independent implementations of the same identity.

**Coils do not move.** The receive array is fixed in scanner coordinates while the
subject moves, so the coil shift ``v`` is applied in the scanner frame and the pose
inside it: ``m_c(k) = sum_v beta_{c,v} exp(-i 2 pi (k - v) . t) rho_hat(R^T (k - v))``.
Rotating the coils with the object is a classic and silent modelling error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from ..coils.fourier_csm import FourierCSM
from ..fourier.gaussian_ft import kspace
from ..representation.gaussian3d import GaussianMixture3D


def rotation_from_euler(angles_deg, device="cpu", dtype=torch.float32) -> Tensor:
    """Intrinsic x-y-z Euler angles in degrees -> ``[3, 3]``; batched ``[..., 3] -> [..., 3, 3]``.

    Batched deliberately: one pose *per k-space sample* is the operator this project
    claims is free, and a Python loop building S rotation matrices would cost more than
    the simulation it feeds. The composition ``Rz @ Ry @ Rx`` and every entry are
    identical to the unbatched form -- this only removes the in-place writes, which were
    what forbade a leading batch dimension.
    """
    a = torch.as_tensor(angles_deg, device=device, dtype=dtype) * torch.pi / 180.0
    if a.shape[-1] != 3:
        raise ValueError(f"expected Euler angles with last dim 3, got {tuple(a.shape)}")
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(c[..., 0]), torch.zeros_like(c[..., 0])

    def _m(rows) -> Tensor:                       # [[..],[..],[..]] -> [..., 3, 3]
        return torch.stack([torch.stack(r, -1) for r in rows], -2)

    Rx = _m([[o, z, z], [z, c[..., 0], -s[..., 0]], [z, s[..., 0], c[..., 0]]])
    Ry = _m([[c[..., 1], z, s[..., 1]], [z, o, z], [-s[..., 1], z, c[..., 1]]])
    Rz = _m([[c[..., 2], -s[..., 2], z], [s[..., 2], c[..., 2], z], [z, z, o]])
    return Rz @ Ry @ Rx


@dataclass
class RigidPoses:
    """One pose per k-space sample: ``R [S, 3, 3]``, ``t [S, 3]`` (mm)."""

    R: Tensor
    t: Tensor

    @staticmethod
    def still(n: int, device="cpu", dtype=torch.float32) -> "RigidPoses":
        return RigidPoses(torch.eye(3, device=device, dtype=dtype).expand(n, 3, 3).contiguous(),
                          torch.zeros(n, 3, device=device, dtype=dtype))

    @staticmethod
    def from_time(times: Tensor, pose_fn: Callable[[float], tuple]) -> "RigidPoses":
        """Sample a pose function ``t -> (R [3,3], t [3])`` at acquisition times."""
        Rs, ts = zip(*[pose_fn(float(x)) for x in times])
        return RigidPoses(torch.stack(Rs), torch.stack(ts))

    def __len__(self) -> int:
        return int(self.t.shape[0])


def moved_kspace(mixture: GaussianMixture3D, k: Tensor, poses: RigidPoses,
                 coeff: Optional[Tensor] = None, chunk: int = 0,
                 backend: str = "auto",
                 r2s_per_ms: Optional[Tensor] = None,
                 dt_ms: Optional[Tensor] = None) -> Tensor:
    """``rho_hat`` at ``k [S, 3]`` with a per-sample pose -> complex ``[S]``.

    Implements identity (*) on the frequency side: ``k -> R^T k`` plus a phase. Both
    are ``O(S)`` and independent of ``N``; the transformed query is then handed to the
    *same* evaluator a static acquisition uses, which is the entire content of the claim
    that a per-sample pose is free.

    ``backend="auto"`` therefore dispatches to the CUDA kernel whenever the input is on
    CUDA. It must: timing this against a static acquisition that ran the CUDA kernel,
    while this one silently ran the PyTorch reference, measures the ~40x backend gap and
    reports it as the cost of motion -- refuting the claim with an artefact.
    """
    kl = torch.einsum("sij,si->sj", poses.R.to(k.dtype), k)              # R^T k
    if backend == "auto":
        backend = "cuda" if k.is_cuda else "torch"
    if backend == "cuda":
        from ..fourier.cuda_ft import kspace_cuda
        val = kspace_cuda(mixture, kl.contiguous(), coeff=coeff, k_chunk=chunk,
                          r2s_per_ms=r2s_per_ms, dt_ms=dt_ms)
    else:
        if r2s_per_ms is not None:
            raise NotImplementedError(
                "the readout decay is CUDA-only; the torch reference has no path "
                "for it, and silently dropping it would understate the signal")
        val = kspace(mixture, kl.contiguous(), coeff=coeff, chunk=chunk)
    ph = 2 * torch.pi * torch.remainder((k.double() * poses.t.double()).sum(-1), 1.0)
    return val * torch.polar(torch.ones_like(ph), -ph).to(val.dtype)


def moved_coil_kspace(mixture: GaussianMixture3D, k: Tensor, csm: FourierCSM,
                      poses: RigidPoses, coeff: Optional[Tensor] = None,
                      mode_chunk: int = 0) -> Tensor:
    """Multi-coil k-space under per-sample rigid motion -> ``[C, S]``.

    The coil modulation is applied in the *scanner* frame (shift by ``v`` first) and
    the pose acts on the object inside it -- see the module docstring.
    """
    v, s = csm.n_modes, k.shape[0]
    beta = csm.coeffs.to(torch.complex64)
    mode_chunk = mode_chunk or v
    out = torch.zeros(csm.n_coils, s, device=k.device, dtype=torch.complex64)
    for a in range(0, v, mode_chunk):
        sh = csm.freqs[a: a + mode_chunk]                                # [v', 3]
        acc = []
        for i in range(sh.shape[0]):
            acc.append(moved_kspace(mixture, k - sh[i][None, :], poses, coeff=coeff))
        out += beta[:, a: a + mode_chunk] @ torch.stack(acc, 0)
    return out
