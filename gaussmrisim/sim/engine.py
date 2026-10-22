"""The acquisition engine: object + trajectory -> k-space, entirely in closed form.

Shaped after SNAKE's ``BaseAcquisitionEngine`` (a ``simple`` and a ``t2star`` model, shots
as the unit of work, noise at a target SNR) so the two are comparable, but every stage
that SNAKE does on a grid is analytic here: the object is a Gaussian mixture, the coil
maps are a Fourier series, and a pose acts on the object rather than on a resampled copy
of it. There is no NUFFT and no gridding anywhere in this file.

**The two models.**

``simple`` (default) evaluates the contrast at the echo and holds it over the readout.
It is one kernel launch: the coefficient is a per-primitive gather into the argument the
FT kernel already takes, so this model costs exactly what a proton-density simulation
costs.

``t2star`` lets the signal decay along the readout, ``exp(-(t_m - TE)/T2*_t)``. Two
implementations are provided, they agree **bit-exactly**, and the measurement corrected
the reasoning that motivated the second one:

* ``per_tissue`` (default) evaluates each tissue and sums, which is the shape SNAKE uses.
* ``kernel`` carries ``1/T2*`` per primitive and folds the decay into the exponential the
  FT kernel already computes.

The design note claimed SNAKE needs T NUFFTs where we would need one launch, so the
kernel form should win by ~T. **That is wrong, and only half of it was ever true.** SNAKE
needs T transforms because its tissues share a voxel grid and each transform is over the
whole volume. Splitting *this* object by tissue splits the **primitives**: T launches over
N/T primitives each is the same N.M work as one launch over N. The per-tissue loop was
never paying a T-fold penalty, so there was nothing for the kernel form to recover.

Measured at N=400k, M=66k, T=11, single coil: simple 0.141 s, per_tissue 0.239 s
(1.69x simple), kernel 0.267 s (1.89x). The kernel form is **0.9x** -- a slowdown, and
for a reason this project already documented: the FT kernel is L2-resident in the
primitive parameters, and ``r2s[n]`` grows that working set from 44 to 48 bytes per
primitive. The advantage over SNAKE is real and unchanged (one pass over primitives
against T full-volume transforms); the advantage of the kernel form over our own loop
does not exist.

``kernel`` is kept because it does not require tissue-pure primitives, which a future
shared-geometry mode would not have.

**Motion with coils: an optimization that looked obvious and does not pay.** Because a
pose is constant within a shot, the coil sum can be rewritten

    m_c(k) = sum_v b_cv rho_hat_moved(k - v)
           = exp(-i2pi k.t) * sum_v [b_cv exp(+i2pi v.t)] * rho_hat(R^T k - R^T v)

so rotating the modes to ``R^T v`` and folding the shot's phase into the coil
coefficients gives **one launch per shot instead of V**. That is implemented below and
verified equivalent to 4.2e-7 -- and it is **not** the default, because measuring it
refuted the premise. Counting launches was the mistake: ``moved_coil_kspace`` performs V
separate N x M evaluations, while the per-shot form performs S launches each covering V
modes over M/S samples. Both are **V.N.M** operations; the per-shot version merely splits
the same work into many small, poorly occupied launches. Measured at N=200k, M=33k,
S=256, V=125, C=8: static coils 10.68 s, V launches 11.23 s (1.05x), per-shot launches
12.54 s (1.17x) -- so the "optimization" is 0.9x, a slowdown.

What the in-kernel coil accumulation actually buys is **memory**, not arithmetic: it
never materializes the [V, M] shifted spectrum. The per-shot path keeps that property and
is therefore retained behind ``per_shot_coils=True`` for cases where the V x M stack is
the binding constraint, not for speed.

The result that does survive: **motion costs 5% over a static multi-coil acquisition**
(11.23 s against 10.68 s), which is the claim H4 makes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..acquisition.trajectory import Trajectory
from ..coils.fourier_csm import FourierCSM
from ..fourier.cuda_ft import kspace_cuda
from ..motion.rigid import RigidPoses, moved_coil_kspace, moved_kspace
from ..representation.gaussian3d import GaussianMixture3D
from .config import SimConfig
from .motion import expand_to_samples
from .noise import coil_noise, cov_for_snr
from .phantom import TissuePhantom

MODELS = ("simple", "t2star")


def _phase(k: Tensor, t: Tensor) -> Tensor:
    """``exp(-i 2 pi k.t)``, reduced mod one cycle in float64 before casting.

    ``k.mu`` reaches hundreds of cycles in physical units; float32 loses ~1e-4 rad there.
    This is the same reduction ``conventions.cycles_mod_one`` performs and it is not
    optional -- AGNC's kernels omit it because a normalized box never needed it.
    """
    ph = 2 * math.pi * torch.remainder((k.double() * t.double()).sum(-1), 1.0)
    return torch.polar(torch.ones_like(ph), -ph).to(torch.complex64)


def _static(mix: GaussianMixture3D, k: Tensor, csm: Optional[FourierCSM],
            coeff: Optional[Tensor], k_chunk: int) -> Tensor:
    return kspace_cuda(mix, k, csm=csm, coeff=coeff, k_chunk=k_chunk)


def _moved_coil_by_shot(mix: GaussianMixture3D, traj: Trajectory, csm: FourierCSM,
                        poses: RigidPoses, coeff: Optional[Tensor],
                        k_chunk: int) -> Tensor:
    """Coils under per-shot motion, one launch per shot. See the module docstring."""
    k, shot = traj.k, traj.shot.long()
    out = torch.zeros(csm.n_coils, k.shape[0], device=k.device, dtype=torch.complex64)
    beta = csm.coeffs.to(torch.complex64)                        # [C, V]
    for s in range(int(shot.max()) + 1):
        sel = shot == s
        if not bool(sel.any()):
            continue
        R, t = poses.R[s], poses.t[s]
        ks = k[sel]
        v_rot = csm.freqs @ R                                    # R^T v, [V, 3]
        ph_v = 2 * math.pi * torch.remainder(
            (csm.freqs.double() * t.double()[None, :]).sum(-1), 1.0)
        beta_s = beta * torch.polar(torch.ones_like(ph_v), ph_v).to(beta.dtype)[None, :]
        shifted = FourierCSM(coeffs=beta_s, freqs=v_rot)
        val = kspace_cuda(mix, (ks @ R).contiguous(), csm=shifted, coeff=coeff,
                          k_chunk=k_chunk)                       # [C, m]
        out[:, sel] = val * _phase(ks, t)[None, :]
    return out


def _evaluate(mix: GaussianMixture3D, traj: Trajectory, csm: Optional[FourierCSM],
              coeff: Optional[Tensor], poses: Optional[RigidPoses],
              k_chunk: int, per_shot_coils: bool = False) -> Tensor:
    """One object -> ``[M]`` or ``[C, M]``, covering all four motion/coil combinations."""
    if poses is None:
        return _static(mix, traj.k, csm, coeff, k_chunk)
    per_sample = expand_to_samples(poses, traj.shot)
    if csm is None:
        return moved_kspace(mix, traj.k, per_sample, coeff=coeff, chunk=k_chunk)
    if per_shot_coils:
        return _moved_coil_by_shot(mix, traj, csm, poses, coeff, k_chunk)
    return moved_coil_kspace(mix, traj.k, csm, per_sample, coeff=coeff)


@dataclass
class AcquisitionEngine:
    """Turn a phantom and a trajectory into k-space."""

    model: str = "simple"
    snr: float = float("inf")
    k_chunk: int = 0
    #: Use the per-shot coil-motion form. Equivalent (4.2e-7) but ~10% SLOWER; its only
    #: advantage is never materializing the [V, M] shifted spectrum. See the module note.
    per_shot_coils: bool = False
    #: "per_tissue" evaluates each tissue and sums; "kernel" applies the decay inside
    #: the FT kernel. They agree bit-exactly and per_tissue is FASTER -- see the module
    #: note. The kernel path is kept because it does not require tissue-pure primitives.
    t2star_impl: str = "per_tissue"

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}; have {MODELS}")

    def acquire(self, phantom: TissuePhantom, traj: Trajectory, cfg: SimConfig,
                csm: Optional[FourierCSM] = None,
                poses: Optional[RigidPoses] = None,
                by_shot: bool = True) -> Tensor:
        """-> ``[S, C, m]`` when the trajectory has regular shots, else ``[C, M]``.

        ``poses`` carries **one pose per shot**, not per sample.
        """
        if traj.shot is None and poses is not None:
            raise ValueError("motion needs a trajectory with shot indices")
        mix, coeff = phantom.mixture, phantom.coeff(cfg.seq, field=cfg.field)

        if self.model == "simple":
            ksp = _evaluate(mix, traj, csm, coeff, poses, self.k_chunk,
                            self.per_shot_coils)
        elif self.t2star_impl == "per_tissue":
            ksp = self._t2star(phantom, traj, cfg, csm, coeff, poses)
        else:
            ksp = self._t2star_kernel(phantom, traj, cfg, csm, coeff, poses)

        if ksp.dim() == 1:
            ksp = ksp[None, :]                                   # [1, M]
        if self.snr > 0 and math.isfinite(self.snr):
            ksp = ksp + self._noise(ksp, cfg)
        return _to_shots(ksp, traj) if by_shot else ksp

    # ---------------------------------------------------------------- t2star
    def _t2star(self, phantom: TissuePhantom, traj: Trajectory, cfg: SimConfig,
                csm: Optional[FourierCSM], coeff: Tensor,
                poses: Optional[RigidPoses]) -> Tensor:
        """SNAKE's semantics: one evaluation per tissue, decayed, then summed.

        The reference arm. It is T times the work, and that is the point of comparison
        for the single-launch version.
        """
        if traj.time_ms is None:
            raise ValueError("the t2star model needs sample times on the trajectory")
        r2s = phantom.r2star_per_ms(field=cfg.field)             # [N]
        dt = (traj.time_ms - float(cfg.seq.te_ms)).to(traj.k.dtype)      # [M]
        out = None
        for lab in phantom.labels:
            sel = phantom.tissue == lab
            if not bool(sel.any()):
                continue
            sub = phantom.mixture.subset(torch.nonzero(sel).flatten())
            val = _evaluate(sub, traj, csm, coeff[sel], poses, self.k_chunk,
                            self.per_shot_coils)
            decay = torch.exp(-dt * float(r2s[sel][0]))           # tissue-pure -> scalar
            val = val * decay if val.dim() == 1 else val * decay[None, :]
            out = val if out is None else out + val
        return out

    # ------------------------------------------------------- t2star, one launch
    def _t2star_kernel(self, phantom: TissuePhantom, traj: Trajectory, cfg: SimConfig,
                       csm, coeff: Tensor, poses) -> Tensor:
        """The readout decay applied inside the FT kernel -- one launch, not T.

        Each primitive carries its own ``1/T2*``, which is well defined only because it
        belongs to exactly one tissue. The factor folds into the exponential the kernel
        already evaluates for the Gaussian magnitude, so the extra cost is a load of
        ``r2s[n]`` and one multiply-add -- no extra transcendental. SNAKE cannot do this:
        its tissues share a voxel grid, so the decay is per tissue and it needs one NUFFT
        each.
        """
        if traj.time_ms is None:
            raise ValueError("the t2star model needs sample times on the trajectory")
        r2s = phantom.r2star_per_ms(field=cfg.field)
        dt = (traj.time_ms - float(cfg.seq.te_ms)).to(traj.k.dtype)
        mix = phantom.mixture
        if poses is None:
            return kspace_cuda(mix, traj.k, csm=csm, coeff=coeff, k_chunk=self.k_chunk,
                               r2s_per_ms=r2s, dt_ms=dt)
        if csm is not None:
            raise NotImplementedError(
                "coils + motion + in-kernel decay is not wired yet; use "
                "t2star_impl='per_tissue', which covers every combination")
        return moved_kspace(mix, traj.k, expand_to_samples(poses, traj.shot),
                            coeff=coeff, chunk=self.k_chunk,
                            r2s_per_ms=r2s, dt_ms=dt)

    # ---------------------------------------------------------------- noise
    def _noise(self, ksp: Tensor, cfg: SimConfig) -> Tensor:
        g = torch.Generator(device=ksp.device).manual_seed(int(cfg.seed))
        power = float((ksp.abs() ** 2).mean())
        cov = cov_for_snr(power, self.snr, ksp.shape[0], device=ksp.device)
        return coil_noise(tuple(ksp.shape), cov, g, device=ksp.device)


def _to_shots(ksp: Tensor, traj: Trajectory) -> Tensor:
    """``[C, M]`` -> ``[S, C, m]`` when every shot has the same length."""
    if traj.shot is None:
        return ksp
    counts = torch.bincount(traj.shot.long())
    if int(counts.min()) != int(counts.max()):
        return ksp
    s, m = int(counts.numel()), int(counts[0])
    return ksp.reshape(ksp.shape[0], s, m).permute(1, 0, 2).contiguous()
