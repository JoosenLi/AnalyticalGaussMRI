"""How much of the object lies outside the acquisition FOV.

A real scanner acquires a subject that fits inside its field of view; whatever sits
outside does not vanish, it **folds back in** as wrap-around, because sampling k-space at
spacing ``1/FOV`` makes the reconstructed image periodic with that period. A simulator
whose object overruns the box therefore produces aliasing that is physically correct but
usually unintended, and the number has to be reported rather than assumed small.

It is not assumed small here for a measured reason: BrainWeb's head fills its 181 mm
z-FOV, and the H4 motion study found primitive centres already 2.13 mm past the edge at
0 degrees, rising to 52.2 mm at 20 degrees. **Rigid motion moves mass out of the FOV**, so
this is a per-pose quantity, not a property of the phantom alone.

Gaussians have infinite support, so "outside the box" is an integral rather than a count
of centres. The marginal of a 3D Gaussian along axis *i* is exactly ``N(mu_i, Sigma_ii)``
whatever the off-diagonal terms are, so each axis is exact via ``erf``; the three events
are not independent unless the covariance is diagonal in the FOV frame, so the box
integral is bracketed rather than guessed:

    max_i P_out_i  <=  P(outside)  <=  sum_i P_out_i

with the product form reported between them as the estimate that is *exact* for an
axis-aligned primitive. Reporting one number without the bracket would be pretending an
independence that rotated primitives do not have.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..conventions import FOV
from ..representation.gaussian3d import GaussianMixture3D


def _axis_inside(mix: GaussianMixture3D, fov: FOV) -> Tensor:
    """``[N, 3]`` probability that each primitive's marginal falls inside the FOV."""
    cov = mix.covariance()                                  # [N, 3, 3], includes dilation
    var = torch.stack([cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2]], -1).clamp_min(1e-30)
    sd = var.sqrt()
    lo = torch.tensor([c - 0.5 * f for c, f in zip(fov.center_mm, fov.fov_mm)],
                      device=mix.mu.device, dtype=mix.mu.dtype)
    hi = torch.tensor([c + 0.5 * f for c, f in zip(fov.center_mm, fov.fov_mm)],
                      device=mix.mu.device, dtype=mix.mu.dtype)
    z = lambda b: (b - mix.mu) / (sd * (2.0 ** 0.5))
    return (0.5 * (torch.erf(z(hi)) - torch.erf(z(lo)))).clamp(0.0, 1.0)


@torch.no_grad()
def out_of_fov(mix: GaussianMixture3D, fov: FOV,
               weight: Optional[Tensor] = None) -> dict:
    """Fraction of the object's signal outside ``fov`` -> lower / estimate / upper.

    ``weight`` is the per-primitive signal ``a_n`` for a sequence; without it the answer
    is proton-density weighted, which is the object's own mass. A tissue invisible in one
    contrast contributes nothing to that contrast's wrap-around, so the sequence-weighted
    number is the one an acquisition actually sees.
    """
    p_in = _axis_inside(mix, fov)                            # [N, 3]
    p_out = 1.0 - p_in
    w = mix.mass if weight is None else (mix.mass * weight)
    w = w.abs()
    tot = w.sum().clamp_min(1e-30)

    lower = (w * p_out.max(dim=-1).values).sum() / tot
    upper = (w * p_out.sum(dim=-1).clamp(max=1.0)).sum() / tot
    est = (w * (1.0 - p_in.prod(dim=-1))).sum() / tot
    per_axis = ((w[:, None] * p_out).sum(dim=0) / tot)
    return dict(lower=float(lower), estimate=float(est), upper=float(upper),
                per_axis=[float(x) for x in per_axis],
                # a coarse but very cheap companion: centres strictly outside the box.
                # It is NOT the same quantity -- a centre inside the box still leaks mass
                # out -- and it is reported only because a large gap between the two says
                # the overrun is wide tails rather than displaced structure.
                centres_outside=float((p_in.min(dim=-1).values < 0.5).float().mean()))


@torch.no_grad()
def out_of_fov_over_poses(mix: GaussianMixture3D, fov: FOV, poses,
                          weight: Optional[Tensor] = None) -> dict:
    """The same, evaluated at every pose -> worst and mean over shots.

    Motion is what makes this interesting: a subject that fits the FOV at rest can be
    rotated out of it, and the acquisition would then wrap a part of the head that the
    static check called clean.
    """
    import dataclasses

    n = int(poses.R.shape[0])
    vals = []
    for i in range(n):
        R, t = poses.R[i], poses.t[i]
        moved = dataclasses.replace(
            mix, mu=mix.mu @ R.T + t,
            quaternion=_rotate_quat(mix.quaternion, R))
        vals.append(out_of_fov(moved, fov, weight)["estimate"])
    v = torch.tensor(vals)
    return dict(worst=float(v.max()), mean=float(v.mean()),
                worst_shot=int(v.argmax()), per_shot=vals)


def _rotate_quat(q: Tensor, R: Tensor) -> Tensor:
    """Compose a rotation onto each primitive's quaternion, via the matrix form.

    Going through the matrix keeps this independent of the quaternion convention used
    elsewhere: a sign or order slip there would rotate the covariances differently from
    the centres and show up as an anisotropy that tracks the pose.
    """
    from ..representation.gaussian3d import (quaternion_from_rotation_batch,
                                             rotation_from_quaternion)

    M = rotation_from_quaternion(q)
    return quaternion_from_rotation_batch(R[None] @ M)
