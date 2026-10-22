"""Per-shot rigid motion: a random walk in pose, applied exactly.

The generator follows SNAKE's ``motion_generator`` so the two simulators can be given
the same motion: speeds are drawn per shot from a normal distribution in mm/s and deg/s
and integrated, which keeps the *amplitude* independent of the time resolution -- halving
TR then halves the per-shot step instead of doubling the total excursion.

What differs is the application. SNAKE adds the pose to the phantom's affine and
resamples the voxel grid; here the pose acts on the object in closed form, so there is no
interpolation anywhere in the chain. Measured (H4): trilinear resampling of the same
object costs 0.022-0.026 NRMSE in band and a prefiltered cubic B-spline 0.0002-0.0009,
against exactly zero here.

A pose is constant **within a shot**, which is both physically right (a shot is short
against head motion) and what makes the coil path efficient -- see ``engine``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from ..motion.rigid import RigidPoses, rotation_from_euler


def shot_poses(n_shots: int, tr_ms: float,
               ts_std_mm_s: Sequence[float] = (0.5, 0.5, 0.5),
               rs_std_deg_s: Sequence[float] = (0.2, 0.2, 0.2),
               seed: int = 0, device="cpu",
               dtype=torch.float32) -> tuple[RigidPoses, Tensor]:
    """A cumulative random-walk pose per shot -> (poses ``[S]``, euler degrees ``[S,3]``).

    ``ts_std_mm_s`` / ``rs_std_deg_s`` are **speeds**, per second, per axis.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    dt = float(tr_ms) / 1000.0
    ts = torch.tensor(ts_std_mm_s, dtype=torch.float64)
    rs = torch.tensor(rs_std_deg_s, dtype=torch.float64)
    t = (torch.randn(int(n_shots), 3, generator=g, dtype=torch.float64) * ts * dt).cumsum(0)
    r = (torch.randn(int(n_shots), 3, generator=g, dtype=torch.float64) * rs * dt).cumsum(0)
    t = t.to(device=device, dtype=dtype)
    r = r.to(device=device, dtype=dtype)
    return RigidPoses(R=rotation_from_euler(r, device=device, dtype=dtype), t=t), r


def expand_to_samples(poses: RigidPoses, shot: Tensor) -> RigidPoses:
    """Broadcast per-shot poses onto their samples -> ``[M]``.

    The evaluator takes one pose per sample; a shot is simply the case where consecutive
    samples share one. Keeping the per-shot object as the source of truth means the pose
    count is the number of shots, which is what a motion trace actually has.
    """
    idx = shot.long()
    return RigidPoses(R=poses.R[idx].contiguous(), t=poses.t[idx].contiguous())
