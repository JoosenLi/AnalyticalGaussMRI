"""K-space trajectories in cycles/mm, with sample times and shot indices.

MRI-NUFFT normalized trajectories are converted once at the physical-unit boundary.
Optional SPARKLING files are supplied by the user through GMRISIM_SPARKLING.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from ..conventions import FOV, traj_to_k

_BUILDERS = {
    "radial": "initialize_3D_phyllotaxis_radial",
    "golden_radial": "initialize_3D_golden_means_radial",
    "cones": "initialize_3D_cones",
    "floret": "initialize_3D_floret",
    "seiffert": "initialize_3D_seiffert_spiral",
}

# Optional Siemens gradient .bin waveforms, read with mrinufft.io.nsp.read_trajectory.
SPARKLING_DIR = Path(os.environ.get(          # override with GMRISIM_SPARKLING
    "GMRISIM_SPARKLING",
    "data/trajectories"))
KMAX = 0.5


@dataclass
class Trajectory:
    k: Tensor                          # [M, 3] cycles/mm
    time_ms: Optional[Tensor] = None   # [M]
    shot: Optional[Tensor] = None      # [M] long
    kind: str = ""

    def __len__(self) -> int:
        return int(self.k.shape[0])

    def to(self, device) -> "Trajectory":
        f = lambda t: None if t is None else t.to(device)
        return Trajectory(self.k.to(device), f(self.time_ms), f(self.shot), self.kind)


def _rescale_under_kmax(arr: np.ndarray, verbose: bool = True) -> np.ndarray:
    """One global factor so ``max|arr| < 0.5`` holds *in float32*.

    mri-nufft maps ``[-0.5, 0.5)`` onto ``[-pi, pi)``; samples beyond that do not clip,
    they **wrap**, aliasing the outermost points onto the opposite edge of k-space where
    they read as noise rather than as an error. The delivered SPARKLING waveforms exceed
    0.5 slightly, so this is mandatory. One global factor preserves the geometry exactly,
    and it is derived from the FULL waveform so that two runs selecting different shot
    counts are not scaled differently (which would silently change their effective FOV).
    """
    out = arr.astype(np.float32)
    peak = float(np.abs(out).max())
    if peak < KMAX:
        return out
    scale = KMAX / peak
    for _ in range(8):
        scaled = (arr * scale).astype(np.float32)
        if float(np.abs(scaled).max()) < KMAX:
            if verbose:
                print(f"  [sparkling] rescaled by {scale:.6f} (was {peak:.6f}; "
                      f"extent shrinks {100 * (1 - scale):.3f}%)")
            return scaled
        scale *= 1.0 - 1e-6
    raise RuntimeError(f"could not bring |k| under {KMAX} (peak {peak})")


def _read_sparkling_file(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    from mrinufft.io.nsp import read_trajectory

    traj, _ = read_trajectory(str(path))
    return np.asarray(traj, dtype=np.float64)


def _load_sparkling(n_shots: int, n_samples: int, path=None, verbose: bool = True) -> Tensor:
    """Uniformly decimate a SPARKLING waveform to ``n_shots x n_samples``.

    Decimated, not truncated, so a reduced trajectory keeps SPARKLING's k-space coverage
    rather than only its first shots. A file has a FIXED shot count, so asking for more
    cannot produce more -- that is warned about loudly, because silently returning a
    quarter of the requested samples invalidates a benchmark without ever looking wrong.
    """
    p = Path(path or os.environ.get("GMRISIM_SPARKLING_TRAJ", SPARKLING_DIR))
    if p.is_dir():
        cands = sorted(list(p.glob("*.bin")) + list(p.glob("*.npy")))
        if not cands:
            raise FileNotFoundError(f"no .bin/.npy SPARKLING trajectory under {p}")
        shots = [(_read_sparkling_file(c).shape[0], c) for c in cands]
        big = [t for t in shots if t[0] >= int(n_shots)]
        p = (min(big, key=lambda t: t[0]) if big else max(shots, key=lambda t: t[0]))[1]
    elif not p.exists():
        raise FileNotFoundError(f"SPARKLING trajectory not found at {p}")

    arr = _rescale_under_kmax(_read_sparkling_file(p), verbose=verbose)
    nc, ns = arr.shape[0], arr.shape[1]
    si = np.unique(np.linspace(0, nc - 1, min(int(n_shots), nc)).round().astype(int))
    ti = np.unique(np.linspace(0, ns - 1, min(int(n_samples), ns)).round().astype(int))
    if int(n_shots) > nc or int(n_samples) > ns:
        warnings.warn(
            f"SPARKLING {p.name} has {nc}x{ns}; request {n_shots}x{n_samples} clamped to "
            f"{len(si)}x{len(ti)} = {len(si) * len(ti):,} samples. Match a sample budget "
            f"by raising n_samples, not n_shots.", stacklevel=2)
    if verbose:
        print(f"  [sparkling] {p.name}: {nc}x{ns} -> {len(si)}x{len(ti)}")
    sel = arr[si][:, ti]
    assert float(np.abs(sel).max()) < KMAX
    return torch.from_numpy(np.ascontiguousarray(sel))          # [nc', ns', 3]


def build(kind: str, n_shots: int, n_samples: int, fov: FOV, tr_ms: float = 5.0,
          device="cpu", verbose: bool = True, **kwargs) -> Trajectory:
    """Any mri-nufft 3D trajectory, or ``sparkling``, converted to cycles/mm.

    ``tr_ms`` sets the sample times: shot ``i`` is acquired at ``i * tr_ms`` and every
    readout point within a shot shares it. That is the segmented approximation; for
    a readout-resolved model pass ``dwell_ms`` instead.
    """
    if kind == "sparkling":
        arr = _load_sparkling(n_shots, n_samples, verbose=verbose, **kwargs)
    else:
        import mrinufft.trajectories as mt

        if kind not in _BUILDERS:
            raise ValueError(f"unknown trajectory {kind!r}; "
                             f"options {sorted(_BUILDERS) + ['sparkling']}")
        arr = torch.from_numpy(np.asarray(
            getattr(mt, _BUILDERS[kind])(Nc=int(n_shots), Ns=int(n_samples), **kwargs),
            dtype=np.float32))
    nc, ns = int(arr.shape[0]), int(arr.shape[1])
    traj = arr.reshape(-1, 3).to(torch.float32).to(device)
    t = (torch.arange(nc, device=device, dtype=torch.float32) * float(tr_ms)
         )[:, None].expand(nc, ns).reshape(-1).contiguous()
    shot = torch.arange(nc, device=device).repeat_interleave(ns)
    return Trajectory(traj_to_k(traj, fov.voxel_mm[0]).contiguous(), t, shot, kind)


def cartesian(fov: FOV, device="cpu") -> Trajectory:
    """The full Cartesian grid matched to the FOV -- the reference acquisition."""
    return Trajectory(fov.cartesian_k(device=device), kind="cartesian")


def heldout(n_points: int, fov: FOV, r_min: float = 0.0, r_max: float = 1.0,
            seed: int = 0, device="cpu", shells: int = 0) -> Tensor:
    """Random k in the shell ``[r_min, r_max] * Nyquist`` -> ``[H, 3]`` cycles/mm.

    Random points are almost surely off any discrete trajectory, which is what makes
    this a projection-free generalization set -- and it exists only because the object
    has a closed form at arbitrary k. ``r_min > 0`` selects an out-of-band shell.

    **``shells > 0`` stratifies by radius, and for a power-weighted metric it is not
    optional.** Plain uniform-in-volume sampling puts the points where the volume is,
    i.e. near the outer edge, while a brain spectrum falls off steeply -- so a
    ``||err|| / ||ref||`` ratio over such a set is dominated by whichever few near-DC
    samples were drawn, and the smallest radius drawn shrinks like ``n^-1/3``.
    Measured on one fixed phantom at 1/4 Nyquist: four seeds gave 0.0263 / 0.0352 /
    0.0470 / 0.0522 at n=4096, and **raising n to 65536 made it worse** (0.0183 /
    0.0140 / 0.0381 / 0.0142, a 2.7x spread). The estimator does not converge in n
    because it is governed by an extreme order statistic of the radius, not by an
    average. Stratifying gives every shell the same number of samples, which removes
    that term; see :func:`gaussmrisim.evaluation.raster.shell_nrmse`.
    """
    g = torch.Generator().manual_seed(int(seed))
    ny = float(fov.k_nyquist[0])
    lo, hi = float(r_min) * ny, float(r_max) * ny
    n = int(n_points)
    if int(shells) > 1:
        # equal COUNT per shell, uniform in volume within each shell: the points stay
        # off-grid and unbiased inside a shell, but the radial distribution is fixed
        # rather than left to the draw.
        s_ = int(shells)
        edges = torch.linspace(0.0, 1.0, s_ + 1) ** (1.0 / 3.0)
        edges = (lo ** 3 + (hi ** 3 - lo ** 3) * torch.linspace(0.0, 1.0, s_ + 1)) ** (1.0 / 3.0)
        per = [n // s_ + (1 if i < n % s_ else 0) for i in range(s_)]
        parts = []
        for i, cnt in enumerate(per):
            if cnt == 0:
                continue
            a3, b3 = float(edges[i]) ** 3, float(edges[i + 1]) ** 3
            u_ = torch.rand(cnt, generator=g)
            parts.append((a3 + (b3 - a3) * u_) ** (1.0 / 3.0))
        r = torch.cat(parts)
        n = int(r.numel())
    else:
        u = torch.rand(n, generator=g)
        r = (lo ** 3 + (hi ** 3 - lo ** 3) * u) ** (1.0 / 3.0)      # uniform in volume
    cos_t = 2.0 * torch.rand(n, generator=g) - 1.0
    sin_t = torch.sqrt((1.0 - cos_t.square()).clamp_min(0.0))
    phi = 2.0 * torch.pi * torch.rand(n, generator=g)
    return torch.stack([r * sin_t * torch.cos(phi), r * sin_t * torch.sin(phi),
                        r * cos_t], dim=-1).to(torch.float32).to(device)
