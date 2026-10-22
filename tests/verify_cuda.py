#!/usr/bin/env python
"""CUDA kernels vs the PyTorch reference. Nothing ships until this passes.

The rule is that CUDA may be faster, never different. Two traps this script exists to
catch, both of which bit during development:

* **A single-primitive test cannot validate the tile binning.** With an identity
  quaternion the principal widths equal ``sqrt(Sigma_jj)``, so an AABB built from the
  wrong one is still correct; only a rotated primitive exposes it.
* **A gradient check needs a well-conditioned loss.** Checking gradients of
  ``sum(w * rho)`` against a random mixture makes the true gradient ~1e-6 while the
  summands are ~1e-1, so float32 cancellation alone produces a 0.35 relative
  "error" and two mathematically identical PyTorch paths disagree. Use a fitting loss
  ``||render - target||^2``, whose gradient is on the scale of its terms.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaussmrisim.acquisition import trajectory as T
from gaussmrisim.coils.fourier_csm import FourierCSM, cartesian_mode_grid, coil_kspace
from gaussmrisim.conventions import FOV
from gaussmrisim.evaluation.raster import nrmse
from gaussmrisim.fitting.renderer3d import Patch, axis_widths, bin_to_tiles, render
from gaussmrisim.fourier.cuda_ft import kspace_cuda
from gaussmrisim.fourier.gaussian_ft import kspace
from gaussmrisim.representation.gaussian3d import random_mixture, single_gaussian

OK = []


def check(name, detail, v, tol):
    OK.append(v <= tol)
    print(f"  [{'ok ' if v <= tol else 'FAIL'}] {name:<44s} {v:9.3e} (tol {tol:.0e})  {detail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable -- fix the environment, never fall back to CPU")

    print("\n[Fourier kernel]")
    fov = FOV(fov_mm=(256.,) * 3, matrix=(128,) * 3)
    k = T.build("radial", 256, 65, fov, device=dev).k
    mix = random_mixture(4000, fov_mm=fov.fov_mm, sigma_range=(0.8, 4.0), seed=2, device=dev)
    check("FT: CUDA vs torch reference", "N=4000, radial", nrmse(kspace_cuda(mix, k), kspace(mix, k)), 1e-6)

    fov_hi = FOV(fov_mm=(200.,) * 3, matrix=(2000,) * 3)          # 100 um
    k_hi = T.build("radial", 64, 65, fov_hi, device=dev).k
    mix_hi = random_mixture(2000, fov_mm=fov_hi.fov_mm, sigma_range=(0.1, 3.0), seed=5, device=dev)
    check("FT phase at 100 um", "k.mu reaches ~900 cycles",
          nrmse(kspace_cuda(mix_hi, k_hi), kspace(mix_hi, k_hi)), 1e-6)

    freqs = cartesian_mode_grid(5, fov.fov_mm, device=dev)
    g = torch.Generator().manual_seed(0)
    beta = (torch.randn(15, freqs.shape[0], generator=g)
            + 1j * torch.randn(15, freqs.shape[0], generator=g)).to(dev) / 125
    csm = FourierCSM(freqs, beta)
    ks = k[:4096]
    check("multi-coil: CUDA vs torch", "C=15, V=125",
          nrmse(kspace_cuda(mix, ks, csm), coil_kspace(mix, ks, csm)), 1e-5)

    print("\n[Tiled renderer]")
    n = 32
    patch = Patch(origin_mm=(-16.,) * 3, voxel_mm=(1.,) * 3, shape=(n,) * 3)
    a = torch.arange(n, device=dev, dtype=torch.float32) - 16.
    gz, gy, gx = torch.meshgrid(a, a, a, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], -1)

    one = single_gaussian(mu=(1.5, -2., 3.), sigma=(2., 3., 1.5), mass=2.0, device=dev)
    check("render: single primitive", "identity rotation (cannot catch AABB bugs)",
          nrmse(render(one, patch, 6.0).reshape(-1), one.image(pts, cutoff_sigma=6.0)), 1e-6)
    rot = random_mixture(2000, fov_mm=(24.,) * 3, sigma_range=(1.0, 4.0), seed=3, device=dev)
    check("render: rotated mixture", "N=2000, random quaternions -- catches AABB bugs",
          nrmse(render(rot, patch, 6.0).reshape(-1), rot.image(pts, cutoff_sigma=6.0)), 1e-5)

    big = single_gaussian(mu=(0., 0., 0.), sigma=(3., 3., 3.), mass=5.0, device=dev)
    check("render integrates to the mass", "8 sigma cutoff, 1 mm voxels",
          abs(float(render(big, patch, 8.0).sum()) - 5.0) / 5.0, 1e-5)

    mix = random_mixture(64, fov_mm=(24.,) * 3, sigma_range=(1.5, 3.0), seed=4, device=dev)
    target = random_mixture(64, fov_mm=(24.,) * 3, sigma_range=(1.5, 3.0), seed=99,
                            device=dev).image(pts)

    def grads(fn):
        t = [mix.mu.clone().requires_grad_(True), mix.log_scale.clone().requires_grad_(True),
             mix.quaternion.clone().requires_grad_(True), mix.mass.clone().requires_grad_(True)]
        m = dataclasses.replace(mix, mu=t[0], log_scale=t[1], quaternion=t[2], mass=t[3])
        ((fn(m).reshape(-1) - target) ** 2).sum().backward()
        return [x.grad for x in t]

    gc = grads(lambda m: render(m, patch, 8.0))
    gt = grads(lambda m: m.image(pts, cutoff_sigma=8.0))
    for nm, x, y in zip(("mu", "log_scale", "quaternion", "mass"), gc, gt):
        check(f"render grad {nm}", "fitting loss ||render - target||^2", nrmse(x, y), 5e-5)

    # ---------------------------------------------- per-sample poses, both backends
    # moved_kspace dispatches to the CUDA kernel on CUDA input. That dispatch is what
    # makes a per-sample pose cost what a static acquisition costs, so it has to be
    # checked against the reference it replaced -- otherwise the motion timing silently
    # becomes a backend benchmark.
    from gaussmrisim.motion.rigid import RigidPoses, moved_kspace, rotation_from_euler

    kk = torch.randn(3000, 3, device=dev) * 0.15
    S = kk.shape[0]
    ang = torch.stack([torch.linspace(0, 17, S, device=dev),
                       torch.linspace(0, -9, S, device=dev),
                       torch.linspace(0, 4, S, device=dev)], -1)
    poses = RigidPoses(R=rotation_from_euler(ang, device=dev),
                       t=torch.linspace(0, 1.5, S, device=dev)[:, None].expand(-1, 3).contiguous())
    a = moved_kspace(mix, kk, poses, backend="cuda")
    b = moved_kspace(mix, kk, poses, backend="torch")
    check("moved_kspace cuda == torch", f"{S} distinct poses, one per sample",
          nrmse(a, b), 1e-5)

    print(f"\n{sum(OK)}/{len(OK)} checks passed")
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
