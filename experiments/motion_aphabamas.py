#!/usr/bin/env python
"""Motion-corrupted 3D acquisition of a realistic brain: analytic vs existing simulators.

APHABAMAS (Wu & Zhang, 2026) assesses motion-artifact simulators against an analytical
ground truth, but could only afford the Shepp-Logan phantom (a polyhedral brain costs
0.4 s per k-sample). A fitted Gaussian subject IS an analytical phantom of a real brain,
so the same protocol runs on anatomy:

  object       BrainWeb subject 04, 1.70 M primitives fitted to the 0.5 mm tissue maps,
               spoiled-GRE T1 weighting (TR/TE/FA 15/3/30)
  acquisition  1 mm 3D Cartesian, readout along z (S-I), lines along x (L-R) form a
               plane, one plane per TR along y (A-P); one rigid pose per TR, rotation
               about the head centre (APHABAMAS Sec. 3.1)
  motion       8 trajectories, 4 mild (slow drift, <= ~1 mm / 1 deg) and 4 severe
               (drift plus sudden jerks, up to ~5 mm / 5 deg)
  ground truth  exp(-i2pi k.t) rho_hat(R^T k) per sample -- closed form, no resampling;
               checked against an independent route (moving the primitives)
  simulators   the motion-free scan (inverse DFT of the analytic static k-space, complex,
               i.e. the best input such a simulator can be given) moved by
                 image-based   trilinear resampling per pose, then the DFT (Fig. 4a)
                 NUFFT type-2  k-space evaluated at R^T k from the scan (Fig. 4b),
                               finufft at eps 1e-9, i.e. the NUFFT is not the error
  metrics      magnitude reconstructions vs the ground truth: NRMSE and 3D SSIM over the
               whole FOV (APHABAMAS uses RMSD/SSIM over all voxels); k-space NRMSE

    python experiments/motion_aphabamas.py --device cuda:0 --traj 0 1 2 3
    python experiments/motion_aphabamas.py --device cuda:0 --traj 4 --vox 0.5 --no-nufft
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaussmrisim.conventions import FOV
from gaussmrisim.fourier.cuda_ft import kspace_cuda
from gaussmrisim.motion.resample import rotate_raster
from gaussmrisim.motion.rigid import RigidPoses, moved_kspace, rotation_from_euler
from gaussmrisim.sim.config import GreConfig
from gaussmrisim.sim.phantom import TissuePhantom

FIT = "outputs/fits/brainweb_s04_fuzzy_0.50mm_measured_area"
CENTRE = (18.0, -18.0, 0.0)
FOV_MM = (216.0, 248.0, 216.0)          # head + margin: no tissue leaves the FOV
VOX = 1.0
SEQ = (15.0, 3.0, 30.0)
OUT = Path("outputs/sim/motion_aphabamas")


def fov() -> FOV:
    m = tuple(int(round(f / VOX)) for f in FOV_MM)
    return FOV(FOV_MM, m, CENTRE)


def raster_fov(F: FOV) -> FOV:
    """The same grid as :func:`to_image` (voxel n at c + (n - N/2) dx) in ``FOV`` terms,
    whose voxel centres sit at c + (n - (N-1)/2) dx: shift the centre by half a voxel."""
    return FOV(F.fov_mm, F.matrix, tuple(c - 0.5 * v for c, v in zip(F.center_mm, F.voxel_mm)))


def kgrid(F: FOV, device) -> torch.Tensor:
    ax = [(torch.arange(n, device=device, dtype=torch.float64) - n // 2) / f
          for n, f in zip(F.matrix, F.fov_mm)]
    kz, ky, kx = torch.meshgrid(ax[2], ax[1], ax[0], indexing="ij")
    return torch.stack([kx, ky, kz], -1)                 # [nz, ny, nx, 3]


def centre_phase(k: torch.Tensor, sign: float) -> torch.Tensor:
    c = torch.tensor(CENTRE, device=k.device, dtype=torch.float64)
    return torch.exp(sign * 2j * math.pi * ((k @ c) % 1.0)).to(torch.complex64)


def to_image(y: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Cartesian samples ``[nz, ny, nx]`` -> complex image (inverse DFT, density units)."""
    v = y * centre_phase(k, +1.0)
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(v))) / VOX ** 3


def to_kspace(img: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Complex image -> Cartesian samples (forward DFT, the inverse of :func:`to_image`)."""
    y = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(img))) * VOX ** 3
    return y * centre_phase(k, -1.0)


def trajectory(seed: int, n: int, severe: bool):
    """Per-TR Euler angles (deg) and translations (mm), ``[n, 3]`` each."""
    rng = np.random.default_rng(seed)

    def drift(scale):
        w = rng.normal(size=(n + 20, 3)).cumsum(0)            # random walk ...
        w = np.stack([np.convolve(w[:, i], np.ones(21) / 21, "valid") for i in range(3)], 1)
        w -= w[n // 2]                                        # ... smoothed                               # centre of k-space at rest
        return scale * w / np.abs(w).max()

    ang, tr = drift(1.0), drift(1.0)
    if severe:
        for _ in range(rng.integers(2, 5)):
            j = int(rng.integers(n // 8, n - n // 8))
            ang[j:] += rng.uniform(-4, 4, 3)
            tr[j:] += rng.uniform(-4, 4, 3)
        ang *= min(1.0, 5.0 / np.abs(ang).max())              # cap at 5 deg / 5 mm
        tr *= min(1.0, 5.0 / np.abs(tr).max())
    return ang, tr


def poses_for(ang, tr, device):
    """Rotation about the head centre: r -> R (r - c) + c + t = R r + (t + c - R c)."""
    R = rotation_from_euler(torch.tensor(ang, dtype=torch.float64), device=device,
                            dtype=torch.float64)
    c = torch.tensor(CENTRE, device=device, dtype=torch.float64)
    t = torch.tensor(tr, device=device, dtype=torch.float64) + c - R @ c
    return R, t


def ssim3d(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, data_range=float(b.max() - b.min())))


def check_finufft():
    """The type-2 call below against a direct sum on a small random grid."""
    import finufft
    rng = np.random.default_rng(0)
    nz, ny, nx = 6, 8, 10
    f = rng.normal(size=(nz, ny, nx)) + 1j * rng.normal(size=(nz, ny, nx))
    q = rng.uniform(-0.5, 0.5, size=(7, 3))
    m = [np.arange(n) - n // 2 for n in (nx, ny, nz)]
    mz, my, mx = np.meshgrid(m[2], m[1], m[0], indexing="ij")
    ref = np.array([(f * np.exp(-2j * np.pi * (qq[0] * mx + qq[1] * my + qq[2] * mz))).sum()
                    for qq in q])
    got = finufft.nufft3d2(2 * np.pi * q[:, 2], 2 * np.pi * q[:, 1], 2 * np.pi * q[:, 0],
                           f, isign=-1, eps=1e-12)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    assert err < 1e-9, f"finufft convention check failed: {err:.2e}"
    return err


def nufft_type2(scan: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``vox^3 sum_n scan_n e^{-i2pi q.r_n}`` at arbitrary ``q [M, 3]`` (cycles/mm)."""
    import finufft
    f = scan.cpu().numpy().astype(np.complex128)
    qn = q.cpu().numpy() * VOX
    val = finufft.nufft3d2(2 * np.pi * qn[:, 2], 2 * np.pi * qn[:, 1], 2 * np.pi * qn[:, 0],
                           f, isign=-1, eps=1e-9, nthreads=16)
    c = np.asarray(CENTRE)
    val = val * np.exp(-2j * np.pi * ((q.cpu().numpy() @ c) % 1.0)) * VOX ** 3
    return torch.from_numpy(val.astype(np.complex64))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--traj", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--vox", type=float, default=1.0,
                    help="acquisition voxel (mm); outputs of a non-1 mm run go to vox{v}/")
    ap.add_argument("--no-nufft", action="store_true",
                    help="analytic and image-based arms only")
    ap.add_argument("--wait-static", action="store_true",
                    help="wait for static.pt from a --static-only run instead of computing it")
    ap.add_argument("--static-only", action="store_true",
                    help="compute and cache the static k-space, then exit (run it on the "
                         "other GPU; the main run waits for the file)")
    args = ap.parse_args()
    global VOX, OUT
    VOX = args.vox
    if VOX != 1.0:
        OUT = OUT / f"vox{VOX:g}"
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.no_nufft:
        print(f"finufft convention check {check_finufft():.1e}", flush=True)
    F = fov()
    nx, ny, nz = F.matrix
    ph = TissuePhantom.load(FIT, device=dev)
    coeff = ph.coeff(GreConfig(tr_ms=SEQ[0], te_ms=SEQ[1], flip_deg=SEQ[2]))
    k = kgrid(F, dev)                                            # [nz, ny, nx, 3]
    kf = k.reshape(-1, 3)
    plane = torch.arange(ny, device=dev)[None, :, None].expand(nz, ny, nx).reshape(-1)

    static = OUT / "static.pt"
    if args.static_only:
        t0 = time.time()
        y0 = kspace_cuda(ph.mixture, kf.float(), coeff=coeff, k_chunk=1 << 20)
        tmp = Path(str(static) + ".tmp")
        torch.save(y0.reshape(nz, ny, nx).cpu(), tmp)
        tmp.rename(static)                                     # appears only when complete
        print(f"static analytic k-space {kf.shape[0]:,} samples in {time.time() - t0:.0f} s",
              flush=True)
        return 0
    FR = raster_fov(F)
    state = {}

    def motion_free_scan():
        """The static k-space (computed here or by a --static-only run) -> the scan."""
        if "scan" in state:
            return state["y0"], state["scan"]
        if not static.exists() and not args.wait_static:
            t0 = time.time()
            y0 = kspace_cuda(ph.mixture, kf.float(), coeff=coeff, k_chunk=1 << 20)
            torch.save(y0.reshape(nz, ny, nx).cpu(), static)
            print(f"static analytic k-space {kf.shape[0]:,} samples in "
                  f"{time.time() - t0:.0f} s", flush=True)
        while not static.exists():
            time.sleep(10)
        y0 = torch.load(static).to(dev)
        scan = to_image(y0, k)                                 # the motion-free scan
        # the scan reproduces the samples it came from: a convention check, not a result
        rt = float((to_kspace(scan, k) - y0).norm() / y0.norm())
        eye, zero = torch.eye(3, device=dev), torch.zeros(3, device=dev)
        ident = torch.complex(rotate_raster(scan.real.contiguous(), FR, eye, zero, "trilinear"),
                              rotate_raster(scan.imag.contiguous(), FR, eye, zero, "trilinear"))
        idn = float((ident - scan).norm() / scan.norm())
        print(f"scan -> k-space round trip {rt:.1e}; identity pose through the resampler "
              f"{idn:.1e}", flush=True)
        assert rt < 1e-5 and idn < 1e-5
        state.update(y0=y0, scan=scan)
        return y0, scan

    for tj in args.traj:
        severe = tj >= 4
        # the motion is defined on the 1 mm protocol's 248 TRs and mapped onto this run's
        # TRs by acquisition time, so a trajectory is the same motion at every resolution
        n1 = int(round(FOV_MM[1] / 1.0))
        ang, tr = trajectory(tj, n1, severe)
        idx = np.minimum((np.arange(ny) * n1) // ny, n1 - 1)
        ang, tr = ang[idx], tr[idx]
        R, t = poses_for(ang, tr, dev)
        rec = {"traj": tj, "severe": severe, "max_rot_deg": float(np.abs(ang).max()),
               "max_trans_mm": float(np.abs(tr).max()), "angles": ang.tolist(),
               "trans": tr.tolist()}

        # ground truth: closed form, one pose per sample
        torch.cuda.synchronize(dev)
        t0 = time.time()
        gt = moved_kspace(ph.mixture, kf.float(),
                          RigidPoses(R[plane].float(), t[plane].float()), coeff=coeff,
                          chunk=1 << 20).reshape(nz, ny, nx)
        torch.cuda.synchronize(dev)
        rec["gt_seconds"] = time.time() - t0
        # independent route on three planes: move the primitives, static evaluator
        diffs = []
        for j in (ny // 4, ny // 2 + 7, 3 * ny // 4):
            mv = ph.mixture.rigid_transform(R[j].float(), t[j].float())
            alt = kspace_cuda(mv, k[:, j].reshape(-1, 3).float(), coeff=coeff)
            diffs.append(float((alt - gt[:, j].reshape(-1)).norm() / alt.norm()))
        rec["routes_agree"] = max(diffs)

        # image-based: move the scan by trilinear resampling, DFT, keep this TR's plane
        y0, scan = motion_free_scan()
        t0 = time.time()
        ib = torch.zeros_like(gt)
        j = 0
        while j < ny:
            j1 = j + 1
            while j1 < ny and bool(torch.equal(R[j1], R[j])) and bool(torch.equal(t[j1], t[j])):
                j1 += 1
            Rj, tj_ = R[j].float(), t[j].float()
            mv = torch.complex(rotate_raster(scan.real.contiguous(), FR, Rj, tj_, "trilinear"),
                               rotate_raster(scan.imag.contiguous(), FR, Rj, tj_, "trilinear"))
            ib[:, j:j1] = to_kspace(mv, k)[:, j:j1]
            j = j1
        rec["image_based_seconds"] = time.time() - t0

        arms = [("image_based", ib)]
        if not args.no_nufft:
            # NUFFT type-2: the scan's spectrum at R^T k, then the translation phase
            t0 = time.time()
            Rs, ts = R[plane], t[plane]
            q = torch.einsum("sij,si->sj", Rs, kf)                     # R^T k
            n2 = nufft_type2(scan, q).to(dev)
            n2 = n2 * torch.exp(-2j * math.pi * (((kf * ts).sum(-1)) % 1.0)).to(
                torch.complex64)
            arms.append(("nufft_type2", n2.reshape(nz, ny, nx)))
            rec["nufft2_seconds"] = time.time() - t0
            del q, Rs, ts

        mg = to_image(gt, k).abs()
        imgs = {"ground_truth": mg}
        for name, y in arms:
            m = to_image(y, k).abs()
            imgs[name] = m
            rec[name] = {"kspace_nrmse": float((y - gt).norm() / gt.norm()),
                         "image_nrmse": float((m - mg).norm() / mg.norm()),
                         # PSNR against the exact image's own peak, whole volume
                         "psnr_db": float(10 * torch.log10(mg.max() ** 2
                                                           / (m - mg).square().mean())),
                         "ssim": ssim3d(m.cpu().numpy(), mg.cpu().numpy())}
        rec["static_vs_moved_nrmse"] = float((to_image(y0, k).abs() - mg).norm() / mg.norm())
        z = nz // 2 + 8
        torch.save({n: v[z].cpu() for n, v in imgs.items()} | {"static": scan[z].abs().cpu()},
                   OUT / f"traj{tj}_slices.pt")
        # the whole magnitude volumes (float16), for sagittal / coronal views
        torch.save({n: v.half().cpu() for n, v in imgs.items()}, OUT / f"traj{tj}_volumes.pt")
        (OUT / f"traj{tj}.json").write_text(json.dumps(rec, indent=2))
        print(f"traj {tj} ({'severe' if severe else 'mild'}, {rec['max_rot_deg']:.1f} deg / "
              f"{rec['max_trans_mm']:.1f} mm, {VOX:g} mm): routes {rec['routes_agree']:.1e} | "
              + " | ".join(f"{n} PSNR {rec[n]['psnr_db']:.1f} dB ssim {rec[n]['ssim']:.4f}"
                           for n, _ in arms) + f" | GT {rec['gt_seconds']:.0f} s", flush=True)
        del gt, ib, arms, imgs
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
