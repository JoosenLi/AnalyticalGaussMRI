#!/usr/bin/env python
"""A fair grid baseline for the noise-floor study: voxel AVERAGES, not centre samples.

``aliasing_recon_noise.py`` builds its grid arm from point samples of the mixture at the
voxel centres (Dirac basis), which is the weakest grid simulator there is. A voxel phantom
on disk (BrainWeb, BigBrain) holds partial-volume averages, so the fair arms are:

  point      c_n = rho(r_n)                         Dirac basis (the original arm)
  box-dirac  c_n = (1/h^3) int_voxel rho            Dirac basis: what a voxel simulator does
  box-sinc   c_n as above, basis = the voxel box    exact FT of the piecewise-constant
             (k-space multiplied by prod_i sinc(k_i h))  voxel object

The voxel average is computed by supersampling: render on an h/s grid and average s^3
blocks; s = 2 and 4 are both run so convergence is visible. Same objects, trajectories,
noise (one sigma from the Cartesian mean power), reconstruction and metrics as the
original study.

    source scripts/env_gpu_nufft.sh
    python experiments/aliasing_box_baseline.py --device cuda:0 --traj cartesian
    python experiments/aliasing_box_baseline.py --device cuda:1 --traj radial
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aliasing_recon_noise import (ACQ, cartesian_mean_power, CENTRE, FIT, FOV_MM, OUT, SEQ,  # noqa
                                  SNRS, Recon, cartesian_k, grid_nufft, nmse_db, radial_k)
from gaussmrisim.conventions import FOV                                  # noqa: E402
from gaussmrisim.datasets.brainweb import VolumeTarget                   # noqa: E402
from gaussmrisim.fitting.fitter import render_full                       # noqa: E402
from gaussmrisim.fourier.cuda_ft import kspace_cuda                      # noqa: E402
from gaussmrisim.sim.config import GreConfig                             # noqa: E402
from gaussmrisim.sim.phantom import TissuePhantom                        # noqa: E402


def render(mixw, h: float, dev) -> np.ndarray:
    """Point samples of the mixture at the centres of an h-mm grid -> [nz, ny, nx]."""
    n = tuple(int(round(f / h)) for f in FOV_MM)
    fov = FOV(tuple(FOV_MM), n, tuple(CENTRE))
    tgt = VolumeTarget(torch.zeros(n[2], n[1], n[0], device=dev), fov, "grid")
    return render_full(mixw, tgt, cutoff_sigma=6.0).cpu().numpy()


def box_average(mixw, h: float, s: int, dev) -> np.ndarray:
    """Voxel averages on the h-mm grid by s^3 supersampling (midpoint rule)."""
    fine = render(mixw, h / s, dev)
    nz, ny, nx = (d // s for d in fine.shape)
    return fine.reshape(nz, s, ny, s, nx, s).mean(axis=(1, 3, 5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--traj", default="cartesian", choices=["cartesian", "radial"])
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--ss", type=int, nargs="+", default=[2, 4])
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    import cupy
    cupy.cuda.Device(dev.index).use()
    ph = TissuePhantom.load(FIT, device=dev)
    coeff = ph.coeff(GreConfig(tr_ms=SEQ[0], te_ms=SEQ[1], flip_deg=SEQ[2]))
    mixw = dataclasses.replace(ph.mixture, mass=ph.mixture.mass * coeff)
    h = args.h

    vols = {"point": render(mixw, h, dev)}
    for s in args.ss:
        t0 = time.time()
        vols[f"box{s}"] = box_average(mixw, h, s, dev)
        print(f"voxel averages, {s}^3 supersampling, in {time.time() - t0:.0f} s", flush=True)
        torch.cuda.empty_cache()

    k, w = cartesian_k() if args.traj == "cartesian" else radial_k()
    t0 = time.time()
    ya = kspace_cuda(ph.mixture, torch.tensor(k, device=dev, dtype=torch.float32),
                     coeff=coeff, k_chunk=1 << 20).cpu().numpy().astype(np.complex128)
    print(f"{args.traj}: {len(k):,} samples, analytic {time.time() - t0:.0f} s", flush=True)
    recon = Recon(k, w)
    xa = recon(ya)
    e_ref = float(np.mean(np.abs(ya) ** 2)) if args.traj == "cartesian" else cartesian_mean_power()
    rng = np.random.default_rng(0)                     # the same realization as the study
    noise = (rng.standard_normal(len(k)) + 1j * rng.standard_normal(len(k))) / np.sqrt(2)
    off = float(10 * np.log10(np.max(np.abs(xa)) ** 2 / np.mean(np.abs(xa) ** 2)))
    rec = {"M": int(len(k)), "h": h, "psnr_offset_db": off, "noise_ref_power": e_ref,
           "analytic": {snr: off - nmse_db(recon(ya + math.sqrt(e_ref / snr) * noise), xa)
                        for snr in SNRS}, "arms": {}}
    box_ft = np.prod(np.sinc(k * h), axis=1)             # voxel-box basis, prod_i sinc(k_i h)
    arms = [("point", "point", False)] + [(f"box{s}-dirac", f"box{s}", False)
                                          for s in args.ss] \
        + [(f"box{s}-sinc", f"box{s}", True) for s in args.ss]
    nz = int(FOV_MM[2] / ACQ)
    slices = {"analytic": np.abs(xa[nz // 2 + 8])}
    for name, vkey, sinc in arms:
        yg = grid_nufft(vols[vkey], h, k) * (box_ft if sinc else 1.0)
        xg = recon(yg)
        r = {"kspace_nmse_db": nmse_db(yg, ya), "ceiling_psnr_db": off - nmse_db(xg, xa),
             "psnr_db": {snr: off - nmse_db(recon(yg + math.sqrt(e_ref / snr) * noise), xa)
                         for snr in SNRS}}
        rec["arms"][name] = r
        slices[f"{name}_diff"] = (np.abs(xg) - np.abs(xa))[nz // 2 + 8]
        print(f"  {name:12s} k-space NMSE {r['kspace_nmse_db']:6.1f} dB | ceiling PSNR "
              f"{r['ceiling_psnr_db']:5.1f} dB | SNR 50: {r['psnr_db'][50.0]:5.1f} dB "
              f"(analytic {rec['analytic'][50.0]:5.1f})", flush=True)
    out = OUT / f"box_baseline_{args.traj}_h{h:g}.json"
    out.write_text(json.dumps(rec, indent=2))
    np.savez_compressed(OUT / f"box_baseline_{args.traj}_h{h:g}_slices.npz", **slices)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
