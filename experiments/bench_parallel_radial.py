#!/usr/bin/env python
"""Typical cost of an analytical parallel-imaging simulation (paper, Sec. 2).

  object    BrainWeb, 1.70 M primitives (0.5 mm tissue fit), T1-weighted spoiled GRE
  coils     20-channel Biot-Savart loop array around the FOV; sensitivities fitted with
            125 Fourier modes (5^3, shared by all channels) on the head support,
            RSS-normalized; the fit error inside the support is reported
  sampling  1 mm FOV 182 x 218 x 182
              radial R = 20   full spokes of 218 samples, Fibonacci directions, the sample
                              count of the fully sampled Cartesian grid divided by 20
              Cartesian       fully sampled (7.2 M samples), for comparison
  timing    kspace_cuda with the coil sum inside the kernel (exact, no V compression,
            no culling); runtime, peak GPU memory, and single-coil runtime for reference

    python experiments/bench_parallel_radial.py --device cuda:0
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aliasing_recon_noise import CENTRE, FIT, FOV_MM, SEQ, cartesian_k    # noqa: E402
from aliasing_box_baseline import render                                   # noqa: E402
from gaussmrisim.coils.biot_savart import biot_savart_maps                 # noqa: E402
from gaussmrisim.coils.fourier_csm import (cartesian_mode_grid, fit_fourier_csm,  # noqa
                                           normalize_rss)
from gaussmrisim.conventions import FOV                                    # noqa: E402
from gaussmrisim.fourier.cuda_ft import kspace_cuda                        # noqa: E402
from gaussmrisim.sim.config import GreConfig                               # noqa: E402
from gaussmrisim.sim.phantom import TissuePhantom                          # noqa: E402

OUT = Path("outputs/sim/bench_parallel_radial")


def radial_k(n_samples_target: int, n_read: int = 218, kmax: float = 0.5) -> np.ndarray:
    ns = int(round(n_samples_target / n_read))
    i = np.arange(ns) + 0.5
    z = i / ns
    phi = np.pi * (1 + 5 ** 0.5) * i
    d = np.stack([np.sqrt(1 - z ** 2) * np.cos(phi), np.sqrt(1 - z ** 2) * np.sin(phi), z], -1)
    r = (np.arange(n_read) - n_read // 2) / n_read * 2 * kmax
    return (d[:, None, :] * r[None, :, None]).reshape(-1, 3)


def timed(fn, dev):
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    out = fn()
    torch.cuda.synchronize(dev)
    return out, time.time() - t0, torch.cuda.max_memory_allocated(dev) / 2 ** 30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--coils", type=int, default=20)
    ap.add_argument("--modes-side", type=int, default=5)
    ap.add_argument("--accel", type=float, default=20.0)
    ap.add_argument("--cartesian", action="store_true", help="also time the full Cartesian")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    ph = TissuePhantom.load(FIT, device=dev)
    coeff = ph.coeff(GreConfig(tr_ms=SEQ[0], te_ms=SEQ[1], flip_deg=SEQ[2]))

    # sensitivities: Biot-Savart on a 4 mm grid, fitted on the head support
    h = 4.0
    n = tuple(int(round(f / h)) for f in FOV_MM)
    fov4 = FOV(tuple(FOV_MM), n, tuple(CENTRE))
    pts = fov4.grid_coords(device=dev).float()
    rho = torch.from_numpy(render(ph.mixture, h, dev).reshape(-1)).to(dev)
    head = rho > 0.05 * rho.max()
    radius = 1.1 * float(np.linalg.norm(FOV_MM) / 2)
    maps = biot_savart_maps(pts, args.coils, radius)
    freqs = cartesian_mode_grid(args.modes_side, FOV_MM, device=dev)
    t0 = time.time()
    csm = fit_fourier_csm(maps[:, head], pts[head], freqs)
    csm = normalize_rss(csm, pts[head])
    fit_s = time.time() - t0
    ref = maps[:, head] / maps[:, head].abs().square().sum(0).sqrt().max()
    got = csm.sensitivity(pts[head])
    scale = (torch.vdot(got.reshape(-1), ref.reshape(-1)) / got.reshape(-1).abs().square().sum())
    fit_nrmse = float((got * scale - ref).norm() / ref.norm())
    print(f"{args.coils} coils, {csm.n_modes} shared Fourier modes, fit {fit_s:.1f} s, "
          f"NRMSE inside the head {fit_nrmse:.4f}", flush=True)

    m_cart = int(np.prod([round(f) for f in FOV_MM]))
    k = torch.tensor(radial_k(m_cart / args.accel), device=dev, dtype=torch.float32)
    rec = {"N": ph.n, "coils": args.coils, "modes": csm.n_modes, "csm_fit_nrmse": fit_nrmse,
           "csm_fit_seconds": fit_s, "radial": {"accel": args.accel, "M": int(k.shape[0])}}
    _, t1, m1 = timed(lambda: kspace_cuda(ph.mixture, k, coeff=coeff, k_chunk=1 << 20), dev)
    y, tc, mc = timed(lambda: kspace_cuda(ph.mixture, k, csm=csm, coeff=coeff,
                                          k_chunk=1 << 18), dev)
    rec["radial"].update(single_coil_s=t1, single_coil_gib=m1, multicoil_s=tc,
                         multicoil_gib=mc, out_shape=list(y.shape))
    print(f"radial R={args.accel:g}: {k.shape[0]:,} samples | single coil {t1:.1f} s "
          f"({m1:.2f} GiB) | {args.coils} coils x {csm.n_modes} modes {tc:.1f} s "
          f"({mc:.2f} GiB)", flush=True)
    if args.cartesian:
        kc, _ = cartesian_k()
        kc = torch.tensor(kc, device=dev, dtype=torch.float32)
        _, tcc, mcc = timed(lambda: kspace_cuda(ph.mixture, kc, csm=csm, coeff=coeff,
                                                k_chunk=1 << 18), dev)
        rec["cartesian"] = {"M": int(kc.shape[0]), "multicoil_s": tcc, "multicoil_gib": mcc}
        print(f"Cartesian full: {kc.shape[0]:,} samples | {args.coils} coils {tcc:.1f} s "
              f"({mcc:.2f} GiB)", flush=True)
    (OUT / "summary.json").write_text(json.dumps(rec, indent=2))
    print(f"-> {OUT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
