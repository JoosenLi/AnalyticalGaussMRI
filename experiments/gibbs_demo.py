#!/usr/bin/env python
"""Gibbs ringing: present in analytical data, absent from grid-simulated data.

A scanner samples the continuous spectrum, so a fully sampled 1 mm Cartesian acquisition
is the object's spectrum truncated at the band edge, and its inverse DFT rings at sharp
edges. A grid simulation that takes the DFT of a 1 mm voxel phantom and inverts it on the
same grid returns the phantom exactly: the high frequencies that cause the ringing were
never there, and the reconstruction looks better than any real acquisition could
(the inverse crime; cf. Guerquin-Kern et al., TMI 2012, Fig. 1).

  object     BrainWeb s04, 1.70 M primitives (0.5 mm tissue fit), T1w GRE 15/3/30
  analytic   closed-form k-space on the 1 mm Cartesian grid (182 x 218 x 182), inverse DFT
  grid       voxel averages of the mixture on the 1 mm grid (4^3 supersampling), forward
             NUFFT on the same grid (= DFT), inverse DFT -- returns the voxel image
  reference  the continuous object rendered at 0.25 mm (point samples), for the profile
  no noise anywhere

    source scripts/env_gpu_nufft.sh
    python experiments/gibbs_demo.py --device cuda:0
    -> outputs/sim/gibbs_demo/slices.npz, summary.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aliasing_box_baseline import box_average, render                        # noqa: E402
from aliasing_recon_noise import (ACQ, FIT, FOV_MM, SEQ, Recon, cartesian_k,  # noqa: E402
                                  grid_nufft)
from gaussmrisim.fourier.cuda_ft import kspace_cuda                           # noqa: E402
from gaussmrisim.sim.config import GreConfig                                  # noqa: E402
from gaussmrisim.sim.phantom import TissuePhantom                             # noqa: E402

OUT = Path("outputs/sim/gibbs_demo")
FINE = 0.25                                       # reference rendering, mm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    import cupy
    cupy.cuda.Device(dev.index).use()
    OUT.mkdir(parents=True, exist_ok=True)
    ph = TissuePhantom.load(FIT, device=dev)
    coeff = ph.coeff(GreConfig(tr_ms=SEQ[0], te_ms=SEQ[1], flip_deg=SEQ[2]))
    mixw = dataclasses.replace(ph.mixture, mass=ph.mixture.mass * coeff)

    k, w = cartesian_k()
    n = (FOV_MM / ACQ).astype(int)
    shape = (n[2], n[1], n[0])
    t0 = time.time()
    ya = kspace_cuda(ph.mixture, torch.tensor(k, device=dev, dtype=torch.float32),
                     coeff=coeff, k_chunk=1 << 20).cpu().numpy().astype(np.complex128)
    print(f"analytic Cartesian k-space, {len(k):,} samples, {time.time() - t0:.0f} s", flush=True)
    vox = box_average(mixw, ACQ, 4, dev)                     # 1 mm voxel phantom
    yg = grid_nufft(vox, ACQ, k)
    recon = Recon(k, w)                                      # unit weights: inverse DFT
    scale = 1.0 / (len(k) * ACQ ** 3)                        # DFT normalization -> intensity
    xa = recon(ya) * scale
    xg = recon(yg) * scale
    # the grid round trip must return the voxel phantom itself
    rt = float(np.linalg.norm(xg - vox) / np.linalg.norm(vox))
    print(f"grid round trip vs voxel phantom: {rt:.1e}; analytic imag/real "
          f"{np.linalg.norm(xa.imag) / np.linalg.norm(xa.real):.1e}", flush=True)
    assert rt < 1e-4

    z = n[2] // 2 + 8                                        # the slice of Figs. 4-5
    torch.cuda.empty_cache()
    fine = render(mixw, FINE, dev)                           # continuous object, 0.25 mm
    zf = int(round((z + 0.5) * ACQ / FINE - 0.5))            # nearest fine slice to z
    ref = fine[zf - 1:zf + 1].mean(0)                        # the two fine slices around it
    np.savez_compressed(OUT / "slices.npz", analytic=xa[z].real, grid=xg[z].real,
                        voxel=vox[z], fine=ref, h=ACQ, fine_h=FINE)
    (OUT / "summary.json").write_text(json.dumps({
        "samples": int(len(k)), "shape_zyx": [int(s) for s in shape], "slice": int(z),
        "grid_roundtrip_rel": rt,
        "analytic_vs_grid_nrmse": float(np.linalg.norm(xa.real - xg.real)
                                        / np.linalg.norm(xg.real)),
        "analytic_min": float(xa.real.min()), "grid_min": float(xg.real.min())}, indent=2))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
