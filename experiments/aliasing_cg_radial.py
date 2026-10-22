#!/usr/bin/env python
"""Radial arm of the noise-floor study with a least-squares reconstruction (LSQR or CG).

The density-compensated adjoint NUFFT of ``aliasing_box_baseline.py`` is not an inverse of
the radial encoding, so its image is not the image the data determine. Here every radial
data set is reconstructed by the least-squares solution on the 1 mm grid,

    x = argmin ||F x - y'||^2   <=>   F^H F x = F^H y',

solved to a tight tolerance (no density weighting, no regularization, no early stopping)
by either
  lsqr  mri-nufft's own solver, ``op.pinv_solver(y, optim="lsqr")`` on the cufinufft
        operator (default), or
  cg    conjugate gradients on the normal equations, F^H F applied exactly through its
        Toeplitz kernel (T = F^H 1 on the 2x grid, then a zero-padded circular convolution
        by FFT; checked against F^H(F x) on a random image) -- an independent check.

Because the converged solve is LINEAR, recon(y + s n) = LS(y) + s LS(n): one solve for the
noiseless analytic data, one per grid arm and one for the shared noise realization give
every SNR, with the same noise, sigma (Cartesian mean power) and metrics as before; one
SNR is also reconstructed directly from the noisy data to confirm it.

    source scripts/env_gpu_nufft.sh
    python experiments/aliasing_cg_radial.py --device cuda:0 --h 1 --ss 4
    python experiments/aliasing_cg_radial.py --device cuda:1 --h 0.5 --ss 2
    -> outputs/sim/aliasing_recon_noise/box_baseline_radial_h{h}_{solver}.json
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

from aliasing_box_baseline import box_average, render                       # noqa: E402
from aliasing_recon_noise import (ACQ, cartesian_mean_power, FIT, FOV_MM, OUT, SEQ,  # noqa: E402
                                  SNRS, GpuNufft, Recon, grid_nufft, nmse_db, radial_k)
from gaussmrisim.fourier.cuda_ft import kspace_cuda                          # noqa: E402
from gaussmrisim.sim.config import GreConfig                                 # noqa: E402
from gaussmrisim.sim.phantom import TissuePhantom                            # noqa: E402


class Normal:
    """F^H F on the 1 mm grid via its Toeplitz kernel (exact up to the NUFFT tolerance)."""

    def __init__(self, k: np.ndarray, shape_zyx, dev):
        n = tuple(int(s) for s in shape_zyx)
        big = tuple(2 * s for s in n)
        T = GpuNufft(k, ACQ, big).adjoint(np.ones(len(k)))          # modes -n .. n-1
        c = torch.from_numpy(np.fft.ifftshift(T)).to(dev)            # mode 0 at index 0
        self.K, self.n, self.big = torch.fft.fftn(c), n, big

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.zeros(self.big, dtype=x.dtype, device=x.device)
        p[: self.n[0], : self.n[1], : self.n[2]] = x
        y = torch.fft.ifftn(self.K * torch.fft.fftn(p))
        return y[: self.n[0], : self.n[1], : self.n[2]]


def cg(A, b: torch.Tensor, tol: float, maxit: int):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rr = torch.vdot(r.reshape(-1), r.reshape(-1)).real
    b2 = rr.clone()
    it = 0
    while it < maxit and float(rr / b2) > tol ** 2:
        Ap = A(p)
        a = rr / torch.vdot(p.reshape(-1), Ap.reshape(-1)).real
        x += a * p
        r -= a * Ap
        rn = torch.vdot(r.reshape(-1), r.reshape(-1)).real
        p = r + (rn / rr) * p
        rr = rn
        it += 1
    return x, it, float((rr / b2).sqrt())


def lsqr_iterates(op, bk, counts, shape, x0=None) -> dict:
    """mri-nufft LSQR, no stopping test; the iterates at each count in ``counts``.

    ``x0`` (with damp = 0) is a warm start: that path subtracts A x0 from the residual and
    leaves the objective unchanged. (mri-nufft's own density warm start does not: it sets
    x = x_init but starts the residual at b, and with a callback it returns before restoring
    the operator's density -- so it is not used here.)
    """
    import cupy as cp
    out, n = {}, [0]

    def keep(x, operator, kspace_data, **kw):
        n[0] += 1
        if n[0] in counts:
            out[n[0]] = cp.asnumpy(x).reshape(shape)
        return 0

    x = op.pinv_solver(bk, optim="lsqr", atol=0.0, btol=0.0, conlim=0, max_iter=max(counts),
                       progressbar=False, callback=keep, damp=0.0, x0=x0)
    x = x[0] if isinstance(x, tuple) else x
    for c in counts:                    # stopped at machine precision: the final iterate
        if c not in out:
            print(f"    LSQR converged at {n[0]} iterations (< {c})", flush=True)
            out[c] = cp.asnumpy(x).reshape(shape)
    return out


def fixed_iterations(args, k, w, ya, vols, h, adj, shape) -> int:
    """Every radial data set reconstructed directly by LSQR with a FIXED iteration count.

    Unregularized least squares is ill-posed here (the radial sphere leaves the corners of
    the 1 mm grid's frequency cube unsampled), so the iteration count is the regularizer,
    as in practice. Each noisy data set is reconstructed on its own (no linearity
    shortcut), and the noiseless analytic reference with the same count.

    ``--dc-init``: start from the density-compensated adjoint (|k|^2 weights), scaled to
    the data as mri-nufft's ``_scaled_dcp`` does, then iterate WITHOUT density: a warm
    start, same objective.
    """
    import cupy as cp
    counts = sorted(set(args.fixed))
    op, ph0, norm, cpx = adj.nu.op, cp.asarray(adj.c), adj.nu.norm, adj.nu.cpx
    wd = cp.asarray(w / w.mean(), dtype=cp.float32)
    rhs = lambda y: (cp.asarray(y) * ph0 / norm).astype(cpx)

    def run(y):
        b = rhs(y)
        x0 = None
        if args.dc_init:
            x0 = op.adj_op(b * wd).reshape(op.img_full_shape)
            x0 *= cp.linalg.norm(b) / cp.linalg.norm(op.op(x0))
        return lsqr_iterates(op, b, counts, shape, x0=x0)
    rng = np.random.default_rng(0)                     # the same realization as the study
    noise = (rng.standard_normal(len(k)) + 1j * rng.standard_normal(len(k))) / np.sqrt(2)
    sig = {snr: math.sqrt(cartesian_mean_power() / snr) for snr in SNRS}
    t0 = time.time()
    xa = run(ya)
    print(f"  analytic, noiseless: {time.time() - t0:.0f} s for {max(counts)} iterations",
          flush=True)
    res = {n: {"psnr_offset_db": float(10 * np.log10(np.max(np.abs(xa[n])) ** 2
                                                     / np.mean(np.abs(xa[n]) ** 2))),
               "analytic": {}, "arms": {}} for n in counts}
    for snr, s in sig.items():
        xs = run(ya + s * noise)
        for n in counts:
            res[n]["analytic"][snr] = res[n]["psnr_offset_db"] - nmse_db(xs[n], xa[n])
    print(f"  analytic, noisy: {[round(res[n]['analytic'][100.0], 1) for n in counts]} dB "
          f"at SNR 100 for {counts} iterations", flush=True)
    arms = [("point", "point")] + [(f"box{s}-dirac", f"box{s}") for s in args.ss]
    for name, vkey in arms:
        yg = grid_nufft(vols[vkey], h, k)
        xg = run(yg)
        for n in counts:
            off = res[n]["psnr_offset_db"]
            res[n]["arms"][name] = {"kspace_nmse_db": nmse_db(yg, ya),
                                    "ceiling_psnr_db": off - nmse_db(xg[n], xa[n]),
                                    "psnr_db": {}}
        for snr, s in sig.items():
            xs = run(yg + s * noise)
            for n in counts:
                res[n]["arms"][name]["psnr_db"][snr] = (res[n]["psnr_offset_db"]
                                                        - nmse_db(xs[n], xa[n]))
        print(f"  {name:12s} ceiling {[round(res[n]['arms'][name]['ceiling_psnr_db'], 1) for n in counts]}"
              f" dB for {counts} iterations", flush=True)
    main_n = args.primary
    rec = {"M": int(len(k)), "h": h, "noise_ref_power": cartesian_mean_power(),
           "recon": f"mri-nufft LSQR (pinv_solver), {main_n} iterations, "
                    + ("warm start from the density-compensated adjoint, " if args.dc_init
                       else "zero start, ")
                    + "no density in the iterations, no regularization; every data set "
                    "reconstructed directly",
           "iterations": main_n, **res[main_n], "by_iterations": res}
    out = OUT / (f"box_baseline_radial_h{h:g}_lsqr_fixed"
                 + ("_dcinit" if args.dc_init else "") + ".json")
    out.write_text(json.dumps(rec, indent=2))
    print(f"-> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--ss", type=int, nargs="+", default=[4])
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--maxit", type=int, default=5000)
    ap.add_argument("--solver", default="lsqr", choices=["lsqr", "cg"])
    ap.add_argument("--fixed", type=int, nargs="*", default=None,
                    help="LSQR with fixed iteration counts (e.g. 10 20 30 50): every data "
                         "set reconstructed directly; the regularized, practical recon")
    ap.add_argument("--dc-init", action="store_true",
                    help="with --fixed: warm start from the density-compensated adjoint")
    ap.add_argument("--primary", type=int, default=30,
                    help="with --fixed: the count reported at top level (the figure's)")
    ap.add_argument("--snap", type=int, default=250,
                    help="LSQR: keep the iterate at this count to show the metrics converged")
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
        vols[f"box{s}"] = box_average(mixw, h, s, dev)
        torch.cuda.empty_cache()

    k, w_dc = radial_k()
    n = (FOV_MM / ACQ).astype(int)
    shape = (n[2], n[1], n[0])
    ya = kspace_cuda(ph.mixture, torch.tensor(k, device=dev, dtype=torch.float32),
                     coeff=coeff, k_chunk=1 << 20).cpu().numpy().astype(np.complex128)
    adj = Recon(k, np.ones(len(k)))                    # F^H (y e^{+i2pi k.r0}), no weights
    toe_err = None
    if args.solver == "cg":
        t0 = time.time()
        A = Normal(k, shape, dev)
        print(f"Toeplitz kernel in {time.time() - t0:.0f} s", flush=True)
        # the Toeplitz normal operator against the direct NUFFT pair
        g = torch.Generator().manual_seed(0)
        xr = torch.complex(torch.randn(shape, generator=g, dtype=torch.float64),
                           torch.randn(shape, generator=g, dtype=torch.float64))
        nu = adj.nu
        direct = torch.from_numpy(nu.adjoint(nu.forward(xr.numpy()))).to(dev)
        toe_err = float((A(xr.to(dev)) - direct).norm() / direct.norm())
        print(f"Toeplitz vs direct F^H F: {toe_err:.1e}", flush=True)
        assert toe_err < 1e-6
    else:
        import cupy as cp
        op = adj.nu.op                                 # mri-nufft cufinufft, fp32, eps 1e-6
        ph0 = cp.asarray(adj.c)                        # e^{+i2pi k.r0}: F = norm * op
        norm = adj.nu.norm

    def solve(y: np.ndarray, name: str) -> np.ndarray:
        t1 = time.time()
        if args.solver == "cg":
            b = torch.from_numpy(adj(y)).to(dev)
            x, it, res = cg(A, b, args.tol, args.maxit)
            x = x.cpu().numpy()
        else:
            # argmin ||norm op(x) - y e^{+i2pi k.r0}||, mri-nufft's LSQR, no density
            calls = []
            bk = (cp.asarray(y) * ph0 / norm).astype(adj.nu.cpx)   # the operator's precision
            bn = float(cp.linalg.norm(op.adj_op(bk)))

            def monitor(x, operator, kspace_data, **kw):
                # every 20 iterations: relative normal-equation residual ||A^H(Ax-b)||/||A^H b||
                calls.append(1)
                if len(calls) == args.snap:
                    snaps[name] = cp.asnumpy(x).reshape(shape)
                if len(calls) % 20 == 0:
                    g = op.adj_op(op.op(x).reshape(bk.shape) - bk)
                    print(f"    {name} it {len(calls):5d}  {time.time() - t1:7.0f} s  normal "
                          f"residual {float(cp.linalg.norm(g)) / bn:.2e}", flush=True)
                return 0

            x = op.pinv_solver(bk, optim="lsqr", atol=args.tol, btol=args.tol, conlim=0,
                               max_iter=args.maxit, progressbar=False, callback=monitor)
            if isinstance(x, tuple):
                x = x[0]
            x = cp.asnumpy(x).reshape(shape)
            it = len(calls)
            r = cp.asnumpy(op.op(cp.asarray(x))).ravel() * norm - y * adj.c
            res = float(np.linalg.norm(adj.nu.adjoint(r)) / np.linalg.norm(adj(y)))
        print(f"  {args.solver.upper()} {name:12s} {it:4d} it, rel. normal residual "
              f"{res:.1e}, {time.time() - t1:.0f} s", flush=True)
        info[name] = {"iterations": it, "rel_normal_residual": res}
        return x

    if args.fixed:
        return fixed_iterations(args, k, w_dc, ya, vols, h, adj, shape)
    info, snaps = {}, {}
    e_ref = cartesian_mean_power()
    rng = np.random.default_rng(0)                     # the same realization as the study
    noise = (rng.standard_normal(len(k)) + 1j * rng.standard_normal(len(k))) / np.sqrt(2)
    xa = solve(ya, "analytic")
    xn = solve(noise, "noise")
    off = float(10 * np.log10(np.max(np.abs(xa)) ** 2 / np.mean(np.abs(xa) ** 2)))
    sig = {snr: math.sqrt(e_ref / snr) for snr in SNRS}
    rec = {"M": int(len(k)), "h": h,
           "recon": {"lsqr": "least squares, mri-nufft pinv_solver(optim='lsqr')",
                     "cg": "least squares, CG on the normal equations"}[args.solver],
           "toeplitz_vs_direct": toe_err, "psnr_offset_db": off, "noise_ref_power": e_ref,
           "analytic": {snr: off - nmse_db(xa + s * xn, xa) for snr, s in sig.items()},
           "arms": {}, "cg": info}
    arms = [("point", "point")] + [(f"box{s}-dirac", f"box{s}") for s in args.ss]
    for name, vkey in arms:
        yg = grid_nufft(vols[vkey], h, k)
        xg = solve(yg, name)
        if name in snaps and "analytic" in snaps:
            # the same metric from the iterates at --snap: converged if they agree
            info[name]["ceiling_psnr_db_at_snap"] = off - nmse_db(snaps[name], snaps["analytic"])
            info[name]["snap_vs_final_nmse_db"] = nmse_db(snaps[name], xg)
        r = {"kspace_nmse_db": nmse_db(yg, ya), "ceiling_psnr_db": off - nmse_db(xg, xa),
             "psnr_db": {snr: off - nmse_db(xg + s * xn, xa) for snr, s in sig.items()}}
        rec["arms"][name] = r
        if vkey != "point":
            # linearity check: reconstruct the SNR-100 data directly
            s = sig[100.0]
            xd = solve(yg + s * noise, f"{name}@100")
            r["direct_snr100_psnr_db"] = off - nmse_db(xd, xa)
            r["direct_vs_linear_nmse_db"] = nmse_db(xd, xg + s * xn)
        print(f"  {name:12s} ceiling PSNR {r['ceiling_psnr_db']:5.1f} dB | SNR 100: "
              f"{r['psnr_db'][100.0]:5.1f} dB (analytic {rec['analytic'][100.0]:5.1f})",
              flush=True)
    out = OUT / f"box_baseline_radial_h{h:g}_{args.solver}.json"
    out.write_text(json.dumps(rec, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
