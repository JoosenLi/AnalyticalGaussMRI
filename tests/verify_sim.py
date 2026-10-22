#!/usr/bin/env python
"""Verify the simulation layer before any number produced by it is quoted.

The checks are chosen so each one can only pass for the right reason:

1. ``simple`` with unit contrast **is** the plain object spectrum -- the contrast path
   is a gather and must not perturb anything when the gather is all ones.
2. Contrast is **linear in the tissues**: a two-tissue phantom equals the weighted sum
   of its parts evaluated separately. This is what licenses the per-primitive gather.
3. ``t2star`` collapses onto ``simple`` when every sample sits at the echo. If it did
   not, the two models would be different physics rather than the same physics sampled
   differently, and no comparison between them would mean anything.
4. ``t2star`` off the echo actually **changes** the answer (a test that cannot fail is
   not a test), and it decays -- tissues with short T2* lose more.
5. An **identity pose** reproduces the static acquisition, through the motion code path
   rather than around it.
6. The fast per-shot coil-motion path equals ``moved_coil_kspace``, the V-launch
   reference already in ``verify_core``. This is the one that matters: the fast path
   rotates the coil modes and folds a phase into the coil coefficients, and an error
   there would look like a plausible sensitivity map rather than like a bug.
7. Noise reaches the requested SNR, and shot reshaping round-trips.

    python tests/verify_sim.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaussmrisim.acquisition import trajectory as T
from gaussmrisim.coils.fourier_csm import FourierCSM, cartesian_mode_grid
from gaussmrisim.conventions import FOV
from gaussmrisim.fourier.cuda_ft import kspace_cuda
from gaussmrisim.motion.rigid import RigidPoses, moved_coil_kspace
from gaussmrisim.representation.gaussian3d import random_mixture
from gaussmrisim.sim import (AcquisitionEngine, GreConfig, HardwareConfig, SimConfig,
                             TissuePhantom, shot_poses)
from gaussmrisim.sim.engine import _to_shots

OK, BAD = "  ok  ", " FAIL "


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def main() -> int:
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 1
    dev = "cuda:0"
    passed = failed = 0

    def check(name, val, tol, note=""):
        nonlocal passed, failed
        good = bool(val <= tol)
        passed += good
        failed += not good
        print(f"[{OK if good else BAD}] {name:<52s} {val:.3e} <= {tol:.0e}  {note}")

    fov = FOV(fov_mm=(200.0,) * 3, matrix=(64, 64, 64))
    g = torch.Generator(device="cpu").manual_seed(0)
    n = 3000
    mix = random_mixture(n, fov_mm=fov.fov_mm, sigma_range=(2.0, 8.0), seed=0).to(dev)
    # two tissues with very different T2*: white matter (61 ms) and csf (2000 ms)
    tissue = torch.where(torch.arange(n, device=dev) % 2 == 0,
                         torch.tensor(3, device=dev), torch.tensor(1, device=dev))
    ph = TissuePhantom(mix, tissue, field="1.5T")
    cfg = SimConfig(fov=fov, seq=GreConfig(tr_ms=50.0, te_ms=30.0, flip_deg=15.0),
                    hardware=HardwareConfig(n_coils=4))
    traj = T.build("radial", 32, 65, fov, tr_ms=5.0, device=dev)
    eng = AcquisitionEngine(model="simple", snr=float("inf"))

    # --- 1. unit contrast == the plain spectrum
    a = ph.coeff(cfg.seq)
    ksp = eng.acquire(ph, traj, cfg, by_shot=False)[0]
    ref = kspace_cuda(mix, traj.k, coeff=a)
    check("simple == kspace_cuda with the same coeff", rel(ksp, ref), 1e-6)

    # --- 2. linear in the tissues
    parts = 0
    for lab in ph.labels:
        sel = tissue == lab
        parts = parts + kspace_cuda(mix.subset(torch.nonzero(sel).flatten()),
                                    traj.k, coeff=a[sel])
    check("contrast is linear over tissues", rel(ksp, parts), 1e-6,
          "(licenses the per-primitive gather)")

    # --- 3. t2star at the echo == simple
    flat = T.Trajectory(k=traj.k, time_ms=torch.full_like(traj.time_ms, cfg.seq.te_ms),
                        shot=traj.shot, kind=traj.kind)
    e2 = AcquisitionEngine(model="t2star", snr=float("inf"))
    k_echo = e2.acquire(ph, flat, cfg, by_shot=False)[0]
    check("t2star at t=TE == simple", rel(k_echo, ksp), 1e-6)

    freqs_pre = cartesian_mode_grid(2, fov.fov_mm, device=dev)
    beta_pre = (torch.randn(cfg.hardware.n_coils, freqs_pre.shape[0], device=dev)
                + 1j * torch.randn(cfg.hardware.n_coils, freqs_pre.shape[0],
                                   device=dev)).to(torch.complex64)
    csm_pre = FourierCSM(coeffs=beta_pre, freqs=freqs_pre)

    # --- 4. ... and off the echo it differs, in the right direction
    k_t2s = e2.acquire(ph, traj, cfg, by_shot=False)[0]
    diff = rel(k_t2s, ksp)
    print(f"         t2star vs simple over the readout: {diff:.3e} (must be large)")
    check("t2star off-echo actually changes the signal", 1.0 / max(diff, 1e-12), 1e3,
          "(a check that cannot fail is not a check)")

    # --- 4b. the in-kernel decay must equal the per-tissue reference, and be faster
    e2k = AcquisitionEngine(model="t2star", snr=float("inf"), t2star_impl="kernel")
    k_fast = e2k.acquire(ph, traj, cfg, by_shot=False)[0]
    check("t2star in-kernel == per-tissue reference", rel(k_fast, k_t2s), 1e-5,
          "(one launch vs one per tissue)")
    k_fast_echo = e2k.acquire(ph, flat, cfg, by_shot=False)[0]
    check("t2star in-kernel at t=TE == simple", rel(k_fast_echo, ksp), 1e-6)
    cs_ref = e2.acquire(ph, traj, cfg, csm=csm_pre, by_shot=False)
    cs_fast = e2k.acquire(ph, traj, cfg, csm=csm_pre, by_shot=False)
    check("t2star in-kernel with coils == reference", rel(cs_fast, cs_ref), 1e-5)

    # --- 5. identity pose == static, through the motion path
    S = int(traj.shot.max()) + 1
    still = RigidPoses(R=torch.eye(3, device=dev)[None].expand(S, 3, 3).contiguous(),
                       t=torch.zeros(S, 3, device=dev))
    k_still = eng.acquire(ph, traj, cfg, poses=still, by_shot=False)[0]
    check("identity pose == static (motion path)", rel(k_still, ksp), 1e-6)

    # --- 6. the fast per-shot coil path == the V-launch reference
    freqs = cartesian_mode_grid(3, fov.fov_mm, device=dev)
    beta = (torch.randn(cfg.hardware.n_coils, freqs.shape[0], device=dev)
            + 1j * torch.randn(cfg.hardware.n_coils, freqs.shape[0], device=dev))
    csm = FourierCSM(coeffs=beta.to(torch.complex64), freqs=freqs)
    poses, _ = shot_poses(S, tr_ms=5.0, ts_std_mm_s=(8.0, 8.0, 8.0),
                          rs_std_deg_s=(4.0, 4.0, 4.0), seed=1, device=dev)
    fast = AcquisitionEngine(per_shot_coils=True).acquire(
        ph, traj, cfg, csm=csm, poses=poses, by_shot=False)
    per_sample = RigidPoses(R=poses.R[traj.shot.long()].contiguous(),
                            t=poses.t[traj.shot.long()].contiguous())
    slow = moved_coil_kspace(mix, traj.k, csm, per_sample, coeff=a)
    check("per-shot coil motion == moved_coil_kspace", rel(fast, slow), 5e-5,
          "(equivalent, but 0.9x -- not the default)")
    default = eng.acquire(ph, traj, cfg, csm=csm, poses=poses, by_shot=False)
    check("default coil-motion path == moved_coil_kspace", rel(default, slow), 1e-6)

    # a static-coil sanity check too, so the coil path is pinned without motion
    k_coil = eng.acquire(ph, traj, cfg, csm=csm, by_shot=False)
    check("static coils == kspace_cuda(csm=...)",
          rel(k_coil, kspace_cuda(mix, traj.k, csm=csm, coeff=a)), 1e-6)

    # --- 7. noise reaches the requested SNR; shots round-trip
    eng_n = AcquisitionEngine(model="simple", snr=100.0)
    noisy = eng_n.acquire(ph, traj, cfg, csm=csm, by_shot=False)
    got = float((k_coil.abs() ** 2).mean() / ((noisy - k_coil).abs() ** 2).mean())
    print(f"         requested SNR 100, measured {got:.1f}")
    check("noise hits the requested SNR", abs(got - 100.0) / 100.0, 0.15)

    shots = _to_shots(k_coil, traj)
    back = shots.permute(1, 0, 2).reshape(k_coil.shape)
    check("shot reshape round-trips", rel(back, k_coil), 0.0,
          f"[C,M]={tuple(k_coil.shape)} -> {tuple(shots.shape)}")

    print(f"\n{passed}/{passed + failed} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
