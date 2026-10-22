#!/usr/bin/env python
"""Does grid-simulation error matter once noise is added? BrainWeb, fully sampled.

Object   BrainWeb s04, 1.70 M primitives fitted to the 0.5 mm tissue maps, T1-weighted GRE
         (15/3/30). The mixture is the ground-truth object: its k-space is closed form.
Data     1 mm, fully sampled:
           cartesian  182 x 218 x 182 at dk = 1/FOV
           radial     3D radial, pi/2 N^2 full spokes (Nyquist), 218 samples per spoke
         analytic   kspace_cuda at every sample (the reference)
         grid h     the mixture rendered at voxel centres on an h-mm grid (Dirac basis),
                    then type-2 NUFFT (mri-nufft/cufinufft, fp32, eps 1e-6) at the same samples -- SNAKE's
                    voxel-phantom path, the NUFFT itself contributing nothing
Recon    the same operator for every data set: weighted adjoint NUFFT (type 1) onto the
         1 mm grid, weights 1 (Cartesian, = inverse DFT) or |k|^2 (radial density)
Noise    Eq. (8): complex white noise, variance mean|y|^2 / SNR, the SAME realization
         added to analytic and grid data, over a range of per-sample SNR
Outputs  k-space NMSE of grid vs analytic; image NMSE of every recon against the
         noiseless analytic recon, per SNR; radial k-shell power of the grid error vs the
         noise floor -> outputs/sim/aliasing_recon_noise/summary.json (+ slices)

    source scripts/env_gpu_nufft.sh
    python experiments/aliasing_recon_noise.py --device cuda:0 --traj cartesian
    python experiments/aliasing_recon_noise.py --device cuda:1 --traj radial
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaussmrisim.conventions import FOV
from gaussmrisim.datasets.brainweb import VolumeTarget
from gaussmrisim.fitting.fitter import render_full
from gaussmrisim.fourier.cuda_ft import kspace_cuda
from gaussmrisim.sim.config import GreConfig
from gaussmrisim.sim.phantom import TissuePhantom

FIT = "outputs/fits/brainweb_s04_fuzzy_0.50mm_measured_area"
CENTRE = np.array([18.0, -18.0, 0.0])
FOV_MM = np.array([182.0, 218.0, 182.0])
ACQ = 1.0                                         # acquisition resolution, mm
SEQ = (15.0, 3.0, 30.0)
OUT = Path("outputs/sim/aliasing_recon_noise")
def cartesian_mean_power():
    """Read the power measured by the Cartesian experiment for this fitted subject."""
    record = OUT / "box_baseline_cartesian_h1.json"
    if not record.exists():
        raise FileNotFoundError("Run the 1 mm Cartesian experiment before the radial study: " + str(record))
    return float(json.loads(record.read_text())["noise_ref_power"])
SNRS = [10.0, 30.0, 50.0, 100.0, 300.0, 1e3, 3e3, 1e4, 1e5]


def cartesian_k():
    n = (FOV_MM / ACQ).astype(int)
    ax = [(np.arange(m) - m // 2) / f for m, f in zip(n, FOV_MM)]
    kz, ky, kx = np.meshgrid(ax[2], ax[1], ax[0], indexing="ij")
    return np.stack([kx, ky, kz], -1).reshape(-1, 3), np.ones(kx.size)


def radial_k():
    kmax, nr = 0.5 / ACQ, int(FOV_MM.max() / ACQ)
    ns = int(round(math.pi / 2 * (FOV_MM.max() / ACQ) ** 2))   # Nyquist on the k_max sphere
    i = np.arange(ns) + 0.5                                         # Fibonacci hemisphere
    z = i / ns
    phi = np.pi * (1 + 5 ** 0.5) * i
    d = np.stack([np.sqrt(1 - z ** 2) * np.cos(phi), np.sqrt(1 - z ** 2) * np.sin(phi), z], -1)
    r = (np.arange(nr) - nr // 2) / nr * 2 * kmax
    k = (d[:, None, :] * r[None, :, None]).reshape(-1, 3)
    w = np.maximum(np.sum(k ** 2, -1), (kmax / nr) ** 2)              # |k|^2 density weights
    return k, w


# mri-nufft's cufinufft precision follows the trajectory dtype; single precision is its
# standard mode, and we use it here: complex64 data, tolerance 1e-6 (relative error ~1e-6,
# i.e. -120 dB, far below every error this study measures).
NUFFT_DTYPE = np.float32
NUFFT_EPS = 1e-6


class GpuNufft:
    """mri-nufft, cufinufft backend, single precision: raw sums with no normalization.

    mri-nufft's op/adj_op compute FINUFFT's isign -1 / +1 sums at x = 2 pi t, t in
    [-0.5, 0.5), divided by ``norm_factor``; multiplying it back gives the plain sums
    (checked against FINUFFT CPU: 2.4e-6 in single precision). Columns of t follow the
    array axes, i.e. (z, y, x) for a [nz, ny, nx] volume.
    """

    def __init__(self, k: np.ndarray, h: float, shape_zyx, eps: float = NUFFT_EPS,
                 dtype=NUFFT_DTYPE):
        import cupy as cp
        from mrinufft import get_operator
        self.cpx = np.complex64 if np.dtype(dtype) == np.float32 else np.complex128
        t = np.ascontiguousarray((k * h)[:, ::-1], dtype=dtype)
        assert t.min() >= -0.5 and t.max() < 0.5, "samples outside the grid's band"
        # mri-nufft defaults to gpu_device_id=0 and switches CuPy to it: pass the current one
        self.op = get_operator("cufinufft")(t, shape=tuple(int(s) for s in shape_zyx),
                                            density=False, n_coils=1, eps=eps,
                                            gpu_device_id=cp.cuda.Device().id)
        self.norm = float(self.op.norm_factor)

    def forward(self, vol: np.ndarray) -> np.ndarray:
        out = np.asarray(self.op.op(vol.astype(self.cpx))).ravel()
        return out.astype(np.complex128) * self.norm

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        out = np.asarray(self.op.adj_op(y.astype(self.cpx))).reshape(self.op.shape)
        return out.astype(np.complex128) * self.norm


def grid_nufft(vol: np.ndarray, h: float, k: np.ndarray) -> np.ndarray:
    """h^3 sum_n vol_n exp(-i 2 pi k.r_n), r_n the voxel centres (GPU NUFFT, single precision, eps 1e-6)."""
    v = GpuNufft(k, h, vol.shape).forward(vol)
    r0 = CENTRE + 0.5 * h                                           # m' = 0 voxel centre
    return v * np.exp(-2j * np.pi * ((k @ r0) % 1.0)) * h ** 3


class Recon:
    """Weighted adjoint NUFFT onto the 1 mm grid -> [nz, ny, nx] (same operator for all)."""

    def __init__(self, k: np.ndarray, w: np.ndarray):
        n = (FOV_MM / ACQ).astype(int)
        self.nu = GpuNufft(k, ACQ, (n[2], n[1], n[0]))
        self.c = w * np.exp(2j * np.pi * ((k @ (CENTRE + 0.5 * ACQ)) % 1.0))

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return self.nu.adjoint(y * self.c)


def nmse_db(a, b):
    return float(10 * np.log10(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2)))
