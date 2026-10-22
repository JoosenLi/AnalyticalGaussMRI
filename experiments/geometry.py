#!/usr/bin/env python
"""What each representation actually is: the geometry of the Shepp-Logan fits in a ROI.

The axial plane z = 0 cuts every model; the cross-section is drawn over the reference
voxels in a 26 x 15 mm window around the three small ellipsoids:
  polyhedral      marching-cubes triangles  -> segments
  tetrahedral     tetrahedra (<= 5 mm^3, and the parameter-matched mesh) -> polygons
  Gaussian        each primitive's conditional 1-sigma ellipse in the plane, opacity
                  exp(-(z0 - mu_z)^2 / (2 Sigma_zz)) (how much of it the plane sees)

    python experiments/geometry.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection, PolyCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from gaussmrisim.datasets.polyhedral import from_voxels           # noqa: E402
from gaussmrisim.fourier import polyhedral_tets as PT             # noqa: E402
from gaussmrisim.representation.gaussian3d import GaussianMixture3D  # noqa: E402
from shepplogan_benchmark import CACHE, ROOT as SL, VOX, make_fov, mc_meshes, phantom  # noqa

FIG = ROOT / "outputs/figures"
WIN = (-14.0, 12.0, -62.0, -47.0)            # x0, x1, y0, y1 in mm (scanner frame)
Z0 = 0.0
COL = "#d95f02"


def plane_segments_tri(V, F, z0):
    """Triangle mesh  cut by z = z0 -> list of 2D segments."""
    tri = V[F]                                            # [F, 3, 3]
    s = tri[..., 2] - z0
    keep = (s.min(1) < 0) & (s.max(1) > 0)
    tri, s = tri[keep], s[keep]
    segs = []
    for t, d in zip(tri, s):
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if d[a] * d[b] < 0:
                w = d[a] / (d[a] - d[b])
                pts.append(t[a, :2] + w * (t[b, :2] - t[a, :2]))
        if len(pts) == 2:
            segs.append(pts)
    return segs


def plane_polys_tet(V, T, z0, win):
    """Tetrahedra cut by z = z0 -> list of 2D polygons inside ``win``."""
    P = V[T]                                              # [T, 4, 3]
    s = P[..., 2] - z0
    keep = (s.min(1) < 0) & (s.max(1) > 0)
    c = P[..., :2].mean(1)
    x0, x1, y0, y1 = win
    keep &= (c[:, 0] > x0 - 3) & (c[:, 0] < x1 + 3) & (c[:, 1] > y0 - 3) & (c[:, 1] < y1 + 3)
    P, s = P[keep], s[keep]
    polys = []
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for t, d in zip(P, s):
        pts = [t[a, :2] + d[a] / (d[a] - d[b]) * (t[b, :2] - t[a, :2])
               for a, b in edges if d[a] * d[b] < 0]
        pts = np.array(pts)
        cen = pts.mean(0)
        pts = pts[np.argsort(np.arctan2(pts[:, 1] - cen[1], pts[:, 0] - cen[0]))]
        polys.append(pts)
    return polys


def tets(step, vol, idx, win=WIN):
    fov = make_fov(phantom())
    if step == 1:
        ms = mc_meshes(phantom(), fov, "cpu")
    else:
        frac = torch.load(CACHE)["channels"]
        ms = [from_voxels(frac[i], fov, 0.5, name=f"E{i + 1}", step=step) for i in idx]
        ms = dict(zip(idx, ms))
    out = []
    for i in idx:
        m = ms[i]
        V, T = PT.tetrahedralize(m, vol)
        out += plane_polys_tet(V.numpy(), T.numpy(), Z0, win)
    return out


def gaussian_ellipses(subject="const", win=WIN, fit="fit_N300k_tv0.0001_floor200"):
    d = torch.load(SL / subject / fit / "mixture.pt", map_location="cpu")
    mix = GaussianMixture3D(**{k: v for k, v in d.items() if k in
                               ("mu", "log_scale", "quaternion", "mass", "dilation_mm")})
    mu = mix.mu.double().numpy()
    R = mix.rotation().double()
    s = mix.effective_sigma().double()
    C = (R @ torch.diag_embed(s ** 2) @ R.transpose(-1, -2)).numpy()
    x0, x1, y0, y1 = win
    near = ((mu[:, 0] > x0 - 2) & (mu[:, 0] < x1 + 2) & (mu[:, 1] > y0 - 2)
            & (mu[:, 1] < y1 + 2) & (np.abs(mu[:, 2] - Z0) < 3 * np.sqrt(C[:, 2, 2])))
    mu, C = mu[near], C[near]
    w = np.exp(-(Z0 - mu[:, 2]) ** 2 / (2 * C[:, 2, 2]))
    cen = mu[:, :2] + C[:, :2, 2] * ((Z0 - mu[:, 2]) / C[:, 2, 2])[:, None]
    Cc = C[:, :2, :2] - C[:, :2, 2:3] @ C[:, 2:3, :2] / C[:, 2, 2][:, None, None]
    th = np.linspace(0, 2 * np.pi, 40)
    circ = np.stack([np.cos(th), np.sin(th)], 0)
    polys = []
    for c, S in zip(cen, Cc):
        L = np.linalg.cholesky(S + 1e-9 * np.eye(2))
        polys.append((c[:, None] + L @ circ).T)
    return polys, w, int(near.sum())


def main():
    ph = phantom()
    fov = make_fov(ph)
    ref = np.load(SL / "const" / "reference_slice.npy")
    nx, ny, nz = fov.matrix
    ox, oy = (c - 0.5 * f for c, f in zip(fov.center_mm[:2], fov.fov_mm[:2]))
    ext = (ox, ox + nx * VOX, oy, oy + ny * VOX)
    small = [7, 8, 9]                                     # the three small ellipsoids
    near_all = list(range(len(ph.value)))
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.25), gridspec_kw={"wspace": 0.04})
    inner = ref[(ref > 0.5) & (ref < 1.8)]
    lo, hi = np.percentile(inner, [0.5, 99.5])
    for a in ax:
        a.imshow(ref, origin="lower", extent=ext, cmap="gray", vmin=lo - 0.05, vmax=hi + 0.05,
                 interpolation="nearest")
        a.set_xlim(WIN[0], WIN[1]); a.set_ylim(WIN[2], WIN[3]); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
    # polyhedral
    ms = mc_meshes(ph, fov, "cpu")
    segs = []
    for i in near_all:
        segs += plane_segments_tri(ms[i].vertices.double().numpy(), ms[i].faces.numpy(), Z0)
    ax[0].add_collection(LineCollection(segs, colors=COL, linewidths=0.6))
    ax[0].set_title("polyhedral (Ngo)", fontsize=6)
    # tetrahedra (only the small ellipsoids and the two that contain them)
    for a, step, vol, name in ((ax[1], 1, 5.0, "tetrahedral"), (ax[2], 2, 8.0,
                                                                 "tetrahedral, matched")):
        polys = tets(step, vol, [1] + small)
        a.add_collection(PolyCollection(polys, facecolors="none", edgecolors=COL,
                                        linewidths=0.25))
        a.set_title(name, fontsize=6)
    # Gaussians
    polys, w, n = gaussian_ellipses()
    order = np.argsort(w)
    ax[3].add_collection(PolyCollection([polys[i] for i in order], facecolors="none",
                                        edgecolors=[(0.85, 0.37, 0.01, float(0.03 + 0.35 * w[i]))
                                                    for i in order], linewidths=0.12))
    ax[3].set_title(f"Gaussian ({n:,} near the plane)", fontsize=6)
    for sp in [s for a in ax for s in a.spines.values()]:
        sp.set_linewidth(0.3)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "geometry.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(FIG / "geometry.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    print(f"-> {FIG / 'geometry.pdf'}  ({n} Gaussians near the plane)")


if __name__ == "__main__":
    main()
