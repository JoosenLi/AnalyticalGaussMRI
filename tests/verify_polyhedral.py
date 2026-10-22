#!/usr/bin/env python
"""Checks for the polyhedral FT port and the mesh voxelizer, against closed forms.

A shifted box has ``F(k) = 8abc sinc(2a kx) sinc(2b ky) sinc(2c kz) exp(-i2pi k.s)`` and a
known volume; both the CUDA kernel (three precisions), the PyTorch reference and the
voxelizer are compared with it, including k parallel to face normals (the kernel's
special branch) and tiny |k| (the cancellation regime).

    python tests/verify_polyhedral.py --device cuda:0
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaussmrisim.conventions import FOV
from gaussmrisim.datasets.polyhedral import Mesh, check_closed, voxelize
from gaussmrisim.fourier import polyhedral_ft as P


def box(a, b, c, s) -> Mesh:
    V = torch.tensor([[x, y, z] for x in (-a, a) for y in (-b, b) for z in (-c, c)],
                     dtype=torch.float64) + s
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4),
             (1, 5, 7, 3)]
    F = [t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]
    m = Mesh(V, torch.tensor(F))
    return m if m.signed_volume() > 0 else Mesh(V, m.faces[:, [0, 2, 1]])


def box_fine(a, b, c, s, n) -> Mesh:
    """The same box with every face cut into n x n squares (2 triangles each), welded."""
    import itertools
    pts, tris = {}, []

    def vid(p):
        key = tuple(round(x, 12) for x in p)
        if key not in pts:
            pts[key] = len(pts)
        return pts[key]
    h = (a, b, c)
    for ax, sg in itertools.product(range(3), (-1, 1)):
        u, w = [i for i in range(3) if i != ax]
        g = [[None] * (n + 1) for _ in range(n + 1)]
        for i, j in itertools.product(range(n + 1), repeat=2):
            p = [0.0, 0.0, 0.0]
            p[ax] = sg * h[ax]
            p[u] = -h[u] + 2 * h[u] * i / n
            p[w] = -h[w] + 2 * h[w] * j / n
            g[i][j] = vid(p)
        for i, j in itertools.product(range(n), repeat=2):
            tris += [(g[i][j], g[i + 1][j], g[i + 1][j + 1]), (g[i][j], g[i + 1][j + 1], g[i][j + 1])]
    V = torch.tensor(list(pts), dtype=torch.float64) + s
    F = torch.tensor(tris)
    # orient each face outward: normal . (centroid - box centre) > 0
    t = V[F]
    nrm = torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=-1)
    flip = (nrm * (t.mean(1) - s)).sum(-1) < 0
    F = torch.where(flip[:, None], F[:, [0, 2, 1]], F)
    return Mesh(V, F)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    dev = ap.parse_args().device
    a, b, c = 10.0, 7.0, 5.0
    s = torch.tensor([3.0, -2.0, 1.5], dtype=torch.float64)
    m = box(a, b, c, s)
    check_closed(m)
    vol = 8 * a * b * c
    g = torch.Generator().manual_seed(0)
    k = (torch.rand(20000, 3, generator=g, dtype=torch.float64) - 0.5) * 0.4
    k[:6] = torch.tensor([[0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.07], [0.02, 0.02, 0],
                          [1e-4, 0, 0], [1e-3, 2e-3, 0]], dtype=torch.float64)
    truth = (vol * torch.sinc(2 * a * k[:, 0]) * torch.sinc(2 * b * k[:, 1])
             * torch.sinc(2 * c * k[:, 2]) * torch.exp(-2j * math.pi * (k @ s))).to(dev)
    rec = P.pack([m], [1.0], device=dev)
    checks = []
    for prec, tol in (("double", 1e-11), ("f32", 1e-5)):
        out = P.polyhedral_ft(rec, k, prec)
        err = float((out - truth).abs().max() / vol)
        checks.append((f"box FT, {prec}, max |err| / volume (incl. k || N, |k|=1e-4)",
                       err, tol))
    ref = P.polyhedral_ft_reference(rec, k[6:])
    checks.append(("box FT, PyTorch reference, rel L2",
                   float((ref - truth[6:]).norm() / truth[6:].norm()), 1e-11))
    # the box cut into grid cells with random constants == sum of sub-box sincs
    from gaussmrisim.fourier.polyhedral_cells import Grid, cell_edges, edge_ft
    md = m.to(dev)
    kd = k[6:3006].to(dev)
    lo = (s - torch.tensor([a, b, c], dtype=torch.float64)).to(dev)
    hi = (s + torch.tensor([a, b, c], dtype=torch.float64)).to(dev)
    for h, org in ((4.0, (-8.3, -9.7, -4.1)), (7.0, (-7.13, -9.31, -3.77))):
        shape = tuple(int(math.ceil((float(hi[i]) - org[i]) / h)) + 1 for i in range(3))
        grid = Grid(org, h, shape)
        vals = torch.rand(grid.n(), generator=torch.Generator().manual_seed(1),
                          dtype=torch.float64).to(dev)
        O = torch.tensor(org, dtype=torch.float64, device=dev)
        idx = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev) for n in shape],
                                         indexing="ij"), -1).reshape(-1, 3)
        clo, chi = torch.maximum(O + idx * h, lo), torch.minimum(O + (idx + 1) * h, hi)
        ok = (chi > clo).all(1)
        L, ctr, w = (chi - clo)[ok], (0.5 * (clo + chi))[ok], vals[grid.flat(idx[ok])]
        tru = (w * L.prod(1) * torch.sinc(kd[:, None, 0] * L[:, 0])
               * torch.sinc(kd[:, None, 1] * L[:, 1]) * torch.sinc(kd[:, None, 2] * L[:, 2])
               * torch.exp(-2j * math.pi * (kd @ ctr.T))).sum(1)
        rec = cell_edges(md, grid, vals)
        for prec, tol in (("double", 1e-11), ("f32", 1e-5)):
            got = edge_ft(rec, kd, prec)
            checks.append((f"box in {h:g} mm cells, random values, {prec}, rel L2",
                           float((got - tru).norm() / tru.norm()), tol))
        ones = edge_ft(cell_edges(md, grid, torch.ones_like(vals)), kd, "double")
        whole = P.polyhedral_ft(P.pack([md], [1.0], device=dev), kd, "double")
        checks.append((f"box in {h:g} mm cells, all ones == uncut mesh",
                       float((ones - whole).norm() / whole.norm()), 1e-11))
    fov = FOV((40.0, 32.0, 24.0), (80, 64, 48), tuple(s.tolist()))
    f = voxelize(m, fov, supersample=4, device=dev)
    checks.append(("voxelizer volume vs exact",
                   abs(float(f.double().sum()) * fov.voxel_volume_mm3 / vol - 1), 1e-6))

    # the box tetrahedralized, one random constant per tetrahedron
    from gaussmrisim.fourier import polyhedral_tets as PT
    # -Y keeps the surface triangles, so the box surface is pre-subdivided (6x6 per
    # face) to leave TetGen room to refine the interior
    V, Tt = PT.tetrahedralize(box_fine(a, b, c, s, 6), max_volume=20.0)
    V, Tt = V.to(dev), Tt.to(dev)
    checks.append((f"tets ({Tt.shape[0]}) fill the box: sum |T| / volume - 1",
                   abs(float(PT.tet_volumes(V, Tt).sum()) / vol - 1), 1e-12))
    vals = torch.rand(Tt.shape[0], generator=torch.Generator().manual_seed(2),
                      dtype=torch.float64).to(dev)
    kt = k[6:].to(dev)
    # Brion divides by w.(v_j - v_l): keep only k at least 1e-2 rad from orthogonal to
    # every tetrahedron edge, so the REFERENCE is well conditioned (eps / 1e-6 at worst)
    E = torch.cat([V[Tt[:, i]] - V[Tt[:, j]] for i in range(4) for j in range(i + 1, 4)])
    good = (2 * math.pi * (kt @ E.T)).abs().min(1).values > 1e-2
    kt = kt[good][:2000]
    ref, mag = PT.simplex_ft_reference(V, Tt, vals, kt, abs_sum=True)  # Brion: independent
    rec = PT.tet_edges(V, Tt, vals)
    got = edge_ft(rec, kt, "double")
    checks.append((f"box in tets, random values, double vs Brion ({kt.shape[0]} k)",
                   float((got - ref).norm() / ref.norm()), 1e-9))
    # fp32: 265 random-valued tets cancel heavily (the sum is much smaller than its
    # terms), so the error is measured against the scale of the terms, sum_T |a_T F_T|
    got = edge_ft(rec, kt, "f32")
    cond = float(mag.norm() / ref.norm())
    checks.append((f"box in tets, random values, f32 vs Brion / term scale (cond {cond:.1f})",
                   float((got - ref).norm() / mag.norm()), 1e-5))
    ones = edge_ft(PT.tet_edges(V, Tt, torch.ones_like(vals)), kt, "double")
    whole = P.polyhedral_ft(P.pack([md], [1.0], device=dev), kt, "double")
    checks.append(("box in tets, all ones == uncut mesh (interior faces cancel)",
                   float((ones - whole).norm() / whole.norm()), 1e-11))
    r_, c_, v_ = PT.tet_voxel_pairs(V, Tt, fov, sub=3)
    rows_sum = torch.zeros(f.numel(), dtype=torch.float64, device=dev).index_add_(0, r_, v_)
    checks.append(("tet partial-volume matrix: row sums vs exact voxelizer, max |diff|",
                   float((rows_sum - f.reshape(-1).to(dev).double()).abs().max()), 0.12))
    a_true = 1 + 0.3 * torch.rand(Tt.shape[0], generator=torch.Generator().manual_seed(3),
                                  dtype=torch.float64).to(dev)
    tgt = PT.tet_image(r_, c_, v_, a_true, f.numel())
    a_fit = PT.fit_tet_values(r_, c_, v_, Tt.shape[0], tgt, ridge=1e-8, iters=2000)
    img = PT.tet_image(r_, c_, v_, a_fit, f.numel())
    checks.append(("tet LS fit reproduces a tet-generated image, rel L2",
                   float((img - tgt).norm() / tgt.norm()), 1e-6))
    ok = 0
    for name, e, tol in checks:
        flag = e <= tol
        ok += flag
        print(f"  [{'ok ' if flag else 'BAD'}] {name:62s} {e:.3e}  (tol {tol:g})")
    print(f"\n{ok}/{len(checks)} checks passed")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
