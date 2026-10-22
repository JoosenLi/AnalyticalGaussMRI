"""A closed-surface tissue tetrahedralized, one constant per tetrahedron, with an exact FT.

The extension of Ngo et al.'s piecewise-constant polyhedral phantom from "one constant per
surface" to "one constant per tetrahedron": the interior of the closed mesh ``Omega`` is
cut into a boundary-conforming tetrahedral mesh (TetGen, ``-pY``: the surface triangles
are kept EXACTLY, so the union of the tetrahedra is ``Omega`` itself), and each
tetrahedron gets its own constant ``a_T``. Unlike the axis-aligned grid cells of
:mod:`polyhedral_cells`, the interior faces follow no fixed direction and the mesh is
graded (small at the surface, large inside).

``rho = sum_T a_T 1_T`` is again a sum of weighted planar polygons, so the same verified
edge kernel computes its FT exactly. Every interior triangle is shared by two tetrahedra
with opposite outward normals, so one record with weight ``a_T1 - a_T2`` carries both;
a boundary triangle carries ``a_T``. Equal neighbours cancel and are dropped.

Checked against an INDEPENDENT closed form, the Brion/Lawrence vertex formula for a
simplex (:func:`simplex_ft_reference`), which shares no code with the face kernel.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from ..datasets.polyhedral import Mesh
from .polyhedral_cells import _edge_records

#: the 4 faces of tetrahedron (0,1,2,3), each listed so that it is OUTWARD when the
#: tetrahedron is positively oriented (det[v1-v0, v2-v0, v3-v0] > 0)
_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def tetrahedralize(mesh: Mesh, max_volume: float, min_ratio: float = 0.0):
    """Boundary-conforming tetrahedral mesh of the inside of ``mesh``.

    ``max_volume`` (mm^3) bounds the interior tetrahedra. The surface triangles are kept
    exactly (``-Y``), so the union of the tetrahedra is ``Omega`` itself -- checked by
    volume and by all-ones weights reproducing the uncut mesh FT. ``min_ratio`` is unused
    (quality refinement needs surface splits; see below). Returns
    ``(vertices [V,3] float64, tets [T,4] long)``, every tetrahedron positively oriented.
    """
    import tetgen

    v = mesh.vertices.detach().cpu().double().numpy()
    f = mesh.faces.detach().cpu().numpy()
    # drop vertices no face uses (marching cubes leaves them after its degenerate faces
    # are removed); TetGen would drop them silently and renumber
    used, f = np.unique(f, return_inverse=True)
    v, f = v[used], f.reshape(-1, 3).astype(np.int32)
    # Raw switches, measured on the marching-cubes surfaces: with -p alone (Steiner
    # points allowed on the surface) every variant fails inside TetGen (`split_subface`),
    # and the keyword interface silently ignores the volume bound (three bounds, one
    # mesh). ``-pY`` keeps the surface triangles EXACTLY (sum of volumes = mesh volume to
    # 1e-16) and ``-a`` then refines the interior; quality (-q) is off because it needs
    # surface splits. Tetrahedra touching the surface may exceed the bound (they cannot
    # be split without moving the surface).
    nodes, elem, *_ = tetgen.TetGen(v, f).tetrahedralize(
        switches=f"pYa{float(max_volume):.9g}Q")
    quality = False
    V = torch.from_numpy(np.ascontiguousarray(nodes, dtype=np.float64))
    T = torch.from_numpy(np.ascontiguousarray(elem[:, :4], dtype=np.int64))
    d = torch.linalg.det(torch.stack([V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]],
                                      V[T[:, 3]] - V[T[:, 0]]], dim=1))
    T = torch.where((d < 0)[:, None], T[:, [0, 2, 1, 3]], T)
    tetrahedralize.last_quality = quality
    return V, T


def tet_volumes(V: Tensor, T: Tensor) -> Tensor:
    a, b, c, d = (V[T[:, i]] for i in range(4))
    return torch.linalg.det(torch.stack([b - a, c - a, d - a], dim=1)) / 6.0


def tet_faces(V: Tensor, T: Tensor, a: Tensor):
    """Merged oriented triangles ``[F, 3]`` (vertex ids) and weights ``[F]``.

    Each face of each tetrahedron is keyed by its sorted vertex triple; a key seen twice
    is an interior face and keeps the first orientation with weight ``a_first - a_second``.
    """
    dev = T.device
    tri = torch.cat([T[:, list(f)] for f in _FACES])                     # [4T, 3]
    w = a.to(dev, torch.float64).repeat(4)
    # rows, not a packed integer key: nv^3 overflows int64 beyond ~2M vertices
    uniq, inv, cnt = torch.unique(tri.sort(dim=1).values, dim=0, return_inverse=True,
                                  return_counts=True)
    if int(cnt.max()) > 2:
        raise RuntimeError("a triangle is shared by more than two tetrahedra")
    # first occurrence of each key keeps its orientation; the second subtracts
    order = torch.argsort(inv, stable=True)
    first = torch.ones_like(order, dtype=torch.bool)
    first[1:] = inv[order[1:]] != inv[order[:-1]]
    sign = torch.empty_like(w)
    sign[order] = torch.where(first, 1.0, -1.0).to(w)
    W = torch.zeros(uniq.shape[0], dtype=torch.float64, device=dev)
    W.index_add_(0, inv, sign * w)
    rep = torch.zeros(uniq.shape[0], dtype=torch.long, device=dev)
    rep[inv[order[first]]] = order[first]
    keep = W != 0
    return tri[rep[keep]], W[keep]


def tet_edges(V: Tensor, T: Tensor, a: Tensor) -> Tensor:
    """Edge records ``[E, 14]`` of ``sum_T a_T 1_T`` for :func:`polyhedral_cells.edge_ft`."""
    tri, w = tet_faces(V, T, a)
    t = V.to(tri.device)[tri]                                            # [F, 3, 3]
    N = torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=-1)
    N = N / N.norm(dim=-1, keepdim=True)
    return torch.cat([_edge_records(t[:, e], t[:, (e + 1) % 3], N, t[:, 0], w)
                      for e in range(3)])


def simplex_ft_reference(V: Tensor, T: Tensor, a: Tensor, k: Tensor,
                         abs_sum: bool = False):
    """Independent fp64 check: Brion/Lawrence, for a generic ``k``

        int_T e^{w.r} dr = 6 |T| sum_j e^{w.v_j} / prod_{l != j} w.(v_j - v_l),  w = -i2pi k

    It divides by ``w.(v_j - v_l)``, so it is ill-conditioned when ``k`` is nearly
    orthogonal to an edge -- fine for random ``k`` in a test, never for production.
    """
    k = k.double()
    W = -2j * math.pi * k.to(torch.complex128)                            # [S, 3]
    P = V.double()[T]                                                    # [T, 4, 3]
    vol = tet_volumes(V.double(), T)
    out = torch.zeros(k.shape[0], dtype=torch.complex128, device=k.device)
    for j in range(4):
        num = torch.exp(W @ P[:, j].T.to(torch.complex128))              # [S, T]
        den = torch.ones_like(num)
        for l in range(4):
            if l != j:
                den = den * (W @ (P[:, j] - P[:, l]).T.to(torch.complex128))
        out += (num / den * (6 * vol * a.double()).to(torch.complex128)).sum(1)
    if not abs_sum:
        return out
    # scale of the individual terms, for a cancellation-aware fp32 tolerance
    mag = torch.zeros(k.shape[0], dtype=torch.float64, device=k.device)
    for t0 in range(T.shape[0]):
        one = torch.zeros_like(a, dtype=torch.float64)
        one[t0] = a[t0]
        mag += simplex_ft_reference(V, T[t0:t0 + 1], one[t0:t0 + 1], k).abs()
    return out, mag


@torch.no_grad()
def tet_voxel_pairs(V: Tensor, T: Tensor, fov, sub: int = 3, max_points: int = 16_000_000):
    """Partial-volume matrix ``A[v, t] = |voxel v ∩ T_t| / |voxel|`` by sub-sampling.

    ``sub**3`` points per voxel at an irrationally jittered sub-grid (a marching-cubes
    surface has vertices on the voxel-centre lattice, so an unjittered sub-point can lie
    exactly on a shared face and be counted twice). Each point is located by a
    barycentric test against the tetrahedra whose bounding box contains it.
    Returns COO ``(rows [P] flat voxel index z-slowest, cols [P], vals [P])``.
    """
    dev = T.device
    nx, ny, nz = fov.matrix
    d = torch.tensor(fov.voxel_mm, dtype=torch.float64, device=dev)
    o = torch.tensor([c - 0.5 * f for c, f in zip(fov.center_mm, fov.fov_mm)],
                     dtype=torch.float64, device=dev)
    jit = torch.tensor([1e-4 * math.pi, 1e-4 * math.e, 1e-4 * math.sqrt(2)],
                       dtype=torch.float64, device=dev)
    hs = d / sub
    P = V.to(dev).double()[T]                                            # [T, 4, 3]
    # sub-point integer coordinates u: position = o + (u + 0.5 + jit) * hs
    lo = torch.ceil((P.min(1).values - o) / hs - 0.5 - jit).long()
    hi = torch.floor((P.max(1).values - o) / hs - 0.5 - jit).long()
    n = (hi - lo + 1).clamp(min=0)
    # one inverse per tetrahedron, so a point costs one 3x3 matvec, not a gather + solve
    Minv = torch.linalg.inv(torch.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0],
                                         P[:, 3] - P[:, 0]], 2))
    p0 = P[:, 0].contiguous()
    del P
    rows, cols, cnts = [], [], []
    per = n.prod(1)
    # chunk by sub-point count, not by tetrahedron count: tetrahedra span 4 orders of size
    cs = torch.cumsum(per, 0)
    marks = torch.arange(max_points, max(int(cs[-1]), max_points) + 1, max_points,
                         device=dev)
    cut = torch.searchsorted(cs, marks) + 1
    bounds = sorted({0, T.shape[0], *[min(int(c), T.shape[0]) for c in cut.tolist()]})
    for a0, a1 in zip(bounds[:-1], bounds[1:]):
        sl = slice(a0, a1)
        chunk = a1 - a0
        cnt = per[sl]
        tot = int(cnt.sum())
        if tot == 0:
            continue
        src = torch.repeat_interleave(torch.arange(cnt.numel(), device=dev), cnt)
        pos = torch.arange(tot, device=dev) - torch.repeat_interleave(
            torch.cumsum(cnt, 0) - cnt, cnt)
        nn = n[sl][src]
        u = lo[sl][src] + torch.stack([pos % nn[:, 0], (pos // nn[:, 0]) % nn[:, 1],
                                       pos // (nn[:, 0] * nn[:, 1])], 1)
        x = o + (u.double() + 0.5 + jit) * hs
        g = src + a0
        lam = torch.einsum("pij,pj->pi", Minv[g], x - p0[g])
        del x, g
        inside = (lam >= 0).all(1) & (lam.sum(1) <= 1)
        u, src = u[inside], src[inside]
        vox = u // sub
        ok = ((vox >= 0) & (vox < torch.tensor([nx, ny, nz], device=dev))).all(1)
        vox, src = vox[ok], src[ok]
        flat = (vox[:, 2] * ny + vox[:, 1]) * nx + vox[:, 0]
        key = flat * chunk + src                       # unique (voxel, tet) in this chunk
        kk, c = torch.unique(key, return_counts=True)
        rows.append(kk // chunk)
        cols.append(kk % chunk + a0)
        cnts.append(c)
    vals = torch.cat(cnts).double() / sub ** 3
    return torch.cat(rows), torch.cat(cols), vals


def tet_adjacency(T: Tensor) -> Tensor:
    """Pairs ``[P, 2]`` of tetrahedra sharing a triangle."""
    nt = T.shape[0]
    tri = torch.cat([T[:, list(f)] for f in _FACES]).sort(dim=1).values
    owner = torch.arange(nt, device=T.device).repeat(4)
    _, inv, cnt = torch.unique(tri, dim=0, return_inverse=True, return_counts=True)
    order = torch.argsort(inv, stable=True)
    inv_s, own_s = inv[order], owner[order]
    pair = (inv_s[1:] == inv_s[:-1])
    return torch.stack([own_s[:-1][pair], own_s[1:][pair]], 1)


def fit_tet_values(rows: Tensor, cols: Tensor, vals: Tensor, n_tets: int,
                   target_flat: Tensor, ridge: float = 0.05, iters: int = 2000,
                   tol: float = 1e-8, adj: Tensor | None = None,
                   smooth: float = 0.05) -> Tensor:
    """Regularized least squares by Jacobi-preconditioned conjugate gradients:

        min ||A a - T||^2 + ridge ||a - a0||^2 + smooth sum_{(i,j) in adj} (a_i - a_j)^2

    Without ``adj`` this is the plain ridge toward the single best constant ``a0`` (Ngo's
    model), as in :func:`polyhedral_cells.fit_cell_values`. That prior is WRONG for a
    tetrahedralized marching-cubes surface: ``-Y`` keeps the 0.5 mm surface triangles, so
    the surface is lined with tetrahedra far smaller than a voxel that the data barely
    see, and a smoothly varying subject differs from its global mean by +-30% there. The
    neighbour term (a graph Laplacian over shared faces) lets them inherit their
    neighbours' value instead; ``ridge`` then only pins the null space.
    """
    t = target_flat.double()[rows]

    def Av(x):
        return vals * x[cols]

    def AT(y):
        return torch.zeros(n_tets, dtype=torch.float64, device=vals.device).index_add_(
            0, cols, vals * y)

    rowsum = torch.zeros_like(target_flat, dtype=torch.float64).index_add_(0, rows, vals)
    a0 = float((target_flat.double() * rowsum).sum() / rowsum.square().sum())
    nvox = target_flat.numel()

    if adj is not None:
        ai, aj = adj[:, 0], adj[:, 1]
        deg = torch.zeros(n_tets, dtype=torch.float64, device=vals.device)
        deg.index_add_(0, ai, torch.ones_like(ai, dtype=torch.float64))
        deg.index_add_(0, aj, torch.ones_like(aj, dtype=torch.float64))

    def lap(x):
        d = x[ai] - x[aj]
        return torch.zeros_like(x).index_add_(0, ai, d).index_add_(0, aj, -d)

    def AtA(x):
        y = torch.zeros(nvox, dtype=torch.float64, device=vals.device).index_add_(
            0, rows, Av(x))
        out = AT(y[rows]) + ridge * x
        return out + smooth * lap(x) if adj is not None else out

    b = AT(t) + ridge * a0
    # Jacobi preconditioner: diag(A^T A) + ridge. Tetrahedra span four orders of volume,
    # so the unpreconditioned system converges far too slowly -- measured: 300 plain CG
    # iterations left the smooth Shepp-Logan at image error 0.10.
    Dinv = 1.0 / (AT(vals) + ridge + (smooth * deg if adj is not None else 0.0))
    x = Dinv * b                                       # per-tet local least squares
    r = b - AtA(x)
    zr = Dinv * r
    p = zr.clone()
    rz = float(r @ zr)
    b2 = float(b @ b)
    it = 0
    for it in range(1, iters + 1):
        Ap = AtA(p)
        al = rz / float(p @ Ap)
        x += al * p
        r -= al * Ap
        if float(r @ r) < tol * tol * b2:
            break
        zr = Dinv * r
        rzn = float(r @ zr)
        p = zr + (rzn / rz) * p
        rz = rzn
    fit_tet_values.last = {"iters": it, "rel_residual": math.sqrt(float(r @ r) / b2)}
    return x


def tet_image(rows: Tensor, cols: Tensor, vals: Tensor, a: Tensor, nvox: int) -> Tensor:
    """``A a`` as a flat voxel vector (z-slowest)."""
    return torch.zeros(nvox, dtype=torch.float64, device=vals.device).index_add_(
        0, rows, vals * a[cols])
