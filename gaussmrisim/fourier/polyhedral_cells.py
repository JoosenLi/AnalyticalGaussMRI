"""A closed-surface tissue cut into grid cells, one constant per cell, with an exact FT.

This is the strongest piecewise-constant polyhedral phantom one can build from data
without a tetrahedral mesher: keep the tissue's true surface ``Omega``, cut its interior
with an axis-aligned grid of cubes ``C_c`` (side ``h``), and give each cell
``Omega ∩ C_c`` its own constant ``a_c``. As ``h`` shrinks it tends to the voxel grid.

``rho = sum_c a_c 1[Omega ∩ C_c]`` is a sum of weighted planar polygons, so by the
divergence theorem its FT needs only oriented boundary EDGES (see the edge kernel in
``cuda/polyhedral_ft_kernel.cu``). The boundary of every cell is

  * surface pieces  -- each triangle of ``Omega`` clipped to the cell (convex polygons),
                       weight ``a_c``, normal = the triangle's outward normal;
  * caps            -- ``Omega ∩ (grid plane) ∩ (square)``; the two cells sharing the
                       square see it with opposite normals, so one region with weight
                       ``a_below - a_above`` and normal ``+e_axis`` carries both. Its
                       boundary is (a) the surface/plane intersection segments inside
                       the square and (b) the parts of the square's own edges that lie
                       inside ``Omega`` -- grid lines, found by ray casting.

No polygon is ever assembled: Green's theorem in the plane only needs the oriented
segments, so caps are just more edges. Checked in ``tests/verify_polyhedral.py``:
all-ones weights must reproduce the uncut mesh exactly, a box cut into random-valued
cells must equal the sum of the sub-box sincs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..datasets.polyhedral import Mesh
from .polyhedral_ft import _ext

EREC = 14


@dataclass
class Grid:
    origin: tuple          # mm, corner of cell (0, 0, 0)
    h: float               # cell side, mm
    shape: tuple           # cells per axis (x, y, z)

    def n(self) -> int:
        return int(self.shape[0] * self.shape[1] * self.shape[2])

    def flat(self, i: Tensor) -> Tensor:
        """``[.., 3]`` (ix, iy, iz) -> flat cell index, x fastest."""
        return (i[..., 2] * self.shape[1] + i[..., 1]) * self.shape[0] + i[..., 0]


# ----------------------------------------------------------------------------- records
def _edge_records(a: Tensor, b: Tensor, N: Tensor, o: Tensor, w: Tensor) -> Tensor:
    """Oriented edges a->b on planes with normal N, region point o, weight w -> [E, 14]."""
    Lt = b - a
    Ln = torch.cross(Lt, N, dim=-1)                       # L (t x N) = L n
    c = 0.5 * (a + b)
    s = (N * torch.cross(a - o, b - o, dim=-1)).sum(-1)
    return torch.cat([N, Ln, Lt, c, s[:, None], w[:, None]], dim=1)


def mesh_edges(mesh: Mesh, value: float = 1.0) -> Tensor:
    """The uncut mesh in edge form (for checking against the face kernel)."""
    t = mesh.tri()
    N = torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=-1)
    N = N / N.norm(dim=-1, keepdim=True)
    recs = [_edge_records(t[:, e], t[:, (e + 1) % 3], N, t[:, 0],
                          torch.full_like(N[:, 0], value)) for e in range(3)]
    return torch.cat(recs)


def edge_ft(rec: Tensor, k: Tensor, precision: str = "double",
            k_chunk: int = 32768) -> Tensor:
    """``precision``: "double" (reference) or "f32" (all float32, compensated phase,
    float64 only for face-k pairs within ~6 degrees of parallel)."""
    ext = _ext()
    if precision == "f32":
        f, r, kk = ext.forward_edges_f32, rec.float().contiguous(), k.float()
    elif precision == "double":
        f, r, kk = ext.forward_edges_double, rec.contiguous(), k.double()
    else:
        raise ValueError(precision)
    kk = kk.to(rec.device).contiguous()
    out = torch.empty(kk.shape[0], dtype=torch.complex128, device=rec.device)
    for a in range(0, kk.shape[0], k_chunk):
        out[a:a + k_chunk] = torch.view_as_complex(f(r, kk[a:a + k_chunk]))
    return out


# ----------------------------------------------------------------------- polygon clip
def _split(V: Tensor, n: Tensor, axis: int, q: Tensor):
    """Cut convex polygons (padded ``V [P, K, 3]``, ``n [P]``) by ``x_axis = q [P]``.
    Returns (below, above), each as (V, n), Sutherland-Hodgman."""
    P, K, _ = V.shape
    dev = V.device
    idx = torch.arange(K, device=dev)[None, :].expand(P, K)
    valid = idx < n[:, None]
    nxt = torch.where(idx + 1 < n[:, None], idx + 1, torch.zeros_like(idx))
    A = V
    B = torch.gather(V, 1, nxt[..., None].expand(P, K, 3))
    da = A[..., axis] - q[:, None]
    db = B[..., axis] - q[:, None]
    cross = valid & (((da < 0) & (db > 0)) | ((da > 0) & (db < 0)))
    t = torch.where(cross, da / torch.where(cross, da - db, torch.ones_like(da)),
                    torch.zeros_like(da))
    X = A + t[..., None] * (B - A)
    X[..., axis] = torch.where(cross, q[:, None].expand(P, K), X[..., axis])
    outs = []
    for keep in (valid & (da <= 0), valid & (da >= 0)):
        emit = torch.stack([keep, cross], dim=2).reshape(P, 2 * K)        # A then X
        pts = torch.stack([A, X], dim=2).reshape(P, 2 * K, 3)
        pos = torch.cumsum(emit.long(), dim=1) - 1
        cnt = emit.sum(1)
        Ko = K + 2
        out = torch.zeros(P, Ko, 3, dtype=V.dtype, device=dev)
        pi = torch.arange(P, device=dev)[:, None].expand(P, 2 * K)
        out[pi[emit], pos[emit]] = pts[emit]
        outs.append((out, cnt))
    return outs


def _poly_area2(V: Tensor, n: Tensor) -> Tensor:
    P, K, _ = V.shape
    idx = torch.arange(K, device=V.device)[None, :].expand(P, K)
    nxt = torch.where(idx + 1 < n[:, None], idx + 1, torch.zeros_like(idx))
    valid = (idx < n[:, None])[..., None]
    a = (V - V[:, :1]) * valid
    b = torch.gather(V, 1, nxt[..., None].expand(P, K, 3)) - V[:, :1]
    return torch.cross(a, b * valid, dim=-1).sum(1).norm(dim=-1)


def clip_to_cells(mesh: Mesh, grid: Grid):
    """Triangles of ``mesh`` cut by every grid plane -> (V [P, K, 3], n [P], N [P, 3],
    cell [P]). Each output polygon lies in exactly one cell."""
    t = mesh.tri()
    N = torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=-1)
    N = N / N.norm(dim=-1, keepdim=True)
    V = torch.zeros(t.shape[0], 3, 3, dtype=t.dtype, device=t.device)
    V[:] = t
    n = torch.full((t.shape[0],), 3, dtype=torch.long, device=t.device)
    o = torch.tensor(grid.origin, dtype=t.dtype, device=t.device)
    h = float(grid.h)
    eps = 1e-9 * h
    done_V, done_n, done_N = [], [], []
    for axis in range(3):
        nxt_V, nxt_n, nxt_N = [], [], []
        work = [(V, n, N)]
        while work:
            V, n, N = work.pop()
            if V.shape[0] == 0:
                continue
            K = V.shape[1]
            valid = (torch.arange(K, device=V.device)[None, :] < n[:, None])
            x = V[..., axis]
            lo = torch.where(valid, x, torch.full_like(x, float("inf"))).min(1).values
            hi = torch.where(valid, x, torch.full_like(x, -float("inf"))).max(1).values
            q = o[axis] + (torch.floor((lo - o[axis]) / h) + 1) * h
            crossing = (q > lo + eps) & (q < hi - eps)
            keep = ~crossing
            nxt_V.append(V[keep]); nxt_n.append(n[keep]); nxt_N.append(N[keep])
            if bool(crossing.any()):
                (Vb, nb), (Va, na) = _split(V[crossing], n[crossing], axis, q[crossing])
                Nc = N[crossing]
                nxt_V.append(Vb); nxt_n.append(nb); nxt_N.append(Nc)
                work.append((Va, na, Nc))                      # may cross the next plane
        K = max(v.shape[1] for v in nxt_V)
        V = torch.cat([torch.nn.functional.pad(v, (0, 0, 0, K - v.shape[1])) for v in nxt_V])
        n, N = torch.cat(nxt_n), torch.cat(nxt_N)
        good = (n >= 3) & (_poly_area2(V, n) > 1e-12)
        V, n, N = V[good], n[good], N[good]
    K = V.shape[1]
    valid = (torch.arange(K, device=V.device)[None, :] < n[:, None])[..., None]
    cen = (V * valid).sum(1) / n[:, None]
    ci = torch.floor((cen - o) / h).long()
    return V, n, N, grid.flat(ci)


# --------------------------------------------------------------------------- the phantom
def cell_edges(mesh: Mesh, grid: Grid, a: Tensor) -> Tensor:
    """Edge records [E, 14] of ``sum_c a_c 1[Omega ∩ C_c]``. ``a``: [grid.n()]."""
    dev = mesh.vertices.device
    dt = torch.float64
    a = a.to(dev, dt)
    o = torch.tensor(grid.origin, dtype=dt, device=dev)
    h = float(grid.h)
    S = grid.shape
    recs = []

    # 1. surface pieces
    V, n, N, cell = clip_to_cells(mesh, grid)
    K = V.shape[1]
    for i in range(K):
        live = n > i
        j = torch.where(i + 1 < n, torch.full_like(n, i + 1), torch.zeros_like(n))
        A_ = V[live, i]
        B_ = V[live].gather(1, j[live][:, None, None].expand(-1, 1, 3))[:, 0]
        recs.append(_edge_records(A_, B_, N[live], V[live, 0], a[cell[live]]))

    t = mesh.tri()
    Nt = torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=-1)
    Nt = Nt / Nt.norm(dim=-1, keepdim=True)
    E = torch.eye(3, dtype=dt, device=dev)

    def cell_value(ix, iy, iz):
        return a[grid.flat(torch.stack([ix, iy, iz], -1))]

    # 2. caps: per axis, per interior plane j (between cells j-1 and j)
    for ax in range(3):
        b1, b2 = [x for x in range(3) if x != ax]
        x = t[..., ax]
        jlo = torch.floor((x.min(1).values - o[ax]) / h).long() + 1
        jhi = torch.floor((x.max(1).values - o[ax]) / h).long()
        cnt = (jhi - jlo + 1).clamp(min=0)
        if int(cnt.sum()) == 0:
            continue
        src = torch.repeat_interleave(torch.arange(t.shape[0], device=dev), cnt)
        jj = jlo[src] + torch.arange(int(cnt.sum()), device=dev) - torch.repeat_interleave(
            torch.cumsum(cnt, 0) - cnt, cnt)
        keep = (jj >= 1) & (jj <= S[ax] - 1)
        src, jj = src[keep], jj[keep]
        q = o[ax] + jj.to(dt) * h
        T = t[src]
        d = T[..., ax] - q[:, None]                                # [P, 3]
        s = d >= 0
        pts = []
        for e in range(3):
            u, v = e, (e + 1) % 3
            c_ = s[:, u] != s[:, v]
            tt = torch.where(c_, d[:, u] / torch.where(c_, d[:, u] - d[:, v],
                                                        torch.ones_like(d[:, u])), 0 * d[:, u])
            p = T[:, u] + tt[:, None] * (T[:, v] - T[:, u])
            pts.append((c_, p))
        has = torch.stack([c for c, _ in pts], 1)                 # exactly two True
        ok = has.sum(1) == 2
        P1 = torch.where(has[:, 0:1], pts[0][1], pts[1][1])
        P2 = torch.where(has[:, 2:3], pts[2][1], pts[1][1])
        P1, P2 = P1[ok], P2[ok]
        src, jj, q = src[ok], jj[ok], q[ok]
        P1[:, ax] = q
        P2[:, ax] = q
        Ncap = E[ax].expand(P1.shape[0], 3)
        m = Nt[src] - (Nt[src][:, ax:ax + 1]) * Ncap               # in-plane outward
        tdir = torch.cross(Ncap, m, dim=-1)
        flip = ((P2 - P1) * tdir).sum(-1) < 0
        A_, B_ = torch.where(flip[:, None], P2, P1), torch.where(flip[:, None], P1, P2)
        # split each segment by the grid lines of the two other axes
        segs = [(A_, B_, jj)]
        for bx in (b1, b2):
            out = []
            while segs:
                A_, B_, jj = segs.pop()
                if A_.shape[0] == 0:
                    continue
                lo = torch.minimum(A_[:, bx], B_[:, bx])
                hi = torch.maximum(A_[:, bx], B_[:, bx])
                qq = o[bx] + (torch.floor((lo - o[bx]) / h) + 1) * h
                cr = (qq > lo + 1e-9 * h) & (qq < hi - 1e-9 * h)
                out.append((A_[~cr], B_[~cr], jj[~cr]))
                if bool(cr.any()):
                    A2, B2, q2 = A_[cr], B_[cr], qq[cr]
                    tt = (q2 - A2[:, bx]) / (B2[:, bx] - A2[:, bx])
                    M_ = A2 + tt[:, None] * (B2 - A2)
                    M_[:, bx] = q2
                    # either half may still cross a further line, depending on the
                    # segment's direction -- re-queue both, not just the upper one
                    segs.append((A2, M_, jj[cr]))
                    segs.append((M_, B2, jj[cr]))
            segs = out
        A_ = torch.cat([s_[0] for s_ in segs]); B_ = torch.cat([s_[1] for s_ in segs])
        jj = torch.cat([s_[2] for s_ in segs])
        L = (B_ - A_).norm(dim=-1)
        g = L > 1e-12
        A_, B_, jj = A_[g], B_[g], jj[g]
        mid = 0.5 * (A_ + B_)
        idx = torch.floor((mid - o) / h).long()
        lo_i, hi_i = idx.clone(), idx.clone()
        lo_i[:, ax] = jj - 1
        hi_i[:, ax] = jj
        w = a[grid.flat(lo_i)] - a[grid.flat(hi_i)]
        po = o.expand(A_.shape[0], 3).clone()
        po[:, ax] = o[ax] + jj.to(dt) * h                          # a point of the plane
        recs.append(_edge_records(A_, B_, E[ax].expand(A_.shape[0], 3), po, w))

    # 3. grid lines parallel to axis c, at interior planes of the other two axes
    for c in range(3):
        p1, p2 = [x for x in range(3) if x != c]
        u0, v0 = t[..., p1], t[..., p2]
        ilo = torch.ceil((u0.min(1).values - o[p1]) / h).long().clamp(min=1)
        ihi = torch.floor((u0.max(1).values - o[p1]) / h).long().clamp(max=S[p1] - 1)
        jlo = torch.ceil((v0.min(1).values - o[p2]) / h).long().clamp(min=1)
        jhi = torch.floor((v0.max(1).values - o[p2]) / h).long().clamp(max=S[p2] - 1)
        wi, wj = (ihi - ilo + 1).clamp(min=0), (jhi - jlo + 1).clamp(min=0)
        cnt = wi * wj
        if int(cnt.sum()) == 0:
            continue
        src = torch.repeat_interleave(torch.arange(t.shape[0], device=dev), cnt)
        pos = torch.arange(int(cnt.sum()), device=dev) - torch.repeat_interleave(
            torch.cumsum(cnt, 0) - cnt, cnt)
        li = ilo[src] + pos % wi[src]
        lj = jlo[src] + pos // wi[src]
        qx = o[p1] + li.to(dt) * h
        qy = o[p2] + lj.to(dt) * h
        T = t[src]
        ax_, ay_ = T[:, 0, p1], T[:, 0, p2]
        bx_, by_ = T[:, 1, p1], T[:, 1, p2]
        cx_, cy_ = T[:, 2, p1], T[:, 2, p2]
        det = (by_ - cy_) * (ax_ - cx_) + (cx_ - bx_) * (ay_ - cy_)
        l1 = ((by_ - cy_) * (qx - cx_) + (cx_ - bx_) * (qy - cy_)) / det
        l2 = ((cy_ - ay_) * (qx - cx_) + (ax_ - cx_) * (qy - cy_)) / det
        l3 = 1 - l1 - l2
        ins = (l1 >= 0) & (l2 >= 0) & (l3 >= 0) & (det != 0)
        zc = (l1 * T[:, 0, c] + l2 * T[:, 1, c] + l3 * T[:, 2, c])[ins]
        sg = (-torch.sign(Nt[src[ins], c])).long()                # +1 entering along +c
        line = (lj[ins] * S[p1] + li[ins])
        # every grid line must cross a closed surface an even number of times; an odd
        # count means a line went exactly through a shared edge (counted twice) and the
        # inside/outside pairing below would silently be wrong -- refuse instead
        _, per_line = torch.unique(line, return_counts=True)
        if bool((per_line % 2 == 1).any()):
            raise RuntimeError(
                f"{int((per_line % 2 == 1).sum())} grid lines parallel to axis {c} cross "
                "the surface an odd number of times: a line passes exactly through a mesh "
                "edge. Offset the grid origin by DIFFERENT irrational amounts per axis "
                "(an equal offset slides along 45-degree edges, e.g. marching cubes).")
        o1 = torch.argsort(zc, stable=True)
        o2 = torch.argsort(line[o1], stable=True)
        order = o1[o2]
        line, zc, sg = line[order], zc[order], sg[order]
        # running count per line: inside after a crossing iff the count is 1
        first = torch.ones_like(line, dtype=torch.bool)
        first[1:] = line[1:] != line[:-1]
        cs = torch.cumsum(sg, 0)
        base = torch.zeros_like(cs)
        seg_start = torch.nonzero(first).flatten()
        base_vals = torch.cat([torch.zeros(1, dtype=cs.dtype, device=dev), cs[seg_start[1:] - 1]])
        base = base_vals[torch.cumsum(first.long(), 0) - 1]
        level = cs - base
        startI = (level[:-1] == 1) & (line[1:] == line[:-1])
        z0, z1 = zc[:-1][startI], zc[1:][startI]
        L_ = line[:-1][startI]
        li_, lj_ = L_ % S[p1], L_ // S[p1]
        # split intervals at the grid planes of axis c
        k0 = torch.floor((z0 - o[c]) / h).long()
        k1 = torch.floor((z1 - o[c]) / h).long()
        cntk = (k1 - k0 + 1).clamp(min=1)
        srck = torch.repeat_interleave(torch.arange(z0.shape[0], device=dev), cntk)
        kk_ = k0[srck] + torch.arange(int(cntk.sum()), device=dev) - torch.repeat_interleave(
            torch.cumsum(cntk, 0) - cntk, cntk)
        za = torch.maximum(z0[srck], o[c] + kk_.to(dt) * h)
        zb = torch.minimum(z1[srck], o[c] + (kk_ + 1).to(dt) * h)
        g = zb > za + 1e-12
        za, zb, kk_, li_, lj_ = za[g], zb[g], kk_[g], li_[srck][g], lj_[srck][g]
        Pa = torch.zeros(za.shape[0], 3, dtype=dt, device=dev)
        Pa[:, p1] = o[p1] + li_.to(dt) * h
        Pa[:, p2] = o[p2] + lj_.to(dt) * h
        Pb = Pa.clone()
        Pa[:, c], Pb[:, c] = za, zb

        def idx3(i1, i2, ic):
            out = torch.zeros(i1.shape[0], 3, dtype=torch.long, device=dev)
            out[:, p1], out[:, p2], out[:, c] = i1, i2, ic
            return out

        # on plane p1 (normal e_p1), between squares p2-index lj-1 | lj
        # D(i2, ic) = a[p1: li-1] - a[p1: li]; edge weight D(lj-1) - D(lj), n = +e_p2
        def D_p1(i2):
            return a[grid.flat(idx3(li_ - 1, i2, kk_))] - a[grid.flat(idx3(li_, i2, kk_))]
        w1 = D_p1(lj_ - 1) - D_p1(lj_)
        N1 = E[p1].expand(za.shape[0], 3)
        tdir = torch.cross(N1, E[p2].expand_as(N1), dim=-1)          # n = +e_p2
        f1 = (tdir[:, c] < 0)
        A1, B1 = torch.where(f1[:, None], Pb, Pa), torch.where(f1[:, None], Pa, Pb)
        po1 = o.expand(za.shape[0], 3).clone(); po1[:, p1] = Pa[:, p1]
        recs.append(_edge_records(A1, B1, N1, po1, w1))
        # on plane p2 (normal e_p2), between squares p1-index li-1 | li, n = +e_p1
        def D_p2(i1):
            return a[grid.flat(idx3(i1, lj_ - 1, kk_))] - a[grid.flat(idx3(i1, lj_, kk_))]
        w2 = D_p2(li_ - 1) - D_p2(li_)
        N2 = E[p2].expand(za.shape[0], 3)
        tdir2 = torch.cross(N2, E[p1].expand_as(N2), dim=-1)
        f2 = (tdir2[:, c] < 0)
        A2, B2 = torch.where(f2[:, None], Pb, Pa), torch.where(f2[:, None], Pa, Pb)
        po2 = o.expand(za.shape[0], 3).clone(); po2[:, p2] = Pa[:, p2]
        recs.append(_edge_records(A2, B2, N2, po2, w2))

    rec = torch.cat(recs)
    return rec[rec[:, 13] != 0].contiguous()


def fit_cell_values(frac: Tensor, target: Tensor, grid_vox: int,
                    ridge: float = 0.05) -> Tensor:
    """Ridge least-squares constant per cell from voxels -> ``[cells_z, cells_y, cells_x]``.

    The grid is aligned with the voxels (``grid_vox`` per cell side), so each voxel lies
    in one cell and the model there is ``a_c * frac``:

        a_c = (sum f T + ridge * a0) / (sum f^2 + ridge),   a0 = sum f T / sum f^2 (tissue)

    The ridge is needed as soon as the mesh is NOT the one that generated the data
    (e.g. marching cubes): a cell the extracted surface only grazes has ``sum f^2 ~ 0``
    but real target signal, and plain least squares gave values up to 8.8e4 there
    (1-2% of cells; true mesh: all in [0.7, 1.3]). The image ``a_c f`` hides it, the
    continuous phantom does not -- its k-space error reached 2.6e2. ``ridge = 0.05`` is
    5% of one full voxel of evidence: under 1% bias for a 1 mm cell fully inside tissue
    (sum f^2 = 8), and a sliver cell falls back to the tissue constant."""
    m = int(grid_vox)
    nz, ny, nx = frac.shape
    pz, py, px = (-nz) % m, (-ny) % m, (-nx) % m
    F = torch.nn.functional.pad(frac.double(), (0, px, 0, py, 0, pz))
    Tt = torch.nn.functional.pad(target.double(), (0, px, 0, py, 0, pz))
    sh = (F.shape[0] // m, m, F.shape[1] // m, m, F.shape[2] // m, m)
    num = (F * Tt).reshape(sh).sum((1, 3, 5))
    den = (F * F).reshape(sh).sum((1, 3, 5))
    a0 = float(num.sum() / den.sum().clamp_min(1e-30))
    a = (num + ridge * a0) / (den + ridge)
    return torch.where(den > 0, a, torch.zeros_like(a))
