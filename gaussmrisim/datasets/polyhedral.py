"""Closed triangle meshes as phantoms: loading, checks, and exact partial-volume voxels.

The polyhedral phantom of Ngo et al. (MRM 2016; code ``github.com/trimngo/polyphantom``)
represents each tissue as the interior of a closed surface with a constant intensity,
and has an exact Fourier transform (:mod:`gaussmrisim.fourier.polyhedral_ft`). That
makes it an **independent analytic reference** for this project: a Gaussian mixture
fitted to its voxels can be scored in k-space against the closed form, not against a
raster DFT that carries the voxel grid's own aliases.

The voxelizer gives each voxel the **fraction of its volume inside the mesh** -- the
box-filtered indicator, which is what an MR voxel integrates. It is exact along z and
supersampled ``s x s`` in x and y: every sub-ray parallel to z crosses the surface at
known heights, and the inside length within each voxel is integrated in closed form
from those crossings. A crossing's sign (enter/exit) comes from the outward normal, so
no sort is needed and odd crossing counts cannot corrupt a column.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..conventions import FOV


@dataclass
class Mesh:
    vertices: Tensor            # [V, 3] float64, mm
    faces: Tensor               # [F, 3] long, outward (counter-clockwise seen from outside)
    name: str = ""

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def to(self, device) -> "Mesh":
        return Mesh(self.vertices.to(device), self.faces.to(device), self.name)

    def tri(self) -> Tensor:
        """``[F, 3, 3]`` vertex coordinates per face."""
        return self.vertices[self.faces]

    def signed_volume(self) -> float:
        """Divergence theorem, ``sum_f v0 . (v1 x v2) / 6``. Positive iff outward."""
        t = self.tri()
        return float((t[:, 0] * torch.cross(t[:, 1], t[:, 2], dim=-1)).sum() / 6.0)

    def area(self) -> float:
        t = self.tri()
        return float(torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=-1)
                     .norm(dim=-1).sum() / 2)

    def bytes(self) -> int:
        """Indexed storage: float32 vertices + int32 faces."""
        return int(self.vertices.shape[0]) * 12 + self.n_faces * 12

    def transformed(self, A: Tensor, b: Tensor) -> "Mesh":
        """``r -> A r + b``; flips the winding if ``det A < 0`` so it stays outward."""
        v = self.vertices @ A.to(self.vertices).T + b.to(self.vertices)
        f = self.faces if float(torch.det(A.double())) > 0 else self.faces[:, [0, 2, 1]]
        return Mesh(v, f, self.name)


def load_stl(path: str | Path, name: str | None = None) -> Mesh:
    """Binary STL -> indexed, welded, outward-oriented mesh. Raises if not closed."""
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    if len(b) != 84 + 50 * n:
        raise ValueError(f"{path}: not a binary STL (or truncated)")
    rec = np.frombuffer(b[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                                 ("a", "<u2")]))
    v = rec["v"].reshape(-1, 3)
    # weld exactly-equal float32 coordinates: STL repeats every vertex per triangle
    uniq, inv = np.unique(v, axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)
    # drop degenerate triangles (two equal corners): zero area, no contribution
    good = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & \
           (faces[:, 0] != faces[:, 2])
    mesh = Mesh(torch.from_numpy(uniq.astype(np.float64)),
                torch.from_numpy(faces[good].astype(np.int64)),
                name or Path(path).stem)
    check_closed(mesh)
    if mesh.signed_volume() < 0:
        mesh = Mesh(mesh.vertices, mesh.faces[:, [0, 2, 1]], mesh.name)
    return mesh


def from_voxels(frac: Tensor, fov: FOV, level: float = 0.5, name: str = "",
                step: int = 1) -> Mesh:
    """Marching cubes on a partial-volume map ``[nz, ny, nx]`` -> closed outward mesh.

    This is what a polyhedral phantom built FROM DATA looks like: the surface is
    extracted from the same voxels a Gaussian fit sees, not borrowed from a ground truth.

    Two details make it closed, both measured: (i) the level is offset by 3e-6, because
    a partial-volume map has many voxels at EXACTLY 0.5 (1755-3942 per surface here),
    and isosurface vertices landing on voxel centres are emitted inconsistently by
    neighbouring cubes -- 9-204 cracked edges per surface otherwise; (ii) degenerate
    triangles are kept through the closedness check (dropping them first opens cracks)
    and removed afterwards, since a zero-area face contributes nothing to any surface
    integral. Welding equal coordinates instead was tried and made it worse.

    ``step > 1`` marches every ``step``-th voxel (skimage ``step_size``): a coarser
    surface with ~``step^2`` fewer triangles, for parameter-matched comparisons.
    """
    import math

    import numpy as np
    from skimage.measure import marching_cubes

    vox = fov.voxel_mm
    v, f, _, _ = marching_cubes(frac.double().cpu().numpy(), level=level + 1e-6 * math.pi,
                                spacing=(vox[2], vox[1], vox[0]), allow_degenerate=True,
                                method="lewiner", step_size=int(step))
    corner = np.array([c - 0.5 * F for c, F in zip(fov.center_mm, fov.fov_mm)])
    xyz = v[:, ::-1] + corner + 0.5 * np.array(vox)       # (z,y,x) -> mm, voxel centres
    mesh = Mesh(torch.from_numpy(np.ascontiguousarray(xyz)),
                torch.from_numpy(f.astype(np.int64)), name)
    info = check_closed(mesh)
    t = mesh.tri()
    area = torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=-1).norm(dim=-1)
    mesh = Mesh(mesh.vertices, mesh.faces[area > 1e-12], name)
    if mesh.signed_volume() < 0:
        mesh = Mesh(mesh.vertices, mesh.faces[:, [0, 2, 1]], name)
    mesh.info = info
    return mesh


def check_closed(mesh: Mesh) -> dict:
    """Every undirected edge in exactly two faces, every directed edge once (consistent
    winding). Both are needed: the FT and the voxelizer are only exact for a closed,
    consistently oriented surface. Raises otherwise."""
    f = mesh.faces
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])           # directed
    nv = int(mesh.vertices.shape[0])
    d_key = e[:, 0] * nv + e[:, 1]
    u_key = torch.minimum(e[:, 0], e[:, 1]) * nv + torch.maximum(e[:, 0], e[:, 1])
    _, u_cnt = torch.unique(u_key, return_counts=True)
    _, d_cnt = torch.unique(d_key, return_counts=True)
    bad_u, bad_d = int((u_cnt != 2).sum()), int((d_cnt != 1).sum())
    if bad_u or bad_d:
        raise ValueError(f"{mesh.name}: not a closed consistently-oriented surface "
                         f"({bad_u} edges not shared by exactly 2 faces, "
                         f"{bad_d} repeated directed edges)")
    return {"edges": int(u_cnt.numel()), "euler": nv - int(u_cnt.numel()) + mesh.n_faces}


@torch.no_grad()
def voxelize(mesh: Mesh, fov: FOV, supersample: int = 4, device="cuda",
             chunk: int = 200_000, max_gib: float = 2.0) -> Tensor:
    """Fraction of each voxel inside the mesh -> ``[nz, ny, nx]`` float32.

    Exact along z; ``supersample**2`` sub-rays per voxel in x-y, placed at the centres of
    an ``s x s`` sub-grid with a tiny irrational offset so no ray passes exactly through
    a mesh edge or vertex (which would count a crossing twice or not at all).

    Sub-ray rows are processed in y-slabs so the two fp64 difference arrays stay under
    ``max_gib``; every ray is independent, so slabbing does not change the result.
    """
    dev = torch.device(device)
    nx, ny, nz = fov.matrix
    dx, dy, dz = fov.voxel_mm
    o = [c - 0.5 * f for c, f in zip(fov.center_mm, fov.fov_mm)]       # FOV corner
    s = int(supersample)
    jit = (1.0 / (s * np.pi * 1e3), 1.0 / (s * np.e * 1e3))           # sub-ray units
    t = mesh.to(dev).tri()                                               # [F, 3, 3]
    nrm = torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=-1)
    keep = nrm[:, 2].abs() > 0                                           # vertical faces
    t, nrm = t[keep], nrm[keep]                                          # carry no z-ray
    sgn = -torch.sign(nrm[:, 2])          # outward normal pointing -z => ray enters (+1)

    # sub-ray coordinates in units of sub-voxels: x_sub = (x - o_x) / (dx / s) - 0.5
    sx = (t[..., 0] - o[0]) / (dx / s) - 0.5 - jit[0]
    sy = (t[..., 1] - o[1]) / (dy / s) - 0.5 - jit[1]
    NX, NY = nx * s, ny * s
    rows = max(s, int(max_gib * 2**30 / (3 * 8 * NX * (nz + 2))) // s * s)
    out = []
    for ys in range(0, NY, rows):
        out.append(_voxelize_slab(sx, sy, t, sgn, o, dz, NX, ys, min(NY, ys + rows), nz,
                                  s, nx, chunk, dev))
    return torch.cat(out, dim=0).permute(2, 0, 1).contiguous().float()   # [nz, ny, nx]


def _voxelize_slab(sx, sy, t, sgn, o, dz, NX, ys, ye, nz, s, nx, chunk, dev) -> Tensor:
    """Sub-ray rows ``[ys, ye)`` (a multiple of ``s``) -> ``[ny_slab, nx, nz]`` fractions."""
    NY = ye - ys
    full = torch.zeros(NY * NX, nz + 2, device=dev, dtype=torch.float64)
    part = torch.zeros(NY * NX, nz + 2, device=dev, dtype=torch.float64)
    for a in range(0, t.shape[0], chunk):
        X, Y = sx[a:a + chunk], sy[a:a + chunk]
        Z = (t[a:a + chunk, :, 2] - o[2]) / dz                           # voxel units
        S = sgn[a:a + chunk]
        x0 = torch.ceil(X.min(1).values).clamp(0, NX).long()
        x1 = torch.floor(X.max(1).values).clamp(-1, NX - 1).long()
        y0 = torch.ceil(Y.min(1).values).clamp(ys, ye).long()
        y1 = torch.floor(Y.max(1).values).clamp(ys - 1, ye - 1).long()
        wx, wy = (x1 - x0 + 1).clamp(min=0), (y1 - y0 + 1).clamp(min=0)
        cnt = wx * wy
        if int(cnt.sum()) == 0:
            continue
        src = torch.repeat_interleave(torch.arange(cnt.numel(), device=dev), cnt)
        pos = torch.arange(int(cnt.sum()), device=dev) - torch.repeat_interleave(
            torch.cumsum(cnt, 0) - cnt, cnt)
        px = x0[src] + pos % wx[src]
        py = y0[src] + pos // wx[src]
        # barycentric test in x-y; half-open on the jittered grid
        ax, ay = X[src, 0], Y[src, 0]
        bx, by = X[src, 1], Y[src, 1]
        cx, cy = X[src, 2], Y[src, 2]
        qx, qy = px.double(), py.double()
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        l1 = ((by - cy) * (qx - cx) + (cx - bx) * (qy - cy)) / det
        l2 = ((cy - ay) * (qx - cx) + (ax - cx) * (qy - cy)) / det
        l3 = 1.0 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        src, px, py = src[inside], px[inside], py[inside]
        l1, l2, l3 = l1[inside], l2[inside], l3[inside]
        zc = l1 * Z[src, 0] + l2 * Z[src, 1] + l3 * Z[src, 2] - 0.5     # voxel-centre units
        # crossing at height p (voxel j covers [j, j+1) in these units, shifted by the
        # half voxel): indicator after p fills voxel floor(p) by (floor(p)+1-p) and all
        # later voxels fully. Clamp to the column; outside crossings saturate.
        p = (zc + 0.5).clamp(0.0, float(nz))
        j = torch.floor(p).long().clamp(max=nz)
        ray = (py - ys) * NX + px
        sg = S[src].double()
        part.index_put_((ray, j), sg * (j.double() + 1.0 - p), accumulate=True)
        full.index_put_((ray, j + 1), sg, accumulate=True)
    frac = torch.cumsum(full, dim=1) + part                              # [rays, nz+2]
    del full, part
    frac = frac[:, :nz].clamp(0.0, 1.0)
    return frac.reshape(NY // s, s, nx, s, nz).mean(dim=(1, 3))          # [ny_s, nx, nz]
