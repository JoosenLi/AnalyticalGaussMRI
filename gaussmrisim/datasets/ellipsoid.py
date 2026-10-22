"""Analytic ellipsoid phantoms: an independent reference with a closed-form transform.

Validation rule 1 forbids checking Gaussian machinery against a Gaussian-generated
object. Ellipsoids are the cleanest independent alternative: uniform density inside a
rotated ellipsoid, image and k-space both exact (Koay, Sarlls & Ozarslan, MRM 2007).

    rho(r)     = sum_i a_i * 1[ |S_i^{-1} R_i^T (r - mu_i)| <= 1 ]
    rho_hat(k) = sum_i a_i V_i * 3 j1(2 pi q_i) / (2 pi q_i) * exp(-i 2 pi k . mu_i),
                 q_i = | S_i R_i^T k |,   V_i = 4/3 pi abc

This is what makes it possible to ask the question a raster reference cannot answer:
fit a mixture to the *rasterized* phantom, then compare its analytic k-space against
the *analytic* k-space of the object the raster came from. The difference between that
and the comparison against the raster's own DFT is precisely the reference error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..conventions import cycles_mod_one
from ..representation.gaussian3d import rotation_from_quaternion


@dataclass
class EllipsoidPhantom:
    centre: Tensor        # [E, 3] mm
    semi: Tensor          # [E, 3] mm
    quat: Tensor          # [E, 4]
    value: Tensor         # [E] additive density inside

    def to(self, device):
        return EllipsoidPhantom(self.centre.to(device), self.semi.to(device),
                                self.quat.to(device), self.value.to(device))

    def image(self, points: Tensor, chunk: int = 0) -> Tensor:
        R = rotation_from_quaternion(self.quat)
        chunk = chunk or max(1, int(4e7 // max(self.centre.shape[0], 1)))
        out = torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)
        for a in range(0, points.shape[0], chunk):
            d = points[a: a + chunk, None, :] - self.centre[None]
            loc = torch.einsum("pej,ejk->pek", d, R) / self.semi[None]
            out[a: a + chunk] = ((loc.square().sum(-1) <= 1.0).to(points.dtype)
                                 @ self.value)
        return out

    def kspace(self, k: Tensor, chunk: int = 0) -> Tensor:
        """Exact continuous FT at arbitrary ``k`` (cycles/mm) -> complex ``[S]``."""
        R = rotation_from_quaternion(self.quat)
        vol = 4.0 / 3.0 * math.pi * self.semi.prod(-1) * self.value
        chunk = chunk or max(1, int(4e7 // max(self.centre.shape[0], 1)))
        out = torch.empty(k.shape[0], device=k.device, dtype=torch.complex64)
        for a in range(0, k.shape[0], chunk):
            kk = k[a: a + chunk]
            q = 2 * math.pi * (torch.einsum("sj,ejk->sek", kk, R) * self.semi[None]
                               ).norm(dim=-1)                      # [s, E]
            # 3 j1(q)/q, with the removable singularity at q -> 0 taken as 1
            small = q < 1e-6
            qs = torch.where(small, torch.ones_like(q), q)
            f = 3.0 * (torch.sin(qs) - qs * torch.cos(qs)) / qs.pow(3)
            f = torch.where(small, torch.ones_like(f), f)
            ph = 2 * math.pi * cycles_mod_one(kk, self.centre)
            out[a: a + chunk] = (torch.polar(f, -ph).to(torch.complex64)
                                 @ vol.to(torch.complex64))
        return out


def shepp_logan_3d(fov_mm: float = 200.0, device="cpu") -> EllipsoidPhantom:
    """The 3D Shepp-Logan head, scaled to a physical FOV (Koay et al., MRM 2007).

    Parameters are in the usual normalized ``[-1, 1]`` convention and are scaled here
    by ``fov_mm / 2`` -- the one place the normalized convention is allowed to appear,
    at the boundary, converted once.
    """
    #   value,    a,     b,     c,      x0,     y0,     z0,   phi(deg about z)
    P = [(2.00, 0.6900, 0.9200, 0.810,  0.000,  0.000,  0.00,   0.0),
         (-0.8, 0.6624, 0.8740, 0.780,  0.000, -0.018,  0.00,   0.0),
         (-0.2, 0.1100, 0.3100, 0.220,  0.220,  0.000,  0.00, -18.0),
         (-0.2, 0.1600, 0.4100, 0.280, -0.220,  0.000,  0.00,  18.0),
         (0.10, 0.2100, 0.2500, 0.410,  0.000,  0.350, -0.15,   0.0),
         (0.10, 0.0460, 0.0460, 0.050,  0.000,  0.100,  0.25,   0.0),
         (0.10, 0.0460, 0.0460, 0.050,  0.000, -0.100,  0.25,   0.0),
         (0.10, 0.0460, 0.0230, 0.050, -0.080, -0.605,  0.00,   0.0),
         (0.10, 0.0230, 0.0230, 0.020,  0.000, -0.606,  0.00,   0.0),
         (0.10, 0.0230, 0.0460, 0.020,  0.060, -0.605,  0.00,   0.0)]
    s = 0.5 * float(fov_mm)
    val = torch.tensor([p[0] for p in P], dtype=torch.float32)
    semi = torch.tensor([[p[1], p[2], p[3]] for p in P], dtype=torch.float32) * s
    ctr = torch.tensor([[p[4], p[5], p[6]] for p in P], dtype=torch.float32) * s
    ang = torch.tensor([p[7] for p in P], dtype=torch.float32) * math.pi / 180.0
    quat = torch.stack([torch.cos(ang / 2), torch.zeros_like(ang),
                        torch.zeros_like(ang), torch.sin(ang / 2)], dim=-1)
    return EllipsoidPhantom(ctr, semi, quat, val).to(device)


def unit_ft64(ph: EllipsoidPhantom, k: Tensor) -> Tensor:
    """FT of each UNIT-density ellipsoid at ``k`` [S, 3] -> complex128 ``[S, E]``, in fp64.

    ``3 j1(q)/q`` cancels catastrophically as ``q -> 0`` (relative error ~ eps/q^2), so
    below ``q = 1e-2`` its Taylor series ``1 - q^2/10 + q^4/280`` is used; the next
    term is ``q^6/15120 < 1e-16`` there.
    """
    k = k.double()
    R = rotation_from_quaternion(ph.quat.double().to(k.device))
    semi, c = ph.semi.double().to(k.device), ph.centre.double().to(k.device)
    q = 2 * math.pi * (torch.einsum("sj,ejk->sek", k, R) * semi[None]).norm(dim=-1)
    qs = q.clamp_min(1e-2)
    f = torch.where(q < 1e-2, 1 - q * q / 10 + q.pow(4) / 280,
                    3.0 * (torch.sin(qs) - qs * torch.cos(qs)) / qs.pow(3))
    vol = 4.0 / 3.0 * math.pi * semi.prod(-1)
    return torch.polar(vol * f, -2 * math.pi * (k @ c.T))


@torch.no_grad()
def voxelize_ellipsoids(ph: EllipsoidPhantom, fov, supersample: int = 4, device="cuda",
                        chunk: int = 1 << 16) -> Tensor:
    """Fraction of each voxel inside each ellipsoid -> ``[E, nz, ny, nx]`` float32.

    Exact along z (each sub-ray parallel to z meets the quadric at two closed-form
    heights, and the inside length per voxel is the overlap of two intervals);
    ``supersample**2`` sub-rays per voxel in x-y. The box-filtered indicator, i.e. what
    an MR voxel integrates.
    """
    dev = torch.device(device)
    nx, ny, nz = fov.matrix
    dx, dy, dz = fov.voxel_mm
    o = [c - 0.5 * f for c, f in zip(fov.center_mm, fov.fov_mm)]
    s = int(supersample)
    R = rotation_from_quaternion(ph.quat.double().to(dev))
    semi, ctr = ph.semi.double().to(dev), ph.centre.double().to(dev)
    out = torch.zeros(ph.semi.shape[0], nz, ny, nx)
    for i in range(ph.semi.shape[0]):
        Q = R[i] @ torch.diag(semi[i].pow(-2)) @ R[i].T
        half = (R[i].square() @ semi[i].square()).sqrt()           # AABB half-extent
        cx, cy, cz = ctr[i].tolist()
        ix0 = max(0, math.ceil((cx - float(half[0]) - o[0]) / (dx / s) - 0.5))
        ix1 = min(nx * s - 1, math.floor((cx + float(half[0]) - o[0]) / (dx / s) - 0.5))
        iy0 = max(0, math.ceil((cy - float(half[1]) - o[1]) / (dy / s) - 0.5))
        iy1 = min(ny * s - 1, math.floor((cy + float(half[1]) - o[1]) / (dy / s) - 0.5))
        j0 = max(0, math.floor((cz - float(half[2]) - o[2]) / dz))
        j1 = min(nz, math.ceil((cz + float(half[2]) - o[2]) / dz))
        iy, ix = torch.meshgrid(torch.arange(iy0, iy1 + 1, device=dev),
                                torch.arange(ix0, ix1 + 1, device=dev), indexing="ij")
        iy, ix = iy.reshape(-1), ix.reshape(-1)
        j = torch.arange(j0, j1, device=dev, dtype=torch.float64)
        acc = torch.zeros(ny * nx, j1 - j0, device=dev, dtype=torch.float64)
        for a in range(0, ix.numel(), chunk):
            X = o[0] + (ix[a:a + chunk] + 0.5) * dx / s - cx
            Y = o[1] + (iy[a:a + chunk] + 0.5) * dy / s - cy
            A = Q[2, 2]
            B = 2 * (Q[2, 0] * X + Q[2, 1] * Y)
            C = Q[0, 0] * X * X + 2 * Q[0, 1] * X * Y + Q[1, 1] * Y * Y - 1
            disc = B * B - 4 * A * C
            hit = disc > 0
            sq = disc.clamp_min(0).sqrt()
            u0 = ((-B - sq) / (2 * A) + cz - o[2]) / dz
            u1 = ((-B + sq) / (2 * A) + cz - o[2]) / dz
            ov = (torch.minimum(u1[:, None], j + 1) - torch.maximum(u0[:, None], j)
                  ).clamp_min(0) * hit[:, None]
            vox = (iy[a:a + chunk] // s) * nx + ix[a:a + chunk] // s
            acc.index_add_(0, vox, ov)
        full = torch.zeros(ny * nx, nz, device=dev, dtype=torch.float64)
        full[:, j0:j1] = acc / (s * s)
        out[i] = full.reshape(ny, nx, nz).permute(2, 0, 1).float().cpu()
        del acc, full
    return out
