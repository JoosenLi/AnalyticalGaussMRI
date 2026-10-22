"""Boundary-aware adaptive density control.

The problem this solves. A block fit allocates primitives in proportion to *occupied
voxels*, so a homogeneous white-matter interior and a CSF/WM boundary of the same volume
get the same budget. That is backwards: a smooth interior is carried efficiently by one
large Gaussian (measured: capping sigma drove band-limited Fourier NRMSE from 0.099 to
0.255, because the big primitives are load-bearing), while an interface needs several
small ones to resolve a step. Globally shrinking sigma buys the boundary at the interior's
expense -- measured on the ellipsoid, and it is what the channel-fitting experiment did by
accident (median sigma 0.682 -> 0.379 mm, interfaces improved, interiors got worse).

So: **split only at boundaries, and split across the boundary**, leaving interior
primitives alone.

Three quantities decide, all per primitive:

* ``p[n, t]`` -- tissue composition, the share of the primitive's support in each label.
  Impurity ``1 - max_t p[n, t]`` (or the normalized entropy) says it straddles an
  interface. Segmentation is **auxiliary supervision only**: it never enters the fitting
  loss, which stays single-contrast.
* ``E[n]`` -- the contribution-weighted local residual, i.e. how much unexplained signal
  sits under this primitive. Impurity alone would split every boundary primitive
  including the ones already doing fine.
* the scale, so a primitive already at the floor is not split into two useless ones.

The split geometry is the part that has to be right. Splitting along the largest
eigenvector -- 3DGS's rule, and what :func:`fitter.densify` does -- is blind to the
interface: for a primitive straddling a flat boundary the useful direction is the
boundary **normal**, which is generally not the long axis. Worse, the existing rule
shrinks all three axes by 0.6, which does not preserve the second moment and so changes
the object's low-frequency content every time it fires.

Here the normal comes from the segmentation, and the split is **exactly second-moment
preserving**, which is what lets it be applied mid-fit without perturbing the spectrum:

    d_max^2 = 1 / (n^T Sigma^{-1} n)        Mahalanobis half-width along the normal
    d = alpha * d_max
    children at mu +/- d n,  each mass m / 2
    Sigma' = Sigma - d^2 n n^T

Then the two-component mixture has mean ``mu`` and covariance
``Sigma' + d^2 n n^T = Sigma`` exactly, for any ``alpha < 1``. The parent's first two
moments survive the split; only the third and higher change, which is precisely the
freedom needed to represent a step.

**The displacement must be scaled by the Mahalanobis width, not the marginal one.**
``Sigma - d^2 n n^T`` is positive definite iff ``d^2 n^T Sigma^{-1} n < 1``, and
``1/sqrt(n^T Sigma^{-1} n) <= sqrt(n^T Sigma n)`` by Cauchy-Schwarz, with equality only
when ``n`` is an eigenvector of ``Sigma``. Using the marginal ``sqrt(n^T Sigma n)``
therefore works for an isotropic primitive and silently produces an **indefinite**
covariance for an anisotropic one whose normal is off-axis -- which is the boundary case
this module exists for. ``Sigma = diag(100, 1, 1)`` with ``n = (1,1,0)/sqrt(2)`` and
``alpha = 0.5`` gives a 2x2 block of determinant -537; ``from_covariance`` then clamps the
negative eigenvalue and the moment identity above quietly stops holding.

Two exact consequences, both checked in ``scripts/verify_boundary.py``:

* tangential variance is untouched: ``t.n = 0`` gives ``t^T Sigma' t = t^T Sigma t``;
* the **conditional** variance along the normal scales by exactly ``1 - alpha^2``, i.e.
  ``1 / (n^T Sigma'^{-1} n) = (1 - alpha^2) / (n^T Sigma^{-1} n)`` (Sherman-Morrison).
  The *marginal* variance ``n^T Sigma' n`` does not scale so cleanly unless ``n`` is an
  eigenvector, which is why the conditional form is the one to assert on.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import torch
from torch import Tensor

from ..datasets.brainweb import VolumeTarget
from ..representation.gaussian3d import GaussianMixture3D, from_covariance
from .renderer3d import Patch, render


def tissue_moments(mixture: GaussianMixture3D, target: VolumeTarget,
                   n_classes: int = 12, cutoff_sigma: float = 4.0,
                   block: int = 128, verbose: bool = False) -> tuple[Tensor, Tensor]:
    """Per-primitive tissue shares ``[N, T]`` and tissue centroids ``[N, T, 3]`` (mm).

    Both come from the renderer's **backward pass**, so no new kernel is involved and the
    numbers inherit the verified forward. ``rho`` is linear in the masses, so
    ``d rho_j / d m_n = N(r_j; mu_n, Sigma_n)``; feeding an incoming gradient ``g_j``
    returns ``sum_j g_j N(r_j; mu_n, Sigma_n)``. With ``g = 1[label_j = t]`` that is the
    unnormalized tissue share, and with ``g = 1[label_j = t] * r_j`` it is the
    unnormalized first moment -- so the centroid of tissue ``t`` *as this primitive sees
    it* costs three more backward passes.

    That centroid is the whole point: the vector between two tissues' centroids under one
    primitive's own Gaussian weighting is the direction that separates them locally, which
    is the interface normal without ever forming a gradient of the label map (whose
    one-voxel-wide gradient is far noisier at 0.5 mm than a Gaussian-weighted moment).
    """
    if target.labels is None:
        raise ValueError("target carries no labels; load the volume with load_crisp()")
    dev = target.volume.device
    nz, ny, nx = target.volume.shape
    d = target.fov.voxel_mm
    base = [-0.5 * f + 0.5 * v + c for f, v, c in
            zip(target.fov.fov_mm, d, target.fov.center_mm)]

    mass = mixture.mass.detach().clone().requires_grad_(True)
    probe = dataclasses.replace(mixture, mass=mass)
    share = torch.zeros(mixture.n, int(n_classes), device=dev)
    mom = torch.zeros(mixture.n, int(n_classes), 3, device=dev)

    for z0 in range(0, nz, block):
        for y0 in range(0, ny, block):
            for x0 in range(0, nx, block):
                lab = target.labels[z0: z0 + block, y0: y0 + block, x0: x0 + block]
                present = torch.unique(lab)
                present = present[present > 0]
                if present.numel() == 0:
                    continue
                sh = (min(block, nx - x0), min(block, ny - y0), min(block, nz - z0))
                org = (base[0] + x0 * d[0], base[1] + y0 * d[1], base[2] + z0 * d[2])
                p = Patch(origin_mm=org, voxel_mm=d, shape=sh)
                rho = render(probe, p, cutoff_sigma)
                # patch coordinates, [nz, ny, nx] to match the render layout
                gz = org[2] + d[2] * torch.arange(sh[2], device=dev, dtype=rho.dtype)
                gy = org[1] + d[1] * torch.arange(sh[1], device=dev, dtype=rho.dtype)
                gx = org[0] + d[0] * torch.arange(sh[0], device=dev, dtype=rho.dtype)
                coord = (gx[None, None, :].expand(sh[2], sh[1], sh[0]),
                         gy[None, :, None].expand(sh[2], sh[1], sh[0]),
                         gz[:, None, None].expand(sh[2], sh[1], sh[0]))
                for t in present.tolist():
                    ind = (lab == t).to(rho.dtype)
                    g, = torch.autograd.grad(rho, mass, ind, retain_graph=True)
                    share[:, int(t)] += g.detach()
                    for ax in range(3):
                        g, = torch.autograd.grad(rho, mass, ind * coord[ax],
                                                 retain_graph=True)
                        mom[:, int(t), ax] += g.detach()
                del rho
        if verbose:
            print(f"    tissue moments z={min(z0 + block, nz)}/{nz}", flush=True)

    s = share.sum(dim=1, keepdim=True)
    w = torch.zeros_like(share)
    hit = s.squeeze(1) > 0
    w[hit] = share[hit] / s[hit]
    w[~hit, 0] = 1.0
    centroid = mom / share.clamp_min(1e-20)[..., None]
    return w, centroid


def impurity(w: Tensor) -> Tensor:
    """``1 - max_t p[n, t]`` -- 0 for a single-tissue primitive, up to 1-1/T at a junction."""
    return 1.0 - w.max(dim=1).values


def normalized_entropy(w: Tensor) -> Tensor:
    """``H(p) / log T`` in ``[0, 1]``. Reported alongside impurity; they rank differently
    when a primitive touches three tissues rather than straddling two."""
    p = w.clamp_min(1e-12)
    t = w.shape[1]
    return -(p * p.log()).sum(dim=1) / float(torch.log(torch.tensor(float(t))))


def boundary_normal(w: Tensor, centroid: Tensor) -> tuple[Tensor, Tensor]:
    """Unit interface normal ``[N, 3]`` and its confidence ``[N]``.

    The direction runs between the centroids of the two most-present tissues, weighted by
    how much of the primitive each holds. A primitive that is essentially pure gets a
    near-zero-length vector, and the confidence -- the raw separation in mm -- is what the
    caller thresholds on: a normal estimated from a 0.01-mm centroid difference is noise,
    and splitting across it would be worse than not splitting at all.
    """
    top = w.topk(2, dim=1)
    i1, i2 = top.indices[:, 0], top.indices[:, 1]
    n = torch.arange(w.shape[0], device=w.device)
    v = centroid[n, i1] - centroid[n, i2]                       # [N, 3] mm
    sep = v.norm(dim=-1)
    normal = v / sep.clamp_min(1e-12)[:, None]
    return normal, sep


def select_for_split(w: Tensor, resid: Tensor, sigma_max: Tensor, sep: Tensor,
                     frac: float = 0.10, min_impurity: float = 0.05,
                     min_sigma_mm: float = 0.0, min_sep_mm: float = 0.0
                     ) -> Tensor:
    """Indices to split: impure **and** badly fitted **and** big enough to be worth it.

    All three conditions are required, and the reason is that each alone is known to
    misfire. Impurity alone splits every interface primitive, including the many that
    already explain their neighbourhood. Residual alone is the existing rule, and it sends
    splits into smooth interiors. Scale alone is 3DGS's size heuristic, measured harmful
    here. ``frac`` then caps the number so the primitive count is a controlled variable --
    without it, "adaptive" and "more capacity" could not be told apart, which is the
    comparison the whole study turns on.
    """
    imp = impurity(w)
    ok = (imp >= min_impurity) & (sigma_max > min_sigma_mm) & (sep > min_sep_mm)
    if not bool(ok.any()):
        return torch.empty(0, dtype=torch.long, device=w.device)
    # rank on the product of the two independent signals, so a primitive must be BOTH
    # impure and unexplained; a sum would let a huge residual alone carry an interior one.
    score = torch.where(ok, imp * resid / resid.mean().clamp_min(1e-30),
                        torch.zeros_like(imp))
    k = min(max(1, int(frac * w.shape[0])), int(ok.sum()))
    return torch.topk(score, k).indices


def split_across_normal(mix: GaussianMixture3D, idx: Tensor, normal: Tensor,
                        alpha: float = 0.5, min_sigma_mm: float = 0.0
                        ) -> GaussianMixture3D:
    """Split ``idx`` across ``normal``, preserving mass and the full second moment.

    ``alpha`` is the displacement as a fraction of the **Mahalanobis** half-width
    ``d_max = (n^T Sigma^{-1} n)^{-1/2}``, which is the largest displacement keeping
    ``Sigma - d^2 n n^T`` positive definite. At ``alpha = 1`` the child collapses in the
    normal direction, so it is clamped below 1. Scaling by the marginal
    ``sqrt(n^T Sigma n)`` instead is wrong for any primitive whose normal is not an
    eigenvector -- see the module docstring.

    Note what is *not* done: the tangential axes are untouched. A primitive lying in the
    plane of a CSF/WM boundary should stay long in that plane -- it is correctly modelling
    a sheet -- and only needs resolution across it.
    """
    a = float(min(max(alpha, 1e-3), 0.95))
    sub = mix.subset(idx)
    nrm = normal[idx]
    nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    cov = _bare_cov(sub)
    # d_max^2 = 1 / (n^T Sigma^{-1} n); solve rather than invert, and in float64 because
    # a near-degenerate primitive makes the quadratic form ill-conditioned exactly where
    # the split matters most.
    sol = torch.linalg.solve(cov.double(), nrm.double()[..., None]).squeeze(-1)
    g = torch.einsum("ni,ni->n", nrm.double(), sol).clamp_min(1e-30)        # n^T Sigma^-1 n
    d2 = ((a * a) / g).to(cov.dtype)                                        # displacement^2
    outer = nrm[:, :, None] * nrm[:, None, :]                               # [K, 3, 3]
    cov_c = cov - d2[:, None, None] * outer

    # keep the shrunken axis above the floor: a split that lands under the sigma floor
    # would be undone by the clamp on the next step and the mass would be misplaced.
    if min_sigma_mm > 0:
        ev = torch.linalg.eigvalsh(cov_c)[:, 0]
        bad = ev < min_sigma_mm ** 2
        if bool(bad.any()):
            cov_c[bad] = cov[bad]                       # split position only, keep width

    shift = d2.clamp_min(0).sqrt()[:, None] * nrm
    half = sub.mass * 0.5
    ca = from_covariance(sub.mu + shift, cov_c, half)
    cb = from_covariance(sub.mu - shift, cov_c, half)

    keep = torch.ones(mix.n, dtype=torch.bool, device=mix.mu.device)
    keep[idx] = False
    rest = mix.subset(torch.nonzero(keep).flatten())
    cat = lambda f: torch.cat([f(rest), f(ca), f(cb)], dim=0)
    return GaussianMixture3D(
        mu=cat(lambda m: m.mu), log_scale=cat(lambda m: m.log_scale),
        quaternion=cat(lambda m: m.quaternion), mass=cat(lambda m: m.mass),
        dilation_mm=mix.dilation_mm)


def _bare_cov(mix: GaussianMixture3D) -> Tensor:
    """``R diag(sigma^2) R^T`` WITHOUT the band-limit term.

    The dilation is a property of the object, applied on top of whatever the parameters
    say; splitting must act on the bare covariance or every split would inflate the
    stored widths by the filter and the filter would be applied twice.
    """
    r = mix.rotation()
    s2 = torch.exp(2.0 * mix.log_scale)
    return r @ torch.diag_embed(s2) @ r.transpose(-1, -2)
