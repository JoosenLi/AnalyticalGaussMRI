"""Image-domain fit of a Gaussian mixture to a volume.

This is the one place where approximation is *allowed and intended*: the fit uses tile
binning and a ``cutoff_sigma``, because the objective is "get the primitives into the
right places", not "define the object". The object is whatever mixture comes out, and
its k-space is then evaluated exactly, untruncated, by the Fourier engine. The two must
not be confused -- a cutoff that leaked into the forward model would make the phantom's
spectrum differ from the phantom.

Strategy, coarse to fine:

1. **Importance-sampled initialization.** Centres drawn with probability proportional to
   the target, so primitives start where the signal is; each then carries an equal share
   of the total mass, which is the unbiased importance-sampling estimate and makes the
   initial total mass exactly right by construction.
2. **Patch-stochastic Adam.** A random box per step rather than the whole volume: at
   1 mm a BrainWeb volume is 12 M voxels and rendering all of it every step wastes the
   binning, most of which is unchanged.
3. **Adaptive density control** between rounds: prune negligible mass, split primitives
   that are large and still badly wrong, clone those that are small and badly wrong.

Multiresolution is handled by the caller (:mod:`scripts.fit_brainweb`): fit a
box-averaged target first, then continue the same primitives on the finer one.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..conventions import FOV
from ..datasets.brainweb import VolumeTarget
from ..representation.gaussian3d import (GaussianMixture3D,
                                         quaternion_from_rotation_batch)
from .renderer3d import Patch, axis_widths, bin_to_tiles, render


@dataclass
class FitConfig:
    n_primitives: int = 50_000
    steps: int = 1500
    patch: int = 64                 # voxels per side of the training patch
    patches_per_step: int = 1
    lr_mu: float = 0.15             # mm per step scale; positions are in millimetres
    lr_log_scale: float = 0.01
    lr_quat: float = 0.005
    lr_mass: float = 0.02
    cutoff_sigma: float = 4.0
    init_sigma_vox: float = 1.0     # initial sigma in target voxels
    min_sigma_vox: float = 0.35     # floor on the BARE sigma (see the note below)
    dilation_vox: float = 0.0       # Mip-Splatting 3D smoothing filter, in target voxels
    # **OFF by default (changed 2026-09-08).** A real subject is not band-limited, and a
    # simulator that band-limits its object hides aliasing a reconstruction would really
    # face -- so a filter, which changes the object, has to earn being a default and no
    # longer does. `scripts/filter_default_study.py`, Shepp-Logan at 1 mm scored against
    # ANALYTIC truth (so no arm can win by reproducing the raster's own aliases):
    #
    #   fit filter   target        1/4Nyq  1/2Nyq   1Nyq   out-of-band / TRUE
    #   0.35         point         0.0303  0.0998  0.4569  0.43
    #   none         point         0.0313  0.1139  0.6167  2.71
    #   none         3^3 average   0.0192  0.0476  0.1819  0.33
    #   0.35         3^3 average   0.0295  0.0896  0.2841  0.06
    #
    # The reading that matters: **the out-of-band fabrication came from the TARGET, not
    # from the missing filter.** Point-sampling a target aliases it, the fit then plants
    # sub-voxel spikes to reproduce those aliases, and the object radiates 2.71x the true
    # power above Nyquist. The filter suppressed that symptom at a real cost in band. With
    # a properly voxel-averaged target the spikes never appear, and unfiltered is then
    # better on every band (1.5x / 1.9x / 1.6x) while still UNDER-shooting the true
    # out-of-band power, which is the safe direction. Filtering on top over-suppresses to
    # 6% of the truth.
    #
    # BrainWeb is already in that regime -- a fuzzy value IS a voxel average -- so the
    # relevant rows are the last two and the filter loses. Set `dilation_vox > 0`
    # deliberately when you WANT a band-limited object (studying gridding, or feeding a
    # method that assumes band-limitation), and say so when reporting.
    #
    # NOTE: keep `min_sigma_vox` at a real floor whenever a dilation IS used. The effective
    # width is sqrt(sigma^2 + s^2), so d(sigma_eff)/d(sigma) = sigma / sigma_eff -> 0 as
    # the bare sigma collapses: a primitive that shrinks below the filter stops receiving
    # a scale gradient and can never grow back. Letting the bare sigma run to 0.05 voxel
    # measured WORSE in band than a plain clamp (0.168 vs 0.053 at 1/4 Nyquist) purely
    # from this frozen-scale effect, not from the band-limit itself.
    max_sigma_vox: float = 24.0
    densify_every: int = 250
    densify_until: float = 0.75     # fraction of steps after which structure is frozen
    # --- boundary-aware adaptive density control (gaussmrisim.fitting.boundary) -------
    # Splits at these fractions of `steps`, across the segmentation's interface normal,
    # only for primitives that are impure AND unexplained AND above the scale floor.
    # Empty tuple = off, which is the default: the uniform fit is the control arm.
    boundary_split_at: tuple = ()
    boundary_split_frac: float = 0.10    # fraction of the CURRENT N split per round
    boundary_alpha: float = 0.5          # displacement in Mahalanobis half-widths
    boundary_min_impurity: float = 0.05
    boundary_min_sep_mm: float = 0.0     # reject a normal estimated from noise
    boundary_min_sigma_vox: float = 0.0  # do not split what is already at the floor
    rescale_on_refine: bool = True  # cap sigma to the new voxel when a level refines
    refine_sigma_vox: float = 2.5   # the cap, in voxels of the new level
    fg_patch_frac: float = 0.9      # fraction of patches centred on foreground
    # --- voxel subsampling (minibatch over VOXELS, not over regions) ------------------
    # `voxel_stride = s` evaluates every s-th voxel along each axis, i.e. 1/s^3 of the
    # volume per step, with the offset redrawn each step so all voxels are seen over
    # training. This is NOT the patch-stochastic scheme that was refuted here: a random
    # spatial patch touches a given primitive only when its own region is drawn, so each
    # primitive gets ~total_visits/volume Adam updates and the momentum dies in between.
    # A strided grid spans the WHOLE FOV every step, so every primitive receives a
    # gradient at every step -- it is proper minibatch SGD, and only the gradient's
    # variance goes up, not its coverage.
    # It also stays inside the tiled renderer: a strided grid is still a regular grid,
    # so it is a Patch with s x the voxel spacing and 1/s the shape. Nothing scatters.
    voxel_stride: int = 1
    mass_weight: float = 1.0        # penalty on total-mass drift (see fit())
    # --- data term ---------------------------------------------------------------
    # "l2" is the maximum-likelihood choice under Gaussian noise and is the default.
    # It is also dominated by the few voxels with the largest residual -- on a brain
    # those are the skull/CSF rim, whose error is an order of magnitude above the
    # interior, so L2 spends the primitive budget flattening the rim while the
    # interior keeps a diffuse texture. "l1" and "huber" weight the interior far more
    # evenly. This CHANGES THE OBJECTIVE, so a fit under one loss may not be compared
    # to a fit under another by its own training loss -- only by the reported metrics.
    loss_kind: str = "l2"           # "l2" | "l1" | "huber"
    # --- Monte-Carlo total variation (gaussmrisim.fitting.continuous_tv) ----------
    # tv_weight * mean_q sqrt(h^2 |grad rho(r_q)|^2 + delta^2) at tv_samples uniform
    # points of the target box per step, analytic gradient, no grid (ported from the
    # ICLR demo). 0 = off (default). It changes the FIT, not how the object is evaluated.
    tv_weight: float = 0.0
    tv_samples: int = 8192
    tv_delta: float = 1e-3
    # "mc": Monte-Carlo points (above). "grid": finite differences of the render the data
    # term already made -- free, but it sees only the voxel centres, i.e. exactly what the
    # data term sees. "grid_jitter": the same on a second render shifted by a random
    # sub-voxel offset each step -- a dense grid BETWEEN the centres, where the
    # oscillation the image loss cannot see lives. One extra render per step.
    tv_kind: str = "mc"
    huber_delta_frac: float = 0.05  # delta, as a fraction of the target's max
    prune_mass_frac: float = 1e-4   # relative to the mean mass
    prune_sigma_vox: float = 0.0    # REFUTED, keep 0 (disabled) -- see densify()
    split_frac: float = 0.05        # fraction of primitives touched per densification
    rebin_margin_vox: float = 0.75   # AABB inflation; rebin once drift exceeds it
    seed: int = 0
    # --- staged optimization (AGNC's rule: make it a sequence of well-conditioned
    # subproblems, not one nonconvex fit). Fractions of `steps` spent with
    # (mass only) / (+ mu, log_scale) / (+ quaternion).
    stages: tuple = (0.10, 0.40, 0.50)
    # --- adaptive density control
    split_score: str = "mass_grad"   # "mass_grad" | "mu_grad" | "both" | "residual"
    edge_init_frac: float = 0.35     # share of initial centres drawn from |grad target|
    # --- sheet initialization for thin tissues (off by default) ----------------------
    # (sigma across, sigma along) the local surface, in voxels. When set, the edge-drawn
    # primitives are oriented by the target gradient with exactly these widths instead
    # of the volume-preserving `aniso_init` shape. Meant for tissues whose thickness
    # 2V/A is a few voxels (around fat 1.35, muscle 2.15, dura 2.24, fat 2.47 at 0.5 mm).
    sheet_init: tuple = ()
    # --- structure control: "split" (3DGS split/clone, N grows) or "relocate"
    # (3DGS-MCMC-style, N fixed: the lowest-energy primitives are moved onto
    # high-residual ones, which are split in two). See relocate().
    densify_mode: str = "split"
    aniso_init: float = 4.0          # >0: flatten edge primitives across the boundary
                                     # (measured 4 > 9 > 1; 9 over-commits to the
                                     #  initial normal, which is noisy at 1 mm)


def _gradient_field(v: Tensor) -> Tensor:
    """Central-difference gradient -> ``[nz, ny, nx, 3]`` ordered (x, y, z)."""
    comp = []
    for d in (2, 1, 0):                      # array dims are (z, y, x); we want (x, y, z)
        comp.append(0.5 * (torch.roll(v, -1, dims=d) - torch.roll(v, 1, dims=d)))
    return torch.stack(comp, dim=-1)


def _gradient_magnitude(v: Tensor) -> Tensor:
    """|grad| of the target -- an edge map, same shape as ``v``."""
    return _gradient_field(v).norm(dim=-1)


def _orthonormal_frame(n: Tensor) -> Tensor:
    """Rotation matrices ``[N, 3, 3]`` whose FIRST column is the unit vector ``n``.

    Built with the branch-free Hughes-Moller trick (pick the smallest component of ``n``
    to cross against) so no primitive gets a degenerate frame, which a fixed reference
    axis would produce wherever the boundary happens to be parallel to it.
    """
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    a = torch.zeros_like(n)
    a.scatter_(1, n.abs().argmin(dim=1, keepdim=True), 1.0)
    e1 = torch.cross(n, a, dim=-1)
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    e2 = torch.cross(n, e1, dim=-1)
    return torch.stack([n, e1, e2], dim=-1)


_MULTINOMIAL_MAX = 1 << 24          # torch.multinomial's category limit


def _sample_weighted(w: Tensor, n: int, gen) -> Tensor:
    """``n`` flat indices drawn proportional to ``w``, with replacement.

    ``torch.multinomial`` refuses more than 2^24 categories, and a 0.5 mm tissue crop
    routinely exceeds that -- dura's bounding box alone is ~36 Mvox. Restricting the
    draw to the support is the natural fix and is also faster, since a zero-weight voxel
    could never have been drawn: it is the same distribution, not an approximation. If
    the support itself is still too large, fall back to inverse-CDF sampling, which has
    no size limit.
    """
    if n <= 0:
        return torch.empty(0, dtype=torch.long, device=w.device)
    nz = torch.nonzero(w > 0).flatten()
    if nz.numel() == 0:
        raise ValueError("no positive weight to sample from")
    if nz.numel() <= _MULTINOMIAL_MAX:
        p = w[nz]
        return nz[torch.multinomial(p / p.sum(), n, replacement=True, generator=gen)]
    c = torch.cumsum(w[nz].double(), 0)
    u = torch.rand(n, device=w.device, generator=gen).double() * float(c[-1])
    return nz[torch.searchsorted(c, u).clamp_max(nz.numel() - 1)]


def initialize(target: VolumeTarget, n: int, sigma_vox: float = 1.0,
               seed: int = 0, edge_frac: float = 0.35, aniso: float = 0.0,
               sheet: tuple = ()) -> GaussianMixture3D:
    """Importance-sampled centres, isotropic scales, equal masses summing to the target.

    Two sampling densities, mixed. Drawing centres proportional to *intensity* alone
    puts primitives where the mass is, which is where a smooth interior already needs
    few of them; the error, however, lives on the boundaries. A share ``edge_frac`` is
    therefore drawn proportional to ``|grad target|``. This matters more than it looks:
    a primitive that starts on the wrong side of an edge has to travel there, and sigma
    barely moves under gradient descent, so initialization largely decides the final
    scale distribution.
    """
    dev = target.volume.device
    g = torch.Generator(device=dev).manual_seed(int(seed))
    w = target.volume.reshape(-1).clamp_min(0)
    if float(w.sum()) <= 0:
        raise ValueError("target has no positive mass")
    n_edge = int(float(edge_frac) * int(n))
    parts = []
    if n_edge:
        e = _gradient_magnitude(target.volume).reshape(-1).clamp_min(0)
        parts.append(_sample_weighted(e, n_edge, g))
    parts.append(_sample_weighted(w, int(n) - n_edge, g))
    idx = torch.cat(parts)
    nz, ny, nx = target.volume.shape
    vz = idx // (ny * nx)
    vy = (idx // nx) % ny
    vx = idx % nx
    d = torch.tensor(target.fov.voxel_mm, device=dev)
    o = torch.tensor([-0.5 * f + 0.5 * v + c for f, v, c in
                      zip(target.fov.fov_mm, target.fov.voxel_mm, target.fov.center_mm)],
                     device=dev)
    jitter = (torch.rand(int(n), 3, device=dev, generator=g) - 0.5)      # within the voxel
    mu = o + (torch.stack([vx, vy, vz], -1).float() + jitter) * d

    sigma = float(sigma_vox) * d.min()
    mass = torch.full((int(n),), target.total_mass / float(n), device=dev)
    q = torch.zeros(int(n), 4, device=dev)
    q[:, 0] = 1.0
    log_scale = torch.log(sigma).expand(int(n), 3).contiguous().clone()

    if sheet and n_edge:
        # Explicit sheet shape: normal axis first, as in the aniso branch below.
        gvec = _gradient_field(target.volume).reshape(-1, 3)[idx[:n_edge]]
        q[:n_edge] = quaternion_from_rotation_batch(_orthonormal_frame(gvec))
        across, along = float(sheet[0]), float(sheet[1])
        log_scale[:n_edge] = torch.log(d.min() * torch.tensor(
            [across, along, along], device=dev))
    elif aniso > 0 and n_edge:
        # Anatomy is made of sheets -- cortex, the GM/WM interface, the skull. An
        # isotropic primitive spends most of its volume on the wrong side of such a
        # boundary; one flattened ACROSS the local gradient and stretched ALONG the
        # surface covers the same sheet with far fewer primitives. The gradient
        # direction is the surface normal, so the frame is free: no structure tensor
        # eigendecomposition needed, just an orthonormal completion of grad/|grad|.
        gvec = _gradient_field(target.volume).reshape(-1, 3)[idx[:n_edge]]
        R = _orthonormal_frame(gvec)                       # column 0 = normal
        q[:n_edge] = quaternion_from_rotation_batch(R)
        a = float(aniso)
        log_scale[:n_edge] = torch.log(sigma) + torch.tensor(
            [-math.log(a), 0.5 * math.log(a), 0.5 * math.log(a)], device=dev)  # volume-preserving
    return GaussianMixture3D(mu=mu, log_scale=log_scale, quaternion=q, mass=mass)


def foreground_index(target: VolumeTarget, thresh: float = 0.02) -> Tensor:
    """Flat indices of voxels carrying signal -- where patches should be drawn from.

    A BrainWeb volume is mostly air: uniformly placed patches spend most steps fitting
    zeros, which costs the same as a real patch and teaches nothing. Drawing patch
    centres from the foreground is the single cheapest fix to the sample efficiency, and
    it does not bias the fit because the loss is still evaluated over the whole patch,
    background included.
    """
    v = target.volume.reshape(-1)
    return torch.nonzero(v > float(thresh) * float(v.max())).flatten()


def _random_patch(target: VolumeTarget, side: int, gen, fg: Tensor | None = None,
                  fg_frac: float = 0.9) -> Patch:
    nz, ny, nx = target.volume.shape
    dev = target.volume.device
    if fg is not None and fg.numel() and float(torch.rand(1, generator=gen, device=dev)) < fg_frac:
        f = int(fg[int(torch.randint(0, fg.numel(), (1,), generator=gen, device=dev))])
        c = [f % nx, (f // nx) % ny, f // (ny * nx)]
        lo = [int(min(max(ci - side // 2, 0), max(0, n - side))) for ci, n in zip(c, (nx, ny, nz))]
    else:
        lo = [int(torch.randint(0, max(1, n - side + 1), (1,), generator=gen, device=dev))
              for n in (nx, ny, nz)]
    o = [-0.5 * f + 0.5 * v + c + i * v for f, v, c, i in
         zip(target.fov.fov_mm, target.fov.voxel_mm, target.fov.center_mm, lo)]
    return Patch(origin_mm=tuple(o), voxel_mm=target.fov.voxel_mm,
                 shape=(min(side, nx), min(side, ny), min(side, nz))), lo


def _strided_view(target: VolumeTarget, s: int, gen) -> tuple:
    """Every ``s``-th voxel along each axis, with a random offset -> (Patch, values).

    The offset is redrawn per call, so over training every voxel is visited; within one
    step the sample is spread over the entire field of view rather than concentrated in
    one region, which is what keeps every primitive receiving a gradient.

    The returned Patch has ``s x`` the voxel spacing and an origin shifted by the offset,
    which is exactly the sub-grid being evaluated -- the renderer needs no change and the
    target values are its own samples at those points, so nothing is interpolated or
    averaged and no aliasing is introduced into the loss.
    """
    nz, ny, nx = target.volume.shape
    dev = target.volume.device
    o = [int(torch.randint(0, s, (1,), generator=gen, device=dev)) for _ in range(3)]
    d = target.fov.voxel_mm
    base = [-0.5 * f + 0.5 * v + c for f, v, c in
            zip(target.fov.fov_mm, d, target.fov.center_mm)]
    shape = ((nx - o[0] + s - 1) // s, (ny - o[1] + s - 1) // s, (nz - o[2] + s - 1) // s)
    patch = Patch(origin_mm=(base[0] + o[0] * d[0], base[1] + o[1] * d[1],
                             base[2] + o[2] * d[2]),
                  voxel_mm=(d[0] * s, d[1] * s, d[2] * s), shape=shape)
    vals = target.volume[o[2]::s, o[1]::s, o[0]::s]
    return patch, vals


def _patch_target(target: VolumeTarget, lo, shape) -> Tensor:
    return target.volume[lo[2]: lo[2] + shape[2], lo[1]: lo[1] + shape[1],
                         lo[0]: lo[0] + shape[0]]


def _patch_channels(target: VolumeTarget, lo, shape) -> Optional[Tensor]:
    if target.channels is None:
        return None
    return target.channels[:, lo[2]: lo[2] + shape[2], lo[1]: lo[1] + shape[1],
                           lo[0]: lo[0] + shape[0]]


@torch.no_grad()
def residual_score(mix: GaussianMixture3D, coords: Tensor, tgt: Tensor,
                   cutoff: float, binning) -> Tensor:
    """``score_n = sum_j |res_j| N(r_j; mu_n, Sigma_n)`` -- unexplained signal under a
    primitive, WITHOUT the sign cancellation that hides a straddling primitive.

    The gradient w.r.t. mass is ``sum_j 2 res_j N_j(n)``, and for a primitive sitting on
    a thin sheet the residual is positive on one side and negative on the other, so that
    sum is near zero however wrong the primitive is. It is the same cancellation AbsGS
    (Ye et al. 2024) identified for the *positional* gradient in 3DGS, one derivative
    over: taking ``.abs()`` after the voxel sum, as the accumulators do, is too late.

    Taking ``|res|`` as the incoming gradient instead gives the uncancelled quantity from
    the already-verified backward pass, because rho is linear in the masses -- the same
    trick ``physics/tissue.py`` uses for tissue weights, and again no new CUDA.
    """
    # densification runs under no_grad; this one backward needs it back on
    with torch.enable_grad():
        ms = mix.mass.detach().clone().requires_grad_(True)
        mm = dataclasses.replace(mix, mass=ms)
        pred = render(mm, coords, cutoff, binning=binning)
        w = (pred.detach() - tgt).abs()
        (pred * w).sum().backward()
    return ms.grad.detach()


def densify(mix: GaussianMixture3D, grad_mu: Tensor, grad_mass: Tensor, cfg: FitConfig,
            voxel_mm: float) -> GaussianMixture3D:
    """Prune, split and clone -- the structural half of the optimization.

    Positional gradient magnitude is the error signal: a primitive that the loss still
    wants to move is one whose neighbourhood is not yet explained. Large ones are
    **split** (halved in scale, offset along their long axis) because they are covering
    structure they cannot resolve; small ones are **cloned** because the region needs
    more of them. This is 3DGS's adaptive density control with the camera parts removed.
    """
    # Mass prune only. A size prune -- 3DGS's max-screen-size rule -- was tried and
    # MEASURED HARMFUL here: capping sigma at 8-10 voxels drove band-limited Fourier
    # NRMSE from 0.099 to 0.255 and collapsed N from 100k to 46k. The large primitives
    # are not degenerate, they are load-bearing: a brain has big smooth interiors and a
    # 20 mm Gaussian is the efficient way to carry that low-frequency mass. In 3DGS the
    # size prune exists to stop screen-space rendering artefacts, which is a statement
    # about a camera, not about a density. Left at 0 (disabled) deliberately; do not
    # re-enable it on the intuition that big primitives look wrong.
    keep = mix.mass > cfg.prune_mass_frac * mix.mass.mean().clamp_min(1e-30)
    if cfg.prune_sigma_vox > 0:
        keep = keep & (mix.sigma().max(dim=-1).values < cfg.prune_sigma_vox * voxel_mm)
    mix = mix.subset(torch.nonzero(keep).flatten())
    grad_mu, grad_mass = grad_mu[keep], grad_mass[keep]
    if mix.n == 0:
        raise RuntimeError("every primitive was pruned; lower prune_mass_frac")

    k = max(1, int(cfg.split_frac * mix.n))
    # Which primitives are failing? The gradient w.r.t. MASS is the residual projected
    # onto a primitive's own footprint -- exactly "how much unexplained signal is sitting
    # under this primitive" -- whereas the positional gradient answers the different
    # question of which way it should move, and is small wherever the residual is
    # symmetric about the centre (a missing sharp edge inside a blob, say). Splitting on
    # the positional gradient sends new primitives into smooth interiors, which is where
    # the previous schedule saturated.
    gmu = grad_mu.norm(dim=-1)
    if cfg.split_score == "mu_grad":
        score = gmu
    elif cfg.split_score == "both":
        score = gmu / gmu.mean().clamp_min(1e-30) + grad_mass / grad_mass.mean().clamp_min(1e-30)
    else:
        score = grad_mass
    cand = torch.topk(score, min(k, mix.n)).indices
    big = mix.sigma()[cand].max(dim=-1).values > 2.0 * cfg.init_sigma_vox * voxel_mm

    sel = cand
    sub = mix.subset(sel)
    axis = sub.rotation()[:, :, 0]                       # first principal direction
    off = 0.5 * sub.sigma().max(dim=-1, keepdim=True).values * axis
    shift = torch.where(big[:, None], off, 0.3 * sub.sigma().min(dim=-1, keepdim=True).values
                        * torch.randn_like(off))
    shrink = torch.where(big[:, None], torch.full_like(sub.log_scale, math.log(0.6)),
                         torch.zeros_like(sub.log_scale))

    a = dataclasses.replace(sub, mu=sub.mu + shift, log_scale=sub.log_scale + shrink,
                            mass=sub.mass * 0.5)
    b = dataclasses.replace(sub, mu=sub.mu - shift, log_scale=sub.log_scale + shrink,
                            mass=sub.mass * 0.5)
    rest_idx = torch.ones(mix.n, dtype=torch.bool, device=mix.mu.device)
    rest_idx[sel] = False
    rest = mix.subset(torch.nonzero(rest_idx).flatten())
    cat = lambda f: torch.cat([f(rest), f(a), f(b)], dim=0)
    return GaussianMixture3D(mu=cat(lambda m: m.mu), log_scale=cat(lambda m: m.log_scale),
                             quaternion=cat(lambda m: m.quaternion), mass=cat(lambda m: m.mass))


def _split_pair(sub: GaussianMixture3D, cfg: FitConfig, voxel_mm: float):
    """The two children of a split, exactly as :func:`densify` makes them."""
    big = sub.sigma().max(dim=-1).values > 2.0 * cfg.init_sigma_vox * voxel_mm
    axis = sub.rotation()[:, :, 0]
    off = 0.5 * sub.sigma().max(dim=-1, keepdim=True).values * axis
    shift = torch.where(big[:, None], off, 0.3 * sub.sigma().min(dim=-1, keepdim=True).values
                        * torch.randn_like(off))
    shrink = torch.where(big[:, None], torch.full_like(sub.log_scale, math.log(0.6)),
                         torch.zeros_like(sub.log_scale))
    a = dataclasses.replace(sub, mu=sub.mu + shift, log_scale=sub.log_scale + shrink,
                            mass=sub.mass * 0.5)
    b = dataclasses.replace(sub, mu=sub.mu - shift, log_scale=sub.log_scale + shrink,
                            mass=sub.mass * 0.5)
    return a, b


def relocate(mix: GaussianMixture3D, score: Tensor, cfg: FitConfig, voxel_mm: float,
             gen) -> GaussianMixture3D:
    """Fixed-budget structure control in the spirit of 3DGS-MCMC (Kheradmand et al. 2024).

    N never changes. The ``k = split_frac * N`` primitives that contribute least are
    moved onto primitives sitting on unexplained signal, and each such target is split
    in two with the same moment-preserving rule as :func:`densify` -- the relocated
    primitive becomes the second child. So the budget is spent where the residual is,
    and the tier sizes stay exactly what was asked for.

    "Contributes least" is the energy of the primitive's own density,
    ``||m N||^2 = m^2 / ((4 pi)^{3/2} prod sigma)``: a low-mass primitive can still be
    a sharp, load-bearing one, so mass alone would be the wrong ranking. Targets are
    drawn WITHOUT replacement with probability proportional to ``score`` (the mass
    gradient, i.e. the residual under the footprint) rather than taken as a top-k, which
    is the MCMC paper's choice and keeps a few large-residual primitives from absorbing
    every relocation.
    """
    n = mix.n
    k = max(1, min(int(cfg.split_frac * n), n // 2))
    energy = mix.mass.clamp_min(0).square() / mix.sigma().prod(dim=-1)
    dead = torch.topk(energy, k, largest=False).indices
    w = score.clamp_min(0).clone()
    w[dead] = 0.0
    if float(w.sum()) <= 0:
        return mix
    k = min(k, int((w > 0).sum()))
    dead = dead[:k]
    tgt = torch.multinomial(w / w.sum(), k, replacement=False, generator=gen)
    a, b = _split_pair(mix.subset(tgt), cfg, voxel_mm)
    out = {f: getattr(mix, f).clone() for f in ("mu", "log_scale", "quaternion", "mass")}
    for f in out:
        out[f][tgt] = getattr(a, f)
        out[f][dead] = getattr(b, f)
    return dataclasses.replace(mix, **out)


def _grid_tv(v: Tensor, delta: float) -> Tensor:
    """Huber-TV of a rendered ``[nz, ny, nx]`` block: mean sqrt(|forward diff|^2 + delta^2),
    in value change per voxel -- the same convention and normalization (a mean over the
    box) as the Monte-Carlo estimator, so one ``tv_weight`` means the same for both."""
    dx = v[:-1, :-1, 1:] - v[:-1, :-1, :-1]
    dy = v[:-1, 1:, :-1] - v[:-1, :-1, :-1]
    dz = v[1:, :-1, :-1] - v[:-1, :-1, :-1]
    return torch.sqrt(dx * dx + dy * dy + dz * dz + delta * delta).mean()


def _data_term(pred: Tensor, tgt: Tensor, cfg: FitConfig, tmax: float) -> Tensor:
    """The fitting data term. See ``FitConfig.loss_kind``.

    Huber's ``delta`` is set from the TARGET's dynamic range rather than from the
    residual, so it is a fixed property of the problem and does not drift as the fit
    improves -- a delta chased off the running residual would anneal L1 into L2 exactly
    as the interior starts to matter.
    """
    r = pred - tgt
    k = cfg.loss_kind
    if k == "l2":
        return r.square().mean()
    if k == "l1":
        return r.abs().mean()
    if k == "huber":
        d = max(cfg.huber_delta_frac * tmax, 1e-8)
        a = r.abs()
        # scaled so the quadratic branch matches 0.5 r^2 and the linear branch continues
        # it with matching value and slope at |r| = delta
        return torch.where(a <= d, 0.5 * r.square(), d * (a - 0.5 * d)).mean()
    raise ValueError(f"unknown loss_kind {k!r}")


def fit(target: VolumeTarget, cfg: FitConfig, mixture: Optional[GaussianMixture3D] = None,
        log_every: int = 100, verbose: bool = True) -> tuple[GaussianMixture3D, dict]:
    """Fit (or continue fitting) a mixture to ``target``. Returns the mixture and a trace."""
    dev = target.volume.device
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    vox = float(min(target.fov.voxel_mm))
    nchan = 0 if target.channels is None else int(target.channels.shape[0])
    if mixture is None:
        mixture = initialize(target, cfg.n_primitives, cfg.init_sigma_vox, cfg.seed,
                             cfg.edge_init_frac, cfg.aniso_init, cfg.sheet_init)
    # The band-limit is a property of the OBJECT, so it is carried on the mixture and
    # survives into the returned phantom -- not applied only while fitting.
    mixture = dataclasses.replace(mixture, dilation_mm=cfg.dilation_vox * vox)

    if mixture is not None and cfg.rescale_on_refine:
        # A mixture arriving from a coarser level carries that level's scales, and
        # sigma barely moves under gradient descent (AGNC measured 1.06-1.16x over a
        # whole run) -- so wherever it starts is where it stays. Capping it to the new
        # voxel is what makes a refinement level actually refine instead of re-fitting
        # the coarse solution.
        cap = math.log(cfg.refine_sigma_vox * vox)
        if float(mixture.log_scale.max()) > cap:
            mixture = dataclasses.replace(mixture, log_scale=mixture.log_scale.clamp(max=cap))

    fg = foreground_index(target)
    tmax = float(target.volume.max()) if target.channels is None else 1.0
    total_mass = (target.channel_mass if nchan
                  else torch.tensor(target.total_mass, device=dev))
    stage0 = cfg.stages[0] > 0
    mu = mixture.mu.clone().requires_grad_(not stage0)
    ls = mixture.log_scale.clone().requires_grad_(not stage0)
    q = mixture.quaternion.clone().requires_grad_(cfg.stages[0] + cfg.stages[1] <= 0)
    if nchan:
        # One mass per (primitive, channel), geometry shared. The exponent is what costs,
        # and it is evaluated once per (primitive, voxel) inside the kernel, so the extra
        # channels only multiply the cheap accumulation -- but the *reference* renderer
        # has one scalar mass, so we call it per channel rather than touch a verified
        # kernel. Channels are the tissue indicators, which makes the fitted m[c, n] the
        # primitive's mass OF tissue c and contrast exact rather than averaged.
        cm = target.channel_mass
        init = mixture.mass[None, :] * (cm / cm.sum().clamp_min(1e-30))[:, None]
        ms = init.clone().requires_grad_(True)
    else:
        ms = mixture.mass.clone().requires_grad_(True)
    # fused=True keeps the whole Adam update in one kernel per group. The fitting loop
    # is launch-latency bound, not compute bound -- the actual render forward+backward is
    # 0.56 ms against ~1 ms of surrounding tiny ops -- so collapsing the optimizer's
    # element-wise chain is worth more here than it would be in a compute-bound trainer.
    def _make_opt(mu, ls, q, ms, mass_scale):
        groups = [{"params": [mu], "lr": cfg.lr_mu * vox},
                  {"params": [ls], "lr": cfg.lr_log_scale},
                  {"params": [q], "lr": cfg.lr_quat},
                  {"params": [ms], "lr": cfg.lr_mass * mass_scale}]
        try:
            return torch.optim.Adam(groups, eps=1e-15, fused=True)
        except (RuntimeError, ValueError):          # CPU tensors, or an older torch
            return torch.optim.Adam(groups, eps=1e-15, foreach=True)

    opt = _make_opt(mu, ls, q, ms, float(mixture.mass.mean()))

    lo_s, hi_s = math.log(cfg.min_sigma_vox * vox), math.log(cfg.max_sigma_vox * vox)
    trace = {"step": [], "loss": [], "nrmse": [], "n": []}
    grad_accum = torch.zeros_like(mu)
    mass_accum = torch.zeros_like(ms)
    # Splitting needs a residual estimate, so never on step 0 -- mass_accum would be
    # zero and the selection would be impurity-only, which is the arm being tested
    # against, not this one.
    split_steps = set()
    if cfg.boundary_split_at:
        if nchan:
            raise NotImplementedError(
                "boundary splitting assumes one mass per primitive; with per-channel "
                "masses split_across_normal would halve only the scalar. "
                "Keep channels as the ablation arm (boundary_split_at=()).")
        if target.labels is None:
            raise ValueError("boundary_split_at needs a labelled target for the normals")
        split_steps = {max(1, int(f * cfg.steps)) for f in cfg.boundary_split_at}
    s1, s2 = int(cfg.stages[0] * cfg.steps), int((cfg.stages[0] + cfg.stages[1]) * cfg.steps)
    binning, last_patch, mu_ref, t0 = None, None, None, time.time()
    tv_lo = torch.tensor([c - 0.5 * f for c, f in zip(target.fov.center_mm, target.fov.fov_mm)],
                         device=dev, dtype=torch.float32)
    tv_hi = tv_lo + torch.tensor(target.fov.fov_mm, device=dev, dtype=torch.float32)

    for step in range(cfg.steps):
        # Staged: mass is a LINEAR parameter given the geometry, so solving it first
        # (stage 1) is a well-conditioned problem whose answer the nonlinear stages then
        # only have to correct. Unfreezing everything at step 0 instead makes the mass
        # chase geometry that is still moving.
        if step == s1:
            mu.requires_grad_(True); ls.requires_grad_(True)
        if step == s2:
            q.requires_grad_(True)
        m = GaussianMixture3D(mu=mu, log_scale=ls, quaternion=q, mass=ms)
        loss = torch.zeros((), device=dev)
        for pi in range(cfg.patches_per_step):
            if cfg.voxel_stride > 1:
                patch, tgt = _strided_view(target, cfg.voxel_stride, gen)
                lo = None
            else:
                patch, lo = _random_patch(target, cfg.patch, gen, fg, cfg.fg_patch_frac)
                tgt = _patch_target(target, lo, patch.shape)
            # Rebinning costs 0.71 ms and is CONSTANT in N: it is ~15 tiny torch ops and
            # is launch-latency bound, which made it 44% of a step while the actual
            # forward+backward was 0.56 ms. When the patch does not change between steps
            # -- which is the whole of block fitting, where the block IS the patch -- the
            # lists can be reused. They are still rebuilt every `rebin_every` steps, and
            # in between the AABB is inflated by the furthest a primitive can drift
            # (Adam's step is bounded by its learning rate), so a reused list still
            # covers every primitive it should. Without that margin a drifted primitive
            # silently stops contributing, which reads as a bad fit rather than a bug.
            margin = cfg.rebin_margin_vox * vox
            # Rebin on DRIFT, not on a step count. A fixed interval has to assume the
            # worst case (Adam can move a primitive by ~lr every step), so the margin
            # grows with the interval and inflates every AABB: at rebin_every=64 that
            # was 9.6 mm of padding, which produced so many extra tile-primitive pairs
            # that the step got 3.3x SLOWER than rebinning every time. Measuring the
            # actual displacement instead is one O(N) reduction and keeps the margin at
            # a fixed 0.75 voxel, which is what makes the reuse sound: a list built with
            # that inflation stays valid until something has moved that far.
            if binning is None or patch != last_patch or mu_ref is None:
                drift_ok = False
            else:
                drift_ok = bool((mu.detach() - mu_ref).abs().max() < margin)
            if not drift_ok:
                binning = bin_to_tiles(mu.detach(), axis_widths(m).detach() + margin,
                                       patch, cfg.cutoff_sigma)
                mu_ref, last_patch = mu.detach().clone(), patch
            b = binning
            if nchan:
                if lo is None:
                    raise NotImplementedError("voxel_stride with channels")
                ch = _patch_channels(target, lo, patch.shape)
                for c in range(nchan):
                    mc = dataclasses.replace(m, mass=ms[c])
                    loss = loss + _data_term(
                        render(mc, patch, cfg.cutoff_sigma, binning=b), ch[c], cfg, tmax)
            else:
                pred = render(m, patch, cfg.cutoff_sigma, binning=b)
                loss = loss + _data_term(pred, tgt, cfg, tmax)
                if cfg.tv_weight > 0 and cfg.tv_kind == "grid":
                    loss = loss + cfg.tv_weight * _grid_tv(pred, cfg.tv_delta)
                elif cfg.tv_weight > 0 and cfg.tv_kind == "grid_jitter":
                    off = torch.rand(3, generator=gen, device=dev).tolist()
                    jp = Patch(origin_mm=tuple(o + f * v for o, f, v in
                                               zip(patch.origin_mm, off, patch.voxel_mm)),
                               voxel_mm=patch.voxel_mm, shape=patch.shape)
                    jb = bin_to_tiles(mu.detach(), axis_widths(m).detach(), jp,
                                      cfg.cutoff_sigma)
                    loss = loss + cfg.tv_weight * _grid_tv(
                        render(m, jp, cfg.cutoff_sigma, binning=jb), cfg.tv_delta)
        if cfg.tv_weight > 0 and cfg.tv_kind == "mc":
            if nchan:
                raise NotImplementedError("tv_weight with per-channel masses")
            from .continuous_tv import sampled_tv
            loss = loss + cfg.tv_weight * sampled_tv(m, tv_lo, tv_hi, cfg.tv_samples, vox,
                                                     cfg.tv_delta, gen)
        if cfg.mass_weight:
            # The patch L2 loss does not constrain the TOTAL mass, and the total mass is
            # exactly rho_hat(0): a mixture that drifts 6% low simulates every image 6%
            # dark, uniformly, which no image-domain metric on a normalized comparison
            # would reveal. One scalar penalty fixes the DC that the patches cannot see.
            got = ms.sum(-1) if nchan else ms.sum()
            loss = loss + cfg.mass_weight * (
                ((got - total_mass) / total_mass.clamp_min(1e-30)).square().sum()
                if nchan else ((got - total_mass) / total_mass).square())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # during the mass-only stage mu has no grad at all; Adam skips it and so do we
        if mu.grad is not None:
            grad_accum += mu.grad.detach().abs()
        if ms.grad is not None:
            mass_accum += ms.grad.detach().abs()
        opt.step()
        with torch.no_grad():
            ls.clamp_(lo_s, hi_s)
            ms.clamp_(min=0.0)

        # Accumulate the running loss ON THE GPU. Calling float() every step forces a
        # device sync that serializes the whole loop -- with a 0.6 ms kernel that sync
        # was a large fraction of the step. Only read it back when actually logging.
        with torch.no_grad():
            cur = (loss / cfg.patches_per_step).detach()
            ema = cur if step == 0 else 0.98 * ema + 0.02 * cur
        if (step % log_every == 0) or step == cfg.steps - 1:
            # sqrt(loss)/rms is an NRMSE only when the loss IS the mean square error.
            # Under l1 or huber it is a different functional and the ratio is not
            # comparable to an L2 run -- printing it as "NRMSE" invites reading a change
            # of objective as a change in fit quality. Normalize each loss by its own
            # target statistic and say which one is being shown.
            if cfg.loss_kind == "l2":
                den = target.volume.square().mean().sqrt().clamp_min(1e-12)
                rel, tag = float(ema.sqrt() / den), "patch-NRMSE"
            elif cfg.loss_kind == "l1":
                den = target.volume.abs().mean().clamp_min(1e-12)
                rel, tag = float(ema / den), "patch-relMAE"
            else:
                den = target.volume.abs().mean().clamp_min(1e-12)
                rel, tag = float(ema / den), "patch-relHuber"
            trace["step"].append(step); trace["loss"].append(float(loss))
            trace["nrmse"].append(rel); trace["n"].append(int(mu.shape[0]))
            if verbose:
                print(f"    step {step:5d}  {tag}(ema) {rel:.4f}  N={mu.shape[0]:,}  "
                      f"{time.time() - t0:6.1f}s", flush=True)

        if step in split_steps:
            # Segmentation enters HERE and only here: it selects and orients the split.
            # It never touches the loss, which stays single-contrast -- so this is
            # auxiliary supervision, not multi-channel fitting.
            from .boundary import (boundary_normal, select_for_split,
                                   split_across_normal, tissue_moments)
            m = GaussianMixture3D(mu=mu.detach(), log_scale=ls.detach(),
                                  quaternion=q.detach(), mass=ms.detach(),
                                  dilation_mm=cfg.dilation_vox * vox)
            w, cen = tissue_moments(m, target, cutoff_sigma=cfg.cutoff_sigma)
            nrm, sep = boundary_normal(w, cen)
            sel = select_for_split(
                w, mass_accum.detach(), m.sigma().max(dim=-1).values, sep,
                frac=cfg.boundary_split_frac, min_impurity=cfg.boundary_min_impurity,
                min_sigma_mm=cfg.boundary_min_sigma_vox * vox,
                min_sep_mm=cfg.boundary_min_sep_mm)
            if sel.numel():
                m = split_across_normal(m, sel, nrm, alpha=cfg.boundary_alpha,
                                        min_sigma_mm=cfg.min_sigma_vox * vox)
                mu = m.mu.clone().requires_grad_(True)
                ls = m.log_scale.clone().requires_grad_(True)
                q = m.quaternion.clone().requires_grad_(True)
                ms = m.mass.clone().requires_grad_(True)
                opt = _make_opt(mu, ls, q, ms, float(ms.mean()))
                grad_accum = torch.zeros_like(mu)
                mass_accum = torch.zeros_like(ms)
                # N changed, so the cached tile binning and its drift reference describe
                # a different primitive set; force a rebuild rather than compare shapes.
                binning, mu_ref = None, None
            if verbose:
                print(f"    boundary split: {sel.numel():,} split -> N={mu.shape[0]:,}",
                      flush=True)

        if (nchan and cfg.densify_every and step
                and step % cfg.densify_every == 0):
            raise NotImplementedError(
                "densify() splits and clones a single mass vector; with per-channel "
                "masses it would silently drop channels. Set densify_every=0.")
        if (cfg.densify_every and step and step % cfg.densify_every == 0
                and step < cfg.densify_until * cfg.steps):
            m = GaussianMixture3D(mu=mu.detach(), log_scale=ls.detach(),
                                  quaternion=q.detach(), mass=ms.detach())
            score = mass_accum
            if cfg.split_score == "residual":
                score = residual_score(m, patch, tgt, cfg.cutoff_sigma, b)
            if cfg.densify_mode == "relocate":
                m = relocate(m, score, cfg, vox, gen)
            else:
                m = densify(m, grad_accum, score, cfg, vox)
            mu = m.mu.clone().requires_grad_(True); ls = m.log_scale.clone().requires_grad_(True)
            q = m.quaternion.clone().requires_grad_(True); ms = m.mass.clone().requires_grad_(True)
            opt = _make_opt(mu, ls, q, ms, float(ms.mean()))
            grad_accum = torch.zeros_like(mu)
            mass_accum = torch.zeros_like(ms)
            binning, mu_ref = None, None          # N changed -- see the split above
            if verbose:
                print(f"    densify -> N={mu.shape[0]:,}", flush=True)

    if nchan:
        # The channels are tissue indicators, so m[c, n] is the primitive's mass of
        # tissue c. Report the object in the SAME form the rest of the pipeline uses:
        # a scalar reference mass and a composition simplex. Taking the reference mass
        # as sum_t PD_t m[t, n] is what makes physics.tissue.contrast() exact here --
        # a_n = sum w S / sum w PD then gives mass * a_n = sum_t S_t m[t, n], the signal
        # of a partial-volume primitive with no averaging approximation left in it.
        from ..datasets.brainweb import TISSUE_PROPS
        mm = ms.detach().clamp_min(0.0)                       # [C, N]
        pdv = torch.tensor([TISSUE_PROPS[int(t)][0] for t in target.channel_labels],
                           device=dev, dtype=mm.dtype)
        w = torch.zeros(mm.shape[1], 12, device=dev, dtype=mm.dtype)
        w[:, list(int(t) for t in target.channel_labels)] = (
            mm / mm.sum(0, keepdim=True).clamp_min(1e-30)).T
        out = GaussianMixture3D(mu=mu.detach(), log_scale=ls.detach(),
                                quaternion=q.detach(), mass=(pdv[:, None] * mm).sum(0),
                                tissue_weights=w, dilation_mm=cfg.dilation_vox * vox)
        out.meta = {"channel_mass": mm}
        return out, trace
    out = GaussianMixture3D(mu=mu.detach(), log_scale=ls.detach(),
                            quaternion=q.detach(), mass=ms.detach(),
                            dilation_mm=cfg.dilation_vox * vox)
    return out, trace


@torch.no_grad()
def render_full(mixture: GaussianMixture3D, target: VolumeTarget,
                cutoff_sigma: float = 5.0, block: int = 128) -> Tensor:
    """Render the mixture over the whole target FOV, in blocks -> ``[nz, ny, nx]``."""
    nz, ny, nx = target.volume.shape
    out = torch.zeros_like(target.volume)
    d = target.fov.voxel_mm
    base = [-0.5 * f + 0.5 * v + c for f, v, c in
            zip(target.fov.fov_mm, d, target.fov.center_mm)]
    for z0 in range(0, nz, block):
        for y0 in range(0, ny, block):
            for x0 in range(0, nx, block):
                sh = (min(block, nx - x0), min(block, ny - y0), min(block, nz - z0))
                p = Patch(origin_mm=(base[0] + x0 * d[0], base[1] + y0 * d[1],
                                     base[2] + z0 * d[2]), voxel_mm=d, shape=sh)
                out[z0: z0 + sh[2], y0: y0 + sh[1], x0: x0 + sh[0]] = render(
                    mixture, p, cutoff_sigma)
    return out


# --------------------------------------------------------------------- block fitting
