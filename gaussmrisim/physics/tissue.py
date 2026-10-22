"""Per-primitive tissue composition from a labelled volume -- pillar 2 of the paper.

The geometry is fitted **once**, to a scalar target. What makes the fitted anatomy
re-imageable under a new sequence is a per-primitive contrast coefficient
``a_n(theta)``, and that coefficient is only meaningful if each primitive knows what
tissue it is made of.

The assignment is the mixture's own responsibility, weighted by its own kernel:

    w[n, t]  =  sum_j 1[label_j = t] N(r_j; mu_n, Sigma_n)  /  sum_t (...)

i.e. the fraction of primitive ``n``'s support that lies in tissue ``t``. Two things
make this the right definition rather than "the label at ``mu_n``":

* a primitive straddling a boundary genuinely *is* partial volume, and the MR signal
  of a region is the sum of the signals of its constituents, so ``a_n(theta) =
  sum_t w[n, t] a_t(theta)`` is exact for the composition, not an approximation of it;
* it never assigns a tissue the primitive barely touches, which a nearest-voxel lookup
  does whenever ``mu_n`` lands one voxel across an interface.

**Never average relaxation times instead of signals.** Averaging T1 across a CSF/WM
interface produces ~2400 ms, which is not a tissue that exists; averaging the two
*signals* is what physically happens in a voxel of mixed content. That is why this
module returns a simplex and :func:`contrast` combines at the signal level.

Implementation note: the numerator is a scatter from voxels to primitives, which is
exactly what the renderer's backward pass already computes -- ``d rho_j / d m_n =
N(r_j; mu_n, Sigma_n)``, since ``rho`` is linear in the masses. Setting the incoming
gradient to the class indicator therefore yields the sum above from the *verified*
kernel, with no new CUDA and no second definition of the kernel to keep in step.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import torch
from torch import Tensor

from ..datasets.brainweb import TISSUE_PROPS, VolumeTarget
from ..representation.gaussian3d import GaussianMixture3D
from ..fitting.renderer3d import Patch, render


def tissue_weights_from_labels(mixture: GaussianMixture3D, target: VolumeTarget,
                               n_classes: int = 12, cutoff_sigma: float = 4.0,
                               block: int = 128, verbose: bool = False) -> Tensor:
    """``[N, T]`` simplex: the share of each primitive's support in each label.

    ``target.labels`` must be present (``datasets.brainweb.load_crisp``). Primitives
    that receive no weight at all -- possible only if a primitive sits entirely outside
    every occupied block -- are left as pure background, which is what they are.
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
    acc = torch.zeros(mixture.n, int(n_classes), device=dev)
    for z0 in range(0, nz, block):
        for y0 in range(0, ny, block):
            for x0 in range(0, nx, block):
                lab = target.labels[z0: z0 + block, y0: y0 + block, x0: x0 + block]
                present = torch.unique(lab)
                present = present[present > 0]           # background carries no signal
                if present.numel() == 0:
                    continue
                sh = (min(block, nx - x0), min(block, ny - y0), min(block, nz - z0))
                p = Patch(origin_mm=(base[0] + x0 * d[0], base[1] + y0 * d[1],
                                     base[2] + z0 * d[2]), voxel_mm=d, shape=sh)
                rho = render(probe, p, cutoff_sigma)
                for t in present.tolist():
                    g, = torch.autograd.grad(rho, mass, (lab == t).to(rho.dtype),
                                             retain_graph=True)
                    acc[:, int(t)] += g.detach()
                del rho
        if verbose:
            print(f"    tissue scatter z={z0 + block}/{nz}", flush=True)
    s = acc.sum(dim=1, keepdim=True)
    w = torch.zeros_like(acc)
    hit = s.squeeze(1) > 0
    w[hit] = acc[hit] / s[hit]
    w[~hit, 0] = 1.0
    return w


def attach_tissue(mixture: GaussianMixture3D, weights: Tensor,
                  props: Optional[dict] = None) -> GaussianMixture3D:
    """Store the simplex and the composition-averaged relaxation times on the mixture.

    The relaxation fields are carried for *diagnostics and for sequences that have no
    closed form per class*; the correct contrast path is :func:`contrast`, which mixes
    signals. They are stored so a primitive can be inspected, not so a T1 average can
    be quietly used as physics.
    """
    props = props or TISSUE_PROPS
    t = weights.shape[1]
    dev = weights.device
    tab = torch.tensor([[props[i][j] for j in range(4)] for i in range(t)],
                       device=dev, dtype=weights.dtype)          # [T, 4]
    avg = weights @ tab
    return dataclasses.replace(mixture, tissue_weights=weights,
                               t1_ms=avg[:, 1], t2_ms=avg[:, 2], t2star_ms=avg[:, 3])


def contrast(mixture: GaussianMixture3D, signal, props: Optional[dict] = None,
             classes: Optional[Sequence[int]] = None) -> Tensor:
    """``a_n(theta)`` for a mixture whose masses were fitted to a **proton-density** map.

    ``signal`` is a callable of one class's ``(PD, T1, T2, T2*)`` returning that class's
    full signal for the sequence, e.g. SPGR ``PD sin(a)(1-E1)/(1-E1 cos a) exp(-TE/T2*)``.

    The coefficient is a **ratio**::

        a_n  =  sum_t w[n, t] S_t  /  sum_t w[n, t] PD_t

    and the denominator is not optional. The fitted object already *is* the proton
    density -- ``sum_n m_n N_n(r) ~ PD(r)`` -- so ``a_n`` must supply only the
    sequence-dependent factor ``S/PD``, or the simulation would carry PD twice. Written
    as a ratio it is the **PD-weighted** average of that factor over the primitive's
    support, which is the physically correct combination for a primitive straddling an
    interface: the signal of a region is the sum of its constituents' signals, so the
    per-spin factor must be averaged with the spins, not with the volume. Averaging by
    volume instead would let a large low-PD share (skull, PD 0.2) pull the coefficient
    as hard as an equal share of CSF.

    A sanity property that falls straight out: for ``signal = PD`` every ``a_n`` is
    exactly 1, i.e. the object as fitted. :func:`gaussmrisim.physics.signal_model` keeps
    ``None`` as the fast path for that case.

    Primitives with no proton density anywhere in their support (pure background) get
    ``a_n = 0``: they carry no spins, so no sequence can make them signal.
    """
    if mixture.tissue_weights is None:
        raise ValueError("mixture has no tissue_weights; run tissue_weights_from_labels")
    props = props or TISSUE_PROPS
    w = mixture.tissue_weights
    idx = list(classes) if classes is not None else list(range(w.shape[1]))
    sig = torch.tensor([float(signal(*props[i])) for i in idx],
                       device=w.device, dtype=w.dtype)
    pd = torch.tensor([float(props[i][0]) for i in idx], device=w.device, dtype=w.dtype)
    num = w[:, idx] @ sig
    den = w[:, idx] @ pd
    return torch.where(den > 1e-8, num / den.clamp_min(1e-8), torch.zeros_like(num))
