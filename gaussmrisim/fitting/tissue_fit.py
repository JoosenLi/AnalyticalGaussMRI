"""Fit one Gaussian mixture per tissue, and concatenate them into one object.

Why per tissue rather than one mixture with per-channel weights, which this project
already has: a primitive that straddles a tissue interface carries **one** averaged
contrast coefficient across an edge that may not exist in the contrast it was fitted to.
Measured on the 0.5 mm fit: geometry fitted to PD alone gave 0.056 on PD but 0.21-0.23 on
T1w/T2w, and the decomposition by distance showed the penalty is concentrated within
1 mm of an interface, because the CSF/WM step is 0.23 in PD and 0.65 in T2w. Fitting each
tissue to its own membership volume removes that by construction: every interface a
tissue has *is* an interface of the object being fitted.

The mixtures are then **concatenated into one** ``GaussianMixture3D`` with a ``tissue``
index, which is what makes the physics cheap:

* contrast becomes a gather, ``a_n = a[tissue_n]``, handed to the kernel's existing
  ``coeff`` argument -- the ``simple`` forward model needs no kernel change at all;
* T2* decay becomes a per-primitive scalar, because each primitive belongs to exactly
  one tissue, instead of SNAKE's one-NUFFT-per-tissue;
* one launch. The FT kernel is L2-resident in the primitive parameters (measured cliff
  at 96 MiB, ~3 M primitives), so T separate launches would stream the parameters T
  times over for no reason.

Each tissue is fitted **inside its own bounding box**. Their volumes sum to 49.5 Mvox
against a 31.2 Mvox foreground union -- 1.6x overlap, not 12x -- which is what makes
this affordable at 0.5 mm at all.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor

from ..datasets.brainweb import VolumeTarget, channel_bbox, crop_channel
from ..representation.gaussian3d import GaussianMixture3D
from .fitter import FitConfig, _gradient_magnitude, fit


# --------------------------------------------------------------------- allocation
def tissue_drivers(target: VolumeTarget, thresh: float = 0.01,
                   device=None) -> Dict[str, Tensor]:
    """The two geometric quantities a budget can be allocated on.

    ``volume`` counts occupied voxels; ``area`` is the total variation, a discrete
    surface measure. They are **not** interchangeable: measured on BrainWeb they rank
    the tissues differently -- muscle/skin has more interface area than grey matter with
    24% less volume -- so which one is used changes the answer, and that is exactly why
    both are reported instead of one being assumed.
    """
    vols, areas = [], []
    for i in range(target.channels.shape[0]):
        # one channel at a time: the full fuzzy target is 2.5 GB and must not all be
        # resident on the device at once
        f = target.channels[i].float()
        if device is not None:
            f = f.to(device)
        vols.append(float((f > thresh).sum()))
        a = 0.0
        for d in range(3):
            a += float(f.diff(dim=d).abs().sum())
        areas.append(a)
        del f
        if device is not None:
            torch.cuda.empty_cache()
    return {"volume": torch.tensor(vols), "area": torch.tensor(areas)}


def allocate(drivers: Dict[str, Tensor], total: int, rule: str = "area",
             floor: int = 2000) -> Tensor:
    """Split a primitive budget across tissues -> ``[T]`` long.

    ``rule`` is ``"area"``, ``"volume"``, or ``"sqrt_area_volume"`` (the geometric mean,
    which is what a shell of fixed thickness would give). All three are **functions of
    geometry only**. That is deliberate: weighting by a sequence's signal would tie the
    fitted geometry to one sequence and destroy the property the whole architecture
    exists for -- fit once, re-image under anything. Some tissues are invisible in some
    sequences (skull contributes exactly zero at TR=50/TE=30/FA=15); that belongs in the
    reporting, not in the allocation.
    """
    if rule == "area":
        w = drivers["area"]
    elif rule == "volume":
        w = drivers["volume"]
    elif rule == "sqrt_area_volume":
        w = (drivers["area"] * drivers["volume"]).sqrt()
    else:
        raise ValueError(f"unknown allocation rule {rule!r}")
    w = w.clamp_min(0)
    n = torch.full_like(w, float(floor))
    rest = max(int(total) - int(n.sum()), 0)
    n = n + rest * w / w.sum().clamp_min(1e-30)
    return n.round().long()


def fit_error_model(runs: Sequence[dict], band: str = "shell_nrmse_0.25"
                    ) -> tuple:
    """Per-tissue ``err(N) = c * N^-p`` from two or more measured fits -> {label: (c, p)}.

    ``band`` defaults to the **shell** metric, not the power-weighted one. The first
    water-filling arm was fitted on ``fourier_nrmse_0.25``, whose per-tissue values carry
    2x draw noise, so it optimized against corrupted inputs and lost to plain
    area-proportional allocation. Fitting a model on a metric that does not converge is
    the kind of mistake that produces a confident, wrong answer.

    A power law because that is what the measurements look like and because it has the
    one property the allocation needs: a closed-form marginal utility. It is fitted per
    tissue, not globally, and that is the whole point -- the exponents differ by 4x
    (vessels 0.10, around-fat 0.45), which is exactly why one proportional rule cannot
    serve every tissue.
    """
    import math

    by_label: Dict[int, list] = {}
    for run in runs:
        for r in run["tissues"]:
            if band in r and r[band] > 0 and r["n"] > 0:
                by_label.setdefault(int(r["label"]), []).append((r["n"], r[band]))
    out = {}
    spans = {}
    for lab, pts in by_label.items():
        pts = sorted(set(pts))
        if len(pts) < 2:
            n, e = pts[0]
            out[lab] = (e * n ** 0.3, 0.3)          # no slope measurable; assume typical
            spans[lab] = (n, n)
            continue
        # least squares on log err = log c - p log N
        xs = [math.log(n) for n, _ in pts]
        ys = [math.log(e) for _, e in pts]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(den, 1e-30)
        p_ = max(-slope, 1e-3)
        c_ = math.exp(my + p_ * mx)
        out[lab] = (c_, p_)
        spans[lab] = (min(n for n, _ in pts), max(n for n, _ in pts))
    return out, spans


def allocate_measured(curves: Dict[int, list], weight: Dict[int, float], total: int,
                      labels: Sequence[int], floor: int = 2000,
                      step: int = 2000) -> Tensor:
    """Budget from the MEASURED err(N) curve, by greedy marginal utility.

    ``allocate_waterfill`` equalizes the marginal utility of an analytic power law, and a
    power law cannot represent a minimum. Measured on a controlled sweep -- same steps,
    same seed, only N varying -- around fat's in-band error bottoms out at ~142k and is
    **46% worse** by 293k, so a rule derived from ``err = c N^-p`` keeps spending budget
    there long after it has started doing harm. Nothing forces the anatomy to be monotone
    and the model should not assume it is.

    ``curves`` is ``{label: [(N, err), ...]}``. Between measured points the curve is
    interpolated in log-log, and **outside the measured range it is held flat** rather
    than extrapolated: a flat segment says "no evidence more helps here", which is the
    honest reading, whereas extrapolating a slope is what stripped skin and marrow to 3-9k
    the first time. Allocation is greedy in blocks of ``step``, always to whichever tissue
    buys the largest reduction in ``sum_t (w_t err_t)^2``, so a tissue past its minimum
    simply stops being chosen -- no special case needed.
    """
    import bisect
    import math

    tables = {}
    for lab in labels:
        pts = sorted((int(n), float(e)) for n, e in curves.get(lab, []))
        if not pts:
            raise ValueError(f"no measured curve for label {lab}")
        tables[lab] = ([math.log(n) for n, _ in pts], [math.log(e) for _, e in pts])

    def err(lab: int, n: float) -> float:
        xs, ys = tables[lab]
        x = math.log(max(n, 1.0))
        if x <= xs[0]:
            return math.exp(ys[0])                     # held flat below the measured range
        if x >= xs[-1]:
            return math.exp(ys[-1])                    # and above it
        i = bisect.bisect_right(xs, x) - 1
        t = (x - xs[i]) / (xs[i + 1] - xs[i])
        return math.exp(ys[i] + t * (ys[i + 1] - ys[i]))

    # Start at the lowest MEASURED point, not at ``floor``. Below the measured range the
    # curve is held flat, so a greedy started at the floor buys exactly zero from its
    # first blocks, finds no tissue worth funding, and stops with everyone at the floor.
    # The flat segment is the honest reading of "no evidence"; it just cannot be the
    # place the search begins.
    n = {l: float(max(floor, min(int(a) for a, _ in curves[l]))) for l in labels}
    w = {l: float(weight.get(l, 1.0)) for l in labels}
    if sum(n.values()) > total:
        sc = total / sum(n.values())
        n = {l: max(float(floor), v * sc) for l, v in n.items()}
    cost = {l: (w[l] * err(l, n[l])) ** 2 for l in labels}
    rest = int(total) - int(sum(n.values()))
    for _ in range(max(rest // step, 0)):
        best, gain = None, 0.0
        for l in labels:
            c = (w[l] * err(l, n[l] + step)) ** 2
            g = cost[l] - c
            if g > gain:
                best, gain = l, g
        if best is None:                               # nothing left that helps
            break
        n[best] += step
        cost[best] = (w[best] * err(best, n[best])) ** 2
    return torch.tensor([round(n[l]) for l in labels], dtype=torch.long)


def allocate_waterfill(model: Dict[int, tuple], weight: Dict[int, float], total: int,
                       labels: Sequence[int], floor: int = 2000,
                       span: Optional[Dict[int, tuple]] = None,
                       extrapolate: float = 2.0) -> Tensor:
    """Budget that equalizes **marginal** error reduction -- the water-filling rule.

    Minimizes ``sum_t (w_t c_t N_t^-p_t)^2`` subject to ``sum_t N_t = total``. Setting the
    derivatives equal gives ``N_t = (2 p_t w_t^2 c_t^2 / lam)^(1/(2 p_t + 1))``, and lam
    falls out of a bisection on the budget.

    ``weight`` is each tissue's contribution to the assembled object, ``||f_t||``, so a
    tissue that is small or faint is not given budget merely because its *relative* error
    is large. It is a norm of the membership volume and therefore **sequence-independent**,
    which the allocation must be: weighting by one sequence's ``a_t`` would tie the fitted
    geometry to that sequence and destroy the fit-once property.
    """
    import math

    ks = [(float(weight.get(l, 1.0)), *model.get(l, (1.0, 0.3))) for l in labels]

    def n_of(lam: float) -> list:
        out = []
        for w_, c_, p_ in ks:
            a = 2.0 * p_ * (w_ * c_) ** 2 / max(lam, 1e-300)
            out.append(max(a ** (1.0 / (2.0 * p_ + 1.0)), float(floor)))
        return out

    # Clamp to the range the power law was actually measured over. Without this the rule
    # strips any tissue whose measured slope is flat down to the floor -- skin, skull and
    # bone marrow came out at 3-9k against a measured range of 77-326k, a 30x
    # extrapolation on a model fitted over 1.6x. A flat slope inside the measured window
    # is evidence that more primitives do not help *there*; it is not evidence that the
    # tissue can be represented by almost none, and the difference is the whole risk of
    # optimizing an extrapolated fit.
    bounds = None
    if span is not None:
        bounds = [(max(float(floor), span[l][0] / extrapolate),
                   span[l][1] * extrapolate) if l in span else (float(floor), 1e18)
                  for l in labels]

    def clamped(lam: float) -> list:
        v = n_of(lam)
        if bounds is None:
            return v
        return [min(max(x, lo_), hi_) for x, (lo_, hi_) in zip(v, bounds)]

    lo, hi = 1e-30, 1e30
    for _ in range(400):
        mid = math.sqrt(lo * hi)
        if sum(clamped(mid)) > total:
            lo = mid
        else:
            hi = mid
    return torch.tensor([round(x) for x in clamped(math.sqrt(lo * hi))],
                        dtype=torch.long)


# --------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate_tissue(mix: GaussianMixture3D, sub: VolumeTarget, n_k: int = 4096,
                    bands: Sequence[float] = (0.25, 0.5, 1.0), seed: int = 0,
                    cutoff_sigma: float = 6.0) -> dict:
    """Per-tissue error against the membership volume it was fitted to.

    The reference is the fuzzy channel and its raster DFT -- never another mixture,
    which would be the representation-level inverse crime. The raster DFT carries the
    target's own aliases (measured 0.019 / 0.052 / 0.148 at 1/4, 1/2 and full Nyquist on
    a 1 mm Shepp-Logan), so a band-limited number below those is measuring the reference
    rather than the fit. Reported per band for that reason.
    """
    from ..acquisition import trajectory as T
    from ..evaluation.raster import nrmse, raster_dft, shell_nrmse
    from ..fourier.cuda_ft import kspace_cuda
    from .fitter import render_full

    dev = sub.volume.device
    rec = render_full(mix.to(dev), sub, cutoff_sigma=cutoff_sigma)
    out = {"image_nrmse": float((rec - sub.volume).norm() / sub.volume.norm()),
           "mass_target": float(sub.total_mass),
           "mass_fit": float(mix.integrated_mass().sum())}
    out["mass_rel_err"] = abs(out["mass_fit"] - out["mass_target"]) / max(
        out["mass_target"], 1e-30)
    del rec
    torch.cuda.empty_cache()
    if n_k > 0 and dev != torch.device("cpu"):
        pts = sub.fov.grid_coords(device=dev, dtype=torch.float64)
        for b in bands:
            k = T.heldout(int(n_k), sub.fov, r_max=float(b), seed=seed, device=dev)
            ref = raster_dft(sub.volume, pts, k, sub.fov.voxel_volume_mm3)
            got = kspace_cuda(mix.to(dev), k).to(torch.complex128)
            out[f"fourier_nrmse_{b:g}"] = float(nrmse(got, ref))
            # Band-uniform companion: the power-weighted ratio above does not converge
            # in the sample count (2.0x seed spread at n=4096, 2.7x at 65536) because it
            # is set by whichever near-DC sample was drawn. The shell mean is stable to
            # 1.02x and is the number an allocation model should be fitted on.
            sh, _mid = shell_nrmse(got, ref, k, float(b), shells=16, fov=sub.fov)
            out[f"shell_nrmse_{b:g}"] = sum(sh) / max(len(sh), 1)
            del ref, got, k
        del pts
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------- the fit
def fit_tissues(target: VolumeTarget, budget: Tensor, cfg: FitConfig,
                device="cuda:0", thresh: float = 0.01, pad: int = 2,
                n_k: int = 4096, verbose: bool = True, thin: Optional[dict] = None
                ) -> tuple[GaussianMixture3D, Tensor, dict]:
    """Fit every channel and concatenate -> (mixture, ``tissue [N]``, report).

    Channels are fitted one at a time and moved off the device immediately: the full
    fuzzy target is 11 x 56.9 Mvox x 4 B = 2.5 GB, which must never all be resident.
    """
    mixes, tags, rows = [], [], []
    for i in range(target.channels.shape[0]):
        label = target.channel_labels[i] if target.channel_labels else i
        box = channel_bbox(target, i, thresh=thresh, margin_vox=pad)
        sub = crop_channel(target, i, box)
        sub = dataclasses.replace(sub, volume=sub.volume.to(device))
        # FULL VOLUME, always: the crop *is* the patch. `fit`'s default patch of 64
        # voxels covers 0.78% of a tissue crop per step, which is the patch-stochastic
        # regime this project already refuted -- each primitive then receives about a
        # dozen Adam updates over a 1500-step run instead of 1500, and the momentum dies
        # in between. Measured on white matter: patch 64 gives 0.316 at 1/4 Nyquist and
        # the full crop gives 0.0122, a factor of 26, for 2.7x the time. It costs 1.8 GiB
        # at 33.8 Mvox, so there is no reason to patch at this scale.
        nz, ny, nx = sub.volume.shape
        c = dataclasses.replace(cfg, n_primitives=int(budget[i]),
                                patch=max(nx, ny, nz), patches_per_step=1)
        # Thin tissues (sheet thickness 2V/A below thin["vox"]) get a sheet-shaped
        # initialization and fixed-budget relocation. Measured per tissue at equal
        # final N (scripts/ablate_fit_strategies.py, 2026-10-14): better in image AND
        # in band for fat / muscle / dura, flat for around fat; relocation alone was
        # worse on thick tissues, so it is applied to thin ones only. Relocation does not
        # grow N, so the budget is scaled by n_scale to match what split/clone reaches.
        thick = float(2 * sub.volume.sum()
                      / _gradient_magnitude(sub.volume).sum().clamp_min(1e-30))
        is_thin = bool(thin) and thick < float(thin["vox"])
        if is_thin:
            sheet = (max(float(thin.get("across_min", 0.35)),
                          float(thin.get("across_frac", 0.3)) * thick),
                     float(thin.get("along", 2.0)))
            c = dataclasses.replace(
                c, sheet_init=sheet, edge_init_frac=float(thin.get("edge_frac", 0.8)),
                densify_mode=thin.get("mode", "relocate"),
                n_primitives=int(thin["n_exact"][int(label)]) if "n_exact" in thin
                else int(round(int(budget[i]) * float(thin.get("n_scale", 1.15)))))
        t0 = time.time()
        mix, tr = fit(sub, c, verbose=False)
        dt = time.time() - t0
        ev = evaluate_tissue(mix, sub, n_k=n_k) if n_k >= 0 else {}
        mixes.append(mix.to("cpu"))
        tags.append(torch.full((mix.n,), int(label), dtype=torch.long))
        mvox = float(sub.volume.numel()) / 1e6
        rows.append({"label": int(label), "n": int(mix.n), "seconds": dt,
                     "thickness_vox": thick, "thin_init": is_thin,
                     "mvox": mvox, "box": list(box),
                     "occupied_mvox": float((ch_occ := (sub.volume > thresh).sum())) / 1e6,
                     **ev})
        if verbose:
            print(f"  tissue {label:2d}  N={mix.n:>8,}  box {mvox:6.2f} Mvox "
                  f"(occ {float(ch_occ) / 1e6:5.2f})  {dt:6.1f} s  "
                  f"img {ev.get('image_nrmse', float('nan')):.4f}  "
                  f"1/4Nyq {ev.get('fourier_nrmse_0.25', float('nan')):.4f}  "
                  f"mass {ev.get('mass_rel_err', float('nan')):.1e}", flush=True)
        del sub, mix
        torch.cuda.empty_cache()

    cat = GaussianMixture3D(
        mu=torch.cat([m.mu for m in mixes]),
        log_scale=torch.cat([m.log_scale for m in mixes]),
        quaternion=torch.cat([m.quaternion for m in mixes]),
        mass=torch.cat([m.mass for m in mixes]),
        dilation_mm=mixes[0].dilation_mm)
    return cat, torch.cat(tags), {"tissues": rows}
