#!/usr/bin/env python
"""Gaussian mixture vs polyhedral cells on a NEUTRAL truth: the 3D Shepp-Logan.

The OASIS benchmark (scripts/polyphantom_benchmark.py) defines its truth with triangle
meshes, i.e. in the polyhedral function class: sharp planar facets that a polyhedron
represents exactly and a Gaussian never does. Here the truth is a sum of ellipsoids
(Koay MRM 2007) -- curved AND sharp, so it lies in NEITHER class -- and its transform is
the closed form ``V 3 j1(q)/q e^{-i2pi k.c}``, evaluated in fp64. Both arms see only the
0.5 mm partial-volume voxels and are scored against that closed form.

Subjects
  const   rho = sum_i a_i 1_i(r)
  smooth  rho = sum_i a_i 1_i(r) [1 + ALPHA cos(2 pi v_i.r + phi_i)]   (modulation theorem:
          three shifted ellipsoid FTs per ellipsoid, still exact). SUPERSEDED: ALPHA = 0.3 on
          every ellipsoid separately, the skull (2.0) included, swings the 1.2 brain interior
          by up to +-70% with 30-60 mm stripes -- no anatomy looks like that.
  natural rho = [sum_i a_i 1_i(r)] g(r): ONE slowly varying field shared by every
          structure, bright centre and darker rim like a head-coil / B1 profile,
          g = 1 - 0.62 eps + (eps/3) sum_{a=x,y,z} cos(2 pi (r_a - c_a)/P), eps = 0.22,
          P = 260 mm: g in [0.98, 1.08] inside the head, std 2.8%, so every edge and
          contrast of the phantom survives. 7 shifted ellipsoid FTs per ellipsoid (one of
          them unshifted, the constant term), still exact. Chosen from four candidates
          (outputs/figures/shepplogan_natural_candidates.png).

Stages (outputs/shepplogan_bench/<subject>/)
  truth   held-out k in 16 equal-count shells to 1/4, 1/2, 1 Nyquist; fp64 closed form
  fit     per-ellipsoid Gaussian fit to the voxels (``--total``), same driver as OASIS
  cells   marching cubes on the voxels -> cut into h mm cells -> one ridge-LS constant
          per cell, exact edge-form FT (``--h``; 0 = one constant per ellipsoid)
  eval    raster DFT + Gaussians, image / ROI / k-space errors, fp32 timing -> json
  plot    contrast_channels layout + error-vs-band / cost panels

    python experiments/shepplogan_benchmark.py truth --subject smooth --device cuda:0
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaussmrisim.acquisition import trajectory as T
from gaussmrisim.conventions import FOV
from gaussmrisim.datasets.brainweb import VolumeTarget
from gaussmrisim.datasets.ellipsoid import shepp_logan_3d, unit_ft64, voxelize_ellipsoids
from gaussmrisim.datasets.polyhedral import from_voxels, voxelize
from gaussmrisim.evaluation.raster import nrmse, raster_dft, shell_nrmse
from gaussmrisim.fitting.fitter import FitConfig, render_full
from gaussmrisim.fitting.tissue_fit import allocate, fit_tissues, tissue_drivers
from gaussmrisim.fourier.cuda_ft import kspace_cuda
from gaussmrisim.fourier.polyhedral_cells import Grid, cell_edges, edge_ft, fit_cell_values
from gaussmrisim.representation.gaussian3d import GaussianMixture3D, rotation_from_quaternion
from benchmark_utils import timed  # noqa: E402

SCALE_MM = 180.0                  # Shepp-Logan [-1,1] -> 180 mm: a brain-sized head
VOX = 0.5
MARGIN = 8.0
ALPHA = 0.3
ROOT = Path("outputs/shepplogan_bench")
CACHE = ROOT / "voxels_0.5mm_ss4.pt"


def phantom(dev="cpu"):
    return shepp_logan_3d(SCALE_MM, device=dev)


def modulation_params(E: int):
    """Fixed per-ellipsoid modulation: random direction, period 30-60 mm, random phase."""
    rng = np.random.default_rng(7)
    d = rng.normal(size=(E, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    period = np.linspace(30.0, 60.0, E)
    return d / period[:, None], rng.uniform(0, 2 * np.pi, E)


NATURAL_EPS = 0.22
NATURAL_PERIOD_MM = 260.0


def modulation_terms(subject: str, E: int) -> list[list[tuple[float, np.ndarray, float]]]:
    """Per ellipsoid, the cosine terms ``(eps, v [cycles/mm], phi)`` of its factor
    ``1 + sum eps cos(2 pi v.r + phi)``; empty for the constant subject."""
    if subject == "const":
        return [[] for _ in range(E)]
    if subject == "smooth":
        v, phi = modulation_params(E)
        return [[(ALPHA, v[i], float(phi[i]))] for i in range(E)]
    c = np.array(make_fov(phantom()).center_mm)
    e, P = NATURAL_EPS, NATURAL_PERIOD_MM
    shared = [(e / 3, np.eye(3)[a] / P, float(-2 * math.pi * c[a] / P)) for a in range(3)]
    shared.append((-0.62 * e, np.zeros(3), 0.0))        # v = 0: the constant offset
    return [shared for _ in range(E)]


def make_fov(ph) -> FOV:
    R = rotation_from_quaternion(ph.quat.double())
    half = (R.square() @ ph.semi.double().square()[..., None]).squeeze(-1).sqrt()
    lo = (ph.centre.double() - half).min(0).values - MARGIN
    hi = (ph.centre.double() + half).max(0).values + MARGIN
    n = [2 * math.ceil(float(h - l) / (2 * VOX)) for l, h in zip(lo, hi)]
    c = ((lo + hi) / 2).tolist()
    return FOV(tuple(VOX * m for m in n), tuple(n), tuple(c))


def target(ph, fov, dev, subject: str) -> VolumeTarget:
    if CACHE.exists():
        ch = torch.load(CACHE)["channels"]
    else:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        ch = voxelize_ellipsoids(ph, fov, supersample=4, device=dev)
        torch.save({"channels": ch, "fov": (fov.fov_mm, fov.matrix, fov.center_mm)}, CACHE)
    if subject != "const":
        # voxel value = exact partial-volume fraction x modulation at the voxel centre;
        # the exact voxel average differs at 2nd order in pi|v|dx: <= 4.6e-4 for
        # ``smooth``, <= 1e-6 for ``natural`` (periods >= 120 mm), stated
        terms = modulation_terms(subject, ch.shape[0])
        pts = fov.grid_coords(device="cpu", dtype=torch.float64).reshape(*ch.shape[1:], 3)

        def factor(tt):
            g = torch.ones(pts.shape[:-1], dtype=torch.float64)
            for e, v, phi in tt:
                g += e * torch.cos(2 * math.pi * (pts @ torch.tensor(v)) + phi)
            return g.float()
        g_shared = factor(terms[0]) if subject == "natural" else None
        ch = torch.stack([ch[i] * (g_shared if g_shared is not None else factor(terms[i]))
                          for i in range(ch.shape[0])])
    vol = sum(float(a) * ch[i] for i, a in enumerate(ph.value))
    return VolumeTarget(vol, fov, f"shepplogan_{SCALE_MM:g}mm_0.5mm", channels=ch,
                        channel_labels=tuple(range(1, ch.shape[0] + 1)))


def truth_ft(ph, k: torch.Tensor, subject: str) -> torch.Tensor:
    a = ph.value.double().to(k.device)
    if subject == "const":
        return unit_ft64(ph, k) @ a.to(torch.complex128)
    terms = modulation_terms(subject, a.numel())
    k = k.double()
    tot = torch.zeros(k.shape[0], dtype=torch.complex128, device=k.device)
    for i in range(a.numel()):
        one = dataclasses.replace(ph, centre=ph.centre[i:i + 1], semi=ph.semi[i:i + 1],
                                  quat=ph.quat[i:i + 1], value=ph.value[i:i + 1])
        f = unit_ft64(one, k)[:, 0]
        for eps, v, phi in terms[i]:           # cos = (e^{i.} + e^{-i.})/2, modulation thm
            e = 0.5 * eps * complex(math.cos(phi), math.sin(phi))
            v = torch.tensor(v, device=k.device, dtype=torch.float64)
            f = f + e * unit_ft64(one, k - v)[:, 0] + e.conjugate() * unit_ft64(one, k + v)[:, 0]
        tot += a[i] * f
    return tot


def mc_meshes(ph, fov, dev):
    f = ROOT / "mc_meshes.pt"
    if f.exists():
        from gaussmrisim.datasets.polyhedral import Mesh
        return [Mesh(v, fc, n) for v, fc, n in torch.load(f)]
    ch = torch.load(CACHE)["channels"]
    ms = []
    for i in range(ch.shape[0]):
        m = from_voxels(ch[i], fov, 0.5, name=f"E{i + 1}")
        print(f"  MC ellipsoid {i + 1}: {m.n_faces:,} faces, volume "
              f"{m.signed_volume():.1f} vs exact "
              f"{4 / 3 * math.pi * float(ph.semi[i].double().prod()):.1f} mm^3", flush=True)
        ms.append(m)
    torch.save([(m.vertices, m.faces, m.name) for m in ms], f)
    return ms


def view(fov):
    """Axial slice through the three small ellipsoids at z = 0, and an ROI around them."""
    nx, ny, nz = fov.matrix
    o = [c - 0.5 * f for c, f in zip(fov.center_mm, fov.fov_mm)]
    z = int(round((0.0 - o[2]) / VOX - 0.5))
    yc = (-0.605 * SCALE_MM / 2 - o[1]) / VOX
    xc = (0.0 - o[0]) / VOX
    roi = [int(xc - 40), int(xc + 40), int(yc - 30), int(yc + 50)]
    return z, roi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["truth", "fit", "cells", "eval"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--subject", default="natural", choices=["const", "smooth", "natural"])
    ap.add_argument("--total", type=int, default=1_000_000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--tv-samples", type=int, default=8192)
    ap.add_argument("--tv-kind", default="mc", choices=["mc", "grid", "grid_jitter"])
    ap.add_argument("--tv-weight", type=float, default=0.0,
                    help="Monte-Carlo TV weight in the fit (gaussmrisim.fitting.continuous_tv)")
    ap.add_argument("--voxel-box", action="store_true",
                    help="fit through a voxel-integrating render: every primitive is seen "
                         "convolved with N(0, dx^2/12 I), the Gaussian matching the voxel box "
                         "to 2nd order (sinc(u) vs exp(-pi^2 u^2/6) differ by pi^4 u^4/180, "
                         "1.3e-4 at 1/4 Nyquist); the STORED object is the deconvolved one, "
                         "so its k-space carries no voxel blur"),
    ap.add_argument("--floor", type=int, default=2000,
                    help="minimum primitives per tissue in the area allocation")
    ap.add_argument("--h", type=float, nargs="+", default=[0, 4, 2, 1])
    ap.add_argument("--bands", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--n-k", type=int, default=4096)
    ap.add_argument("--n-time", type=int, default=32768)
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    root = ROOT / args.subject
    root.mkdir(parents=True, exist_ok=True)
    ph = phantom()
    fov = make_fov(ph)
    tgt = target(ph, fov, dev, args.subject)
    z, roi = view(fov)
    x0, x1, y0, y1 = roi

    if args.stage == "truth":
        ch = torch.load(CACHE)["channels"]
        vol_err = [abs(float(ch[i].double().sum()) * fov.voxel_volume_mm3
                       / (4 / 3 * math.pi * float(ph.semi[i].double().prod())) - 1)
                   for i in range(ch.shape[0])]
        print(f"FOV {fov.fov_mm} mm, matrix {fov.matrix}; voxelizer volume rel err max "
              f"{max(vol_err):.1e}", flush=True)
        # the series/closed-form switch must be continuous: compare across q = 1e-2
        out = {"bands": args.bands, "k": {}, "sharp": {}, "voxel_volume_err": vol_err}
        for b in args.bands:
            k = T.heldout(args.n_k, fov, r_max=float(b), seed=0, device=dev, shells=16)
            out["k"][b] = k.cpu()
            out["sharp"][b] = truth_ft(ph, k.double(), args.subject).cpu()
        if args.subject != "const":
            terms = modulation_terms(args.subject, len(ph.value))
            g = []
            for i in range(len(ph.value)):
                one = dataclasses.replace(ph, centre=ph.centre[i:i + 1], semi=ph.semi[i:i + 1],
                                          quat=ph.quat[i:i + 1], value=ph.value[i:i + 1])
                V = 4 / 3 * math.pi * float(ph.semi[i].double().prod())
                gi = 1.0
                for eps, v, phi in terms[i]:
                    e = 0.5 * eps * complex(math.cos(phi), math.sin(phi))
                    F = complex(unit_ft64(one, -torch.tensor(v[None]))[0, 0])
                    gi += 2 * (e * F).real / V
                g.append(gi)
            out["mean_g"] = g
        torch.save(out, root / "truth.pt")
        print(f"truth -> {root / 'truth.pt'}")
        return 0

    tr = torch.load(root / "truth.pt")

    def kscore(get) -> dict:
        r = {}
        for b in args.bands:
            k = tr["k"][b].to(dev)
            sharp = tr["sharp"][b].to(dev)
            g = get(k).to(torch.complex128)
            sh, _ = shell_nrmse(g, sharp, k, float(b), shells=16, fov=fov)
            r[f"shell_{b:g}"] = float(sum(sh) / len(sh))
            r[f"nrmse_{b:g}"] = float(nrmse(g, sharp))
        return r

    def iscore(img: torch.Tensor) -> dict:
        ref = tgt.volume
        s, rs = img[z].numpy(), ref[z].numpy()
        return {"image_nrmse": float((img - ref).norm() / ref.norm()),
                "roi_nrmse": float(np.linalg.norm(s[y0:y1, x0:x1] - rs[y0:y1, x0:x1])
                                   / np.linalg.norm(rs[y0:y1, x0:x1]))}

    if args.stage == "fit":
        budget = allocate(tissue_drivers(tgt, 0.01, device=dev), args.total, "area", args.floor)
        print(f"budget {budget.tolist()}", flush=True)
        cfg = FitConfig(steps=args.steps, cutoff_sigma=4.0, tv_weight=args.tv_weight,
                        tv_samples=args.tv_samples, tv_kind=args.tv_kind,
                        dilation_vox=(1 / math.sqrt(12)) if args.voxel_box else 0.0)
        t0 = time.time()
        mix, tissue, rep = fit_tissues(tgt, budget, cfg, device=dev, n_k=0)
        dt = time.time() - t0
        out = root / (f"fit_N{args.total // 1000}k" + ("_box" if args.voxel_box else "")
                      + (f"_tv{args.tv_weight:g}" if args.tv_weight > 0 else "")
                      + (f"_m{args.tv_samples // 1024}k" if args.tv_weight > 0
                         and args.tv_samples != 8192 and args.tv_kind == "mc" else "")
                      + (f"_{args.tv_kind}" if args.tv_weight > 0 and args.tv_kind != "mc"
                         else "")
                      + (f"_floor{args.floor}" if args.floor != 2000 else ""))
        out.mkdir(parents=True, exist_ok=True)
        mix = dataclasses.replace(mix, dilation_mm=0.0) if args.voxel_box else mix
        torch.save(dict(mix.__dict__), out / "mixture.pt")
        torch.save(tissue, out / "tissue.pt")
        (out / "summary.json").write_text(json.dumps(
            {"total_requested": args.total, "n": int(mix.n), "seconds": dt,
             "fit_seconds": sum(r["seconds"] for r in rep["tissues"]),
             "voxel_box": bool(args.voxel_box), "tv_weight": args.tv_weight,
             "tv_samples": args.tv_samples, "tv_kind": args.tv_kind,
             "peak_gib": torch.cuda.max_memory_allocated(dev) / 2**30, **rep}, indent=2))
        print(f"N={mix.n:,} in {dt:.0f} s -> {out}")
        return 0

    if args.stage == "cells":
        ms = [m.to(dev) for m in mc_meshes(ph, fov, dev)]
        mem = [voxelize(m, fov, supersample=4, device=dev).cpu() for m in ms]
        nx, ny, nz = fov.matrix
        corner = [c - 0.5 * f for c, f in zip(fov.center_mm, fov.fov_mm)]
        # different irrational sub-nm offsets per axis keep grid lines off MC edges
        origin = tuple(x + d for x, d in zip(corner, (1e-7 * math.pi, 1e-7 * math.e * 0.37,
                                                      1e-7 * math.sqrt(2) * 0.61)))
        kt = T.heldout(args.n_time, fov, r_max=1.0, seed=1, device=dev)
        rows = {"subject": args.subject, "faces": [m.n_faces for m in ms], "arms": []}
        for h in args.h:
            t0 = time.time()
            recs, img = [], torch.zeros_like(tgt.volume)
            for i, (mesh, c) in enumerate(zip(ms, ph.value.tolist())):
                f, Tt = mem[i].double(), tgt.channels[i].double()
                if h == 0:
                    from gaussmrisim.fourier.polyhedral_cells import mesh_edges
                    a0 = float((f * Tt).sum() / (f * f).sum())
                    recs.append(mesh_edges(mesh, c * a0).float())
                    img += c * (a0 * f).float()
                    continue
                m = int(round(h / VOX))
                grid = Grid(origin, m * VOX, (math.ceil(nx / m), math.ceil(ny / m),
                                              math.ceil(nz / m)))
                a = fit_cell_values(mem[i], tgt.channels[i], m)
                recs.append(cell_edges(mesh, grid, (c * a).reshape(-1).to(dev)).float())
                up = a.repeat_interleave(m, 0).repeat_interleave(m, 1).repeat_interleave(m, 2)
                img += c * (up[:nz, :ny, :nx] * f).float()
                torch.cuda.empty_cache()
            rec = torch.cat(recs)
            del recs
            r = {"arm": "MC polyhedral, " + ("one constant per ellipsoid" if h == 0
                                             else f"cells {h:g} mm"),
                 "params": sum(3 * m.vertices.shape[0] + 1 for m in ms) if h == 0 else None,
                 "h_mm": h, "edges": int(rec.shape[0]), "bytes": int(rec.shape[0]) * 14 * 4,
                 "build_s": time.time() - t0, **iscore(img),
                 **kscore(lambda k: edge_ft(rec, k, "f32"))}
            r["time_s"], r["peak_gib"] = timed(lambda: edge_ft(rec, kt, "f32"), dev, reps=1)
            rows["arms"].append(r)
            np.save(root / f"cells_h{h:g}_slice.npy", img[z].numpy())
            print(f"h={h:g}  edges {r['edges']:,}  image {r['image_nrmse']:.4f}  ROI "
                  f"{r['roi_nrmse']:.4f}  " + "  ".join(
                      f"{b:g}Nyq {r[f'shell_{b:g}']:.4f}" for b in args.bands)
                  + f"  {args.n_time} k {r['time_s']:.2f} s", flush=True)
            del rec
            torch.cuda.empty_cache()
        (root / "cells.json").write_text(json.dumps(rows, indent=2))
        return 0

    if args.stage == "eval":
        rows = {"fov": [fov.fov_mm, fov.matrix, fov.center_mm], "bands": args.bands,
                "slice_z": z, "roi": roi, "arms": []}
        pts = fov.grid_coords(device=dev, dtype=torch.float64)
        vol = tgt.volume.to(dev)
        r = {"arm": "raster DFT of the voxels",
             **kscore(lambda k: raster_dft(vol, pts, k.float(), fov.voxel_volume_mm3))}
        rows["arms"].append(r)
        print(f"raster DFT  " + "  ".join(f"{b:g}Nyq {r[f'shell_{b:g}']:.4f}"
                                           for b in args.bands), flush=True)
        del vol, pts
        torch.cuda.empty_cache()
        kt = T.heldout(args.n_time, fov, r_max=1.0, seed=1, device=dev)
        lut = torch.zeros(len(ph.value) + 1, device=dev)
        lut[1:] = ph.value.to(dev)
        for fd in sorted(root.glob("fit_N*k*"), key=lambda p: (p.name[5:].split("k", 1)[1],
                                                              int(p.name[5:].split("k")[0]))):
            if not fd.is_dir():
                continue
            d = torch.load(fd / "mixture.pt", map_location=dev)
            mix = GaussianMixture3D(**{k: v for k, v in d.items() if k in
                                       ("mu", "log_scale", "quaternion", "mass",
                                        "dilation_mm")})
            coeff = lut[torch.load(fd / "tissue.pt", map_location=dev)]
            summ = json.loads((fd / "summary.json").read_text())
            img = render_full(dataclasses.replace(mix, mass=mix.mass * coeff),
                              dataclasses.replace(tgt, volume=tgt.volume.to(dev),
                                                  channels=None), cutoff_sigma=6.0).cpu()
            box = bool(summ.get("voxel_box"))
            tvw = float(summ.get("tv_weight", 0.0))
            r = {"arm": f"Gaussian N={mix.n:,}" + (", voxel-box fit" if box else "")
                 + (f", TV {tvw:g}" if tvw > 0 else "")
                 + ((f" x{summ.get('tv_samples', 8192) // 1024}k pts"
                     if summ.get("tv_kind", "mc") == "mc" else f" {summ['tv_kind']}")
                    if tvw > 0 else ""),
                 "voxel_box": box, "tv_weight": tvw, "tv_samples": summ.get("tv_samples", 8192),
                 "tv_kind": summ.get("tv_kind", "mc"),
                 "n": int(mix.n), "bytes": int(mix.n) * 44,
                 "fit_s": summ["fit_seconds"], **iscore(img),
                 **kscore(lambda k: kspace_cuda(mix, k.float(), coeff=coeff))}
            r["time_s"], r["peak_gib"] = timed(lambda: kspace_cuda(mix, kt.float(),
                                                                   coeff=coeff), dev)
            np.save(root / f"{fd.name}_slice.npy", img[z].numpy())
            rows["arms"].append(r)
            print(f"{r['arm']:22s} image {r['image_nrmse']:.4f}  ROI {r['roi_nrmse']:.4f}  "
                  + "  ".join(f"{b:g}Nyq {r[f'shell_{b:g}']:.4f}" for b in args.bands)
                  + f"  {args.n_time} k {r['time_s']:.3f} s  fit {r['fit_s']:.0f} s",
                  flush=True)
            del mix, coeff, img
            torch.cuda.empty_cache()
        np.save(root / "reference_slice.npy", tgt.volume[z].numpy())
        (root / "benchmark.json").write_text(json.dumps(rows, indent=2))
        print(f"wrote {root / 'benchmark.json'}")
        return 0

    raise SystemExit("Use experiments/figures.py shepplogan for the paper figure")


if __name__ == "__main__":
    raise SystemExit(main())
