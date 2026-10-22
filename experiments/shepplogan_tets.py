#!/usr/bin/env python
"""Ngo's polyhedral phantom extended to one constant per TETRAHEDRON, on the Shepp-Logan.

Same truth, voxels, marching-cubes surfaces, metrics and timing as
experiments/shepplogan_benchmark.py; only the interior model differs from the grid cells:
each MC surface is tetrahedralized (TetGen, graded, max volume ``--vol`` mm^3), each
tetrahedron gets a ridge-LS constant from the voxels, and the FT is the exact edge kernel
(interior faces weighted by the jump between their two tetrahedra).

    python experiments/shepplogan_tets.py --subject smooth --vol 64 8 1 --device cuda:1
"""
from __future__ import annotations

import argparse
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
from gaussmrisim.evaluation.raster import nrmse, shell_nrmse
from gaussmrisim.fourier import polyhedral_ft as P
from gaussmrisim.fourier import polyhedral_tets as PT
from gaussmrisim.fourier.polyhedral_cells import edge_ft
from benchmark_utils import timed  # noqa: E402
from shepplogan_benchmark import ROOT, make_fov, mc_meshes, phantom, target, view  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--subject", default="smooth", choices=["const", "smooth", "natural"])
    ap.add_argument("--vol", type=float, nargs="+", default=[64.0, 8.0, 1.0],
                    help="TetGen max tetrahedron volume, mm^3 (a cube of side h has h^3)")
    ap.add_argument("--min-ratio", type=float, default=2.0)
    ap.add_argument("--sub", type=int, default=3)
    ap.add_argument("--smooth", type=float, default=0.05,
                    help="neighbour (graph-Laplacian) weight; 0 = ridge to the global "
                         "constant only, as the grid cells use")
    ap.add_argument("--n-time", type=int, default=32768)
    ap.add_argument("--step", type=int, default=1,
                    help="marching-cubes step (voxels); 2 gives ~4x fewer surface triangles, "
                         "used for the parameter-matched arm")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    root = ROOT / args.subject
    ph = phantom()
    fov = make_fov(ph)
    tgt = target(ph, fov, "cpu", args.subject)
    tr = torch.load(root / "truth.pt")
    if args.step == 1:
        ms = mc_meshes(ph, fov, dev)
    else:
        from gaussmrisim.datasets.polyhedral import from_voxels
        frac = torch.load(ROOT / "voxels_0.5mm_ss4.pt")["channels"]   # as mc_meshes
        ms = [from_voxels(frac[i], fov, 0.5, name=f"E{i + 1}", step=args.step).to(dev)
              for i in range(len(ph.value))]
    sfx = "" if args.step == 1 else f"_st{args.step}"
    z, roi = view(fov)
    x0, x1, y0, y1 = roi
    nx, ny, nz = fov.matrix
    nvox = nx * ny * nz
    kt = T.heldout(args.n_time, fov, r_max=1.0, seed=1, device=dev)
    out = root / "tets.json"
    rows = json.loads(out.read_text()) if out.exists() else {"subject": args.subject,
                                                             "arms": []}
    for vol in args.vol:
        t0 = time.time()
        recs, img, ntet, check = [], torch.zeros(nvox, dtype=torch.float64, device=dev), 0, []
        nvert = 0
        no_quality, cg = [], []
        for i, (mesh, c) in enumerate(zip(ms, ph.value.tolist())):
            V, Tt = PT.tetrahedralize(mesh, vol, args.min_ratio)
            V, Tt = V.to(dev), Tt.to(dev)
            vsum = float(PT.tet_volumes(V, Tt).sum())
            check.append(abs(vsum / mesh.signed_volume() - 1))
            r_, c_, v_ = PT.tet_voxel_pairs(V, Tt, fov, sub=args.sub)
            a = PT.fit_tet_values(r_, c_, v_, Tt.shape[0],
                                  tgt.channels[i].reshape(-1).to(dev),
                                  **({"adj": PT.tet_adjacency(Tt), "ridge": 1e-3,
                                      "smooth": args.smooth} if args.smooth > 0 else {}))
            cg.append(PT.fit_tet_values.last)
            img += c * PT.tet_image(r_, c_, v_, a, nvox)
            recs.append(PT.tet_edges(V, Tt, c * a).float())
            if i == 0:          # the largest surface: interior faces must cancel exactly
                kc = tr["k"][0.5][:512].to(dev)
                one = edge_ft(PT.tet_edges(V, Tt, torch.ones_like(a)), kc, "double")
                unc = P.polyhedral_ft(P.pack([mesh.to(dev)], [1.0], device=dev), kc,
                                      "double")
                check_ones = float((one - unc).norm() / unc.norm())
            ntet += int(Tt.shape[0])
            nvert += int(V.shape[0])
            if not PT.tetrahedralize.last_quality:
                no_quality.append(i + 1)
            print(f"  vol {vol:g}: ellipsoid {i + 1} {Tt.shape[0]:,} tets "
                  f"(volume err {check[-1]:.1e}, quality "
                  f"{PT.tetrahedralize.last_quality})", flush=True)
            del V, Tt, r_, c_, v_, a
            torch.cuda.empty_cache()
        rec = torch.cat(recs)
        del recs
        build = time.time() - t0
        img = img.float().cpu().reshape(nz, ny, nx)
        ref = tgt.volume
        s, rs = img[z].numpy(), ref[z].numpy()
        r = {"arm": f"MC polyhedral{' (MC step ' + str(args.step) + ')' if args.step > 1 else ''}, tets <= {vol:g} mm^3"
                    + (", neighbour prior" if args.smooth > 0 else ""), "max_volume_mm3": vol,
             "smooth": args.smooth, "step": args.step, "verts": nvert,
             "params": ntet + 3 * nvert,
             "tets": ntet, "edges": int(rec.shape[0]), "bytes": int(rec.shape[0]) * 14 * 4,
             "build_s": build, "max_volume_err": max(check), "all_ones_vs_uncut": check_ones,
             "no_quality_refinement": no_quality,
             "cg_max_rel_residual": max(x["rel_residual"] for x in cg),
             "cg_max_iters": max(x["iters"] for x in cg),
             "image_nrmse": float((img - ref).norm() / ref.norm()),
             "image_note": f"model voxelized by {args.sub}^3 sub-sampling",
             "roi_nrmse": float(np.linalg.norm(s[y0:y1, x0:x1] - rs[y0:y1, x0:x1])
                                / np.linalg.norm(rs[y0:y1, x0:x1]))}
        for b in tr["bands"]:
            k = tr["k"][b].to(dev)
            sharp = tr["sharp"][b].to(dev)
            g = edge_ft(rec, k, "f32").to(torch.complex128)
            sh, _ = shell_nrmse(g, sharp, k, float(b), shells=16, fov=fov)
            r[f"shell_{b:g}"] = float(sum(sh) / len(sh))
            r[f"nrmse_{b:g}"] = float(nrmse(g, sharp))
        r["time_s"], r["peak_gib"] = timed(lambda: edge_ft(rec, kt, "f32"), dev, reps=1)
        np.save(root / f"tets_v{vol:g}_s{args.smooth:g}{sfx}_slice.npy", s)
        rows["arms"] = [x for x in rows["arms"] if not (x["max_volume_mm3"] == vol and
                        x.get("smooth", 0.0) == args.smooth and
                        x.get("step", 1) == args.step)] + [r]
        out.write_text(json.dumps(rows, indent=2))
        print(f"vol={vol:g}  tets {ntet:,}  edges {r['edges']:,}  image {r['image_nrmse']:.4f}"
              f"  ROI {r['roi_nrmse']:.4f}  " + "  ".join(
                  f"{b:g}Nyq {r[f'shell_{b:g}']:.4f}" for b in tr["bands"])
              + f"  {args.n_time} k {r['time_s']:.2f} s  (all-ones {check_ones:.1e})",
              flush=True)
        del rec
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
