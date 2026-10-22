#!/usr/bin/env python
"""Fit one Gaussian mixture per BrainWeb tissue at 0.5 mm, and report the allocation.

The object produced here is the phantom of the SNAKE-style forward model: one
concatenated mixture plus a tissue index, so contrast is a gather and T2* decay is a
per-primitive scalar (see docs/design/2026-09-07_snake_style_forward_model.md).

The allocation rule is the thing under test. ``--rule`` picks how the primitive budget
is split across tissues, and all the candidates are functions of **geometry only** --
weighting by a sequence's signal would tie the fitted geometry to one sequence and
destroy the property the architecture exists for.

    python experiments/fit_tissues_brainweb.py --device cuda:0 --total 1500000 --rule area
    python experiments/fit_tissues_brainweb.py --drivers-only          # just the table
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaussmrisim.datasets.brainweb import load_fuzzy
from gaussmrisim.fitting.fitter import FitConfig
from gaussmrisim.fitting.tissue_fit import allocate, fit_tissues, tissue_drivers
from gaussmrisim.physics.signal_model import gre_from_table
from gaussmrisim.physics.tissue_props import LABELS



def shard_of(budget: torch.Tensor, i: int, n: int) -> list:
    """Which tissue indices shard ``i`` of ``n`` should fit.

    Greedy longest-processing-time: sort by primitive count descending and give each
    tissue to the currently lightest shard. Round-robin would put grey matter and
    muscle/skin (the two largest) on the same GPU whenever the count is even, which is
    the whole imbalance one is trying to avoid. Cost tracks primitives, not voxels --
    measured, see the design note -- so the primitive count is the right load estimate.
    """
    load = [0] * int(n)
    out = [[] for _ in range(int(n))]
    for idx in torch.argsort(budget, descending=True).tolist():
        j = min(range(int(n)), key=lambda k: load[k])
        out[j].append(idx)
        load[j] += int(budget[idx])
    return sorted(out[int(i)])


def merge_shards(root: Path) -> int:
    """Concatenate the per-shard mixtures into the one object the engine consumes."""
    shards = sorted(root.glob("shard*"))
    if not shards:
        raise FileNotFoundError(f"no shards under {root}")
    mus, lss, qs, ms, ts, reps = [], [], [], [], [], []
    dil = None
    for sd in shards:
        d = torch.load(sd / "mixture.pt", map_location="cpu")
        mus.append(d["mu"]); lss.append(d["log_scale"])
        qs.append(d["quaternion"]); ms.append(d["mass"])
        dil = d.get("dilation_mm", dil)
        ts.append(torch.load(sd / "tissue.pt", map_location="cpu"))
        reps.append(json.loads((sd / "summary.json").read_text()))
    mix = {"mu": torch.cat(mus), "log_scale": torch.cat(lss),
           "quaternion": torch.cat(qs), "mass": torch.cat(ms), "dilation_mm": dil}
    tissue = torch.cat(ts)
    # sort by tissue so the concatenated object is grouped, which makes a per-tissue
    # coefficient gather contiguous and keeps any later per-tissue slice trivial
    order = torch.argsort(tissue, stable=True)
    mix = {k: (v[order] if torch.is_tensor(v) else v) for k, v in mix.items()}
    tissue = tissue[order]
    torch.save(mix, root / "mixture.pt")
    torch.save(tissue, root / "tissue.pt")
    rows = [r for rep in reps for r in rep["tissues"]]
    rows.sort(key=lambda r: r["label"])
    (root / "summary.json").write_text(json.dumps(
        {"n_primitives": int(tissue.numel()), "n_shards": len(shards),
         "seconds": max(r["seconds"] for r in reps),
         "shard_seconds": [r["seconds"] for r in reps],
         "peak_gib": max(r.get("peak_gib", float("nan")) for r in reps),
         "steps": reps[0].get("steps"), "rule": reps[0].get("rule"),
         "device": reps[0].get("device"),
         "labels": sorted({int(r["label"]) for r in rows}),
         "tissues": rows}, indent=2))
    print(f"merged {len(shards)} shards -> {tissue.numel():,} primitives in {root}")
    for r in rows:
        print(f"  tissue {r['label']:2d}  N={r['n']:>8,}  img {r['image_nrmse']:.4f}  "
              f"1/4Nyq {r.get('fourier_nrmse_0.25', float('nan')):.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--subject", type=int, default=4)
    ap.add_argument("--total", type=int, default=1_500_000)
    ap.add_argument("--rule", default="area",
                    choices=["area", "volume", "sqrt_area_volume"])
    ap.add_argument("--floor", type=int, default=2000)
    ap.add_argument("--budget-file", default=None,
                    help="a .pt with {labels, budget}, e.g. the water-filling rule; "
                         "overrides --rule for the fit but not for the printed table")
    ap.add_argument("--steps", type=int, default=1500,
                    help="1500 is converged for full-volume tissue fits: 6000 measured "
                         "0.0123 at 1/4 Nyquist against 0.0122, for 4x the time")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="fit only shard I of N, balancing primitive count across "
                         "shards; run one process per GPU and then --merge")
    ap.add_argument("--merge", action="store_true",
                    help="concatenate the shards written for this tag/rule")
    ap.add_argument("--cutoff-sigma", type=float, default=4.0)
    ap.add_argument("--thresh", type=float, default=0.01)
    ap.add_argument("--drivers-only", action="store_true")
    ap.add_argument("--n-k", type=int, default=4096)
    ap.add_argument("--only", type=int, nargs="+", default=None,
                    help="fit only these tissue labels (for a pilot)")
    ap.add_argument("--thin-vox", type=float, default=0.0,
                    help="tissues with sheet thickness 2V/A below this (voxels) get the "
                         "sheet init + fixed-budget relocation; 0 = off (default)")
    ap.add_argument("--thin-along", type=float, default=2.0)
    ap.add_argument("--thin-n-from", default=None,
                    help="a merged fit dir: give each thin tissue exactly the final N it "
                         "reached there (equal-N comparison against a split/clone run)")
    ap.add_argument("--tag", default="tissues")
    ap.add_argument("--out", default="outputs/fits")
    args = ap.parse_args()

    tgt = load_fuzzy(args.subject)
    labels = list(tgt.channel_labels)
    print(f"{tgt.name}: {tgt.fov.matrix} @ {tgt.fov.voxel_mm[0]:.2f} mm, "
          f"centre {tuple(round(c, 2) for c in tgt.fov.center_mm)}, "
          f"{len(labels)} tissues", flush=True)

    t0 = time.time()
    dr = tissue_drivers(tgt, args.thresh, device=args.device)
    print(f"drivers in {time.time() - t0:.1f} s")

    a_gre = gre_from_table("1.5T", tr_ms=50.0, te_ms=30.0,
                           flip_deg=15.0)[list(labels)]

    alloc = {r: allocate(dr, args.total, r, args.floor)
             for r in ("area", "volume", "sqrt_area_volume")}
    print(f"\n{'idx':>3} {'tissue':14s} {'Mvox':>7} {'area':>10} {'a(GRE)':>8}"
          f" {'N_area':>9} {'N_vol':>9} {'N_sqrt':>9}")
    for i, lab in enumerate(labels):
        print(f"{lab:3d} {LABELS[lab]:14s} {dr['volume'][i] / 1e6:7.3f} "
              f"{dr['area'][i]:10.3e} {a_gre[i]:8.5f} "
              f"{int(alloc['area'][i]):9,} {int(alloc['volume'][i]):9,} "
              f"{int(alloc['sqrt_area_volume'][i]):9,}")
    for r, v in alloc.items():
        print(f"  total {r:18s} {int(v.sum()):,}")

    # How much of the budget goes to tissues invisible in this sequence -- reported,
    # never used to allocate.
    dead = a_gre < 1e-6
    if bool(dead.any()):
        frac = float(alloc[args.rule][dead].sum()) / float(alloc[args.rule].sum())
        names = ", ".join(LABELS[labels[i]] for i in range(len(labels)) if dead[i])
        print(f"  {100 * frac:.1f}% of the '{args.rule}' budget goes to tissues with "
              f"zero signal at TR=50/TE=30/FA=15 ({names})")

    if args.drivers_only:
        return 0

    budget = alloc[args.rule]
    if args.budget_file:
        bf = json.loads(Path(args.budget_file).read_text()) if str(args.budget_file).endswith(".json") else torch.load(args.budget_file, map_location="cpu")
        order = {int(l): i for i, l in enumerate(bf["labels"])}
        budget = torch.tensor([int(bf["budget"][order[int(l)]]) for l in labels],
                              dtype=torch.long)
        print(f"\nbudget from {args.budget_file}: {int(budget.sum()):,} primitives")
    keep = list(range(len(labels)))

    out_root = Path(args.out) / f"{tgt.name}_{args.tag}_{args.rule}"
    if args.merge:
        return merge_shards(out_root)

    if args.shard is not None:
        si, sn = (int(x) for x in args.shard.split("/"))
        keep = shard_of(budget, si, sn)
        tgt.channels = tgt.channels[keep]
        tgt.channel_labels = tuple(labels[i] for i in keep)
        budget = budget[keep]
        print(f"\nshard {si}/{sn}: tissues {tgt.channel_labels}, "
              f"{int(budget.sum()):,} primitives")
    elif args.only is not None:
        keep = [i for i, l in enumerate(labels) if l in set(args.only)]
        tgt.channels = tgt.channels[keep]
        tgt.channel_labels = tuple(labels[i] for i in keep)
        budget = budget[keep]
        print(f"\npilot: fitting only {tgt.channel_labels}")

    cfg = FitConfig(steps=args.steps, cutoff_sigma=args.cutoff_sigma)
    print(f"\nfitting {len(budget)} tissues, {int(budget.sum()):,} primitives, "
          f"{args.steps} steps", flush=True)
    t0 = time.time()
    thin = ({"vox": args.thin_vox, "along": args.thin_along} if args.thin_vox > 0
            else None)
    if thin and args.thin_n_from:
        ref = json.loads((Path(args.thin_n_from) / "summary.json").read_text())
        thin["n_exact"] = {int(r["label"]): int(r["n"]) for r in ref["tissues"]}
    mix, tissue, rep = fit_tissues(tgt, budget, cfg, device=args.device,
                                   thresh=args.thresh, n_k=args.n_k, thin=thin)
    total_s = time.time() - t0
    peak_gib = torch.cuda.max_memory_allocated(args.device) / 2**30
    print(f"total {total_s:.1f} s, N = {mix.n:,}, peak {peak_gib:.2f} GiB")

    out = out_root if args.shard is None else out_root / f"shard{args.shard.split('/')[0]}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({**{k: v for k, v in mix.__dict__.items()}, }, out / "mixture.pt")
    torch.save(tissue, out / "tissue.pt")
    (out / "summary.json").write_text(json.dumps(
        {"subject": args.subject, "rule": args.rule, "total_requested": args.total,
         "n_primitives": int(mix.n), "steps": args.steps,
         "cutoff_sigma": args.cutoff_sigma, "seconds": total_s, "thin": thin,
         "peak_gib": peak_gib, "device": torch.cuda.get_device_name(args.device),
         "labels": [int(x) for x in tgt.channel_labels],
         "budget": [int(x) for x in budget],
         "volume_mvox": [float(x) / 1e6 for x in dr["volume"][keep]],
         "area": [float(x) for x in dr["area"][keep]],
         **rep}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
