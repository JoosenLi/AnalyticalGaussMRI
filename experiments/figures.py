#!/usr/bin/env python
"""Generate paper Figures 3-6 and the Shepp-Logan table from experiment outputs."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sl_psnr_offset(subject):
    """PSNR = offset - NMSE(dB) for a Shepp-Logan subject: 10 log10(peak^2 / mean(f^2))."""
    from shepplogan_benchmark import make_fov, phantom, target
    ph = phantom()
    v = target(ph, make_fov(ph), os.environ.get("GMRISIM_DEVICE", "cuda:0"), subject).volume.double()
    return 10 * math.log10(float(v.max()) ** 2 / float(v.square().mean()))


def db(nrmse):
    """NMSE in dB, 10 log10(||e||^2 / ||ref||^2) = 20 log10(NRMSE)."""
    return 20 * np.log10(np.asarray(nrmse, dtype=float))

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs/figures"
GEN = ROOT / "outputs/tables"
SL = ROOT / "outputs/shepplogan_bench"
COL_W = 3.39                                          # IEEE column width, inches
TXT_W = 7.0

plt.rcParams.update({
    "font.size": 7, "font.family": "serif", "mathtext.fontset": "cm",
    "axes.linewidth": 0.5, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "legend.fontsize": 6, "legend.frameon": False, "lines.linewidth": 1.0,
    "figure.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42, "ps.fonttype": 42})          # embedded TrueType, no Type 3 fonts

C_TRUE, C_ANA, C_GRID, C_GRID2 = "0.55", "#1b7837", "#d95f02", "#7570b3"


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    print(f"-> {FIG / name}.pdf")


def inset(ax, img, box, loc="upper right", size="44%", corners=(2, 4), **kw):
    """Zoomed ROI ``box = (x0, x1, y0, y1)`` (pixel coords of ``img``) in a corner;
    ``corners`` are the two connected corners (1 upper right, 2 upper left, 3 lower left,
    4 lower right)."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    ins = inset_axes(ax, width=size, height=size, loc=loc, borderpad=0.15)
    ins.imshow(img, origin="lower", interpolation="nearest", **kw)
    x0, x1, y0, y1 = box
    ins.set_xlim(x0, x1); ins.set_ylim(y0, y1)
    ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_edgecolor("#f2c14e"); sp.set_linewidth(0.6)
    mark_inset(ax, ins, loc1=corners[0], loc2=corners[1], fc="none", ec="#f2c14e", lw=0.4)
    return ins


def label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=7.5, fontweight="bold",
            va="bottom", ha="right")


# --------------------------------------------------------------------------- Fig. 1






# --------------------------------------------------------------------------- Fig. 3


VOX_SL = 0.5
GAUSS_FIT = "fit_N300k_tv0.0001_floor200"  # area allocation, >= 200 primitives per region
GEO_WIN = (0.6, 4.6, -56.5, -52.5)         # mm: rim of the 2 mm sphere, gap, rim of its neighbour
GEO_COL = os.environ.get("GEO_COL", "#4dd2ff")


def geometry_overlays(subjects):
    """What each model is, cut by the slice plane z = 0 inside GEO_WIN (mm, 2D)."""
    import geometry as PG
    from shepplogan_benchmark import make_fov, mc_meshes, phantom
    ph = phantom()
    fov = make_fov(ph)
    x0, x1, y0, y1 = GEO_WIN
    pad = 1.0
    ox, oy = (c - 0.5 * f for c, f in zip(fov.center_mm[:2], fov.fov_mm[:2]))
    gx = ox + VOX_SL * np.arange(math.floor((x0 - pad - ox) / VOX_SL),
                                 math.ceil((x1 + pad - ox) / VOX_SL) + 1)
    gy = oy + VOX_SL * np.arange(math.floor((y0 - pad - oy) / VOX_SL),
                                 math.ceil((y1 + pad - oy) / VOX_SL) + 1)
    vox = [[(x, gy[0]), (x, gy[-1])] for x in gx] + [[(gx[0], y), (gx[-1], y)] for y in gy]
    out = {"voxels": ("lines", vox)}
    ms = mc_meshes(ph, fov, "cpu")
    segs = []
    for m in ms.values() if isinstance(ms, dict) else ms:
        segs += PG.plane_segments_tri(m.vertices.double().numpy(), m.faces.numpy(), PG.Z0)
    out["cells_h0"] = ("lines", segs)
    win = (x0 - pad, x1 + pad, y0 - pad, y1 + pad)
    small = [1, 7, 8, 9]
    out["tets_v5_s0"] = ("polys", PG.tets(1, 5.0, small, win))
    out["tets_v8_s0_st2"] = ("polys", PG.tets(2, 8.0, small, win))
    for s in subjects:
        polys, w, n = PG.gaussian_ellipses(s, win, GAUSS_FIT)
        out[(GAUSS_FIT, s)] = ("gauss", (polys, w))
    return out


def geo_inset(ax, parent, img, box, overlay, px, size="53%", connect=True, corners=(1, 4),
              **kw):
    """Second zoom, ``box`` inside the ROI inset ``parent``: the model's own geometry."""
    from matplotlib.collections import LineCollection, PolyCollection
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    ins = inset_axes(ax, width=size, height=size, loc="upper left", borderpad=0.15)
    ins.imshow(img, origin="lower", interpolation="nearest", **kw)
    x0, x1, y0, y1 = box
    ins.set_xlim(x0, x1); ins.set_ylim(y0, y1)
    ins.set_xticks([]); ins.set_yticks([])
    kind, data = overlay
    if kind == "lines":
        ins.add_collection(LineCollection([px(np.asarray(s)) for s in data], colors=GEO_COL,
                                          linewidths=0.35))
    elif kind == "polys":
        ins.add_collection(PolyCollection([px(p) for p in data], facecolors="none",
                                          edgecolors=GEO_COL, linewidths=0.25))
    else:
        polys, w = data
        o = [k for k in np.argsort(w) if w[k] > 0.5]       # primitives the plane cuts near their centre
        ins.add_collection(PolyCollection(
            [px(polys[k]) for k in o], facecolors="none", linewidths=0.3,
            edgecolors=[(*matplotlib.colors.to_rgb(GEO_COL), float(w[k] ** 2)) for k in o]))
    for sp in ins.spines.values():
        sp.set_edgecolor(GEO_COL); sp.set_linewidth(0.6)
    if connect:
        mark_inset(parent, ins, loc1=corners[0], loc2=corners[1], fc="none", ec=GEO_COL,
                   lw=0.4)
    else:                                  # the window marked by colour only, no lines
        from matplotlib.patches import Rectangle
        x0, x1, y0, y1 = box
        parent.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=GEO_COL,
                                   lw=0.6))
    return ins


def shepplogan():
    subj = {"const": "constant", "natural": "smooth"}
    rows = {}
    for s in subj:
        b = json.loads((SL / s / "benchmark.json").read_text())
        c = json.loads((SL / s / "cells.json").read_text())
        t = json.loads((SL / s / "tets.json").read_text())["arms"]
        poly = next(a for a in c["arms"] if a["h_mm"] == 0)
        # one least-squares constant per tetrahedron, ridge to the region constant only
        tet = next(a for a in t if a["max_volume_mm3"] == 5.0 and a.get("smooth") == 0.0
                   and a.get("step", 1) == 1)
        tm = next(a for a in t if a["max_volume_mm3"] == 8.0 and a.get("smooth") == 0.0
                  and a.get("step", 1) == 2)
        g = next(a for a in b["arms"] if a.get("dir") == GAUSS_FIT)
        g["params"] = 11 * g["n"]
        rows[s] = {"roi": b["roi"], "arms": [
            ("polyhedral (Ngo)", "cells_h0", poly),
            ("tetrahedral", "tets_v5_s0", tet),
            ("tetrahedral, matched", "tets_v8_s0_st2", tm),
            ("Gaussian (ours)", GAUSS_FIT, g)]}
    geo = geometry_overlays(subj)
    off = {s: sl_psnr_offset(s) for s in subj}
    from shepplogan_benchmark import make_fov, phantom
    fov = make_fov(phantom())
    ox, oy = (c - 0.5 * f for c, f in zip(fov.center_mm[:2], fov.fov_mm[:2]))
    # single column, smooth subject: every panel is the model's image with two zooms,
    # the ROI error map (upper right) and, inside the ROI, the model's own geometry
    s = "natural"
    ref = np.load(SL / s / "reference_slice.npy")
    ys, xs = np.nonzero(ref > 0.05)
    y0, y1, x0, x1 = ys.min() - 4, ys.max() + 5, xs.min() - 4, xs.max() + 5
    inner = ref[(ref > 0.5) & (ref < 1.8)]
    lo, hi = np.percentile(inner, [0.5, 99.5])
    lo, hi = lo - 0.1, hi + 0.1
    vmax = float(np.percentile(ref, 99.5))
    rx0, rx1, ry0, ry1 = rows[s]["roi"]
    cx, cy = 0.5 * (rx0 + rx1) - x0, 0.5 * (ry0 + ry1) - y0 - 4
    box = (cx - 24, cx + 24, cy - 24, cy + 24)              # 24 mm around the small ellipsoids
    px = lambda p: np.stack([(p[..., 0] - ox) / VOX_SL - 0.5 - x0,       # mm -> crop px
                             (p[..., 1] - oy) / VOX_SL - 0.5 - y0], -1)
    box2 = tuple(px(np.array([[GEO_WIN[0], GEO_WIN[2]], [GEO_WIN[1], GEO_WIN[3]]])).T.ravel())
    gray = dict(cmap="gray", vmin=lo, vmax=hi)
    EMAX = 0.02                                             # narrow error range, +-2 %
    ekw = dict(cmap="RdBu_r", vmin=-EMAX, vmax=EMAX)
    fig, ax = plt.subplots(2, 3, figsize=(COL_W, 2.5),
                           gridspec_kw={"wspace": 0.04, "hspace": 0.24})
    panels = [("reference (voxels)", None, None, None)] + list(rows[s]["arms"])
    im = None
    for a, (name, stem, arm) in zip(ax.flat, [(p[0], p[1], p[2]) if len(p) == 3 else
                                             (p[0], None, None) for p in panels]):
        img = ref if stem is None else np.load(SL / s / f"{stem}_slice.npy")
        a.imshow(img[y0:y1, x0:x1], origin="lower", **gray)
        if stem is None:
            ins = inset(a, img[y0:y1, x0:x1], box, size="46%", **gray)
            g, title = geo["voxels"], name
        else:
            d = ((img - ref) / vmax)[y0:y1, x0:x1]
            ins = inset(a, d, box, size="46%", **ekw)
            im = ins.images[0]
            g = geo[stem] if stem in geo else geo[(stem, s)]
            title = (f"{name}\n{arm['params'] / 1e6:.2f} M, "
                     f"{off[s] - db(arm['image_nrmse']):.1f} dB")
        # lines from the box's upper-left and lower-right corners (matplotlib locs 2, 4)
        geo_inset(a, a, img[y0:y1, x0:x1], box2, g, px, size="46%", corners=(2, 4), **gray)
        a.set_title(title, fontsize=5.8, linespacing=1.05, pad=2,
                    fontweight="bold" if stem == GAUSS_FIT else "normal")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_linewidth(0.3)
    key = ax[1, 2]
    key.axis("off")
    cax = key.inset_axes([0.04, 0.22, 0.07, 0.62])
    cb = fig.colorbar(im, cax=cax, ticks=[-EMAX, 0, EMAX])
    cb.ax.set_yticklabels([f"$-${100 * EMAX:g}%", "0", f"{100 * EMAX:g}%"], fontsize=5.3)
    cb.solids.set_rasterized(False)
    cb.outline.set_linewidth(0.3)
    key.text(0.44, 0.84, "upper right:\nerror, 24 mm ROI", fontsize=5.4, va="top",
             transform=key.transAxes, color="#b8860b", linespacing=1.1)
    key.text(0.44, 0.46, "upper left:\ngeometry, 4 mm", fontsize=5.4, va="top",
             transform=key.transAxes, color="#1f8fbf", linespacing=1.1)
    save(fig, "shepplogan")

    m = {}
    for name, _, _ in rows["const"]["arms"]:
        a = {s: next(x for n, _, x in rows[s]["arms"] if n == name) for s in subj}
        m[name] = {"params": a["natural"]["params"],
                   "img_const": 100 * a["const"]["image_nrmse"],
                   "img_nat": 100 * a["natural"]["image_nrmse"],
                   "psnr_const": off["const"] - float(db(a["const"]["image_nrmse"])),
                   "psnr_nat": off["natural"] - float(db(a["natural"]["image_nrmse"])),
                   "mib": a["natural"]["bytes"] / 2**20, "t": a["natural"]["time_s"],
                   "peak": a["natural"]["peak_gib"]}
    (GEN / "shepplogan.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=1))
    return m


# --------------------------------------------------------------------------- Fig. 4






# --------------------------------------------------------------------------- Fig. 5
SNR = 100.0                   # per-sample signal-to-noise power (20 dB), SNAKE's definition




# --------------------------------------------------------------------------- Fig. 6
MOTION_ARROW = {"axial": ((0.30, 0.84), (-1, -1)),      # scalp/skull band, upper left
                "coronal": ((0.60, 0.68), (1, -1))}      # just inside the skull


def motion(show=4, vox=1.0):
    """Per-TR rigid motion on the BrainWeb Gaussian phantom: the analytical acquisition,
    image-based simulation (trilinear resampling into each pose), and their difference,
    in an axial and (when the run saved the volumes) a coronal slice."""
    import torch
    from scipy.ndimage import uniform_filter
    root = ROOT / "outputs/sim/motion_aphabamas"             # one trajectory, 1 mm
    if vox != 1.0:
        root = root / f"vox{vox:g}"
    rec = json.loads((root / f"traj{show}.json").read_text())
    sl = torch.load(root / f"traj{show}_slices.pt")
    gt = sl["ground_truth"].numpy()
    ys, xs = np.nonzero(gt > 0.05 * gt.max())
    cy, cx = slice(ys.min() - 3, ys.max() + 4), slice(xs.min() - 3, xs.max() + 4)
    vmax = np.percentile(gt[cy, cx], 99.7)
    rows = [("axial", gt[cy, cx], sl["image_based"].numpy()[cy, cx])]
    vfile = root / f"traj{show}_volumes.pt"
    if vfile.exists():                            # coronal slice from the whole volumes
        vol = {n: v.float().numpy() for n, v in torch.load(vfile).items()}
        g3 = vol["ground_truth"]
        zs = np.nonzero((g3 > 0.05 * g3.max()).any((1, 2)))[0]
        cz = slice(max(zs.min() - 3, 0), zs.max() + 4)
        yc = g3.shape[1] // 2
        rows.append(("coronal", g3[:, yc, :][cz, cx], vol["image_based"][:, yc, :][cz, cx]))
    ratios = [r[1].shape[0] / r[1].shape[1] for r in rows]
    fig, axs = plt.subplots(len(rows), 3, figsize=(COL_W, 1.42 * sum(ratios) / ratios[0]),
                            gridspec_kw={"wspace": 0.05, "hspace": 0.04,
                                         "height_ratios": ratios}, squeeze=False)
    gray = dict(cmap="gray", vmin=0, vmax=vmax)
    R_ = int(round(20 / vox))                     # half-width of a 40 mm box, in voxels
    for i, (ax, (name, g2, ib2)) in enumerate(zip(axs, rows)):
        # 40 mm ROI where the error energy is largest: argmax of the 40-pixel box mean of
        # the squared difference over the head
        e2 = uniform_filter((ib2 - g2) ** 2, 2 * R_)
        e2[uniform_filter((g2 > 0.1 * vmax).astype(float), 2 * R_) < 0.6] = 0
        keep = np.zeros_like(e2, dtype=bool)
        keep[R_:-R_, R_:-R_] = True               # the whole box inside the crop
        if name == "coronal":                     # skull and cortex, not the neck at the
            keep[:int(0.4 * len(keep))] = False   # FOV's inferior edge
        e2[~keep] = 0
        ry, rx = np.unravel_index(np.argmax(e2), e2.shape)
        box = (rx - R_, rx + R_, ry - R_, ry + R_)
        corner = "lower left" if rx > g2.shape[1] / 2 else "lower right"
        # connect the two ROI corners that do not face the inset
        lines = (2, 4) if corner == "lower left" else (1, 3)
        err = (ib2 - g2) / vmax
        panels = [(g2, gray, "analytical"), (ib2, gray, "image-based"),
                  (err, dict(cmap="RdBu_r", vmin=-0.08, vmax=0.08), "difference")]
        # the arrow: an edge blurred by resampling, picked by eye in each ROI (fractions of
        # the box, and the direction the arrow comes from)
        (fx, fy), (sx, sy) = MOTION_ARROW[name]
        ax_, ay = rx - R_ + 2 * R_ * fx, ry - R_ + 2 * R_ * fy
        d = 0.35 * R_                              # arrow length, voxels
        for a, (img, kw, title) in zip(ax, panels):
            a.imshow(img, origin="lower", interpolation="nearest", **kw)
            ins = inset(a, img, box, loc=corner, size="46%", corners=lines, **kw)
            ins.annotate("", xy=(ax_, ay), xytext=(ax_ + sx * d, ay + sy * d),
                         arrowprops=dict(arrowstyle="-|>", color="#4dd2ff", lw=0.9,
                                         mutation_scale=5, shrinkA=0, shrinkB=1))
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(title, fontsize=6.3, pad=2)
            for sp in a.spines.values():
                sp.set_linewidth(0.3)
        if len(rows) > 1:
            ax[0].set_ylabel(name, fontsize=6.3, labelpad=1)
    axs[0, 2].text(0.97, 0.03, f"{rec['image_based']['psnr_db']:.1f} dB",
                   transform=axs[0, 2].transAxes, fontsize=6.2, va="bottom", ha="right")
    save(fig, "motion")
    summ = {"vox_mm": vox, "trajectory": show + 1,
            "max_rot_deg": rec["max_rot_deg"], "max_trans_mm": rec["max_trans_mm"],
            "psnr_image_based": rec["image_based"]["psnr_db"],
            "ssim_image_based": rec["image_based"]["ssim"],
            "routes_agree": rec["routes_agree"], "gt_seconds": rec["gt_seconds"]}
    (GEN / "motion.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))
    return summ


# --------------------------------------------------------------------------- resolution


# --------------------------------------------------------------------------- noise floor
def noisefloor():
    """BrainWeb, fully sampled 1 mm Cartesian and 3D radial: is the grid error hidden by noise?

    The grid arm is the fair voxel simulator: voxel AVERAGES of the mixture (supersampled,
    4^3 at 1 mm, 2^3 at 0.5 mm; converged to 0.2 dB at 1 mm) treated as points and
    transformed by NUFFT (experiments/aliasing_box_baseline.py). Every data set gets the same
    receiver noise (one sigma, from the Cartesian mean power) and the same reconstruction:
    inverse DFT for Cartesian, 30 LSQR iterations from the density-compensated adjoint
    for radial (aliasing_cg_radial.py).
    """
    root = ROOT / "outputs/sim/aliasing_recon_noise"
    arms = {("cartesian", 1.0): "box4-dirac", ("radial", 1.0): "box4-dirac",
            ("cartesian", 0.5): "box2-dirac", ("radial", 0.5): "box2-dirac"}
    # Cartesian: inverse DFT (exact inverse); radial: 30 LSQR iterations (mri-nufft) from
    # the density-compensated adjoint (experiments/aliasing_cg_radial.py --fixed --dc-init)
    B = {(t, h): json.loads((root / (f"box_baseline_{t}_h{h:g}"
                                     + ("_lsqr_fixed_dcinit" if t == "radial" else "")
                                     + ".json")).read_text())
         for t, h in arms}
    assert all(B[("radial", h)]["iterations"] == 30 for h in (1.0, 0.5))
    fig, b = plt.subplots(figsize=(COL_W, 1.95))
    SMAX = 1e4                         # axis up to 40 dB acquisition SNR
    snr = np.array(sorted(float(x) for x in B[("cartesian", 1.0)]["analytic"] if float(x) <= SMAX))
    sdb = 10 * np.log10(snr)           # acquisition SNR (power ratio, Cartesian sigma), dB
    val = lambda rec: np.array([rec[k] for k in sorted(rec, key=float) if float(k) <= SMAX])
    for tn, mk, ls, lab in (("cartesian", "o", "-", "Cartesian"),
                            ("radial", "^", (0, (3, 1.5)), "radial")):
        b.plot(sdb, val(B[(tn, 1.0)]["analytic"]), color=C_ANA, ls=ls, lw=1.1, marker=mk,
               ms=2.2, label=f"{lab}, analytic", zorder=3)
        for hh, col in ((1.0, C_GRID), (0.5, C_GRID2)):
            b.plot(sdb, val(B[(tn, hh)]["arms"][arms[(tn, hh)]]["psnr_db"]), color=col,
                   ls=ls, lw=0.9, marker=mk, ms=2.2, label=f"{lab}, {hh:g} mm grid")
    b.set_xlim(sdb[0] - 1.5, sdb[-1] + 1.5)
    b.set_xticks([10, 20, 30, 40])
    # indicative scanner values on the same scale (1 mm MPRAGE, multi-channel arrays;
    # approximate, colleague's estimate): ~19 dB at 3 T and ~27 dB at 7 T
    for xv, lab_ in ((19.0, r"$\approx$3T"), (27.0, r"$\approx$7T")):
        b.axvline(xv, color="0.55", lw=0.6, ls=(0, (2, 2)), zorder=1)
        b.text(xv + 0.5, 0.04, lab_, fontsize=5.4, color="0.4",
               transform=b.get_xaxis_transform())
    lo = min(B[k]["psnr_offset_db"] for k in B) + 8
    b.set_ylim(lo, lo + 56)
    b.set_xlabel("acquisition SNR (dB)", labelpad=1)
    b.set_ylabel("image PSNR (dB)")
    b.legend(loc="upper left", handlelength=2.0, borderaxespad=0.2, labelspacing=0.15,
             fontsize=5.0, ncol=2, columnspacing=0.8)
    b.grid(alpha=0.25, lw=0.3)
    save(fig, "noisefloor")


def gibbs(col=145, yr=(10, 62), zoom=(119, 171, 10, 62)):
    """Fully sampled 1 mm Cartesian, no noise: the analytical data ring, the grid data do not.

    Slices from experiments/gibbs_demo.py: the 1 mm voxel reference (voxel averages of the
    mixture, the input of the grid simulation), the inverse DFT of the analytic k-space,
    and that of the voxel reference (its DFT on the same grid, inverted: returns it). The
    profile also shows the continuous object (0.25 mm rendering). Bottom: a profile along the vertical line
    ``col`` across the posterior scalp and skull, where the ringing is strongest.
    """
    d = np.load(ROOT / "outputs/sim/gibbs_demo/slices.npz")
    a, g, f, v = d["analytic"], d["grid"], d["fine"], d["voxel"]
    s = int(round(float(d["h"]) / float(d["fine_h"])))         # fine pixels per voxel
    vmax = float(np.percentile(g, 99.7))
    fig = plt.figure(figsize=(COL_W, 1.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.62], hspace=0.06, wspace=0.03)
    x0, x1, y0, y1 = zoom
    ekw = dict(cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    gray = dict(cmap="gray", vmin=0, vmax=vmax)
    panels = [(a, "analytical", gray), ((a - v) / vmax, "analytical error", ekw),
              (g, "grid", gray), ((g - v) / vmax, "grid error", ekw)]
    for j, (img, title, kw) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img[y0:y1, x0:x1], origin="lower", interpolation="nearest", **kw)
        ax.axvline(col - x0, color="#4dd2ff", lw=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=6, pad=1.5)
    b = fig.add_subplot(gs[1, :])
    yc = np.arange(yr[0], yr[1]) + 0.5 - yr[0]
    ea, eg = ((img[yr[0]:yr[1], col] - v[yr[0]:yr[1], col]) / vmax for img in (a, g))
    b.plot(yc, ea, color=C_ANA, lw=1.0, marker="o", ms=1.8, label="analytical $-$ voxel")
    b.plot(yc, eg, color=C_GRID, lw=1.0, ls=(0, (3, 1.5)), marker="s", ms=1.6,
           label="grid $-$ voxel")
    b.axhline(0, color="0.6", lw=0.5)
    b.set_xlim(0, yr[1] - yr[0])
    b.set_xlabel("position along the line (mm)", labelpad=0.5, fontsize=6)
    b.set_ylabel("error / max", fontsize=6, labelpad=1)
    b.tick_params(labelsize=5.5, pad=1)
    b.set_ylim(-0.12, 0.15)
    b.legend(loc="upper left", fontsize=5.2, frameon=False, ncol=2, handlelength=1.8,
             borderaxespad=0.1, columnspacing=1.0)
    b.grid(alpha=0.25, lw=0.3)
    save(fig, "gibbs")




if __name__ == "__main__":
    which = sys.argv[1:] or ["shepplogan", "gibbs", "noisefloor", "motion"]
    for w in which:
        globals()[w]()
