"""BrainWeb volumes as fitting targets, in physical millimetres.

Optional NIfTI inputs (downloaded separately; not bundled):

* ``*_t1w.nii.gz`` / ``brainweb_<id>_T1.nii.gz`` -- the 1 mm T1w intensity volume;
* ``brainweb_s05_crisp.nii.gz`` -- the 0.5 mm discrete tissue labels (0-11).

The label volume is the more interesting target and the one AGNC never used: it carries
tissue identity, so a mixture fitted to it can be given per-primitive (PD, T1, T2) and
re-imaged under any sequence, instead of baking one T1w contrast into the geometry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..conventions import FOV

# BrainWeb is not redistributable, so it is read in place and never copied into this
# repository. Point GMRISIM_BRAINWEB at your own copy; the default is data/brainweb relative to the working directory.
BRAINWEB_DIR = Path(os.environ.get(
    "GMRISIM_BRAINWEB",
    "data/brainweb"))

# BrainWeb "crisp" discrete model, labels 0-11.
LABELS = ("background", "csf", "grey matter", "white matter", "fat", "muscle",
          "skin", "skull", "vessels", "around fat", "dura", "bone marrow")

# Nominal 3 T properties per label, ms. Sources vary; these are development defaults and
# must be cited or replaced before any published contrast claim. PD is relative.
TISSUE_PROPS = {                       # label: (PD, T1_ms, T2_ms, T2star_ms)
    0:  (0.00, 1.0,    1.0,   1.0),
    1:  (1.00, 4000.0, 2000.0, 200.0),   # CSF
    2:  (0.86, 1330.0, 80.0,   66.0),    # grey matter
    3:  (0.77, 830.0,  70.0,   53.0),    # white matter
    4:  (1.00, 370.0,  130.0,  60.0),    # fat
    5:  (1.00, 1230.0, 30.0,   25.0),    # muscle
    6:  (1.00, 900.0,  60.0,   40.0),    # skin
    7:  (0.20, 250.0,  1.0,    1.0),     # skull
    8:  (0.90, 1600.0, 180.0,  80.0),    # vessels
    9:  (0.77, 370.0,  130.0,  60.0),    # around fat
    10: (0.90, 2000.0, 100.0,  60.0),    # dura
    11: (1.00, 370.0,  130.0,  60.0),    # bone marrow
}


@dataclass
class VolumeTarget:
    """A fitting target: values on a grid, plus the FOV that places it in millimetres."""

    volume: Tensor        # [nz, ny, nx] -- kernel/render layout, x fastest
    fov: FOV
    name: str
    labels: Tensor | None = None        # [nz, ny, nx] long, when the source is `crisp`
    channels: Tensor | None = None      # [C, nz, ny, nx], a vector-valued target
    channel_labels: tuple = ()          # the tissue label each channel stands for

    @property
    def total_mass(self) -> float:
        """``integral rho dV`` -- what a fitted mixture's masses must sum to."""
        return float(self.volume.sum()) * self.fov.voxel_volume_mm3

    def to(self, device) -> "VolumeTarget":
        return VolumeTarget(self.volume.to(device), self.fov, self.name,
                            None if self.labels is None else self.labels.to(device),
                            None if self.channels is None else self.channels.to(device),
                            self.channel_labels)

    @property
    def channel_mass(self) -> Tensor:
        """``integral rho_c dV`` per channel -- what each channel's masses must sum to."""
        if self.channels is None:
            raise ValueError("target has no channels")
        return self.channels.sum(dim=(1, 2, 3)) * self.fov.voxel_volume_mm3


def _as_render_layout(a: np.ndarray) -> np.ndarray:
    """BrainWeb NIfTI axis 0 is axial, so ``(i, j, k)`` is already ``[nz, ny, nx]``.

    A no-op transpose, kept explicit because the *shape* must then be reversed to build
    the FOV: array shape is (nz, ny, nx) while :class:`FOV` is (x, y, z) everywhere.
    Getting that backwards put the object in a 181 mm x-FOV that holds 256 voxels, which
    left the fit itself correct (it works in the target's own frame) and made every
    whole-volume evaluation wrong -- centre of mass off by 38 mm, Fourier NRMSE above 1.
    """
    return np.ascontiguousarray(np.transpose(a, (0, 1, 2)))


def _fov_from_nifti(a: np.ndarray, zooms, affine: np.ndarray | None = None) -> FOV:
    """Array in ``[nz, ny, nx]`` + NIfTI zooms ``(dz, dy, dx)`` -> an (x, y, z) FOV.

    ``affine``, when given, places the FOV where the scanner puts it instead of at the
    origin. It is optional because every earlier caller works in the object's own frame,
    where the offset cancels; it matters as soon as something rotates, because a rigid
    pose is applied about the **origin**, and rotating a head that is centred at
    ``(18, -18, 0)`` about ``(0, 0, 0)`` is a different motion. BrainWeb's 0.5 mm grid is
    off-centre in exactly that way, so the fuzzy loader passes it.
    """
    shape = tuple(int(s) for s in reversed(a.shape))            # (nx, ny, nz)
    vox = tuple(float(z) for z in reversed(tuple(zooms)[:3]))   # (dx, dy, dz)
    centre = (0.0, 0.0, 0.0)
    if affine is not None:
        # centre of the voxel-CENTRE span along each NIfTI axis, then reversed to (x,y,z)
        c = [float(affine[i, 3]) + 0.5 * (a.shape[i] - 1) * float(tuple(zooms)[i])
             for i in range(3)]
        centre = (c[2], c[1], c[0])
    return FOV(fov_mm=tuple(s * v for s, v in zip(shape, vox)), matrix=shape,
               center_mm=centre)


def load_intensity(subject: int | str = 4, path: str | Path | None = None,
                   normalize: str = "p999", device="cpu") -> VolumeTarget:
    """1 mm T1w intensity volume as a proton-density-like target.

    ``normalize='p999'`` divides by the 99.9th percentile and clips, which is the same
    normalization AGNC's contour tracer used -- keeping it means the two phantoms are
    built from the same numbers even though the representations differ.
    """
    import nibabel as nib

    if path is None:
        for pat in (f"subject{int(subject):02d}_t1w.nii.gz", f"brainweb_{int(subject)}_T1.nii.gz"):
            if (BRAINWEB_DIR / pat).exists():
                path = BRAINWEB_DIR / pat
                break
        else:
            raise FileNotFoundError(f"no BrainWeb T1w for subject {subject} under {BRAINWEB_DIR}")
    img = nib.load(str(path))
    a = np.asarray(img.dataobj, dtype=np.float32)
    if normalize == "p999":
        a = np.clip(a / np.percentile(a, 99.9), 0.0, 1.0)
    vol = _as_render_layout(a)
    fov = _fov_from_nifti(vol, img.header.get_zooms())
    return VolumeTarget(torch.from_numpy(vol).to(device), fov,
                        f"brainweb_s{int(subject):02d}_t1w_{fov.voxel_mm[0]:.1f}mm")


def load_crisp(path: str | Path | None = None, contrast: str = "pd",
               device="cpu") -> VolumeTarget:
    """0.5 mm discrete tissue labels, mapped to a scalar target and kept as labels.

    ``contrast='pd'`` gives proton density; any other key must be produced through the
    physics layer rather than hard-coded here, so that a fitted geometry is never tied
    to one sequence.
    """
    import nibabel as nib

    p = Path(path or BRAINWEB_DIR / "brainweb_s05_crisp.nii.gz")
    img = nib.load(str(p))
    lab = np.asarray(img.dataobj).astype(np.int64)
    pd = np.zeros(12, dtype=np.float32)
    for k, v in TISSUE_PROPS.items():
        pd[k] = v[0]
    if contrast != "pd":
        raise ValueError("only 'pd' is defined here; use gaussmrisim.physics for sequences")
    a = pd[lab]
    vol = _as_render_layout(a)
    fov = _fov_from_nifti(vol, img.header.get_zooms())
    return VolumeTarget(torch.from_numpy(vol).to(device), fov,
                        f"brainweb_crisp_{fov.voxel_mm[0]:.1f}mm",
                        labels=torch.from_numpy(_as_render_layout(lab)).to(device))


def load_seg_contrast(sequence, path: str | Path | None = None, factor: int = 1,
                      device="cpu", name: str = "seg") -> VolumeTarget:
    """A **single-contrast** target on the label grid, with the labels alongside it.

    ``sequence(pd, t1, t2, t2star) -> signal`` maps each tissue class to one intensity,
    so the returned volume is a genuine single-contrast image (e.g., T1-weighted) and
    ``labels`` is the exact segmentation of that same grid.

    Why this rather than the acquired ``subject05_t1w.nii.gz``: BrainWeb ships the T1w at
    1 mm on a (181, 256, 256) grid with a 181x256x256 mm FOV, and the crisp labels at
    0.5 mm on (362, 434, 362) with a 181x217x181 mm FOV -- same subject, different grid
    *and* different field of view. Resampling one onto the other introduces a
    registration whose error would land silently in the interface normals, which is the
    one quantity boundary-aware splitting cannot tolerate being wrong. Here the labels
    and the intensity are the same array by construction.

    What it costs, stated plainly: at ``factor=1`` the target is piecewise constant, so
    every interface is an ideal step. That is the *easiest* case for boundary-aware
    splitting and the result must not be reported as if it were an acquired image.
    ``factor > 1`` box-averages the intensity -- giving real partial-volume voxels -- and
    takes the majority label, which is the harder and more realistic setting.
    """
    import nibabel as nib

    p = Path(path or BRAINWEB_DIR / "brainweb_s05_crisp.nii.gz")
    img = nib.load(str(p))
    lab = np.asarray(img.dataobj).astype(np.int64)
    tab = np.array([float(sequence(*TISSUE_PROPS[i])) for i in range(12)],
                   dtype=np.float32)
    a = tab[lab]
    vol = torch.from_numpy(_as_render_layout(a))
    lb = torch.from_numpy(_as_render_layout(lab))
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])

    f = int(factor)
    if f > 1:
        nz, ny, nx = vol.shape
        cz, cy, cx = nz // f * f, ny // f * f, nx // f * f
        vol = (vol[:cz, :cy, :cx]
               .reshape(cz // f, f, cy // f, f, cx // f, f).mean(dim=(1, 3, 5)))
        # majority label, so the segmentation still describes the coarse voxel it labels
        blk = (lb[:cz, :cy, :cx]
               .reshape(cz // f, f, cy // f, f, cx // f, f)
               .permute(0, 2, 4, 1, 3, 5).reshape(cz // f, cy // f, cx // f, -1))
        lb = torch.mode(blk, dim=-1).values
        zooms = tuple(z * f for z in zooms)

    vol = vol.contiguous()
    # _fov_from_nifti reverses the zooms itself -- NIfTI axis 0 is axial here, so
    # get_zooms() is (dz, dy, dx) and must be handed over in that order, not pre-reversed.
    fov = _fov_from_nifti(vol.numpy(), zooms)
    return VolumeTarget(vol.to(device), fov,
                        f"brainweb_s05_{name}_{fov.voxel_mm[0]:.2f}mm",
                        labels=lb.contiguous().to(device))


def downsample(t: VolumeTarget, factor: int) -> VolumeTarget:
    """Box-average by an integer factor -- the coarse level of a multiresolution fit.

    Averaging, not subsampling: the coarse target must be the band-limited version of
    the fine one, or the coarse fit chases aliases and the refinement starts from a
    worse place than a fresh init would.
    """
    f = int(factor)
    if f == 1:
        return t
    nz, ny, nx = t.volume.shape
    cz, cy, cx = nz // f * f, ny // f * f, nx // f * f
    v = t.volume[:cz, :cy, :cx].reshape(cz // f, f, cy // f, f, cx // f, f).mean(dim=(1, 3, 5))
    shape = (cx // f, cy // f, cz // f)                        # FOV.matrix is (x, y, z)
    # Truncating the remainder crops the HIGH end of each axis, so the kept region's
    # centre moves; carrying the old centre would offset the coarse level from the fine
    # one by up to half a coarse voxel and make the refinement start misregistered.
    drop = (t.volume.shape[2] - cx, t.volume.shape[1] - cy, t.volume.shape[0] - cz)
    centre = tuple(c - 0.5 * dr * vx for c, dr, vx
                   in zip(t.fov.center_mm, drop, t.fov.voxel_mm))
    fov = FOV(fov_mm=tuple(s * vx * f for s, vx in zip(shape, t.fov.voxel_mm)),
              matrix=shape, center_mm=centre)
    return VolumeTarget(v.contiguous(), fov, f"{t.name}_ds{f}")


# The fuzzy model ships one uint16 channel per tissue on [0, 4095]. Only the vessels
# channel carries a constant floor -- measured 2026-09-07 on subject 04: every voxel in
# the volume reads at least 16 there (the corner voxel is pure background, bck = 4095,
# and still reads vessels = 16), while the other eleven channels have mode 0 in the
# background region. Left in, it is a constant 0.39% field filling all 56.9 Mvox: a fit
# would spend primitives on it and the phantom would claim vessels everywhere in the
# head. Removing it takes the vessels occupancy from 100% to 3.83%.
FUZZY_FLOOR = {8: 16}
FUZZY_FULL_SCALE = 4095.0


def load_fuzzy(subject: int | str = 4, path: str | Path | None = None,
               tissues: "tuple[int, ...] | None" = None, drop_background: bool = True,
               device="cpu", center_from_affine: bool = True) -> VolumeTarget:
    """0.5 mm **fuzzy** tissue memberships -> a multi-channel target.

    This is the phantom source SNAKE uses (``brainweb_dl``, ``contrast='fuzzy'``), and
    it is the right target for a per-tissue mixture: each channel is the partial-volume
    fraction of one tissue, so a mixture fitted to it represents *that tissue's own
    geometry*, edges included. Fitting one mixture to a PD map instead and deriving
    per-primitive composition afterwards is what produced the 4x penalty on non-PD
    contrasts recorded in M5 -- the geometry had never been shown the edges that only
    exist in T2w.

    **Scaling.** Channels are divided by 4095 and nothing else. With the vessels floor
    removed the model is already a partition of unity: measured median of ``sum_t f_t``
    is exactly 4095, mean |deviation| 5.03 (0.12%), p99 21. Renormalizing per voxel
    would amplify quantization noise in near-empty voxels and change the total mass, so
    the residual 0.12% is accepted and is the floor for any mass-conservation check
    here -- not a bug to chase.

    ``drop_background`` removes label 0, whose ``pd`` is 0 at every field strength, so
    it contributes exactly nothing to any GRE contrast. It is 45.6% of the volume.

    ``center_from_affine`` places the FOV where the NIfTI affine says it is rather than
    at the isocentre. BrainWeb's phantom is **not** centred -- the affine offset is
    (-90.25, -126.25, -72.25) against a (181, 217, 181) mm FOV -- and the difference
    matters twice: a comparison against SNAKE picks it up as a linear phase that reads
    like a resolution mismatch, and a rigid rotation is taken about the origin, so an
    off-centre object rotates about a different point than a centred one.
    """
    import nibabel as nib

    if path is None:
        cands = [Path(os.path.expanduser("~/.cache/brainweb"))
                 / f"brainweb_s{int(subject):02d}_fuzzy.nii.gz",
                 BRAINWEB_DIR / f"brainweb_s{int(subject):02d}_fuzzy.nii.gz"]
        for c in cands:
            if c.exists():
                path = c
                break
        else:
            raise FileNotFoundError(
                f"no BrainWeb fuzzy volume for subject {subject}; looked in "
                f"{[str(c) for c in cands]}. Fetch it with brainweb_dl "
                f"(get_mri(sub_id, contrast='fuzzy')).")
    img = nib.load(str(path))
    a = np.asarray(img.dataobj)
    if a.ndim != 4 or a.shape[-1] != 12:
        raise ValueError(f"expected a [.., .., .., 12] fuzzy volume, got {a.shape}")

    keep = tuple(range(1, 12) if drop_background else range(12)) if tissues is None \
        else tuple(int(t) for t in tissues)
    chans = np.empty((len(keep),) + a.shape[:3], dtype=np.float32)
    for j, t in enumerate(keep):
        v = a[..., t].astype(np.float32) - float(FUZZY_FLOOR.get(t, 0))
        chans[j] = _as_render_layout(np.clip(v, 0.0, None) / FUZZY_FULL_SCALE)

    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    fov = _fov_from_nifti(chans[0], zooms)
    if center_from_affine:
        # first voxel centre = center - fov/2 + voxel/2, per axis, and the NIfTI
        # offset is that same point. Axis 0 of a BrainWeb volume is z, so the affine
        # rows come back in (z, y, x) and the FOV wants (x, y, z).
        off = np.asarray(img.affine)[:3, 3][::-1]
        c = tuple(float(o + 0.5 * f - 0.5 * v)
                  for o, f, v in zip(off, fov.fov_mm, fov.voxel_mm))
        fov = FOV(fov_mm=fov.fov_mm, matrix=fov.matrix, center_mm=c)

    vol = torch.from_numpy(chans.sum(axis=0)).to(device)      # total occupancy
    return VolumeTarget(vol, fov,
                        f"brainweb_s{int(subject):02d}_fuzzy_{fov.voxel_mm[0]:.2f}mm",
                        channels=torch.from_numpy(chans).to(device),
                        channel_labels=keep)


def channel_bbox(target: VolumeTarget, index: int, thresh: float = 0.01,
                 margin_vox: int = 4) -> tuple:
    """Index bounds ``(z0, z1, y0, y1, x0, x1)`` of where channel ``index`` is present."""
    if target.channels is None:
        raise ValueError("target has no channels")
    m = target.channels[index] > float(thresh)
    if not bool(m.any()):
        raise ValueError(f"channel {index} is empty at threshold {thresh}")
    out = []
    for ax in (0, 1, 2):
        p = m.any(dim=tuple(a for a in (0, 1, 2) if a != ax))
        nz = torch.nonzero(p).flatten()
        lo = max(int(nz[0]) - margin_vox, 0)
        hi = min(int(nz[-1]) + 1 + margin_vox, m.shape[ax])
        out += [lo, hi]
    return tuple(out)


def crop_channel(target: VolumeTarget, index: int, bbox: tuple) -> VolumeTarget:
    """One channel of ``target``, cropped to ``bbox``, as a standalone scalar target.

    The FOV is re-derived so the crop keeps its **absolute** position in millimetres.
    Getting this wrong is invisible in the fit -- it converges perfectly in whatever
    frame it is given -- and then every k-space evaluation and every tissue's mixture
    lands somewhere else, which is the failure mode already paid for once (rule 2c).
    """
    z0, z1, y0, y1, x0, x1 = bbox
    vol = target.channels[index][z0:z1, y0:y1, x0:x1].contiguous()
    f, v = target.fov, target.fov.voxel_mm
    matrix = (x1 - x0, y1 - y0, z1 - z0)
    fov_mm = tuple(m * vv for m, vv in zip(matrix, v))
    lo = (x0, y0, z0)
    centre = tuple(c - 0.5 * F + 0.5 * vv + l * vv + 0.5 * F2 - 0.5 * vv
                   for c, F, vv, l, F2 in
                   zip(f.center_mm, f.fov_mm, v, lo, fov_mm))
    lab = None
    if target.labels is not None:
        lab = target.labels[z0:z1, y0:y1, x0:x1].contiguous()
    return VolumeTarget(vol, FOV(fov_mm=fov_mm, matrix=matrix, center_mm=centre),
                        f"{target.name}_t{index}", labels=lab)
