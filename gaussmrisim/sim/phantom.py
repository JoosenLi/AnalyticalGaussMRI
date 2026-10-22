"""The subject: one Gaussian mixture whose primitives each belong to one tissue.

This is the whole reason the architecture is cheap. Because a primitive is tissue-pure,

* **contrast is a gather.** ``a_n = a[tissue_n]``, handed to the FT kernel's existing
  ``coeff`` argument. Re-imaging the same anatomy under a new sequence costs one indexing
  operation and touches no geometry -- the ``simple`` forward model needs no kernel
  change whatsoever.
* **T2* decay is a per-primitive scalar.** SNAKE must run one NUFFT per tissue for its
  T2* model, because its tissues share a voxel grid and the decay differs per tissue.
  Here ``exp(-t_m * r2s_n)`` is a per-(primitive, sample) factor.
* **it is one mixture, not T of them.** The FT kernel is L2-resident in the primitive
  parameters (measured cliff at 96 MiB, ~3 M primitives), so T separate launches would
  stream the parameters T times over for nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor

from ..physics.signal_model import gre_from_table
from ..physics.tissue_props import LABELS, arrays
from ..representation.gaussian3d import GaussianMixture3D


@dataclass
class TissuePhantom:
    """A mixture plus the tissue each primitive belongs to."""

    mixture: GaussianMixture3D
    tissue: Tensor                      # [N] long, BrainWeb index 0-11
    field: str = "1.5T"

    def __post_init__(self) -> None:
        if self.tissue.shape[0] != self.mixture.n:
            raise ValueError(f"{self.tissue.shape[0]} tissue tags for "
                             f"{self.mixture.n} primitives")

    @property
    def n(self) -> int:
        return int(self.mixture.n)

    @property
    def labels(self) -> tuple:
        return tuple(int(x) for x in torch.unique(self.tissue).tolist())

    @property
    def device(self):
        return self.mixture.device

    def to(self, device) -> "TissuePhantom":
        return TissuePhantom(self.mixture.to(device), self.tissue.to(device), self.field)

    # ------------------------------------------------------------------ physics
    def _lut(self, values: Tensor) -> Tensor:
        """Scatter a 12-long per-tissue table onto the primitives -> ``[N]``."""
        return values.to(self.tissue.device)[self.tissue]

    def coeff(self, seq, *, field: Optional[str] = None) -> Tensor:
        """Contrast coefficient per primitive ``a_n(theta)`` -> ``[N]``.

        At the echo, so the ``simple`` model is complete with this alone and the
        ``t2star`` model multiplies it by a readout factor that is 1 at ``t = TE``.
        """
        a = gre_from_table(field or self.field, tr_ms=seq.tr_ms, te_ms=seq.te_ms,
                           flip_deg=seq.flip_deg, device=self.tissue.device)
        return self._lut(a)

    def r2star_per_ms(self, *, field: Optional[str] = None) -> Tensor:
        """``1 / T2*`` per primitive -> ``[N]``, for the readout decay."""
        _pd, _t1, _t2, t2s = arrays(field or self.field, device=self.tissue.device)
        return self._lut(1.0 / t2s.clamp_min(1e-12))

    def tissue_name(self, index: int) -> str:
        return LABELS[int(index)]

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path, device="cpu", field: str = "1.5T") -> "TissuePhantom":
        root = Path(path)
        d = torch.load(root / "mixture.pt", map_location=device)
        mix = GaussianMixture3D(**{k: v for k, v in d.items() if k in
                                   ("mu", "log_scale", "quaternion", "mass",
                                    "dilation_mm")})
        tissue = torch.load(root / "tissue.pt", map_location=device)
        return cls(mix, tissue.long(), field)

    def save(self, path: str | Path) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        torch.save({"mu": self.mixture.mu, "log_scale": self.mixture.log_scale,
                    "quaternion": self.mixture.quaternion, "mass": self.mixture.mass,
                    "dilation_mm": self.mixture.dilation_mm}, root / "mixture.pt")
        torch.save(self.tissue, root / "tissue.pt")
