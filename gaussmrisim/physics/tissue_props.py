"""Tissue relaxation properties, per field strength, on the BrainWeb label indexing.

The numbers are **not** ours: the 1.5 T table is SNAKE's
``snake/core/phantom/data/tissues_properties_1T5.csv``, reproduced here so that a GPS
simulation and a SNAKE simulation of the same subject are driven by identical physics
and any disagreement between them is attributable to the *engine*, not to the table.


Field strength is part of the key, never a default. A T2* of 61 ms (WM at 1.5 T) and
53 ms (WM at 3 T) look similar enough that a silently wrong field would produce a
plausible image, which is exactly the failure mode this project cannot tolerate.

Indexing is BrainWeb's ``BrainWebTissuesV2`` order, 0-11, the same order the fuzzy
volume stores its channels in. SNAKE's CSV is keyed by name and its rows are in a
different order; the mapping is spelled out in ``_SNAKE_1T5_ROWS`` rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

# BrainWebTissuesV2 order -- the channel order of the fuzzy volume.
LABELS = ("background", "csf", "grey matter", "white matter", "fat", "muscle",
          "skin", "skull", "vessels", "around fat", "dura", "bone marrow")

# SNAKE's CSV rows, name -> (T1, T2, T2s, rho, chi). Verbatim from
# tissues_properties_1T5.csv; the name -> BrainWeb index mapping follows.
_SNAKE_1T5_ROWS = {
    "vessels":      (1600.0,   80.0,   70.0, 1.00, -9.00),
    "wm":           ( 500.0,   70.0,   61.0, 0.77, -9.05),
    "muscles":      ( 900.0,   50.0,   40.0, 1.00, -9.05),
    "skull":        ( 300.0,    1.0,    1.0, 0.10, -8.86),
    "dura":         (1000.0,  100.0,   50.0, 0.60, -9.05),
    "marrow":       (1000.0,   50.0,   40.0, 1.00, -9.05),
    "csf":          (4000.0, 2000.0, 2000.0, 1.00, -9.05),
    "muscles_skin": (2569.0,  329.0,   58.0, 1.00, -9.05),
    "gm":           ( 900.0,   90.0,   69.0, 0.86, -9.05),
    "fat":          ( 250.0,   70.0,   60.0, 1.00, -7.79),
    "bck":          ( 800.0,  800.0,  800.0, 0.00,  0.36),
    "fat2":         ( 250.0,   70.0,   60.0, 1.00, -7.79),
}

# SNAKE's row name -> BrainWeb index. Written out because guessing it from the CSV
# order would be silently wrong: the CSV starts at "vessels" (index 8), not at
# background, and "fat2" is BrainWeb's "around fat" (9), not a second fat class.
_SNAKE_NAME_TO_INDEX = {
    "bck": 0, "csf": 1, "gm": 2, "wm": 3, "fat": 4, "muscles": 5,
    "muscles_skin": 6, "skull": 7, "vessels": 8, "fat2": 9, "dura": 10, "marrow": 11,
}


@dataclass(frozen=True)
class TissueProperty:
    """One tissue at one field strength. Times in ms, ``pd`` relative, ``chi`` in ppm."""

    name: str
    t1_ms: float
    t2_ms: float
    t2star_ms: float
    pd: float
    chi_ppm: float = 0.0


def _build_1t5() -> dict[int, TissueProperty]:
    out: dict[int, TissueProperty] = {}
    for name, (t1, t2, t2s, rho, chi) in _SNAKE_1T5_ROWS.items():
        i = _SNAKE_NAME_TO_INDEX[name]
        out[i] = TissueProperty(LABELS[i], t1, t2, t2s, rho, chi)
    if len(out) != 12:
        raise RuntimeError(f"1.5 T table covers {len(out)} of 12 BrainWeb labels")
    return out


# Development defaults at 3 T, kept from the earlier single-contrast work. Sources vary
# and these must be cited or replaced before any published 3 T contrast claim -- unlike
# the 1.5 T table, they are not traceable to a reference implementation.
_PROPS_3T: dict[int, TissueProperty] = {
    0:  TissueProperty(LABELS[0],    1.0,    1.0,   1.0, 0.00),
    1:  TissueProperty(LABELS[1], 4000.0, 2000.0, 200.0, 1.00),
    2:  TissueProperty(LABELS[2], 1330.0,   80.0,  66.0, 0.86),
    3:  TissueProperty(LABELS[3],  830.0,   70.0,  53.0, 0.77),
    4:  TissueProperty(LABELS[4],  370.0,  130.0,  60.0, 1.00),
    5:  TissueProperty(LABELS[5], 1230.0,   30.0,  25.0, 1.00),
    6:  TissueProperty(LABELS[6],  900.0,   60.0,  40.0, 1.00),
    7:  TissueProperty(LABELS[7],  250.0,    1.0,   1.0, 0.20),
    8:  TissueProperty(LABELS[8], 1600.0,  180.0,  80.0, 0.90),
    9:  TissueProperty(LABELS[9],  370.0,  130.0,  60.0, 0.77),
    10: TissueProperty(LABELS[10], 2000.0, 100.0,  60.0, 0.90),
    11: TissueProperty(LABELS[11],  370.0, 130.0,  60.0, 1.00),
}

_TABLES: dict[str, dict[int, TissueProperty]] = {"1.5T": _build_1t5(), "3T": _PROPS_3T}


def table(field: str = "1.5T") -> dict[int, TissueProperty]:
    """Properties for every BrainWeb label at ``field`` ('1.5T' or '3T')."""
    key = str(field).upper().replace(" ", "").rstrip("T") + "T"
    if key not in _TABLES:
        raise ValueError(f"no tissue table for field {field!r}; have {list(_TABLES)}")
    return _TABLES[key]


def arrays(field: str = "1.5T", device="cpu"):
    """The table as tensors ``(pd, t1_ms, t2_ms, t2star_ms)``, each ``[12]``."""
    import torch

    t = table(field)
    cols = [[getattr(t[i], a) for i in range(12)]
            for a in ("pd", "t1_ms", "t2_ms", "t2star_ms")]
    return tuple(torch.tensor(c, dtype=torch.float32, device=device) for c in cols)
