"""Sequence-dependent contrast: primitive properties -> coefficient ``a_n(theta)``.

Deliberately thin. The point of the architecture is that geometry (`mu, Sigma, m`) is
fitted **once** and contrast is a per-acquisition scalar per primitive, so a T1w and a
T2w simulation of the same subject share one anatomy. Bloch/EPG coefficients plug in
here later without touching the Fourier engine -- which is the honest boundary between
"analytical spatial encoding" and "numerical magnetization physics". Do not let a
sequence bake itself into the geometry.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..representation.gaussian3d import GaussianMixture3D


def t2star_decay(props: Tensor, time_ms: Tensor, te_ms: float) -> Tensor:
    """Readout T2* decay relative to the echo, per tissue -> ``[T, M]``.

    ``exp(-(t_m - TE) / T2*_t)``, so it is exactly 1 at the echo and the ``simple``
    model is its ``t_m = TE`` special case rather than a different physics. Splitting it
    out this way is what lets the two models share one contrast term: ``simple`` folds
    ``exp(-TE/T2*)`` into ``a_t`` and stops, ``t2star`` additionally carries this factor
    along the readout.
    """
    dt = (time_ms.to(props.dtype) - float(te_ms))[None, :]
    return torch.exp(-dt / props[:, 2].clamp_min(1e-12)[:, None])


def proton_density(mixture: GaussianMixture3D) -> Optional[Tensor]:
    """``a_n = 1``: the object as fitted. ``None`` is the fast path for this."""
    return None


def spgr(mixture: GaussianMixture3D, tr_ms: float, te_ms: float,
         flip_deg: float) -> Tensor:
    """Spoiled gradient echo, closed form.

        a = sin(alpha) (1 - E1) / (1 - E1 cos alpha) * exp(-TE / T2*)

    Requires ``t1_ms`` and ``t2star_ms`` on the mixture; raising here rather than
    defaulting is intentional, because a silent default would produce a plausible
    image with no physics in it.
    """
    if mixture.t1_ms is None or mixture.t2star_ms is None:
        raise ValueError("spgr needs per-primitive t1_ms and t2star_ms")
    a = torch.as_tensor(float(flip_deg) * torch.pi / 180.0, device=mixture.device)
    e1 = torch.exp(-float(tr_ms) / mixture.t1_ms.clamp_min(1e-6))
    return (torch.sin(a) * (1 - e1) / (1 - e1 * torch.cos(a)).clamp_min(1e-12)
            * torch.exp(-float(te_ms) / mixture.t2star_ms.clamp_min(1e-6)))


def gre(pd: Tensor, t1_ms: Tensor, t2star_ms: Tensor, *, tr_ms: float, te_ms: float,
        flip_deg: float) -> Tensor:
    """Spoiled GRE steady state, per tissue -> ``a_t``.

        a = rho sin(alpha) exp(-TE/T2*) (1 - E1) / (1 - cos(alpha) E1),  E1 = exp(-TR/T1)

    Uses the same factor order and degree convention as SNAKE's ``_contrast_gre``.

    This is the same expression as :func:`spgr`, which takes it from a mixture's
    per-primitive properties; the difference is only where the properties come from. A
    per-tissue table is what the SNAKE-shaped engine wants, since every primitive in a
    tissue shares one coefficient.
    """
    a = torch.as_tensor(float(flip_deg) * torch.pi / 180.0, device=pd.device,
                        dtype=pd.dtype)
    x = -float(tr_ms) / t1_ms.clamp_min(1e-6)
    e1 = torch.exp(x)
    # ``1 - E1`` is the whole numerator and it cancels catastrophically when TR << T1,
    # which is the normal regime: at TR = 10 ms against CSF's T1 = 4000 ms it is
    # 1 - 0.99750, and float32 keeps only ~3 significant digits of it. ``expm1`` gives
    # the same quantity with no cancellation. Measured against the naive form, this is
    # the entire 6.4e-6 relative gap between us and SNAKE at short TR -- their error,
    # not a difference of formula, which is why the cross-check is done in float64.
    one_minus_e1 = -torch.expm1(x)
    return (pd * torch.sin(a) * torch.exp(-float(te_ms) / t2star_ms.clamp_min(1e-6))
            * one_minus_e1 / (1.0 - torch.cos(a) * e1).clamp_min(1e-12))


def gre_from_table(field: str = "1.5T", *, tr_ms: float, te_ms: float,
                   flip_deg: float, device="cpu") -> Tensor:
    """:func:`gre` evaluated on a whole field-strength table -> ``[12]``."""
    from .tissue_props import arrays

    pd, t1, _t2, t2s = arrays(field, device=device)
    return gre(pd, t1, t2s, tr_ms=tr_ms, te_ms=te_ms, flip_deg=flip_deg)


def ir_gre(pd: Tensor, t1_ms: Tensor, t2star_ms: Tensor, *, tr_ms: float, ti_ms: float,
           te_ms: float) -> Tensor:
    """Inversion-recovery GRE, idealized, per tissue -> ``a_t`` (magnitude).

        a = rho |1 - 2 exp(-TI/T1) + exp(-TR/T1)| exp(-TE/T2*)

    A 180-degree inversion, recovery for TI, then one 90-degree gradient-echo readout
    per TR, with full relaxation in between except for the exp(-TR/T1) term: the
    single-shot inversion-recovery signal, without the readout train of an MP-RAGE.
    Choosing TI at a tissue's null, ``T1 ln(2 / (1 + exp(-TR/T1)))``, suppresses it --
    with CSF nulled and a long TE this is a FLAIR-like contrast. **Approximate**: an
    idealized model for showing contrast separation, not a validated sequence.
    """
    t1 = t1_ms.clamp_min(1e-6)
    mz = 1.0 - 2.0 * torch.exp(-float(ti_ms) / t1) + torch.exp(-float(tr_ms) / t1)
    return pd * mz.abs() * torch.exp(-float(te_ms) / t2star_ms.clamp_min(1e-6))


def ir_gre_from_table(field: str = "1.5T", *, tr_ms: float, ti_ms: float, te_ms: float,
                      device="cpu") -> Tensor:
    """:func:`ir_gre` evaluated on a whole field-strength table -> ``[12]``."""
    from .tissue_props import arrays

    pd, t1, _t2, t2s = arrays(field, device=device)
    return ir_gre(pd, t1, t2s, tr_ms=tr_ms, ti_ms=ti_ms, te_ms=te_ms)


def null_ti(t1_ms: float, tr_ms: float) -> float:
    """Inversion time that nulls a tissue of the given T1 in :func:`ir_gre`."""
    import math
    return float(t1_ms) * math.log(2.0 / (1.0 + math.exp(-float(tr_ms) / float(t1_ms))))
