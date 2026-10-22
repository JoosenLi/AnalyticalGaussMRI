"""The continuous object: a mixture of unit-integral 3D Gaussian primitives.

    rho(r) = sum_n a_n * m_n * N(r; mu_n, Sigma_n)

``m_n`` is the primitive's integrated proton-density-like mass (mm^3 . signal) and is
fixed by the anatomy fit; ``a_n`` is a *contrast coefficient* supplied per acquisition
by :mod:`gaussmrisim.physics`, so one fitted anatomy can be re-imaged under different
sequences without refitting geometry. Keeping the two separate is the whole reason the
representation is not just "a fitted T1w image".

Covariance is stored as ``(log_scale, quaternion)`` rather than as six free numbers:
that parameterization is positive-definite by construction, keeps the three principal
widths interpretable in millimetres (they drive the spectral envelope and therefore
the culling bound), and rotates exactly under rigid motion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch
from torch import Tensor


def rotation_from_quaternion(q: Tensor) -> Tensor:
    """Unit-quaternion ``[N, 4]`` (w, x, y, z) -> rotation matrices ``[N, 3, 3]``."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


def quaternion_from_rotation(R: Tensor) -> Tensor:
    """Rotation matrix ``[3, 3]`` -> unit quaternion ``[4]`` (w, x, y, z)."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if float(t) > 0:
        s = torch.sqrt(t + 1.0) * 2
        return torch.stack([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                            (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(torch.argmax(torch.stack([R[0, 0], R[1, 1], R[2, 2]])))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = torch.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
    q = torch.zeros(4, device=R.device, dtype=R.dtype)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i], q[1 + j], q[1 + k] = 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    return q


def quaternion_from_rotation_batch(R: Tensor) -> Tensor:
    """Batched rotation matrices ``[N, 3, 3]`` -> unit quaternions ``[N, 4]`` (w, x, y, z).

    Shepperd's method: form all four candidate quaternions and pick, per matrix, the one
    whose denominator is largest. The scalar version branches on the trace, which is
    correct but is a Python loop -- at 350k primitives that alone dominated
    initialization, so this exists for the same reason any batched routine does.
    """
    m = R.reshape(-1, 3, 3)
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    cand = torch.stack([1.0 + t, 1.0 + 2 * m[:, 0, 0] - t,
                        1.0 + 2 * m[:, 1, 1] - t, 1.0 + 2 * m[:, 2, 2] - t], dim=-1)
    pick = cand.argmax(dim=-1)
    s4 = torch.sqrt(cand.gather(1, pick[:, None]).clamp_min(1e-20)).squeeze(1) * 2.0
    d21, d12 = m[:, 2, 1] - m[:, 1, 2], m[:, 1, 0] - m[:, 0, 1]
    d02, d20 = m[:, 0, 2] - m[:, 2, 0], m[:, 2, 1] + m[:, 1, 2]
    a01, a02, a12 = m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1]
    q = torch.empty(m.shape[0], 4, device=m.device, dtype=m.dtype)
    b = pick == 0
    q[b] = torch.stack([0.25 * s4, d21 / s4, d02 / s4, d12 / s4], -1)[b]
    b = pick == 1
    q[b] = torch.stack([d21 / s4, 0.25 * s4, a01 / s4, a02 / s4], -1)[b]
    b = pick == 2
    q[b] = torch.stack([d02 / s4, a01 / s4, 0.25 * s4, a12 / s4], -1)[b]
    b = pick == 3
    q[b] = torch.stack([d12 / s4, a02 / s4, a12 / s4, 0.25 * s4], -1)[b]
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@dataclass
class GaussianMixture3D:
    """``N`` primitives in millimetres. All tensors share device and dtype.

    ``mass`` is real and non-negative for a physical object; it is stored raw (not
    through a softplus) so that a fit can be inspected and so the closed-form
    transforms stay linear in it. Tissue/relaxation fields are optional and only
    consumed by :mod:`gaussmrisim.physics`.
    """

    mu: Tensor                       # [N, 3] mm
    log_scale: Tensor                # [N, 3] log mm (principal widths)
    quaternion: Tensor               # [N, 4] (w, x, y, z), normalized on use
    mass: Tensor                     # [N] integrated mass
    tissue_weights: Optional[Tensor] = None   # [N, T] simplex over tissue classes
    t1_ms: Optional[Tensor] = None            # [N]
    t2_ms: Optional[Tensor] = None            # [N]
    t2star_ms: Optional[Tensor] = None        # [N]
    dilation_mm: float = 0.0                  # isotropic band-limit, see covariance()
    meta: Optional[dict] = None

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return int(self.mu.shape[0])

    @property
    def device(self):
        return self.mu.device

    def sigma(self) -> Tensor:
        """Principal widths ``[N, 3]`` in mm."""
        return torch.exp(self.log_scale)

    def rotation(self) -> Tensor:
        return rotation_from_quaternion(self.quaternion)

    def covariance(self) -> Tensor:
        """``Sigma = R diag(sigma^2) R^T + s^2 I`` -> ``[N, 3, 3]`` mm^2.

        ``s = dilation_mm`` is a **3D low-pass filter convolved into every primitive**,
        i.e. the object is defined as the mixture blurred by ``N(0, s^2 I)``. Because a
        Gaussian convolved with a Gaussian is a Gaussian, this costs nothing: it is one
        addition on the covariance and every closed form downstream is unchanged.

        It exists because a fit driven only by an image-domain loss will happily produce
        primitives far narrower than the voxel grid -- they are nearly invisible to that
        loss (they fall between sample points) yet they radiate at high ``|k|``, and a
        simulator that fabricates energy above the sampling rate will alias it into every
        undersampled acquisition. Measured on a 1 mm Shepp-Logan against analytic truth:
        an unfiltered fit carried **3.4x** the true power at 1-2.5x Nyquist.

        This is Mip-Splatting's 3D smoothing filter (Yu et al., CVPR 2024), which caps a
        primitive's frequency content at the sampling rate, imported for a different
        reason: there it removes zoom-in aliasing artefacts, here it is what makes the
        phantom's band-limit a *stated property* rather than an accident of the fit.
        Preferable to clamping ``sigma``, which is non-differentiable at the boundary and
        piles primitives up on it (measured: 43% of them).
        """
        R = self.rotation()
        # s^2 I commutes with any rotation, so the filter is absorbed into the principal
        # widths: R diag(sigma^2) R^T + s^2 I = R diag(sigma^2 + s^2) R^T. That keeps the
        # eigen form everywhere -- no matrix inverse, no determinant, and the filtered
        # object is still exactly a Gaussian mixture in the same parameterization.
        return (R @ torch.diag_embed(self.effective_sigma().square())
                @ R.transpose(-1, -2))

    def effective_sigma(self) -> Tensor:
        """Principal widths of the *filtered* primitive, ``[N, 3]`` mm."""
        if not self.dilation_mm:
            return self.sigma()
        return torch.sqrt(self.sigma().square() + float(self.dilation_mm) ** 2)

    def to(self, device=None, dtype=None) -> "GaussianMixture3D":
        f = lambda t: None if t is None else t.to(
            device=device or t.device,
            dtype=(dtype if (dtype is not None and t.is_floating_point()) else t.dtype))
        return replace(self, mu=f(self.mu), log_scale=f(self.log_scale),
                       quaternion=f(self.quaternion), mass=f(self.mass),
                       tissue_weights=f(self.tissue_weights), t1_ms=f(self.t1_ms),
                       t2_ms=f(self.t2_ms), t2star_ms=f(self.t2star_ms))

    def subset(self, idx: Tensor) -> "GaussianMixture3D":
        f = lambda t: None if t is None else t[idx]
        return replace(self, mu=self.mu[idx], log_scale=self.log_scale[idx],
                       quaternion=self.quaternion[idx], mass=self.mass[idx],
                       tissue_weights=f(self.tissue_weights), t1_ms=f(self.t1_ms),
                       t2_ms=f(self.t2_ms), t2star_ms=f(self.t2star_ms))

    # ------------------------------------------------------------------- image
    def image(self, points: Tensor, coeff: Optional[Tensor] = None,
              chunk: int = 0, cutoff_sigma: float = 0.0) -> Tensor:
        """Evaluate ``rho`` at continuous ``points [P, 3]`` (mm) -> ``[P]``.

        ``cutoff_sigma > 0`` truncates each primitive at that Mahalanobis radius.
        That is an **image-domain fitting convenience only**; it changes the object,
        so it must never leak into the k-space forward model, which is defined on the
        untruncated mixture (see the note in :mod:`gaussmrisim.fourier.gaussian_ft`).
        """
        c = torch.ones(self.n, device=points.device, dtype=points.dtype) if coeff is None else coeff
        w = (self.mass.to(c.dtype) * c)
        # must go through covariance(), not sigma(), or the band-limiting dilation is
        # silently absent from the image while present in the k-space -- the two would
        # then describe different objects
        R, s = self.rotation(), self.effective_sigma()
        prec = R @ torch.diag_embed(s.square().reciprocal()) @ R.transpose(-1, -2)
        norm = (2 * torch.pi) ** 1.5 * s.prod(dim=-1)
        chunk = chunk or max(1, int(5e7 // max(self.n, 1)))
        out = torch.zeros(points.shape[0], device=points.device, dtype=w.dtype)
        for a in range(0, points.shape[0], chunk):
            d = points[a: a + chunk, None, :] - self.mu[None, :, :]        # [p, N, 3]
            m2 = torch.einsum("pnj,njk,pnk->pn", d, prec, d)               # Mahalanobis^2
            g = torch.exp(-0.5 * m2) / norm[None, :]
            if cutoff_sigma > 0:
                g = torch.where(m2 <= cutoff_sigma ** 2, g, torch.zeros_like(g))
            out[a: a + chunk] = g.to(w.dtype) @ w
        return out

    def integrated_mass(self, coeff: Optional[Tensor] = None) -> Tensor:
        return self.mass.sum() if coeff is None else (self.mass.to(coeff.dtype) * coeff).sum()

    # ------------------------------------------------------------------ motion
    def rigid_transform(self, R: Tensor, t: Tensor) -> "GaussianMixture3D":
        """Move the object by ``r -> R r + t``: exact, no resampling.

        ``mu -> R mu + t`` and ``Sigma -> R Sigma R^T``, which for this
        parameterization means the principal widths are untouched and only the
        quaternion rotates. Mass is invariant because ``|det R| = 1``.
        """
        Rn = R.to(self.mu.dtype)
        q = quaternion_from_rotation(Rn)
        qw, qv = q[0], q[1:]
        pw, pv = self.quaternion[:, 0], self.quaternion[:, 1:]
        new = torch.cat([(qw * pw - (qv[None] * pv).sum(-1, keepdim=True).squeeze(-1))[:, None],
                         qw * pv + pw[:, None] * qv[None] + torch.cross(
                             qv[None].expand_as(pv), pv, dim=-1)], dim=-1)
        return replace(self, mu=self.mu @ Rn.T + t.to(self.mu.dtype), quaternion=new)

    def affine_transform(self, A: Tensor, b: Tensor) -> "GaussianMixture3D":
        """General affine ``r -> A r + b``: ``mu -> A mu + b``, ``Sigma -> A Sigma A^T``.

        **Mass is unchanged**, and that is the physics: this is the *pushforward* of the
        density, so a compressed region holds the same spins in a smaller volume (its
        peak rises, its integral does not). Scaling mass by ``|det A|`` instead -- the
        natural-looking choice -- breaks the Fourier identity below by exactly that
        factor and would make a deformed phantom gain or lose signal.

        Frequency-side identity, exact and testable:
        ``rho_hat_A(k) = exp(-i 2 pi k . b) rho_hat(A^T k)``, the same shape as the rigid
        case. Use :meth:`rigid_transform` when ``A`` is a rotation; it is exact and
        avoids the eigendecomposition this needs.

        This is also the local linearization of a non-linear deformation ``phi``:
        take ``A = J_phi(mu_n)`` per primitive. That is an approximation (the Gaussian
        is deformed by the Jacobian at its centre, not by ``phi`` itself) and must be
        labelled as one -- it is exact only for affine ``phi``.
        """
        An = A.to(self.mu.dtype)
        if An.ndim == 2:                              # one global map
            An = An[None].expand(self.n, 3, 3)
        # per-primitive A: mu_n -> A_n mu_n, which is an einsum, NOT a matmul against a
        # batched matrix -- the broadcasting silently produces the wrong shape.
        cov = An @ self.covariance() @ An.transpose(-1, -2)
        mu = torch.einsum("nij,nj->ni", An, self.mu) + b.to(self.mu.dtype)
        return from_covariance(mu, cov, self.mass,
                               tissue_weights=self.tissue_weights, t1_ms=self.t1_ms,
                               t2_ms=self.t2_ms, t2star_ms=self.t2star_ms)


def from_covariance(mu: Tensor, cov: Tensor, mass: Tensor, **kw) -> "GaussianMixture3D":
    """Build a mixture from full covariances ``[N, 3, 3]`` -- the inverse of the packing.

    Needed because a general **affine** map does not act on ``(log_scale, quaternion)``
    directly: under ``r -> A r + b`` the covariance goes to ``A Sigma A^T``, which is
    still symmetric positive definite but whose principal axes are *not* the rotated old
    axes unless ``A`` is orthogonal. Recovering the parameterization therefore needs a
    symmetric eigendecomposition -- cheap ([N, 3, 3] ``eigh``), differentiable, and done
    once per deformation rather than per sample.

    So the choice of ``cov6`` as the kernel's wire format costs nothing in generality:
    it is strictly more general than the quaternion form, and this function is the way
    back. Rigid motion still takes the exact quaternion path
    (:meth:`GaussianMixture3D.rigid_transform`), which needs no eigendecomposition.
    """
    evals, evecs = torch.linalg.eigh(cov.double())
    sigma = evals.clamp_min(1e-12).sqrt().to(mu.dtype)
    R = evecs.to(mu.dtype)
    # eigh may return a reflection; a quaternion describes rotations only
    flip = torch.where(torch.linalg.det(R) < 0, -1.0, 1.0)
    R = torch.cat([R[..., :2], R[..., 2:] * flip[:, None, None]], dim=-1)
    q = torch.stack([quaternion_from_rotation(R[i]) for i in range(R.shape[0])])
    return GaussianMixture3D(mu=mu, log_scale=torch.log(sigma), quaternion=q, mass=mass, **kw)


def single_gaussian(mu=(0.0, 0.0, 0.0), sigma=(1.0, 1.0, 1.0), mass=1.0,
                    quaternion=(1.0, 0.0, 0.0, 0.0), device="cpu",
                    dtype=torch.float32) -> GaussianMixture3D:
    """One primitive -- the object every closed-form identity is first checked on."""
    t = lambda v: torch.as_tensor(v, device=device, dtype=dtype).reshape(1, -1)
    return GaussianMixture3D(
        mu=t(mu), log_scale=torch.log(t(sigma)), quaternion=t(quaternion),
        mass=torch.as_tensor([float(mass)], device=device, dtype=dtype))


def random_mixture(n: int, fov_mm=(200.0, 200.0, 200.0), sigma_range=(1.0, 6.0),
                   seed: int = 0, device="cpu", dtype=torch.float32) -> GaussianMixture3D:
    """A random non-negative mixture for scaling/accuracy benchmarks."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    f = torch.as_tensor(fov_mm, dtype=torch.float64)
    mu = (torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5) * 0.6 * f
    lo, hi = float(sigma_range[0]), float(sigma_range[1])
    s = torch.exp(torch.rand(n, 3, generator=g, dtype=torch.float64)
                  * (torch.log(torch.tensor(hi)) - torch.log(torch.tensor(lo)))
                  + torch.log(torch.tensor(lo)))
    q = torch.randn(n, 4, generator=g, dtype=torch.float64)
    return GaussianMixture3D(
        mu=mu.to(device, dtype), log_scale=torch.log(s).to(device, dtype),
        quaternion=(q / q.norm(dim=-1, keepdim=True)).to(device, dtype),
        mass=torch.rand(n, generator=g, dtype=torch.float64).to(device, dtype),
    )
