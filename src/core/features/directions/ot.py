from dataclasses import dataclass

import torch


def sqrtm(A: torch.Tensor) -> torch.Tensor:
    """
    Square root of a symmetric PSD matrix through its eigendecomposition.
    Real by construction, unlike a general-purpose scipy sqrtm.
    """
    lam, Q = torch.linalg.eigh(0.5 * (A + A.mT))
    return (Q * lam.clamp_min(0).sqrt()) @ Q.mT


def gaussian(feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mean and covariance of a feature data matrix.
    """
    return feats.mean(0), torch.cov(feats.mT)


def cov_displacement(sigma_x, sigma_y):
    """
    The covariance part of the displacement.
    """
    A, B = sqrtm(sigma_x), sqrtm(sigma_y)
    U, _, Vh = torch.linalg.svd(B @ A)
    K = A @ (U @ Vh).mT @ B
    return sigma_x + sigma_y - K - K.mT


@dataclass
class Displacement:
    M: torch.Tensor  # second moment, tr M = FID
    Mbar: torch.Tensor  # covariance part
    dmu: torch.Tensor  # mean shift part

    def fid(self) -> torch.Tensor:
        return self.M.trace()

    def fid_w(self, w: torch.Tensor) -> torch.Tensor:
        """
        Directional FID w^T M w along unit direction(s) w. A 2D w [N, D] gives the
        FID of each row, so w = eigen_axes() yields the per-axis charge.
        """
        w = w / w.norm(dim=-1, keepdim=True)
        return torch.einsum("...i,ij,...j->...", w, self.M, w)

    def fid_subspace(self, rows: torch.Tensor) -> torch.Tensor:
        """
        FID energy captured by the subspace spanned by rows [N, D].
        The rows need not be orthonormal; QR gives a basis for their span.
        """
        Q = torch.linalg.qr(rows.mT).Q
        return (Q.mT @ self.M @ Q).trace()

    def fid_w_baseline(self) -> torch.Tensor:
        """
        Expectation of directional FID along a random unit direction.
        """
        return self.fid() / self.M.shape[0]

    def _oriented_axes(self, A: torch.Tensor) -> torch.Tensor:
        """
        Returned rows are the eigenvectors of symmetric A in descending order,
        each oriented toward the generated mean.
        """
        V = torch.linalg.eigh(A)[1].mT.flip([0])
        s = torch.copysign(torch.ones(()), V @ -self.dmu)
        return V * s[:, None]

    def eigen_axes(self) -> torch.Tensor:
        return self._oriented_axes(self.M)

    def mean_axis(self) -> torch.Tensor:
        return self.dmu / self.dmu.norm()

    def stretch_axes(self) -> torch.Tensor:
        return self._oriented_axes(self.Mbar)


def displacement(feats_x: torch.Tensor, feats_y: torch.Tensor) -> Displacement:
    mu_x, sigma_x = gaussian(feats_x)
    mu_y, sigma_y = gaussian(feats_y)

    dmu = mu_x - mu_y
    Mbar = cov_displacement(sigma_x, sigma_y)
    return Displacement(Mbar + torch.outer(dmu, dmu), Mbar, dmu)


def shift_spread(
    disp: Displacement, ref: torch.Tensor, gen: torch.Tensor, w: torch.Tensor
):
    """
    How the generated set differs from the reference along w, in four numbers.

    The mean shift is a location claim: Cohen's d of the projections (+ means
    generated sits toward +w), reported with its share of the directional FID,
    since the "which end is generated" reading is only well-defined when that
    share is large.  The variance ratio is a spread claim that needs no
    orientation: >1 generated is more diverse along w, <1 more concentrated.
    On covariance-part axes, where the means coincide, spread is the claim.

    The last number is how much of FD(w) the two projections explain: their own
    Frechet distance over FD(w), at most 1, since FD(w) is paid by the coupling
    of the full distributions and also charges dependence across directions.
    """
    w = w / w.norm()
    pr, pg = ref @ w, gen @ w
    d = pg.mean() - pr.mean()
    nr, ng = pr.numel(), pg.numel()
    # df-weighted pooled within-group SD, so unequal set sizes are handled
    # (reduces to sqrt((var_r + var_g) / 2) when the two sets are the same size).
    sd = (((nr - 1) * pr.var() + (ng - 1) * pg.var()) / (nr + ng - 2)) ** 0.5
    cohen = (d / sd).item()
    fw = disp.fid_w(w)
    frac = (d**2 / fw).item()  # mean-shift share of directional FID
    projected = ((d**2 + (pg.std() - pr.std()) ** 2) / fw).item()
    return cohen, frac, (pg.var() / pr.var()).item(), projected
