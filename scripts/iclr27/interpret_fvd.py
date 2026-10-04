"""
Split FVD (I3D) and CD-FVD (VideoMAE-v2) into their appearance and flicker parts,
the video analog of interpret_directions.py's concept decomposition of FID.

The probes are a cached `DistortionSweep` over the reference: cd-fvd's decoupled
distortions (Ge et al. 2024, section 3), elastic transform and motion blur, each a
SEPARATE family (never composed, matching their Tables 5-6) at five ImageNet-C
severities two ways:

  spatial (s):          one field redrawn every num_frames -- the SAME distortion on
                        every frame, so frame quality drops but stays coherent.
  spatio-temporal (st): the field redrawn every frame -- the same per-frame quality
                        drop, now flickering, a pure temporal inconsistency.

Two transports define the bands, in the language of the displacement matrix M
(tr M = FVD): M(ref, s) is the appearance transport and M(s, st) the flicker
transport, a genuine transport of the temporal inconsistency alone since the two
sets are frame-quality matched. A family's bands are the eigen-axes of these
transports fit by `--fit`: `holdout` pools the severities OTHER than the one being
read (leave-one-level-out, the default; splitting clips instead would leave N < D
for VideoMAE and a singular covariance), `pool` pools all five (in-sample), `self`
uses the severity's own transport (in-sample, no pooling).

Readings, all as cumulative shares C_{1:k} = sum_{i<=k} w_i^T M w_i / tr M of the
top-k band against the k/D random baseline:

  capture:  how much of the flicker set's FVD, M(ref, st), the flicker band and the
            appearance band hold. cd-fvd's content bias is the flicker band's share,
            low for I3D and high for VideoMAE at every k.
  transfer: how much of one family's flicker transport the other family's flicker
            band holds, against the family's own band.
  overlap:  the two bands are fit independently and overlap, so at k = 10 their
            largest principal-angle cosine, the share of their union, and each band's
            share once orthogonalized against the other.

Severity is kept as a dose axis: cd-fvd's mean over levels is reported only for
their scalar ratio. Tables print at a few k. Features cache to i3d.npy/videomae.npy
per set (multi-GPU under accelerate) and displacements to samples/sweeps/, so
re-runs are cheap linear algebra.

Usage: ACCELERATE_USE_CPU=true python scripts/iclr27/interpret_fvd.py
           [--reference UCF101] [--severity 3] [--fit holdout|pool|self]
           [--generated PATH]
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from core import data
from core.data import DISTORTIONS, VideoDataset
from core.features.directions.ot import Displacement, displacement
from core.features.directions.sweep import DistortionSweep
from core.features.extractors import Extractor, I3DExtractor, VideoMAEExtractor
from core.features.metrics import accelerator, load_cache

SEVERITIES = DistortionSweep.severities  # ImageNet-C severities, the sweep's dose axis
BANDS = {"s": "appearance", "st": "flicker"}  # transport mode -> band name
KS = (1, 10)  # band widths tabulated
FITS = ("holdout", "pool", "self")  # which severities a band is fit on, see `band`

# The two distributional video metrics.
EXTRACTORS = {
    "FVD (I3D)": I3DExtractor,
    "CD-FVD (VideoMAE)": VideoMAEExtractor,
}


def eigen_axes(M: torch.Tensor) -> torch.Tensor:
    """
    Rows: eigenvectors of symmetric M in descending order.
    """
    return torch.linalg.eigh(M)[1].mT.flip(0)


def cumulative(d: Displacement, axes: torch.Tensor) -> np.ndarray:
    """
    C_{1:k} / tr M for k = 1..D along `axes` rows.
    """
    return (d.fid_w(axes).cumsum(0) / d.fid()).cpu().numpy()


class Transports:
    """
    One extractor's transports over the distortion sweep. `ref[fam, mode, sev]` is
    M(ref, probe) from the sweep cache; `flick[fam, sev]` is M(s, st), built from
    the probes' cached features since the sweep holds only reference pairs.
    """

    def __init__(
        self, reference: VideoDataset, extractor: Extractor, fit: str = "holdout"
    ) -> None:
        self.fit = fit
        self.ref = DistortionSweep(reference, extractor)
        feats = lambda fam, mode, sev: load_cache(
            extractor, self.ref.probes[fam, mode, sev].get_path()
        ).double()
        self.flick = {
            (fam, sev): displacement(feats(fam, "s", sev), feats(fam, "st", sev))
            for fam in DISTORTIONS
            for sev in SEVERITIES
        }
        self.dim = self.flick[next(iter(self.flick))].M.shape[0]

    def transport(self, fam: str, mode: str, sev: int) -> Displacement:
        return self.ref[fam, "s", sev] if mode == "s" else self.flick[fam, sev]

    def band(self, fam: str, mode: str, sev: int | None = None) -> torch.Tensor:
        """
        Eigen-axes of the family's `mode` transport pooled (mean M) over the
        severities `fit` selects around `sev`; all five when `sev` is None.
        """
        if sev is None or self.fit == "pool":
            sevs = SEVERITIES
        elif self.fit == "self":
            sevs = [sev]
        else:
            sevs = [s for s in SEVERITIES if s != sev]
        M = torch.stack([self.transport(fam, mode, s).M for s in sevs]).mean(0)
        return eigen_axes(M)


def scalar_table(T: Transports) -> pd.DataFrame:
    """
    cd-fvd's spatial vs spatio-temporal FVD per severity, plus their mean row.
    """
    rows = {
        (fam, sev): {
            "FVD_s": T.ref[fam, "s", sev].fid().item(),
            "FVD_st": T.ref[fam, "st", sev].fid().item(),
        }
        for fam in DISTORTIONS
        for sev in SEVERITIES
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    mean = df.groupby(level=0).mean()
    mean.index = pd.MultiIndex.from_product([mean.index, ["mean"]])
    df = pd.concat([df, mean]).sort_index()
    df["ratio"] = df["FVD_st"] / df["FVD_s"]
    return df.rename_axis(["family", "severity"])


def capture_table(T: Transports) -> pd.DataFrame:
    """
    Share of the flicker set's FVD on each band at the tabulated k, with the
    flicker transport's size and how one-dimensional it is.
    """
    rows = {}
    for fam in DISTORTIONS:
        for sev in SEVERITIES:
            d, flick = T.ref[fam, "st", sev], T.flick[fam, sev]
            row = {
                "FVD(s,st)": flick.fid().item(),
                "lam1/tr": (torch.linalg.eigvalsh(flick.M)[-1] / flick.fid()).item(),
            }
            for mode, name in BANDS.items():
                c = cumulative(d, T.band(fam, mode, sev))
                row |= {f"{name} k={k}": c[k - 1] for k in KS}
            rows[fam, sev] = row
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.rename_axis(["family", "severity"])


def transfer_table(T: Transports, sev: int) -> pd.DataFrame:
    """
    Share of family b's flicker transport on family a's flicker band.
    """
    rows = {
        (k, a): {
            b: cumulative(T.flick[b, sev], T.band(a, "st", sev))[k - 1]
            for b in DISTORTIONS
        }
        for k in KS
        for a in DISTORTIONS
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["k", "family"])


def cosine_table(T: Transports) -> pd.DataFrame:
    """
    |cos| between the top axes of every (family, band) pair.
    """
    names = [f"{fam} {name}" for fam in DISTORTIONS for name in BANDS.values()]
    tops = torch.stack([T.band(fam, mode)[0] for fam in DISTORTIONS for mode in BANDS])
    cos = F.cosine_similarity(tops[:, None], tops[None], dim=-1).abs()
    return pd.DataFrame(cos.cpu().numpy(), index=names, columns=names)


def generated_table(T: Transports, gen: Displacement) -> pd.DataFrame:
    """
    Share of a generated set's FVD on each family's full bands.
    """
    rows = {
        (fam, name): {f"k={k}": cumulative(gen, T.band(fam, mode))[k - 1] for k in KS}
        for fam in DISTORTIONS
        for mode, name in BANDS.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["family", "band"])


def overlap(T: Transports, d: Displacement, fam: str, sev: int | None) -> dict:
    """
    The two bands are fit independently and overlap, so their shares are alignments
    rather than an allocation: at k = 10, the largest principal-angle cosine between
    them, the share of their union, and each band's share once orthogonalized
    against the other (the union's share minus the other band's).
    """
    k = max(KS)
    a, f = T.band(fam, "s", sev)[:k], T.band(fam, "st", sev)[:k]
    share = lambda rows: (d.fid_subspace(rows) / d.fid()).item()
    union = share(torch.cat([a, f]))
    return {
        "cos": torch.linalg.svdvals(a @ f.mT).max().item(),
        "union": union,
        "flicker | appearance": union - share(a),
        "appearance | flicker": union - share(f),
    }


def overlap_table(T: Transports, sev: int, gen: Displacement | None) -> pd.DataFrame:
    """
    `overlap` for the flicker set at `sev` on held-out bands, and for a generated set
    on the bands pooled over all severities.
    """
    rows = {
        (fam, "flicker set"): overlap(T, T.ref[fam, "st", sev], fam, sev)
        for fam in DISTORTIONS
    }
    if gen is not None:
        rows |= {(fam, "generated"): overlap(T, gen, fam, None) for fam in DISTORTIONS}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["family", "set"])


def report(name: str, T: Transports, sev: int, gen: Displacement | None) -> None:
    print(f"\n== {name}: feature dim {T.dim}")
    print("\ncd-fvd scalar (mean row: their Tables 5)")
    print(scalar_table(T))
    print(f"\ncapture: share of M(ref, st) on {T.fit}-fit bands")
    print(capture_table(T))
    print(f"\ntransfer at severity {sev}: share of b's M(s, st) on a's flicker band")
    print(transfer_table(T, sev))
    print("\n|cos| between top axes")
    print(cosine_table(T))
    print(f"\noverlap of the k={max(KS)} bands, flicker set at severity {sev}")
    print(overlap_table(T, sev, gen))
    if gen is not None:
        print(f"\ngenerated set: FVD {gen.fid():.1f}, share on the full bands")
        print(generated_table(T, gen))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", default="UCF101", help="reference dataset class")
    p.add_argument("--severity", type=int, default=3, help="severity of the transfer")
    p.add_argument("--fit", default="holdout", choices=FITS, help="band fit")
    p.add_argument("--generated", default=None, help="generated set path to split")
    args = p.parse_args()
    reference = getattr(data, args.reference)()

    transports, gens = {}, {}
    for name, cls in EXTRACTORS.items():
        extractor = cls()
        transports[name] = Transports(reference, extractor, args.fit)
        if args.generated is not None:
            feats = lambda path: load_cache(extractor, path).double()
            gens[name] = displacement(
                feats(reference.get_path()), feats(args.generated)
            )
    if not accelerator.is_main_process:
        return

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    for name, T in transports.items():
        report(name, T, args.severity, gens.get(name))


if __name__ == "__main__":
    main()
