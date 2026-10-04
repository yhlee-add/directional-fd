"""
Where the FID-versus-quality reversal lives, read in a fixed direction basis.

For each reference we freeze the eigenbasis of the 50-step displacement M and carry
it back along the step sweep.  Because the basis is orthonormal, FID(step) = tr M is
the exact sum of the directional charges C_i(step) = w_i^T M(step) w_i over the fixed
axes, so the reversal decomposes additively over directions.

Along the top axis e_1 we also track the signed motion of the generated mean,
gap_1(step) = (mu_g(step) - mu_r) . e_1 (oriented toward the 50-step generation): a
growing gap is the cloud drifting away from the reference, a shrinking gap is it
drifting home, which is the sign of the reversal.

Two cross-dataset objects then decide why some references reverse and some do not:
  - the generator-motion vectors mu_g(50) - mu_g(15), reference-free, which turn out
    to be a shared quality-drift axis (aligned across datasets);
  - the fixed top axes e_1, which are reference-specific (two clusters, photographic
    vs faces).
The reversal is the shared drift projected onto each reference's expensive axis:
away from the reference (dispersed sets) worsens FID, toward it (curated sets) helps.

Usage: python scripts/iclr27/reversal_basis.py
"""

import torch
import numpy as np
import pandas as pd
from scipy.stats import linregress

from core.features.directions.sweep import StableDiffusionSweep

REFS = ["COCO30K", "CelebAHQ", "FFHQ512", "Flickr30K", "ImageNet", "MJHQ30K"]
STEPS = list(range(15, 55, 5))


def collect(sweep: StableDiffusionSweep, ref: str) -> dict[str, np.ndarray]:
    """
    Fixed-basis charges and top-axis motion for one reference across the step sweep.
    """
    axes = sweep[ref, 50].eigen_axes()  # rows: M(50) eigvecs, descending, oriented

    fid = torch.stack([sweep[ref, n].fid() for n in STEPS])
    charge = torch.stack([sweep[ref, n].fid_w(axes) for n in STEPS])
    # gap on each direction toward generator
    gap = torch.stack([-(axes @ sweep[ref, n].dmu) for n in STEPS])
    e1 = axes[0]
    # reference-free generator motion; mu_r cancels in the dmu difference
    drift = sweep[ref, 15].dmu - sweep[ref, 50].dmu

    res = dict(fid=fid, charge=charge, gap=gap, e1=e1, drift=drift)
    return {k: v.cpu().numpy() for k, v in res.items()}


def cos_frame(vecs: dict) -> pd.DataFrame:
    """
    Gram matrix of the (normalized) vectors, indexed by name.
    """
    names = list(vecs)
    V = np.stack([vecs[k] for k in names])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return pd.DataFrame(V @ V.T, index=names, columns=names)


def main() -> None:
    sweep = StableDiffusionSweep("inception")
    store = {r: collect(sweep, r) for r in REFS}
    steps = np.asarray(STEPS, dtype=float)
    span = steps[-1] - steps[0]
    # replace endpoint deltas with a least-squares trend over the whole sweep
    chg = lambda a: linregress(steps, a).slope * span  # noqa: E731
    # share of the reversal a channel carries: slope of its charge against the
    # total FID across steps, as in reversal_shares.py (shares sum to 1 exactly)
    share = lambda c, fid: linregress(fid, c)  # noqa: E731

    A, B = {}, {}
    for r in REFS:
        s = store[r]
        fid = s["fid"]
        c1, c8 = s["charge"][:, 0], s["charge"][:, :8].sum(1)
        d = chg(fid)
        f1, f8 = share(c1, fid), share(c8, fid)
        A[r] = {
            "dFID": d,
            "dC1": f1.slope * d,
            "sh1%": f1.slope,
            "dC1-8": f8.slope * d,
            "sh8%": f8.slope,
            "dTail": (1 - f8.slope) * d,
            "shT%": 1 - f8.slope,
            "R2_1": f1.rvalue**2,
        }
        g = s["gap"][:, 0]
        B[r] = {
            "gap1@15": g[0],
            "gap1@50": g[-1],
            "d gap1": chg(g),
            "dC1": f1.slope * d,
            "FID": "up" if d > 0 else "down",
        }

    fmt = "{:+.2f}".format
    pct = {c: "{:.0%}".format for c in ("sh1%", "sh8%", "shT%")}
    pct["R2_1"] = "{:.2f}".format

    print("A. Fixed 50-step basis, share of the FID change carried by the top axes.")
    print(pd.DataFrame(A).T.to_string(float_format=fmt, formatters=pct))

    print("\nB. Motion of the generated mean along the fixed top axis e1.")
    print(pd.DataFrame(B).T.to_string(float_format=fmt))

    print("\nC. cos between reference expensive axes e1.")
    print(cos_frame({r: store[r]["e1"] for r in REFS}).to_string(float_format=fmt))

    print("\nD. cos between generator quality-drift axes.")
    print(cos_frame({r: store[r]["drift"] for r in REFS}).to_string(float_format=fmt))


if __name__ == "__main__":
    main()
