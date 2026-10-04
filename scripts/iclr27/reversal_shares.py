"""
How much of the FID reversal each top eigenvector carries, as a regression slope.

Freezes the 50-step displacement eigenbasis and, across the step sweep, scatters the
directional FID C_i = w_i^T M(step) w_i against the total FID = tr M, one panel per
eigenvector.

On equal 1:1 axes the regression slope is C_i's share of the FID change (the reversal),
read against a dashed slope-1 line for the 100% reference. A last panel continues the
slope past the top axes: the cumulative slope s_{1:i} of the top i axes, the same slope
for their summed directional FID, which ends at 1 since all D of them sum to tr M.

Also prints, per step, the mean shift along e_1 as Cohen's d, and why the tail (the
axes past the peak of s_{1:i}) falls: the generated minus reference variance summed
over the tail subspace. Then the slope against the total FID of the comp-simplicity
concept direction of interpret_directions.py, and of its mean part.

Usage: python scripts/iclr27/reversal_shares.py [--reference COCO30K]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import linregress

from core import data
from core.features.directions.grouping import ClipTextEncoder, ClipTextGrouper
from core.features.directions.ot import shift_spread
from core.features.directions.sweep import StableDiffusionSweep
from core.features.metrics import load_cache
from core.utils import figures
from interpret_directions import CONCEPTS, QUANTILE

STEPS = list(range(15, 55, 5))
NAXES = 3

XLIM = (13.0, 15.0)  # fixed FD range, shared across panels
NTICKS = 5  # ticks per axis, first and last land on the frame
REF_COLOR = "crimson"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference", default="COCO30K", help="reference dataset class name"
    )
    args = p.parse_args()

    sweep = StableDiffusionSweep("inception")
    ref = args.reference
    basis = sweep[ref, 50].eigen_axes()  # fixed 50-step eigenbasis, rows descending
    fid = torch.stack([sweep[ref, n].fid() for n in STEPS])
    charge = torch.stack([sweep[ref, n].fid_w(basis) for n in STEPS])
    fid, charge = fid.cpu().numpy(), charge.cpu().numpy()
    # cumulative slope: the per-axis fit on the summed FD of the top i axes, so it
    # starts at e_1's slope and ends at 1 (the slopes add, and all D axes sum to FID)
    share = np.polyfit(fid, np.cumsum(charge, axis=1), 1)[0]

    # per step: the mean shift along e_1 as Cohen's d (+ toward generated), and the
    # generated excess variance over the tail, summed so it is a trace of the tail
    # subspace and does not depend on how its noisy eigenvectors rotate within it
    tail = basis[share.argmax() + 1 :]
    print(
        f"cumulative slope peaks at {share.max():.2f} on axis {share.argmax() + 1}; "
        f"tail: axes {share.argmax() + 2} to {share.size}"
    )
    ref_path = getattr(data, ref)().get_path()
    feats = load_cache("inception", ref_path).to(basis)
    var_ref = (feats @ tail.T).var(0).sum()
    for n in STEPS:
        gen = load_cache("inception", sweep.prefix.format(ref=ref, n=n)).to(basis)
        d = shift_spread(sweep[ref, n], feats, gen, basis[0])[0]
        var_gen = (gen @ tail.T).var(0).sum()
        print(
            f"{n} steps: e_1 Cohen's d {d:+.2f}, tail excess variance"
            f" {var_gen - var_ref:+.3f} (ratio {var_gen / var_ref:.4f})"
        )

    # a concept direction built without reference to M, so its slope measures the
    # concept itself rather than the eigenvector it names
    slope = lambda f: np.polyfit(fid, [f(sweep[ref, n]).item() for n in STEPS], 1)[0]
    pos, neg = CONCEPTS["comp-simplicity"]
    simple = ClipTextGrouper(ClipTextEncoder(), pos, neg, q=QUANTILE).concept(
        load_cache("clip", ref_path), feats
    )
    print(
        f"comp-simplicity: slope {slope(lambda d: d.fid_w(simple)):.2f}, "
        f"mean part {slope(lambda d: (simple @ d.dmu) ** 2):.2f}"
    )

    figures.use_paper_style()
    fig, axes = figures.row(NAXES + 1)
    span = XLIM[1] - XLIM[0]  # square box, same units on both axes
    xs = np.array(XLIM)
    xticks = np.linspace(*XLIM, NTICKS)
    for i, ax in enumerate(axes[:NAXES]):
        x, y = fid, charge[:, i]
        fit = linregress(x, y)

        # y window: same span as x, centered on a round 0.5 multiple near the data
        base = round(y.mean() * 2) / 2
        yticks = np.linspace(base - span / 2, base + span / 2, NTICKS)
        ax.plot(xs, fit.slope * xs + fit.intercept, "k-", lw=1.1, zorder=2)
        ax.plot(xs, (yticks[0], yticks[-1]), color="0.4", ls="--", lw=0.8, zorder=1)
        ax.scatter(x, y, color=REF_COLOR, s=16, zorder=3)

        ax.set(xlim=XLIM, ylim=(yticks[0], yticks[-1]))
        # every other x label, so five fit a narrow panel
        ax.set_xticks(
            xticks, [f"{t:g}" if k % 2 == 0 else "" for k, t in enumerate(xticks)]
        )
        ax.set_yticks(yticks)
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.set(
            xlabel="FD",
            ylabel=f"FD($e_{i + 1}$)",
            title=f"$e_{i + 1}$: slope {fit.slope:.2f}",
        )

    ax, D = axes[-1], share.size
    ax.axhline(1, color="0.4", ls="--", lw=0.8, zorder=1)
    ax.plot(np.arange(1, D + 1), share, color=REF_COLOR, lw=1.2, zorder=3)
    ax.set(xlim=(0, D), ylim=(0, 2), xlabel="eigenvector $i$", ylabel="$s_{1:i}$")
    ax.set_xticks(np.linspace(0, D, NTICKS), ["0", "", "$D/2$", "", "$D$"])
    ax.set_yticks(np.linspace(0, 2, NTICKS))
    ax.set_title("cumulative slope")
    ax.set_axisbelow(True)
    ax.grid(True)

    figs = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(figs, exist_ok=True)
    out = os.path.join(figs, f"reversal_shares_{args.reference}.pdf")
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
