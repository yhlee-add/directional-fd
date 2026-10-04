"""
Where the reversal and the cleanup live along the fixed-basis spectrum.

For each reference we freeze the 50-step displacement eigenbasis and fit, across the
step sweep, the least-squares trend of each cumulative charge sum
C_{1:i}(step) = sum_{j<=i} w_j^T M(step) w_j.

Plotting that trend against i shows where the FID change is built: it climbs over the
head axes (the reversal) and drains back over the tail (the cleanup), ending on the
total FID trend at i = D.

All six references as a 2x3 grid, for the appendix.

Usage: ACCELERATE_USE_CPU=true python scripts/iclr27/reversal_cumulative.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from core.features.directions.sweep import StableDiffusionSweep
from core.utils import figures

REFS = ["COCO30K", "Flickr30K", "ImageNet", "MJHQ30K", "CelebAHQ", "FFHQ512"]
STEPS = list(range(15, 55, 5))

steps = np.array(STEPS, dtype=float)
sc = steps - steps.mean()


def trend(Y: np.ndarray) -> np.ndarray:
    """
    Least-squares slope per step, per column of Y [nsteps, D].
    """
    return (sc @ (Y - Y.mean(0))) / (sc @ sc)


def collect(sweep: StableDiffusionSweep, ref: str) -> np.ndarray:
    """
    Cumulative charge trends [3, D] for total / mean / covariance parts, in the
    fixed 50-step eigenbasis, along the step sweep.
    """
    axes = sweep[ref, 50].eigen_axes()  # rows: M(50) eigvecs, descending, oriented

    def diag(d):
        return torch.stack(
            [
                d.fid_w(axes),
                # (axes @ d.dmu) ** 2,
                # torch.einsum("ni,ij,nj->n", axes, d.Mbar, axes),
            ]
        )

    S = (
        torch.stack([diag(sweep[ref, n]) for n in STEPS]).cpu().numpy()
    )  # [nsteps, k, D]
    cum = np.cumsum(S, axis=2)
    return np.stack([trend(cum[:, k]) for k in range(S.shape[1])])  # [k, D]


XTICKS = list(range(0, 2049, 512))  # 5 ticks (4x4 grid)
YSTEP = 0.03  # common y tick interval, per-panel start


def window(lo: float, hi: float) -> np.ndarray:
    """
    As many y ticks as x ticks, YSTEP apart, centered on [lo, hi].
    """
    span = YSTEP * (len(XTICKS) - 1)
    start = round(((lo + hi) / 2 - span / 2) / YSTEP) * YSTEP
    return np.round(start + YSTEP * np.arange(len(XTICKS)), 10)


def draw(ax, C: np.ndarray, yticks: np.ndarray) -> None:
    """
    One cumulative-slope panel: beta_{1:i} against i, dashed at the total beta_FD.
    """
    ax.axhline(0, color="k", lw=0.8, zorder=2)
    ax.axhline(C[-1], color="0.4", ls="--", lw=0.8, zorder=1)
    ax.plot(np.arange(1, C.size + 1), C, color="crimson", lw=1.2, zorder=3)
    ax.text(
        0.97,
        0.97,
        "$\\beta_\\mathrm{FD} = " + f"{C[-1]:+.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.set(xlim=(XTICKS[0], XTICKS[-1]), xticks=XTICKS)
    ax.set_xticklabels(["0", "", "$D/2$", "", "$D$"])
    ax.set(ylim=(yticks[0], yticks[-1]), yticks=yticks, xlabel="eigenvector $i$")
    ax.set_ylabel(r"$\beta_{1:i}$")
    ax.set_axisbelow(True)
    ax.grid(True)


def main() -> None:
    figures.use_paper_style()
    sweep = StableDiffusionSweep("inception")
    store = {r: collect(sweep, r)[0] for r in REFS}

    # 2x3 whose columns align with the 3-in-a-row figures; the four diverse sets
    # share one window, the two faces keep their own
    diverse = REFS[:4]
    shared = window(
        min(store[r].min() for r in diverse), max(store[r].max() for r in diverse)
    )
    fig, axes = figures.grid(2, 3)
    for k, (r, ax) in enumerate(zip(REFS, axes.ravel())):
        C = store[r]
        draw(ax, C, shared if r in diverse else window(C.min(), C.max()))
        ax.set_title(r)
        if k % 3:  # y labels on the left column only
            ax.set_ylabel("")
        if k < 3:  # x labels on the bottom row only
            ax.set_xlabel("")

    figs = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(figs, exist_ok=True)
    out = os.path.join(figs, "reversal_cumulative.pdf")
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
