"""
Sign reversal under two standard knobs on SD/COCO, for the Preliminaries figure.

Four panels in a row.  The two knobs each get a FID panel and an ImageReward panel,
so a single y-axis carries each metric and the reversal reads as the two panels of a
knob moving the same way while their meanings disagree:

  guidance   CFG scale omega on SD1.5, from Table 8 of lee2025tortoise; FID and
             ImageReward both climb monotonically as omega grows.
  steps      our SD1.5 COCO-30k sampling-step sweep (scripts/iclr27/image_sweep.py);
             more steps raise FID while ImageReward keeps rising.

FID rises (worse) exactly where ImageReward rises (better): the metric moves opposite
to the per-sample quality it is meant to track.

Usage: python scripts/iclr27/reversal_sweep.py
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.utils import figures

NTICKS = 4  # y-axis tick count; first and last land on the frame

# Table 8 of lee2025tortoise (arXiv 2511.04117): SD1.5 + DDIM, CFG scale sweep.
OMEGA = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
OMEGA_FID = [8.438, 9.143, 10.644, 12.030, 13.222, 14.133, 14.902]
OMEGA_IR = [-0.28577, -0.09190, 0.00670, 0.07195, 0.11582, 0.14764, 0.17431]

# SD1.5 + DDIM on COCO30K, sampling-step sweep: FID and ImageReward of image_sweep.py.
STEPS = [15, 20, 25, 30, 35, 40, 45, 50]
STEP_FID = [13.3335, 13.6857, 14.0081, 13.9012, 13.8242, 14.4352, 14.2837, 14.7027]
STEP_IR = [0.0856843, 0.141696, 0.158, 0.159949, 0.162647, 0.169354, 0.168962, 0.172816]

FID_COLOR, IR_COLOR = "crimson", "mediumblue"


def _ticks(lo, hi, n=NTICKS):
    """
    n ticks on a nice step enclosing [lo, hi], ends landing on the frame.

    The step is the smallest nice multiple (1, 2, 3, 5 x a power of ten) for
    which n consecutive multiples of it, anchored at floor(lo / step), still cover
    hi -- so the axis reads on round numbers with the data inside the frame. No 2.5,
    which would force an extra decimal onto every label and widen the tick column.
    """
    mag = 10 ** math.floor(math.log10((hi - lo) / (n - 1)))
    for m in (1, 2, 3, 5, 10, 20, 30, 50, 100):
        step = m * mag
        base = math.floor(lo / step) * step
        if base + step * (n - 1) >= hi - 1e-9:
            return [round(base + step * i, 10) for i in range(n)]


def axis_grid(ax, xticks, y):
    yt = _ticks(min(y), max(y))
    ax.set(xlim=(xticks[0], xticks[-1]), ylim=(yt[0], yt[-1]), xticks=xticks, yticks=yt)
    ax.set_box_aspect(1)
    ax.set_axisbelow(True)
    ax.grid(True)


def panel(ax, x, y, color, marker, xlabel, ylabel, xticks):
    ax.plot(x, y, "-", marker=marker, color=color, lw=1.2, ms=3, zorder=3)
    axis_grid(ax, xticks, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, color=color)


def main():
    figures.use_paper_style()
    fig, axes = figures.row(4, top=0.1)  # no titles
    OMEGA_X = r"guidance scale $\omega$"
    STEP_X = r"sampling steps $N$"
    FID_Y = r"FID $\downarrow$"
    IR_Y = r"ImageReward $\uparrow$"
    OMEGA_TICKS = [2, 4.5, 7, 9.5]
    STEP_TICKS = [10, 25, 40, 55]

    panel(axes[0], OMEGA, OMEGA_FID, FID_COLOR, "o", OMEGA_X, FID_Y, OMEGA_TICKS)
    panel(axes[1], OMEGA, OMEGA_IR, IR_COLOR, "s", OMEGA_X, IR_Y, OMEGA_TICKS)
    panel(axes[2], STEPS, STEP_FID, FID_COLOR, "o", STEP_X, FID_Y, STEP_TICKS)
    panel(axes[3], STEPS, STEP_IR, IR_COLOR, "s", STEP_X, IR_Y, STEP_TICKS)

    figs = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(figs, exist_ok=True)
    out = os.path.join(figs, "reversal_sweep.pdf")
    fig.savefig(out, dpi=300)
    fig.savefig(out[:-4] + ".png", dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
