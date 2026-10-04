"""
Shared style for the paper figures: one width, one font scale, square panels.

Call ``use_paper_style()`` before drawing and build the axes with ``grid()`` or ``row()``.
Every figure is then drawn at one common width and font size and included at
``\\linewidth``, so they all scale down by the same factor and a font point size renders
the same across figures. Never let ``savefig`` rescale (no ``bbox_inches="tight"``),
which would break that shared scale.

Margins are in inches, not figure fractions: the labels they hold have a fixed size in
points, so the room they need does not grow or shrink with the figure. Panels are square
via ``set_box_aspect(1)``, and grids with the same column count line up column-for-column
when stacked on a page.
"""

import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import ScaledTranslation

FONT = {
    # Times to match the body text; Nimbus Roman / TeX Gyre Termes are its free clones
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
}


def use_paper_style() -> None:
    # Pin matplotlib's PDF CreationDate so a redraw only changes the file when the
    # figure itself changed, not on every run. setdefault lets an explicit env win.
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    plt.rcParams.update(FONT)


# Per-paper layout, in inches. width is the authoring width, drawn wider than the
# paper's \linewidth so FONT prints smaller once included. left/bottom hold the tick and
# axis labels, top a title, and wspace/hspace the next panel's labels. ylabel pins each
# y label that far from its spine: matplotlib places it just past the panel's widest
# tick label, so it drifts with the ticks, and it only has to clear the tick labels
# level with it, not the end ticks above or below.
PRESETS: dict[str, dict[str, float]] = {
    "iclr27": dict(
        width=5.5 * 1.25,  # 1.25x \linewidth, so 10pt labels print at 8pt
        left=0.515,
        right=0.063,
        bottom=0.46,
        top=0.25,
        wspace=0.64,
        hspace=0.45,
        ylabel=0.34,
    ),
}


def grid(nrows: int, ncols: int, preset: str = "iclr27", **kwargs):
    """
    An nrows x ncols grid of square panels, laid out by a PRESETS entry. The height is
    solved so each cell is exactly square, so no space is wasted and set_box_aspect(1)
    has nothing to shrink. Pass preset keys (inches) to override.
    """
    m = {**PRESETS[preset], **kwargs}
    w = m["width"]
    box = (w - m["left"] - m["right"] - (ncols - 1) * m["wspace"]) / ncols
    height = m["bottom"] + nrows * box + (nrows - 1) * m["hspace"] + m["top"]
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, height))
    fig.subplots_adjust(
        left=m["left"] / w,
        right=1 - m["right"] / w,
        bottom=m["bottom"] / height,
        top=1 - m["top"] / height,
        wspace=m["wspace"] / box,
        hspace=m["hspace"] / box,
    )
    shift = ScaledTranslation(-m["ylabel"], 0, fig.dpi_scale_trans)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_box_aspect(1)
        ax.yaxis.set_label_coords(0, 0.5, ax.transAxes + shift)
    return fig, axes


row = partial(grid, 1)  # a 1xn row of square panels: row(n) is grid(1, n)
