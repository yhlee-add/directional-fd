import argparse
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

from core.features.directions.ot import displacement, shift_spread
from core.features.directions.grouping import ClipTextEncoder, ClipTextGrouper
from core.features.directions.semantics import GroupConcepts, ImageNetConcepts

from core.features.metrics import load_cache
from core.utils import figures
from core import data

# concept -> (positive prompts, negative prompts), ensembled per pole. The
# dictionary a discovered direction is sparsely decomposed over (SPLICE-style).
CONCEPTS = {
    "comp-simplicity": (
        [
            "a single subject on a plain background",
            "a simple clean iconic photo of one object",
            "a photo with one clear subject and an empty background",
            "a minimal close-up of a lone subject",
        ],
        [
            "a cluttered scene with many objects",
            "a busy messy photo full of things",
            "a crowded real-world scene with lots of clutter",
            "many overlapping objects and people",
        ],
    ),
    "num-objects": (["a photo of one object"], ["a photo of many objects"]),
    "crowd-density": (["a photo of one person"], ["a large crowd of many people"]),
    "has-people": (["a photo of people"], ["a photo with no people"]),
    "indoor-outdoor": (
        ["a photo taken indoors", "inside a room or building"],
        ["a photo taken outdoors", "an outdoor scene under the open sky"],
    ),
    "nature-urban": (
        ["a natural landscape, field or wilderness"],
        ["a city street or urban built environment"],
    ),
    "animals": (["a photo of an animal"], ["a photo with no animals"]),
    "food": (["a photo of food or a meal"], ["a photo with no food"]),
    "vehicles": (
        ["a photo of a vehicle, car, train or plane"],
        ["a photo with no vehicles"],
    ),
    "furniture": (
        ["a photo of furniture, chairs and tables"],
        ["a photo with no furniture"],
    ),
    "electronics": (
        ["a photo of electronic devices, a laptop, tv or phone"],
        ["a photo with no electronics"],
    ),
    "sports-equip": (
        ["a photo with sports equipment, a ball, racket or skateboard"],
        ["a photo with no sports equipment"],
    ),
    "scale-closeup": (
        ["an extreme close-up or macro photo"],
        ["a wide-angle distant establishing shot"],
    ),
    "colorful": (
        ["a vivid colorful saturated photo"],
        ["a muted grayish desaturated photo"],
    ),
    "warm-cool": (["a warm-toned orange and red photo"], ["a cool-toned blue photo"]),
    "bright-dark": (["a bright well-lit photo"], ["a dark dim low-light photo"]),
    "water-sky": (
        ["a photo dominated by sky, sea or water"],
        ["a photo with no sky or water"],
    ),
    "text-signs": (
        ["a photo containing text, signs or writing"],
        ["a photo with no text or signs"],
    ),
    "staged-candid": (
        ["a staged studio product photo"],
        ["a candid everyday real-world snapshot"],
    ),
    "centered": (
        ["a centered composition, subject in the middle"],
        ["an off-center cluttered composition"],
    ),
    "depth-blur": (
        ["a photo with a blurred background and shallow depth of field"],
        ["a photo with everything in sharp focus"],
    ),
    "weather": (
        ["clear sunny bright weather"],
        ["rain, snow, fog or overcast weather"],
    ),
    "night-day": (["a daytime scene"], ["a nighttime scene"]),
    "portrait-scene": (
        ["a portrait of a single person or animal face"],
        ["a wide scene with no clear main subject"],
    ),
}

# random-word poles for the null: an equal-size dictionary of arbitrary caption
# splits, decomposed the same way, to separate real semantics from basis richness.
VOCAB = """xylophone quarterly cerulean mitochondria asphalt tuesday velcro parabola
espresso tectonic lozenge syntax gravel meridian walnut flannel cobalt referendum
trombone sediment aperture molasses quorum tapestry kelvin isthmus gasket vellum""".split()

GENERATED = "samples/StableDiffusionModel_HuggingFaceSolver_COCO30K_steps50"
ALPHA = 5e-5  # concept-lasso strength, tuned to ~5-8 atoms
QUANTILE = 0.1  # fraction of reference images taken at each pole of a concept
CLASS_ALPHA = 7e-5  # class-lasso strength over unit W rows, tuned to a few classes

NTICKS = 5  # shared tick count per axis; first and last land on the frame
REF_COLOR = "crimson"  # reference color
GEN_COLOR = "mediumblue"  # generated color


def _ticks(lo, hi, n=NTICKS):
    """
    n equally spaced integer ticks enclosing [lo, hi] and 0, ends on the frame.

    The step is the smallest integer for which n consecutive multiples of it
    cover the data; anchoring the ticks on multiples of the step puts 0 on the
    grid, a fixed reference for the shift and spread readings.
    """
    lo, hi = np.floor(min(lo, 0)), np.ceil(max(hi, 0))
    step = np.ceil((hi - lo) / (n - 1))
    while np.floor(lo / step) * step + step * (n - 1) < hi:
        step += 1
    base = np.floor(lo / step) * step
    return np.linspace(base, base + step * (n - 1), n)


def square_grid(ax, lo, hi):
    ticks = _ticks(lo, hi)
    ax.set(xlim=(ticks[0], ticks[-1]), ylim=(ticks[0], ticks[-1]))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_axisbelow(True)
    ax.grid(True)


def yaxis_grid(ax, lo, hi):
    ticks = _ticks(lo, hi)
    ax.set_ylim(ticks[0], ticks[-1])
    ax.set_yticks(ticks)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")


def plot_projection(ax, ref, gen, w, title=""):
    """
    A jointplot of the paired sets projected onto w in one square axes: each
    caption i is a point (ref_i . w, gen_i . w), since ref[i] and gen[i] are the
    same prompt.  The two marginals live inside the scatter, drawn in a blended
    transform (data coords along w, axes-fraction coords for bar height) so they
    grow inward from the bottom (real) and left (generated) edges.  The y=x line
    is the no-shift reference: mass above it drifted toward +w under generation.
    """
    w = w / w.norm()
    pr, pg = (ref @ w).cpu().numpy(), (gen @ w).cpu().numpy()
    lo, hi = min(pr.min(), pg.min()), max(pr.max(), pg.max())
    edges = np.linspace(lo, hi, 60)
    centers, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
    hr, _ = np.histogram(pr, bins=edges)
    hg, _ = np.histogram(pg, bins=edges)
    frac = 0.22 / max(hr.max(), hg.max())  # tallest bar spans 22% of the axis

    # each point is a (ref, gen) pair, not one side, so the cloud is neutral;
    # the crimson/blue identity lives on the marginals below.  rasterized so the
    # 30k-point cloud embeds as one image in the PDF (axes/text stay vector).
    ax.scatter(
        pr,
        pg,
        s=1,
        alpha=0.2,
        color="#111",
        edgecolors="none",
        zorder=1,
        rasterized=True,
    )
    square_grid(ax, lo, hi)
    a, b = ax.get_xlim()
    ax.plot([a, b], [a, b], color="0.4", lw=1, ls="--", zorder=0)
    ax.set(title=title, xlabel="Reference", ylabel="Generated")

    xdata_yaxes = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ydata_xaxes = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.bar(
        centers,
        hr * frac,
        width,
        transform=xdata_yaxes,
        color=REF_COLOR,
        alpha=0.6,
        zorder=2,
    )
    ax.barh(
        centers,
        hg * frac,
        width,
        transform=ydata_xaxes,
        color=GEN_COLOR,
        alpha=0.6,
        zorder=2,
    )


def plot_violin(ax, ref, gen, w, title=""):
    """
    Ref and gen projections onto w as two violins, real vs generated side by
    side; the tall axes reads the shift as a vertical offset of the two bodies.
    """
    w = w / w.norm()
    pr, pg = (ref @ w).cpu().numpy(), (gen @ w).cpu().numpy()
    parts = ax.violinplot([pr, pg], showextrema=False, showmeans=True)
    for body, c in zip(parts["bodies"], (REF_COLOR, GEN_COLOR)):
        body.set_facecolor(c)
        body.set_alpha(0.6)
    ax.set_xticks([1, 2], ["ref", "gen"])
    yaxis_grid(ax, min(pr.min(), pg.min()), max(pr.max(), pg.max()))
    ax.set_title(title)


def interpret(reference: str, generated: str, plot: bool, extractor: str) -> None:
    ref_path = getattr(data, reference)().get_path()
    ref = load_cache(extractor, ref_path).double()
    gen = load_cache(extractor, generated).double()

    disp = displacement(ref, gen)
    # Directions to name: the top eigenvectors of M, its orthogonal charge-ordered
    # directions, whose quadratic form is directional FID and trace is FID.
    eigvecs = disp.eigen_axes()[:3]

    ref_clip = load_cache("clip", ref_path)
    encoder = ClipTextEncoder()  # loaded once, shared across groupers

    def groupers(prompts: dict) -> dict[str, ClipTextGrouper]:
        # each pole pair splits the reference images by CLIP similarity
        return {
            k: ClipTextGrouper(encoder, p, n, q=QUANTILE)
            for k, (p, n) in prompts.items()
        }

    # Two equal-size concept spaces: the named concepts, and a null of random-word
    # splits that calibrates how much any reconstruction actually means. The axes
    # live in the extractor's feature space (ref), grouped by CLIP nearness (ref_clip).
    rng = random.Random(0)
    nonsense_prompts = {
        f"rand{i}": ([f"a photo of {a}"], [f"a photo of {b}"])
        for i, (a, b) in enumerate(rng.sample(VOCAB, 2) for _ in CONCEPTS)
    }

    named = GroupConcepts(groupers(CONCEPTS), ref_clip, ref)
    nonsense = GroupConcepts(groupers(nonsense_prompts), ref_clip, ref)
    # ImageNet-class tools live in inception space; skip them on other extractors
    imagenet = ImageNetConcepts(ref.device) if extractor == "inception" else None

    fid = disp.fid()
    print(f"FID {fid:.3f}  baseline {disp.fid_w_baseline():.5f}")
    simple = named.rows[named.names.index("comp-simplicity")]
    print(
        f"comp-simplicity axis: {disp.fid_w(simple) / fid:.1%} of FID, "
        f"cos(e_1) {simple @ eigvecs[0]:+.2f}"
    )

    if imagenet is not None:
        cls_fid = disp.fid_subspace(imagenet.W)
        print(f"row(W) subspace: FID {cls_fid:.3f} ({cls_fid / fid:.1%} of FID)")

    if plot:
        figures.use_paper_style()
        fig, axes = figures.row(len(eigvecs))
        vfig, vaxes = plt.subplots(1, len(eigvecs), figsize=(2.5 * len(eigvecs), 6))

    for i, t in enumerate(eigvecs, start=1):
        charge = disp.fid_w(t)  # the eigenvalue: FID carried along this eigenvector
        print(
            f"\neig_{i}: FID_w {charge:.3f} ({charge / fid:.1%} of FID), "
            f"|cos| to the mean difference {(t @ disp.mean_axis()).abs():.3f}"
        )

        cos, terms = named.decompose(t, ALPHA)
        null_cos, _ = nonsense.decompose(t, ALPHA)
        # CONCEPTS decomposition
        print(
            f"    concepts (cos {cos:.3f} vs nonsense {null_cos:.3f}, {len(terms)} atoms): "
            + "  ".join(f"{w:+.2f} {k}" for k, w in terms)
        )

        if imagenet is not None:
            # Top margin classes
            sig = imagenet.signature(t)
            print("    imagenet: " + ", ".join(f"{c}({r:+.1f})" for c, r in sig))

            # Imagenet classes decomposition
            cls_cos, cls_atoms = imagenet.decompose(t, CLASS_ALPHA)
            print(
                f"    class-lasso (cos {cls_cos:.3f}, {len(cls_atoms)} classes): "
                + "  ".join(f"{v:+.2f} {c}" for c, v in cls_atoms)
            )

        # Projection to w
        cohen, frac, ratio, projected = shift_spread(disp, ref, gen, t)
        spread = "more diverse" if ratio > 1 else "more concentrated"
        print(
            f"    gen vs ref: mean-shift d {cohen:+.2f} ({frac:.0%} of FID_w), "
            f"var {ratio:.2f}x -> generated {spread}; "
            f"projections explain {projected:.0%} of FID_w"
        )
        if plot:
            title = f"$e_{i}$"
            plot_projection(axes[i - 1], ref, gen, t, title=title)
            plot_violin(vaxes[i - 1], ref, gen, t, title=f"eig_{i}")

    if plot:
        vaxes[0].set_ylabel("proj . w")
        figs = os.path.join(os.path.dirname(__file__), "figs")
        os.makedirs(figs, exist_ok=True)
        for f, name in ((fig, "projections"), (vfig, "violins")):
            f.savefig(os.path.join(figs, f"{name}.pdf"), dpi=300)
            plt.close(f)
            print(f"\nsaved {os.path.join(figs, name)}.pdf")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference", default="COCO30K", help="reference dataset class name"
    )
    p.add_argument("--generated", default=GENERATED, help="generated set path")
    p.add_argument(
        "--plot", action="store_true", help="save projection and violin plots"
    )
    p.add_argument(
        "--extractor",
        default="inception",
        help="feature extractor to interpret in (inception enables ImageNet-class tools)",
    )
    args = p.parse_args()
    interpret(args.reference, args.generated, args.plot, args.extractor)


if __name__ == "__main__":
    main()
