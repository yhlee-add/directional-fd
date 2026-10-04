"""
Directional FID on proteins: name the top displacement directions of protein FID
on La-Proteina against the ESM3 reference, and track them across the step sweep.

Built on the same grouping and concept-space machinery as
scripts/iclr27/interpret_directions.py. `--space` picks where
the naming runs: the joint PCA-32 space that defines protein FID (default, the paper's
main text), with the optimal transport recomputed there, or the raw 1536-d ESM3 space
(the paper's appendix). Concept axes come from CATH S40 real domains in the same space,
projected onto the PCA basis for pca32: a `CategoryGrouper` splits the helix-vs-strand
axis by CATH class (`c`, 1 mainly-alpha vs 2 mainly-beta), a `QuantileGrouper` splits
each structural metric at its terciles, and `GroupConcepts` turns them into a
`ConceptSpace` whose atoms a discovered direction is decomposed over.
A `RandomGrouper` dictionary is the null that calibrates the charges.

The reference and CATH S40 sets are prepared on first run. Any set without an
esm3.npy is embedded then (~45 min on one GPU for the CATH set; ESM3Extractor embeds
one structure per file, so row i is file i and prompts.csv row i). Structural metrics
cache alongside as geometry.npy via GeometryExtractor. Reruns read both.
"""

import argparse
import os
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from core.data import CATHDomains, ESMTrainRef
from core.features.extractors import ESM3Extractor, GeometryExtractor
from core.features.metrics import load_cache
from core.features.directions.ot import (
    displacement,
    Displacement,
    gaussian,
    shift_spread,
)
from core.features.directions.grouping import (
    CategoryGrouper,
    QuantileGrouper,
    RandomGrouper,
)
from core.features.directions.semantics import GroupConcepts, ConceptSpace

warnings.filterwarnings("ignore")

REF = ESMTrainRef().get_path()
CATH = CATHDomains().get_path()
GEN = "samples/iclr27/laproteina_steps{}_467"
STEPS = [100, 200, 300, 400, 500, 600]
# lasso strength per space; the PCA-32 concept axes are collinear enough that a weak
# penalty trades large opposite weights among the helix, strand, and coil fractions
ALPHA = {"pca32": 1e-3, "raw": 1e-5}

unit = lambda v: F.normalize(v, dim=0)
cosv = lambda a, b: F.cosine_similarity(a, b, dim=0).item()


def geometry_frame(path: str) -> pd.DataFrame:
    """
    Per-structure metrics as a named frame, cached as geometry.npy.
    """
    feats = load_cache(GeometryExtractor(), path).cpu().numpy()
    return pd.DataFrame(feats, columns=GeometryExtractor.FIELDS)


def pca_fit(ref: torch.Tensor, gen: torch.Tensor, dim: int = 32) -> PCA:
    """
    The joint top-`dim` PCA of ref and gen, the actual protein-FID space.
    """
    return PCA(dim, svd_solver="full").fit(torch.cat([ref, gen]).cpu().numpy())


def to_pca(ref: torch.Tensor, gen: torch.Tensor, *more: torch.Tensor, dim: int = 32):
    """
    Project the sets onto the joint top-`dim` PCA basis of ref and gen.
    """
    p = pca_fit(ref, gen, dim)
    t = lambda x: torch.from_numpy(p.transform(x.cpu().numpy())).double()
    return tuple(t(x) for x in (ref, gen, *more))


def denominators(ref: torch.Tensor, gen: torch.Tensor):
    """
    FID, its mean-shift share, and the top-eigenvector share, raw and in PCA-32, then
    the raw directional FD of the PCA-32 subspace, which bounds protein FID from above.
    """
    print("\n=== denominators ===")
    for name, (a, b) in (("raw", (ref, gen)), ("pca32", to_pca(ref, gen))):
        d = displacement(a, b)
        FID = d.fid().item()
        ms = d.dmu.dot(d.dmu).item() / FID
        e1 = d.fid_w(d.eigen_axes()[0]).item() / FID
        print(f"{name:5s} FID {FID:>12,.0f}  mean-shift {ms:.0%}  eigvec1 {e1:.0%}")
    U = torch.from_numpy(pca_fit(ref, gen).components_).to(ref)
    M = displacement(ref, gen).M
    print(f"raw FD of the PCA-32 subspace {(U @ M @ U.mT).trace():,.0f}")


def diagnose_e1(r: torch.Tensor, g: torch.Tensor):
    """
    What kind of direction eigenvector 1 is: where it sits in the reference variance
    spectrum and its mass on the top reference principal components, evidence that it
    is a bulk reference direction rather than a thin off-manifold offset. Its mean
    shift and spread are read against the other axes in name_directions.
    """
    e1 = displacement(r, g).eigen_axes()[0]
    Sr = gaussian(r)[1]
    lam, Q = torch.linalg.eigh(Sr)  # ascending eigenvalues
    lam, Q = lam.flip(0), Q.flip(1)  # PC1 = most variance
    vr = (e1 @ Sr @ e1).item()
    mass = (Q.mT @ e1) ** 2
    pcs = ", ".join(f"PC{int(i) + 1} {mass[i]:.0%}" for i in mass.topk(2).indices)
    print(
        f"\ne1 diagnostic: above {(lam < vr).float().mean():.1%} of the {lam.numel()} "
        f"ref directions by variance; mass {pcs}"
    )


def name_directions(
    disp: Displacement,
    ref: torch.Tensor,
    gen: torch.Tensor,
    space: ConceptSpace,
    null: ConceptSpace,
    alpha: float,
):
    """
    Charge, cosine, and sparse decomposition of the top eigendirections of M over the
    concept dictionary, with how the two sets differ along each (Cohen's d of the
    projections and the spread ratio, both read from the reference and generated
    embeddings), then the secondary-structure and length charges against the null.
    """
    e = disp.eigen_axes()
    FID = disp.fid().item()
    base = disp.fid_w_baseline().item()
    mean_shift = disp.dmu.dot(disp.dmu).item() / FID
    print(
        f"\n=== eigenvector naming (dim {len(disp.dmu)}, FID {FID:,.0f}, "
        f"mean-shift {mean_shift:.0%}) ==="
    )

    rows = []
    for name, w in zip(space.names, space.rows):
        s = disp.fid_w(w).item()
        rows.append(
            {"axis": name, "share": s / FID, "xbase": s / base}
            | {f"cos(e{k + 1})": cosv(w, e[k]) for k in range(3)}
            | {"cos(dmu)": cosv(w, disp.dmu)}
        )
    signed = {c: "{:+.2f}".format for c in rows[0] if c.startswith("cos")}
    print(
        pd.DataFrame(rows).to_string(
            index=False,
            formatters={"share": "{:.1%}".format, "xbase": "{:.0f}".format, **signed},
        )
    )

    # most collinear concept pairs, which can trade weight in the sparse fit
    G = space.rows @ space.rows.mT
    i, j = torch.triu_indices(*G.shape, 1)
    pairs = G[i, j].abs().topk(4).indices
    print(
        "most collinear concepts: "
        + ", ".join(
            f"{space.names[i[p]]}/{space.names[j[p]]} {G[i[p], j[p]]:+.2f}"
            for p in pairs
        )
    )

    Q = torch.linalg.qr(space.rows.mT).Q  # basis of the concept span
    for i in range(3):
        cover = (Q.mT @ unit(e[i])).norm().item() ** 2
        cos_r, terms = space.decompose(e[i], alpha)
        top = ", ".join(f"{n}{w:+.2f}" for n, w in terms)
        share = disp.fid_w(e[i]).item() / FID
        cohen, frac, ratio, projected = shift_spread(disp, ref, gen, e[i])
        print(
            f"eigvec{i + 1} (share {share:.1%}) span {cover:.0%} | "
            f"decompose cos {cos_r:.2f}: {top}\n"
            f"          shift d {cohen:+.2f} ({frac:.0%} of FD_w), "
            f"spread {ratio:.2f}x, "
            f"projections explain {projected:.0%} of FD_w, "
            f"|cos| to the mean difference {(e[i] @ disp.mean_axis()).abs():.3f}"
        )

    def concept(name: str):
        w = space.rows[space.names.index(name)]
        cohen, frac, ratio, _ = shift_spread(disp, ref, gen, w)
        return disp.fid_w(w).item() / FID, cohen, frac, ratio

    nch = torch.stack([disp.fid_w(r) for r in null.rows]) / FID
    ss_share, ss_d, _, _ = concept("SS")
    print(
        f"\nSS axis share {ss_share:.1%} (d {ss_d:+.2f}) "
        f"vs null mean {nch.mean():.1%} 95pct {nch.quantile(0.95):.1%}"
    )
    # length ("n") is grid-sampled over {100..500}, ~94 each (mean ~299), matching the
    # reference's own 100-500 range, so it should carry little charge; measured to confirm.
    n_share, n_d, n_frac, n_ratio = concept("n")
    print(
        f"length axis share {n_share:.1%}  shift d {n_d:+.2f} "
        f"({n_frac:.0%} of FD_w), spread {n_ratio:.2f}x"
    )


def composition(gen: pd.DataFrame):
    """
    Generated-only structural composition, descriptive color that needs no reference:
    the mean geometry fields and the share of samples with no beta strand.
    """
    print("\n=== generated composition (step set) ===")
    print(
        gen[list(GeometryExtractor.FIELDS)]
        .mean()
        .to_string(float_format="{:.3f}".format)
    )
    print(f"zero-strand fraction: {(gen['strand'] == 0).mean():.0%}")


def sweep(ref: torch.Tensor, cath: pd.DataFrame, esm3: ESM3Extractor):
    print("\n=== per-nsteps sweep ===")
    e1s, rows = {}, []
    for s in STEPS:
        gen = load_cache(esm3, GEN.format(s)).double()
        d = displacement(ref, gen)
        e1s[s] = d.eigen_axes()[0]
        m = geometry_frame(GEN.format(s)).mean(numeric_only=True)
        rows.append(
            {
                "nsteps": s,
                "FID_raw": d.fid().item(),
                "FID_pca32": displacement(*to_pca(ref, gen)).fid().item(),
            }
            | {c: m[c] for c in GeometryExtractor.FIELDS}
        )
    # CATH domains are single folds, not the whole-chain FID reference, so this row
    # is an illustrative composition anchor for the drift, not a discrepancy target.
    rows.append(
        {"nsteps": "cath*"} | {c: cath[c].mean() for c in GeometryExtractor.FIELDS}
    )
    pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x:.3f}")
    print(pd.DataFrame(rows).set_index("nsteps").T.to_string())
    print(
        "e1 stability cos(e1@s, e1@400):",
        {s: round(cosv(e1s[s], e1s[400]), 3) for s in STEPS},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step", type=int, default=400, help="generated set to name eigenvectors on"
    )
    parser.add_argument(
        "--space",
        choices=ALPHA,
        default="pca32",
        help="space to name eigenvectors in: the joint PCA-32 of protein FID or raw ESM3",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="lasso strength for ConceptSpace.decompose (default per space)",
    )
    args = parser.parse_args()

    for dataset in (ESMTrainRef(), CATHDomains()):
        dataset.prepare()
    esm3 = ESM3Extractor()
    ref = load_cache(esm3, REF).double()
    pool = load_cache(esm3, CATH).double()
    frame = geometry_frame(CATH).assign(
        c=pd.read_csv(os.path.join(CATH, "prompts.csv"))["c"].to_numpy()
    )

    groupers = {
        "SS": CategoryGrouper("c", 2, 1),
        **{m: QuantileGrouper(m) for m in GeometryExtractor.FIELDS},
    }
    gen = load_cache(esm3, GEN.format(args.step)).double()
    denominators(ref, gen)

    r, g, cath = (ref, gen, pool) if args.space == "raw" else to_pca(ref, gen, pool)
    named = GroupConcepts(groupers, frame, cath)
    null = GroupConcepts({f"rand{i}": RandomGrouper(i) for i in range(50)}, frame, cath)
    diagnose_e1(r, g)
    alpha = args.alpha or ALPHA[args.space]
    name_directions(displacement(r, g), r, g, named, null, alpha)
    composition(geometry_frame(GEN.format(args.step)))
    sweep(ref, frame, esm3)


if __name__ == "__main__":
    main()
