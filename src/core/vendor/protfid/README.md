# protfid

Vendored copy of [protfid](https://github.com/ffaltings/protfid), the protein Frechet Inception Distance from "Protein FID: Improved Evaluation of Protein Structure Generative Models" (Faltings et al., arXiv 2505.08041).
Structures are embedded with ESM3 structure tokens (mean-pooled, 1536-d), projected to 32 PCA dimensions fit jointly on the reference and evaluation sets, and compared with the standard Frechet distance.

## Provenance

Source: `github.com/ffaltings/protfid` @ `d6d8ac84a6f901f7d60484a50b3281b17b23b99a` (2025-06-24).
Only `protfid/fid.py` and `protfid/residue_constants.py` are vendored.
The examples, CLI packaging, and plotting are not.

## Local modifications

`fid.py`: `import protfid.residue_constants` became `from . import residue_constants`, so the module resolves under `core.vendor.protfid`.

`fid.py`: `load_chains` uses a `ThreadPoolExecutor` instead of a `ProcessPoolExecutor`.
The extractor runs on every accelerate rank with CUDA/NCCL already live, where forking worker processes is unsafe.

`fid.py`: `compute_fid_from_embeddings` fits PCA with `svd_solver="full"`.
At our embedding shape the default `"auto"` picks the randomized solver, whose unseeded projection makes the FID vary ~0.02% between identical calls; `"full"` is exact and deterministic.

## Reference set

The paper's PDB reference is published as precomputed embeddings at `zenodo.org/records/15660186/files/esm_train_ref.ckpt`, a `torch.save` of a raw [N, 1536] tensor.
The Methods text says 4,991 reference structures, but the released ckpt holds 4,943 (shape [4943, 1536]).
`fid.py:main` downloads it on demand.

## Runtime dependencies

`esm`, `scikit-learn`, `scipy`, `numpy`, `torch`, `tqdm`, `appdirs`, `requests`, all in the project environment.
ESM3 weights download from HuggingFace on first use with no auth.

## License

The upstream repository ships no license file.
The paper it accompanies (arXiv 2505.08041) is released under Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.

