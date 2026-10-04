# Directional Fréchet distance

Code for the submission "What Does Fréchet Distance Measure? A Directional
Decomposition". It generates the image, video, and protein sets, computes the
features and Fréchet distances, and reproduces every table and figure.

## Environment

The Docker image provides CUDA 12.8, the Python environment, a separate La-Proteina
environment, and the MongoDB instance that Sacred logs every run to.

```bash
docker build -t directional-fd .
docker run -d --gpus=all --shm-size=8g -v $(pwd):/workspace directional-fd
docker exec -it <container> bash
cd /workspace
source /ddiff-base/py3.10-torch2.9.0/.venv/bin/activate
pip install -e . --no-deps --no-build-isolation
```

Run every command from the repository root. Datasets and model weights download on
first use.

## 1. Generate the sets and their features (GPU)

Each run generates a set with `experiments/generate.py`, computes one metric, and
caches the reference and generated features next to the samples in `samples/`.

```bash
# Images: SD1.5 + DDIM at 15, 20, ..., 50 steps on six references, scored with
# FID, FD over CLIP and DINOv2 features, and ImageReward
python scripts/iclr27/image_sweep.py

# Video: ModelScope zero-shot on the UCF-101 class names, FVD over I3D and VideoMAE-v2
accelerate launch experiments/generate.py with configs/modelscope.yaml
accelerate launch experiments/generate.py with configs/modelscope.yaml metric_name=CDFVD

# Proteins: La-Proteina at 100 to 600 steps, 467 structures each, Protein FID
python scripts/iclr27/protein_sweep.py
```

Generate each protein set in one uninterrupted run on a single GPU. La-Proteina draws
a whole set from one seed and spreads it over five chain lengths, while the pipeline
resumes a partial set by generating only its missing structures, so a resumed or
multi-GPU run gives different structures than a single full run. Delete a partial
`samples/iclr27/laproteina_steps*_467/` directory and rerun it rather than resuming.

## 2. Reproduce the tables and figures

The analysis reads the cached features. Once they exist it runs on CPU with
`ACCELERATE_USE_CPU=true`; figures are written to `scripts/iclr27/figs/`. The first
run of `interpret_fvd.py` embeds the distorted UCF-101 probe sets, and the first run
of `protein_directions.py` embeds the CATH S40 domains (about 45 minutes on one GPU).

| Paper | Command |
| --- | --- |
| Figure 1 | `python scripts/iclr27/reversal_sweep.py` |
| Table 1, Figure 2, Table 4, Appendix D | `python scripts/iclr27/interpret_directions.py --plot` |
| DINOv2 check (Section 4.1.1) | `python scripts/iclr27/interpret_directions.py --extractor dinov2` |
| Figure 3, Section 4.1.2, Appendix D slope | `python scripts/iclr27/reversal_shares.py --reference COCO30K` |
| Figure 4, Appendix E | `python scripts/iclr27/reversal_cumulative.py` and `python scripts/iclr27/reversal_basis.py` |
| Tables 2, 5, 6 | `python scripts/iclr27/interpret_fvd.py --generated samples/modelscope` |
| Table 3, Section 4.3, Table 7 (PCA-32 rows), Appendix G | `python scripts/iclr27/protein_directions.py` |
| Table 7 (raw rows) | `python scripts/iclr27/protein_directions.py --space raw` |

The directional Fréchet distance itself is `src/core/features/directions/ot.py`:
`displacement(ref, gen)` returns the displacement second-moment matrix $M$, and
`fid_w(w)` its quadratic form along a direction.
