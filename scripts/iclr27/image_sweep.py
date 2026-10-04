import argparse
import itertools
import os
import shutil
import subprocess
from typing import Any

import yaml

# Part 1: User-editable config lists
# ICLR27 image sweep: SD1.5 + DDIM over the six image references, varying only the
# number of sampling steps. FID, FDCLIP, and FDDINOv2 cache the reference and generated
# features the analysis scripts read (inception.npy, clip.npy, dinov2.npy); ImageReward
# is the per-sample quality of Figure 1. The sample paths are the prefix of
# core.features.directions.sweep.StableDiffusionSweep.

MODEL_KEY = "stable-diffusion-v1-5/stable-diffusion-v1-5"

MODEL: dict[str, Any] = {
    "model_name": "StableDiffusionModel",
    "model_kwargs": {"model_key": MODEL_KEY, "target": "accelerator"},
}

NUM_INFERENCE_STEPS = [15, 20, 25, 30, 35, 40, 45, 50]

DATASETS = ["COCO30K", "Flickr30K", "ImageNet", "MJHQ30K", "CelebAHQ", "FFHQ512"]

EXPERIMENTS: list[dict[str, Any]] = [
    MODEL
    | {
        "solver_name": "HuggingFaceSolver",
        "solver_kwargs": {
            "model_key": MODEL_KEY,
            "scheduler": "DDIMScheduler",
            "num_inference_steps": steps,
        },
        "dataset_name": dataset,
        "path": f"samples/StableDiffusionModel_HuggingFaceSolver_{dataset}_steps{steps}",
    }
    for dataset in DATASETS
    for steps in NUM_INFERENCE_STEPS
]

METRICS: list[dict[str, Any]] = [
    {"metric_name": "FID"},
    {"metric_name": "FDCLIP"},
    {"metric_name": "FDDINOv2"},
    {"metric_name": "ImageReward"},
]

# One image per reference caption; the batch size the sample sets were generated with
COMMON: dict[str, Any] = {
    "batch_size": 30,
    "seed": 42,
}


# Part 2: Combinator


def combinate() -> list[dict[str, Any]]:
    """
    Cartesian product of EXPERIMENTS and METRICS, merged with COMMON.
    """
    return [
        COMMON | experiment | metric
        for experiment, metric in itertools.product(EXPERIMENTS, METRICS)
    ]


def experiment_names() -> list[str]:
    """
    Sacred experiment names produced by run(), in canonical order.
    """
    return [
        f"{c['path'].removeprefix('samples/')}_{c['metric_name']}" for c in combinate()
    ]


# Part 3: YAML writing + subprocess launch

BATCH_DIR = "configs/batch"


def run(dry_run: bool = False) -> None:
    if os.path.exists(BATCH_DIR):
        shutil.rmtree(BATCH_DIR)
    os.makedirs(BATCH_DIR)

    configs = combinate()
    names = experiment_names()

    for i, (name, config) in enumerate(zip(names, configs), 1):
        yaml_path = os.path.join(BATCH_DIR, f"{name}.yaml")

        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"\t[{i}/{len(configs)}] {name}")
        print(f"\tconfig: {yaml_path}")

        if dry_run:
            print("\t(dry run, skipping launch)")
            continue

        subprocess.run(
            [
                "accelerate",
                "launch",
                "experiments/generate.py",
                "-n",
                name,
                "with",
                yaml_path,
            ],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep sampling steps for SD1.5 over the six image references."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write YAML configs without launching experiments.",
    )
    run(dry_run=parser.parse_args().dry_run)
