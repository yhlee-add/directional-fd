import argparse
import itertools
import os
import shutil
import subprocess
from typing import Any

import yaml

# Part 1: User-editable config lists
# ICLR27 protein sweep: La-Proteina unconditional generation over the ESM3 reference
# set, varying only the number of integration steps (the image sweep's N, ported to
# protein structure). FaltingsProteinFID is the distributional metric.

MODEL: dict[str, Any] = {
    "model_name": "LaProteinaModel",
    "model_kwargs": {"base_config": "inference_ucond_notri"},
}

NSTEPS = [100, 200, 300, 400, 500, 600]

DATASETS = ["ESMTrainRef"]

EXPERIMENTS: list[dict[str, Any]] = [
    MODEL
    | {
        "solver_name": "LaProteinaSolver",
        "solver_kwargs": {"nsteps": nsteps},
        "dataset_name": dataset,
        "path": f"samples/iclr27/laproteina_steps{nsteps}_467",
    }
    for dataset in DATASETS
    for nsteps in NSTEPS
]

METRICS: list[dict[str, Any]] = [
    {"metric_name": "FaltingsProteinFID"},
]

# Match the protein FID paper setup (467 generated structures)
COMMON: dict[str, Any] = {
    "num_samples": 467,
    "batch_size": 467,
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
        f"iclr27_{c['path'].removeprefix('samples/iclr27/')}_{c['metric_name']}"
        for c in combinate()
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
        description="Sweep nsteps for La-Proteina over the protein reference set."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write YAML configs without launching experiments.",
    )
    run(dry_run=parser.parse_args().dry_run)
