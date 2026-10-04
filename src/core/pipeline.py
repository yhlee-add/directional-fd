from typing import Callable

from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
import pandas as pd

from core import data, models, solvers
from core.features import metrics
from core.utils.gloo import gloo_barrier, gloo_main_process_first
from core.utils.layout import done_indices

Condition = Callable[[list[int] | None], dict]

accelerator = Accelerator()


def load_dataset(
    dataset_name: str,
    dataset_kwargs: dict,
) -> data.Dataset:
    dataset: data.Dataset = getattr(data, dataset_name)(**dataset_kwargs)
    dataset.prepare()
    return dataset


def load_conditions(
    condition_kwargs: dict,
    seed: int,
    dataset: data.Dataset,
) -> Condition:
    df = pd.read_csv(dataset.get_csv_path())
    df["seeds"] = df.index + seed

    def condition(batch: list[int] | None = None) -> dict:
        return (df.iloc[batch] if batch else df).to_dict("list") | condition_kwargs

    return condition


def generate_samples(
    path: str,
    num_samples: int | None,
    batch_size: int,
    solver_name: str,
    solver_kwargs: dict,
    model_name: str,
    model_kwargs: dict,
    condition: Condition,
) -> None:
    if num_samples is None:
        num_samples = len(condition()["seeds"])

    # Resume: skip indices already on disk, so a restart continues a partial run.
    done = done_indices(path)
    indices = [i for i in range(num_samples) if i not in done]
    if not indices:
        return

    model: models.DiffusionModel = getattr(models, model_name)(**model_kwargs)
    solver: solvers.Solver = getattr(solvers, solver_name)(**solver_kwargs)

    with accelerator.split_between_processes(indices) as index_split:
        batches = [
            index_split[i : i + batch_size]
            for i in range(0, len(index_split), batch_size)
        ]
        num_batches = max(gather_object([len(batches)]))
        for step in tqdm(range(num_batches), disable=not accelerator.is_main_process):
            if step < len(batches):
                batch = batches[step]
                samples = solver.generate(model, **condition(batch)).output
                for i, sample in zip(batch, samples):
                    model.save_output(sample, path, i)
            gloo_barrier()


def compute_metric(
    path: str,
    num_samples: int | None,
    metric_name: str,
    metric_kwargs: dict,
    dataset: data.Dataset,
    condition: Condition,
) -> float:
    resolved_kwargs = {"ref_path": dataset.get_path()} | metric_kwargs
    metric: metrics.Metric = getattr(metrics, metric_name)(**resolved_kwargs)
    return metric.compute(
        path, **condition(None if num_samples is None else list(range(num_samples)))
    )


def assess(
    path: str,
    num_samples: int | None,
    batch_size: int,
    solver_name: str,
    solver_kwargs: dict,
    model_name: str,
    model_kwargs: dict,
    metric_name: str,
    metric_kwargs: dict,
    condition_kwargs: dict,
    seed: int,
    dataset_name: str,
    dataset_kwargs: dict,
) -> float:

    # Rank 0 converts, the rest find it done
    with gloo_main_process_first():
        dataset = load_dataset(dataset_name, dataset_kwargs)
        condition = load_conditions(condition_kwargs, seed, dataset)

    generate_samples(
        path,
        num_samples,
        batch_size,
        solver_name,
        solver_kwargs,
        model_name,
        model_kwargs,
        condition,
    )
    return compute_metric(
        path, num_samples, metric_name, metric_kwargs, dataset, condition
    )
