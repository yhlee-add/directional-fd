"""
On-disk layout of a sample set: `<path>/png/000000.png`, indexed files in a
payload folder named by their extension, where the index ties each sample to
its row in the sibling `prompts.csv`. The payload folder keeps the samples
clear of the set's other artifacts (`prompts.csv`, `.npy` feature caches,
metric outputs).
"""

import glob
import os


def sample_dir(path: str, ext: str) -> str:
    return os.path.join(path, ext)


def sample_path(path: str, index: int, ext: str, *, write: bool = False) -> str:
    """
    Path of sample `index`; `write=True` creates the payload folder for writers.
    """
    if write:
        os.makedirs(sample_dir(path, ext), exist_ok=True)
    return os.path.join(sample_dir(path, ext), f"{index:06d}.{ext}")


def count_samples(path: str, ext: str) -> int:
    return sum(f.endswith(f".{ext}") for f in os.listdir(sample_dir(path, ext)))


def done_indices(path: str) -> set[int]:
    """
    Indices already written to any payload folder under `path`, so a restart can
    resume a partial run. Matches only `NNNNNN.*` sample files, ignoring the set's
    other artifacts (`prompts.csv`, `.npy` caches, metric outputs).
    """
    files = glob.glob(os.path.join(path, "*", "[0-9]" * 6 + ".*"))
    return {int(os.path.basename(f)[:6]) for f in files}
