import os
from abc import ABC, abstractmethod

from datasets import load_dataset
from huggingface_hub import snapshot_download


class Dataset(ABC):
    """
    Base class for all datasets.
    """

    @abstractmethod
    def prepare(self) -> None:
        """
        Materialize dataset into samples/<name>/. No-op if already present.
        """
        pass

    def get_name(self) -> str:
        return type(self).__name__.lower()

    def get_path(self) -> str:
        return os.path.join("samples", self.get_name())

    def get_csv_path(self) -> str:
        return os.path.join(self.get_path(), "prompts.csv")


class HuggingFaceDataset(Dataset):
    """
    Dataset downloaded from HuggingFace Hub and saved locally.
    """

    def __init__(self, hf_key: str, split: str = "train") -> None:
        self.hf_key = hf_key
        self.split = split

    def resolve_snapshot(self, *args: str) -> str:
        """
        Join a path inside the snapshot, downloading the repo if needed.

        hf_key may be a repo id or an already-materialized snapshot directory.
        """
        root = (
            self.hf_key
            if os.path.isdir(self.hf_key)
            else snapshot_download(self.hf_key, repo_type="dataset")
        )
        return os.path.join(root, *args)

    def prepare(self) -> None:
        if os.path.exists(self.get_csv_path()):  # csv is written last, so it means done
            return
        os.makedirs(self.get_path(), exist_ok=True)
        self.convert_dataset(self.sources())

    def sources(self):
        """
        The raw dataset `convert_dataset` consumes; the full HF split by default.
        """
        return load_dataset(self.hf_key, split=self.split)

    @abstractmethod
    def convert_dataset(self, dataset) -> None:
        pass


class HuggingFaceSplitDataset(HuggingFaceDataset):
    """
    HuggingFace parquet dataset that downloads only its own split.

    load_dataset(split=...) still materializes every split; scoping data_files to
    the split's shards avoids pulling the rest (e.g. a large unused train split).
    """

    def sources(self):
        return load_dataset(
            self.hf_key,
            data_files={self.split: f"data/{self.split}-*.parquet"},
            split=self.split,
            verification_mode="no_checks",
        )


class LocalDataset(Dataset):
    """
    Dataset already present on disk.
    """

    def prepare(self) -> None:
        pass
