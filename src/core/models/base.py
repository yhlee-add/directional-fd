from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Iterable
from enum import Enum
from dataclasses import dataclass, field

from accelerate import Accelerator
from diffusers import DiffusionPipeline
import torch
from PIL.Image import Image

from core.utils.layout import sample_path

Condition = dict[str, Any]  # encoded prompt conditions
Feature = dict[str, Any]  # internal block outputs, keyed by block name


T = TypeVar("T")


accelerator = Accelerator()


class PredictionType(Enum):
    EPSILON = "epsilon"
    SAMPLE = "sample"
    VELOCITY = "v_prediction"


@dataclass
class ModelOutput:
    """
    Network prediction: tensor, its type, and optional extracted features.
    """

    pred: torch.Tensor
    pred_type: PredictionType
    feat: Feature = field(default_factory=dict)
    aux: dict[str, Any] = field(default_factory=dict)


class DiffusionModel(ABC, Generic[T]):
    """
    Base class for all diffusion models.
    """

    @abstractmethod
    def get_network(self) -> torch.nn.Module:
        """
        Returns the neural network used in the diffusion model.
        """
        pass

    @abstractmethod
    def encode_condition(self, **kwargs) -> Condition:
        pass

    @abstractmethod
    def get_prior(self, seeds: list[int]) -> torch.Tensor:
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor, t: torch.Tensor, cond: Condition) -> ModelOutput:
        """
        Prediction with the neural network.
        May not always predict v; could be epsilon or x depending on the model.
        """
        return ModelOutput(x, PredictionType.SAMPLE)  # dummy implementation

    @abstractmethod
    def postprocess(self, x: torch.Tensor) -> Iterable[T]:
        """
        Converts final tensor x_T to desired output format (e.g., image).
        Must call pipe.maybe_free_model_hooks() if using diffusers pipelines.
        """
        pass

    @abstractmethod
    def save_output(self, output: T, folder: str, index: int) -> None:
        """
        Saves the output in the folder with the given index.
        """
        pass

    def predict_with(
        self, x: torch.Tensor, t: torch.Tensor, cond: Condition, f: Feature
    ) -> ModelOutput:
        """
        Prediction with the internal feature.
        Corresponds to \\tilde{v}(x, t, F) in the paper.

        By default, just calls the regular predict function.
        """
        return self.predict(x, t, cond)


class HuggingFaceModel(DiffusionModel[T]):
    """
    A wrapper diffusion model that uses a HuggingFace pipeline.
    """

    def __init__(
        self,
        model_key: str,
        target: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.pipe = DiffusionPipeline.from_pretrained(model_key, torch_dtype=dtype)
        self.pipe.set_progress_bar_config(disable=True)
        pipeline_to(self.pipe, target)


def save_png(output: Image, folder: str, index: int) -> None:
    output.save(sample_path(folder, index, "png", write=True))


def pipeline_to(pipe, option: str | None) -> None:
    if option is None:
        return
    elif option == "accelerator":
        pipe.to(accelerator.device)
    elif option == "model_offload":
        pipe.enable_model_cpu_offload(device=accelerator.device)
    elif option == "sequential_offload":
        pipe.enable_sequential_cpu_offload(device=accelerator.device)
    else:
        raise ValueError(f"Unknown pipeline option: {option}")
