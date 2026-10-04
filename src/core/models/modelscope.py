import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler

from core.data.video import write_ffv1
from core.utils.layout import sample_path
from core.models.base import (
    HuggingFaceModel,
    Condition,
    ModelOutput,
)


class ModelScopeModel(HuggingFaceModel[np.ndarray]):
    """
    ModelScope text-to-video (`ali-vilab/text-to-video-ms-1.7b`), a 1.7B UNet whose
    pipeline natively emits 16-frame clips at 256px, matching the video extractor's
    window. Sampling is delegated to `PipeDefaultSolver`, so the per-step
    `DiffusionModel` hooks are unused and left unimplemented; only `save_output`
    (lossless FFV1 mkv, the format the extractor reads) is provided.
    """

    def __init__(
        self,
        model_key: str,
        target: str | None = None,
        dtype: torch.dtype = torch.float16,
        fps: int = 8,
    ) -> None:
        super().__init__(model_key, target, dtype)
        # DPMSolver reaches ModelScope's quality in ~25 steps (its README default)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.fps = fps

    def save_output(self, output: np.ndarray, folder: str, index: int) -> None:
        frames = (output * 255).round().clip(0, 255).astype(np.uint8)
        write_ffv1(frames, sample_path(folder, index, "mkv", write=True), self.fps)

    def get_network(self) -> torch.nn.Module:
        raise NotImplementedError

    def encode_condition(self, **kwargs) -> Condition:
        raise NotImplementedError

    def get_prior(self, seeds: list[int]) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, x: torch.Tensor, t: torch.Tensor, cond: Condition) -> ModelOutput:
        raise NotImplementedError

    def postprocess(self, x: torch.Tensor) -> np.ndarray:
        raise NotImplementedError
