from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import diffusers
import torch

from core.models.base import Condition, DiffusionModel, HuggingFaceModel


@dataclass
class SolverOutput:
    """
    Final output plus auxiliary data (e.g. the sampling trajectory).
    """

    output: Any
    aux: dict[str, Any] = field(default_factory=dict)


class Solver(ABC):
    """
    Base class for all solvers.
    """

    @abstractmethod
    def denoising_loop(
        self, prior: torch.Tensor, cond: Condition, model: DiffusionModel
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Performs the denoising loop given the prior, condition, and model.
        """
        return prior, {}

    @torch.no_grad()
    def generate(
        self, model: DiffusionModel, seeds: list[int], **kwargs
    ) -> SolverOutput:
        """
        Generates samples using the solver and the given diffusion model.
        """
        cond = model.encode_condition(**kwargs)
        prior = model.get_prior(seeds)
        x, aux = self.denoising_loop(prior, cond, model)
        return SolverOutput(output=model.postprocess(x), aux=aux)


class PipeDefaultSolver(Solver):
    """
    Delegates entirely to the HuggingFace pipeline's __call__.
    """

    def __init__(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps

    def denoising_loop(
        self, prior: torch.Tensor, cond: Condition, model: DiffusionModel
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError("PipeDefaultSolver uses generate() directly")

    @torch.no_grad()
    def generate(
        self, model: DiffusionModel, seeds: list[int], **kwargs
    ) -> SolverOutput:
        assert isinstance(model, HuggingFaceModel)
        prompts: list[str] = kwargs["prompts"]
        output = model.pipe(
            prompts,
            num_inference_steps=self.num_inference_steps,
            generator=[torch.Generator().manual_seed(s) for s in seeds],
        )
        # A pipeline's primary output is its first field (`images`, `frames`, ...).
        return SolverOutput(output=output[0])


class HuggingFaceSolver(Solver):
    """
    Wraps a diffusers scheduler to run the diffusion process.
    """

    def __init__(self, model_key: str, scheduler: str, num_inference_steps: int):

        self.scheduler = getattr(diffusers, scheduler).from_pretrained(
            model_key, subfolder="scheduler"
        )
        self.num_inference_steps = num_inference_steps

    def set_timesteps(self, device: torch.device, model: DiffusionModel) -> None:
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)

    def denoising_loop(
        self, prior: torch.Tensor, cond: Condition, model: DiffusionModel
    ) -> tuple[torch.Tensor, dict[str, Any]]:

        self.set_timesteps(prior.device, model)
        x = prior
        for t in self.scheduler.timesteps:
            model_input = self.scale_model_input(x, t)
            model_output = model.predict(model_input, t, cond)
            x = self.scheduler.step(model_output.pred, t, x, return_dict=False)[0]
        return x, {}

    def scale_model_input(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if hasattr(self.scheduler, "scale_model_input"):
            return self.scheduler.scale_model_input(x, t)
        return x
