import math
from typing import Any

from accelerate import Accelerator
import torch

from core.models.base import DiffusionModel, Condition
from core.models.laproteina import LaProteinaModel
from core.solvers.base import Solver, SolverOutput

accelerator = Accelerator()


class LaProteinaSolver(Solver):
    """
    Drives `LaProteinaModel` for unconditional protein generation.
    Delegates the whole sampling process to La-Proteina's subprocess.
    """

    def __init__(
        self,
        nsteps: int = 400,
        nres_lens: tuple[int, ...] = (100, 200, 300, 400, 500),
        self_cond: bool = True,
    ) -> None:
        """
        Args:
            nsteps (int):
                Integration-step count, the sampling knob to sweep.
            nres_lens (tuple[int, ...]):
                Residue lengths to sample; each rank spreads its shard evenly across
                them, matching La-Proteina's per-length sampling. Set the pipeline's
                `batch_size` to `num_samples` so each rank's shard generates in a
                single subprocess launch.
            self_cond (bool):
                Whether to use self-conditioning during sampling.
        """
        self.nsteps = nsteps
        self.nres_lens = list(nres_lens)
        self.self_cond = self_cond

    def denoising_loop(
        self, prior: torch.Tensor, cond: Condition, model: DiffusionModel
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError("LaProteinaSolver samples via subprocess")

    @torch.no_grad()
    def generate(
        self, model: DiffusionModel, seeds: list[int], **kwargs
    ) -> SolverOutput:
        assert isinstance(model, LaProteinaModel)

        # TODO support multi-gpu
        assert accelerator.num_processes == 1

        per_length = math.ceil(len(seeds) / len(self.nres_lens))
        generation = {
            "args": {"nsteps": self.nsteps, "self_cond": self.self_cond},
            "dataset": {
                "nsamples": per_length,
                "nlens_cfg": {"nres_lens": self.nres_lens},
            },
        }
        pdbs = model.sample(generation, seed=seeds[0])
        return SolverOutput(output=pdbs[: len(seeds)])
