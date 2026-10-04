import os
import shutil
import subprocess
from glob import glob
from typing import Iterable
from uuid import uuid4

import torch
import yaml
from accelerate import Accelerator
from huggingface_hub import hf_hub_download

from core.models.base import Condition, DiffusionModel, ModelOutput
from core.utils.layout import sample_path
from core.utils.venv import laproteina_python

accelerator = Accelerator()


class LaProteinaModel(DiffusionModel[str]):
    """
    Wrapper of the La-Proteina unconditional generator (arXiv 2507.09466).

    La-Proteina is a Lightning/Hydra codebase pinned to its own torch 2.7/cu118
    environment, so it runs as a subprocess in $LAPROTEINA_VENV rather than in-process:
    `sample()` shells out to its `generate.py`, and each produced `.pdb` is filed into
    the canonical `<path>/pdb/000000.pdb` layout.

    `base_config` selects one of La-Proteina's own inference configs (e.g.
    `inference_ucond_notri` for the LD1 checkpoint), which pins the latent-diffusion
    and autoencoder checkpoint pair; the solver's sampling overrides layer on top.
    `ld_ckpt`/`ae_ckpt` are the latent-diffusion and autoencoder weights, resolved
    from `hf_repo` in the HF cache and pointed at directly, so the cloned repo is
    never written to.
    """

    def __init__(
        self,
        base_config: str = "inference_ucond_notri",
        ld_ckpt: str = "LD1_ucond_notri_512.ckpt",
        ae_ckpt: str = "AE1_ucond_512.ckpt",
        hf_repo: str = "nvidia/NV-La-Proteina-Ucond-v1",
        repo_dir: str = "/opt/la-proteina",
    ) -> None:
        self.base_config = base_config
        self.repo_dir = os.environ.get("LAPROTEINA_REPO") or repo_dir
        self.ld_path = hf_hub_download(hf_repo, ld_ckpt)
        self.ae_path = hf_hub_download(hf_repo, ae_ckpt)

    def sample(self, generation: dict, seed: int) -> list[str]:
        """
        Run one La-Proteina generation job, extending `base_config` with the given
        `generation` overrides, and return the produced PDB paths.

        generate.py reads a Hydra config by name from the repo's `configs/`, seeds
        globally off `cfg.seed`, and writes to `./inference/<config_name>/` relative
        to its cwd, so a uniquely named temp config is dropped in and cleaned up.

        The `ckpt_*` keys repoint the base config's checkpoints at the HF cache.
        """
        name = f"laproteina_{uuid4().hex[:8]}"
        config = {
            "defaults": [self.base_config, "_self_"],
            "ckpt_path": os.path.dirname(self.ld_path),
            "ckpt_name": os.path.basename(self.ld_path),
            "autoencoder_ckpt_path": self.ae_path,
            "seed": seed,
            "generation": generation,
        }
        config_path = os.path.join(self.repo_dir, "configs", f"{name}.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        try:
            subprocess.run(
                [
                    laproteina_python(),
                    "proteinfoundation/generate.py",
                    "--config_name",
                    name,
                ],
                cwd=self.repo_dir,
                env=self._subprocess_env(),
                check=True,
            )
            return sorted(
                glob(os.path.join(self.repo_dir, "inference", name, "job_*", "*.pdb"))
            )
        finally:
            os.remove(config_path)

    @staticmethod
    def _subprocess_env() -> dict:
        """
        Environment for the generate.py subprocess. Pin it to this rank's GPU:
        accelerate exposes the whole GPU set to every rank, so generate.py's
        `Trainer(devices=1)` would otherwise land every rank's job on the first one.
        DATA_PATH satisfies a `${oc.env:DATA_PATH}` interpolation in the checkpoint's
        config; it is a training path, unused for sampling.
        """
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = (
            visible.split(",")
            if visible
            else list(map(str, range(torch.cuda.device_count())))
        )
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": devices[accelerator.local_process_index],
        }
        env.setdefault("DATA_PATH", "/tmp")
        return env

    def save_output(self, output: str, folder: str, index: int) -> None:
        """
        Move output out of the repo's inference/ dir, then drop the now-empty job dir
        generate.py made (one PDB per dir), so a run leaves nothing behind.
        """
        shutil.move(output, sample_path(folder, index, "pdb", write=True))
        os.rmdir(os.path.dirname(output))

    # La-Proteina owns its whole sampling loop in-subprocess, so the in-process
    # diffusion interface is never driven; only `sample`/`save_output` are used.

    def get_network(self) -> torch.nn.Module:
        raise NotImplementedError

    def encode_condition(self, **kwargs) -> Condition:
        raise NotImplementedError

    def get_prior(self, seeds: list[int]) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, x: torch.Tensor, t: torch.Tensor, cond: Condition) -> ModelOutput:
        raise NotImplementedError

    def postprocess(self, x: torch.Tensor) -> Iterable[str]:
        raise NotImplementedError
