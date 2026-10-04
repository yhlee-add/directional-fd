"""
Cached displacements for a sweep of generated-vs-reference feature sets.

A sweep is a family of displacements sharing one regime (e.g. the reversal step sweep
over inference steps). displacement() is the cost -- a cov, two sqrtm and an svd per
member -- so a sweep is computed once and cached to a single file of Mbar and dmu per
member. Downstream reductions (charge, drift, cumulative spectra) are then cheap
matmuls against a member's eigen_axes(), via Displacement.fid_w.
"""

import os

import torch

from core.utils.gloo import gloo_main_process_first
from core import data
from core.data import VideoDataset, DISTORTIONS, DistortedVideoDataset
from core.features.directions.ot import Displacement, displacement
from core.features.extractors import Extractor
from core.features.metrics import accelerator, load_cache

CACHE = "samples/sweeps"


def displacements(
    extractor: Extractor | str, gen_to_ref: dict[str, str]
) -> dict[str, Displacement]:
    return {
        gen_path: displacement(
            load_cache(extractor, ref_path).double(),
            load_cache(extractor, gen_path).double(),
        )
        for gen_path, ref_path in gen_to_ref.items()
    }


class Sweep:

    def __init__(
        self, name: str, gen_to_ref: dict[str, str], extractor: Extractor | str
    ):
        cache = os.path.join(CACHE, f"{name}.pt")
        if os.path.exists(cache):
            self.load_dict(torch.load(cache, map_location=accelerator.device))
        else:
            self.disps = displacements(extractor, gen_to_ref)
            if accelerator.is_main_process:
                os.makedirs(CACHE, exist_ok=True)
                torch.save(self.as_dict(), cache)
            accelerator.wait_for_everyone()

    def as_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {k: dict(Mbar=v.Mbar, dmu=v.dmu) for k, v in self.disps.items()}

    def load_dict(self, d: dict[str, dict[str, torch.Tensor]]) -> None:
        self.disps = {
            k: Displacement(
                v["Mbar"] + torch.outer(v["dmu"], v["dmu"]), v["Mbar"], v["dmu"]
            )
            for k, v in d.items()
        }

    def __getitem__(self, key):
        return self.disps[key]


class StableDiffusionSweep(Sweep):

    prefix = "samples/StableDiffusionModel_HuggingFaceSolver_{ref}_steps{n}"
    references = ["COCO30K", "Flickr30K", "ImageNet", "MJHQ30K", "CelebAHQ", "FFHQ512"]
    steps = list(range(15, 55, 5))

    def __init__(self, extractor: Extractor | str):
        super().__init__(
            f"stablediffusion_{extractor}",
            {
                self.prefix.format(ref=ref, n=n): getattr(data, ref)().get_path()
                for ref in self.references
                for n in self.steps
            },
            extractor,
        )

    def __getitem__(self, key):
        ref, n = key
        return super().__getitem__(self.prefix.format(ref=ref, n=n))


class DistortionSweep(Sweep):
    """
    Cached clean-vs-distorted displacements over a reference set: per family, the
    spatial (coherent, redraw = num_frames) and spatio-temporal (flicker, redraw = 1)
    probe at every ImageNet-C severity, each a DistortedVideoDataset materialized
    beside the reference. Keyed by (family, mode, severity), mode in {"s", "st"}.

    An Extractor instance lets load_cache extract and cache features npy on the
    first build. The displacements cache to
    `samples/sweeps/<reference>_distortion_<extractor>.pt`.
    """

    severities = (1, 2, 3, 4, 5)

    def __init__(self, reference: VideoDataset, extractor: Extractor | str):
        self.probes: dict[tuple[str, str, int], DistortedVideoDataset] = {}
        for family in DISTORTIONS:
            for sev in self.severities:
                self.probes[family, "s", sev] = DistortedVideoDataset(
                    reference, family, sev, redraw=reference.num_frames
                )
                self.probes[family, "st", sev] = DistortedVideoDataset(
                    reference, family, sev, redraw=1
                )

        # Materialize rank-0-first (slow ffmpeg encodes); the ranks then extract the
        # shared mkv in parallel. gloo's long timeout dodges the NCCL watchdog abort.
        with gloo_main_process_first():
            reference.prepare()
            for probe in self.probes.values():
                probe.prepare()

        name = extractor if isinstance(extractor, str) else extractor.get_name()
        super().__init__(
            f"{reference.get_name()}_distortion_{name}",
            {p.get_path(): reference.get_path() for p in self.probes.values()},
            extractor,
        )

    def __getitem__(self, key):
        return super().__getitem__(self.probes[key].get_path())
