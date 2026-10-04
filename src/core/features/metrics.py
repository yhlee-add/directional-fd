import os
import contextlib
from abc import ABC, abstractmethod

from accelerate import Accelerator
from huggingface_hub import snapshot_download
import torch
from tqdm import tqdm
import numpy as np

from core.utils.layout import sample_path

# FID, CMMD, KID, FDCLIP, FDDINOv2
from cleanfid.fid import fid_from_feats, kernel_distance
from core.vendor.cmmd import distance
from core.features.extractors import (
    Extractor,
    InceptionExtractor,
    ClipExtractor,
    Dinov2Extractor,
    I3DExtractor,
    VideoMAEExtractor,
    ESM3Extractor,
    index_loader,
)

# IS
from torchmetrics.image.inception import InceptionScore as _InceptionScore
from core.features.extractors import InceptionLogitsExtractor

# CLIP Score
from torchmetrics.functional.multimodal.clip_score import clip_score
from transformers import CLIPProcessor, CLIPModel

# ImageReward
import ImageReward as RM

# PSNR, SSIM, LPIPS
from torchvision.io import read_image
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)

accelerator = Accelerator()


@contextlib.contextmanager
def quiet():
    """
    Silence the per-parameter chatter libraries print while loading a checkpoint.
    """
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        yield


class Metric(ABC):
    """
    Base class for all metrics.
    """

    @abstractmethod
    def compute(self, path: str, **kwargs) -> float:
        """
        Computes the metric given the path to the generated samples.
        """
        return 0.0


def load_cache(extractor: Extractor | str, path: str) -> torch.Tensor:
    name = extractor if isinstance(extractor, str) else extractor.get_name()

    cache = os.path.join(path, f"{name}.npy")
    if os.path.exists(cache):
        return torch.asarray(np.load(cache), device=accelerator.device)
    if isinstance(extractor, str):
        raise FileNotFoundError(f"no cached {name}.npy in {path}")

    feats = extractor.extract_all(path)
    if accelerator.is_main_process:
        np.save(cache, feats.numpy())
    accelerator.wait_for_everyone()
    return feats.to(accelerator.device)


class Distance(Metric):
    """
    Distance metric between reference features and evaluated features,
    extracted by a shared `extractor` and cached per folder.
    """

    extractor_cls: type[Extractor]

    def __init__(self, ref_path: str = "samples/coco30k", **kwargs):
        self.ref_path = ref_path
        self.extractor = self.extractor_cls(**kwargs)

    @abstractmethod
    def distance(self, ref: torch.Tensor, eval: torch.Tensor) -> float:
        pass

    @torch.no_grad()
    def compute(self, path: str, **kwargs) -> float:
        ref_feats = load_cache(self.extractor, self.ref_path)
        eval_feats = load_cache(self.extractor, path)
        return self.distance(ref_feats, eval_feats)


class FID(Distance):
    extractor_cls = InceptionExtractor

    def distance(self, ref: torch.Tensor, eval: torch.Tensor) -> float:
        return fid_from_feats(ref.cpu().numpy(), eval.cpu().numpy()).item()


class KID(Distance):
    extractor_cls = InceptionExtractor

    def distance(self, ref: torch.Tensor, eval: torch.Tensor) -> float:
        return kernel_distance(ref.cpu().numpy(), eval.cpu().numpy())


class CMMD(Distance):
    extractor_cls = ClipExtractor

    def distance(self, ref: torch.Tensor, eval: torch.Tensor) -> float:
        return distance.mmd(ref, eval).item()


class FDCLIP(FID):
    extractor_cls = ClipExtractor


class FDDINOv2(FID):
    extractor_cls = Dinov2Extractor


class FVD(FID):
    """
    Frechet Video Distance: FID over I3D features, as everyone reports it.
    """

    extractor_cls = I3DExtractor


class CDFVD(FID):
    """
    Content-debiased FVD: the same distance over VideoMAE-v2 features,
    which respond to temporal quality that `FVD` misses.
    """

    extractor_cls = VideoMAEExtractor


class FaltingsProteinFID(Distance):
    """
    Protein Frechet distance (Faltings et al., arXiv 2505.08041): FID over ESM3
    structure-token embeddings, projected to `pca_dim` PCA dimensions fit jointly
    on the reference and evaluation sets. The joint projection is why this is not a
    plain `FID` over a fixed feature cache.
    """

    extractor_cls = ESM3Extractor

    def __init__(
        self, ref_path: str = "samples/esmtrainref", pca_dim: int = 32, **kwargs
    ):
        super().__init__(ref_path=ref_path, **kwargs)
        self.pca_dim = pca_dim

    def distance(self, ref: torch.Tensor, eval: torch.Tensor) -> float:
        from core.vendor.protfid.fid import compute_fid_from_embeddings

        mean, std = compute_fid_from_embeddings(
            ref.cpu(), eval.cpu(), pca_dim=self.pca_dim
        )
        return mean.item()


class IS(Metric):

    def __init__(self, model_path: str | None = None, **kwargs):
        self.extractor = InceptionLogitsExtractor(model_path=model_path)

    @torch.no_grad()
    def compute(self, path: str, **kwargs) -> float:
        eval_logits = load_cache(self.extractor, path)

        torch.random.manual_seed(0)
        mean, std = _InceptionScore(feature=torch.nn.Identity())(eval_logits)
        return mean.item()


class CLIPScore(Metric):

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        batch_size: int = 30,
        **kwargs,
    ):
        self.model = CLIPModel.from_pretrained(model_name).to(accelerator.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.batch_size = batch_size

    def get_model(self):
        return self.model, self.processor

    @torch.no_grad()
    def compute(self, path: str, **kwargs) -> float:
        prompts = kwargs["prompts"]
        indices = list(range(len(prompts)))

        with accelerator.split_between_processes(indices) as index_split:
            weighted_sum = torch.zeros((), device=accelerator.device)
            for i in tqdm(
                range(0, len(index_split), self.batch_size),
                desc="CLIPScore",
                disable=not accelerator.is_main_process,
            ):
                batch_idx = index_split[i : i + self.batch_size]
                images = torch.stack(
                    [read_image(sample_path(path, j, "png")) for j in batch_idx]
                ).to(accelerator.device)
                texts = [prompts[j] for j in batch_idx]
                score = clip_score(images, texts, model_name_or_path=self.get_model)
                weighted_sum += score * len(batch_idx)

        gathered = accelerator.gather(weighted_sum.unsqueeze(0))
        return (gathered.sum() / len(indices)).item()


class ImageReward(Metric):

    def __init__(self, **kwargs):
        snap = snapshot_download("THUDM/ImageReward")
        with quiet():
            self.model = RM.load(
                os.path.join(snap, "ImageReward.pt"),
                device=accelerator.device,
                med_config=os.path.join(snap, "med_config.json"),
            )

    @torch.no_grad()
    def compute(self, path: str, **kwargs) -> float:
        prompts = kwargs["prompts"]
        results = []
        for batch_idx in tqdm(
            index_loader(len(prompts), 1),
            desc="ImageReward",
            disable=not accelerator.is_main_process,
        ):
            i = batch_idx.item()
            score = self.model.score(prompts[i], sample_path(path, i, "png"))
            gathered = accelerator.gather_for_metrics(
                torch.tensor([score], device=accelerator.device)
            )
            results.append(gathered)
        return torch.cat(results).mean().item()


class Similarity(Metric):

    def __init__(self, gt_path: str, **kwargs):
        self.gt_path = gt_path

    @abstractmethod
    def similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        pass

    @torch.no_grad()
    def compute(self, path: str, **kwargs) -> float:
        prompts = kwargs["prompts"]
        results = []
        for batch_idx in tqdm(
            index_loader(len(prompts), 1),
            desc=self.__class__.__name__,
            disable=not accelerator.is_main_process,
        ):
            i = batch_idx.item()
            img = read_image(sample_path(path, i, "png")).to(
                dtype=torch.float32, device=accelerator.device
            )[None]
            gt_img = read_image(sample_path(self.gt_path, i, "png")).to(
                dtype=torch.float32, device=accelerator.device
            )[None]
            gathered = accelerator.gather_for_metrics(
                self.similarity(img, gt_img).unsqueeze(0)
            )
            results.append(gathered)
        return torch.cat(results).mean().item()


class PSNR(Similarity):

    def __init__(self, gt_path: str, **kwargs):
        super().__init__(gt_path)
        self.psnr = PeakSignalNoiseRatio(
            data_range=255.0,
        ).to(accelerator.device)

    def similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return self.psnr(img1, img2)


class SSIM(Similarity):

    def __init__(self, gt_path: str, **kwargs):
        super().__init__(gt_path)
        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=255.0,
        ).to(accelerator.device)

    def similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return self.ssim(img1, img2)


class LPIPS(Similarity):
    def __init__(self, gt_path: str, **kwargs):
        super().__init__(gt_path)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            normalize=True  # normalize=True expects [0,1]
        ).to(accelerator.device)

    def similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return self.lpips(img1 / 255.0, img2 / 255.0)  # convert to [0,1]


# H.264 quality for the VBench transcode: near-visually-lossless but a standard
# 8-bit 4:2:0 stream, the format also used by videocrafter-1 sampled videos.
VBENCH_H264_CRF = 10
