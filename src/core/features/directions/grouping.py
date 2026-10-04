from abc import ABC, abstractmethod
from collections.abc import Sequence
import re

from accelerate import Accelerator
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

from core.data.base import Dataset

accelerator = Accelerator()


def load_prompts(dataset: Dataset) -> list[str]:
    return pd.read_csv(dataset.get_csv_path())["prompts"].tolist()


class Grouper(ABC):

    @abstractmethod
    def group(self, seq: Sequence) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Check if elements of `seq` is in the positive or negative group.

        Args:
            seq (Sequence): sequence of target elements.
        Returns:
            Two torch bool tensors. Each is a mask for the positive and negative group.
        """
        pass

    def concept(
        self, group_on: Sequence, feats: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Concept axis w_c = normalize(mu_+ - mu_-) over this grouper's two groups.

        Args:
            group_on (Sequence):
                what to split on; the pos/neg masks come from it.
            feats (torch.Tensor | None):
                the space the axis lives in, indexed by the masks;
                defaults to `group_on` when the split and axis share a space.
        Returns:
            The unit concept axis in the space of `feats`.
        """
        pos, neg = self.group(group_on)
        feats = group_on if feats is None else feats
        return F.normalize(feats[pos].mean(0) - feats[neg].mean(0), dim=0)


class RegexGrouper(Grouper):

    def __init__(self, pos_regex: str, neg_regex: str):
        self.pos = re.compile(pos_regex, re.I)
        self.neg = re.compile(neg_regex, re.I)

    def group(self, seq: Sequence) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(seq[0], str):
            raise ValueError("seq must be list[str]")

        pm = torch.tensor([self.pos.search(t) is not None for t in seq]).to(
            accelerator.device
        )
        nm = torch.tensor([self.neg.search(t) is not None for t in seq]).to(
            accelerator.device
        )
        return pm & ~nm, nm & ~pm


class ClipTextEncoder:
    """
    The CLIP text encoder: embeds prompts into the joint space, mean-ensembling from
    several prompts. Expensive to load, so build one and share it across many
    `ClipTextGrouper`s that reuse it.
    """

    def __init__(
        self,
        hf_key: str = "openai/clip-vit-large-patch14-336",
        batch_size: int = 512,
    ):
        self.tok = CLIPTokenizer.from_pretrained(hf_key)
        self.model = CLIPTextModelWithProjection.from_pretrained(hf_key).eval()
        self.model.to(accelerator.device)
        self.batch_size = batch_size

    @torch.no_grad()
    def embed(self, prompts: list[str]) -> torch.Tensor:
        res = []
        for i in range(0, len(prompts), self.batch_size):
            batch = self.tok(
                prompts[i : i + self.batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(accelerator.device)
            res.append(self.model(**batch).text_embeds)
        return torch.cat(res)

    def ensemble(self, prompt: str | list[str]) -> torch.Tensor:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        return self.embed(prompt).mean(0)  # [D]


class ClipTextGrouper(Grouper):
    """
    Split items by their CLIP similarity to a fixed text axis.
    The top and bottom `q` quantiles are the positive and negative groups.
    """

    def __init__(
        self,
        encoder: ClipTextEncoder,
        pos_prompt: str | list[str],
        neg_prompt: str | list[str] = "",
        q: float = 0.25,
    ):
        self.encoder = encoder
        self.axis = encoder.ensemble(pos_prompt) - encoder.ensemble(neg_prompt)
        self.q = q

    def group(self, seq: Sequence) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(seq[0], str):
            feats = self.encoder.embed(seq)  # [N, D]
        elif isinstance(seq, torch.Tensor):
            feats = seq
        else:
            raise ValueError("seq must be one of list[str] or torch.Tensor")

        similarity = F.cosine_similarity(feats, self.axis)  # [N]
        lo = torch.quantile(similarity, self.q)
        hi = torch.quantile(similarity, 1 - self.q)
        return (similarity >= hi), (similarity <= lo)


class QuantileGrouper(Grouper):
    """
    Split a numeric attribute into its top and bottom `q` quantiles, the poles for a
    scalar concept (protein length, radius of gyration, coil fraction, ...). `group_on`
    is keyed (a DataFrame or mapping), so one shared `group_on` drives many keyed
    groupers under `GroupConcepts`.
    """

    def __init__(self, key: str, q: float = 1 / 3):
        self.key, self.q = key, q

    def group(self, group_on: Sequence) -> tuple[torch.Tensor, torch.Tensor]:
        v = torch.as_tensor(
            np.asarray(group_on[self.key], dtype=float), device=accelerator.device
        )
        lo, hi = torch.quantile(v, self.q), torch.quantile(v, 1 - self.q)
        return v >= hi, v <= lo


class CategoryGrouper(Grouper):
    """
    Split by exact value of a categorical attribute: pos where `key == pos`, neg where
    `key == neg` (e.g. CATH class c==2 mainly-beta vs c==1 mainly-alpha).
    """

    def __init__(self, key: str, pos, neg):
        self.key, self.pos, self.neg = key, pos, neg

    def group(self, group_on: Sequence) -> tuple[torch.Tensor, torch.Tensor]:
        v = torch.as_tensor(np.asarray(group_on[self.key]), device=accelerator.device)
        return v == self.pos, v == self.neg


class RandomGrouper(Grouper):
    """
    Split the items into two random halves, ignoring content: a null concept whose
    charge calibrates how much any difference-of-means axis means, the tabular analogue
    of the random-word poles in the image dictionary.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def group(self, group_on: Sequence) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(group_on)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed))
        pos = torch.zeros(n, dtype=torch.bool)
        pos[perm[: n // 2]] = True
        return pos.to(accelerator.device), (~pos).to(accelerator.device)
