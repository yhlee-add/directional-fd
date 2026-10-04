from abc import ABC
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from sklearn.linear_model import Lasso
from cleanfid.inception_torchscript import InceptionV3W

from core.features.directions.grouping import Grouper
from core.vendor.inception.labels import imagenet_labels


class ConceptSpace(ABC):
    """
    Named unit axes that a direction is sparsely decomposed over.

    Subclasses populate `self.names` and `self.rows` ([K, D] unit atoms); the base
    provides the sparse signed lasso decomposition shared by every such space.
    """

    names: list[str]
    rows: torch.Tensor  # [K, D]

    def decompose(
        self, target: torch.Tensor, alpha: float, threshold: float = 1e-6
    ) -> tuple[float, list[tuple[str, float]]]:
        """
        Sparse signed weights reconstructing target from the atoms.

        Signed, not nonnegative: each atom's orientation is arbitrary, so its sign
        is the fit's to choose.

        Args:
            target (torch.Tensor): [D] direction to reconstruct.
            alpha (float): lasso strength; larger keeps fewer atoms.
            threshold (float): drop atoms with |weight| at or below this.
        Returns:
            The reconstruction cosine and the surviving (name, weight) atoms,
            ranked by descending |weight| with the ~zero atoms dropped.
        """
        A = self.rows.mT  # [D, K]
        fit = Lasso(alpha=alpha, fit_intercept=False, max_iter=50000)
        fit.fit(A.cpu().numpy(), target.cpu().numpy())

        coef = torch.as_tensor(fit.coef_, device=A.device, dtype=A.dtype)
        coef[coef.abs() <= threshold] = 0
        cos = F.cosine_similarity(A @ coef, target, dim=0).item()

        terms = [(name, w) for name, w in zip(self.names, coef.tolist()) if w != 0]
        return cos, sorted(terms, key=lambda kw: abs(kw[1]), reverse=True)


class GroupConcepts(ConceptSpace):
    """
    Concept axes named by groupers: each grouper splits `group_on` into poles and
    the difference-of-means over `feats` is that atom's axis.
    """

    def __init__(
        self,
        groupers: dict[str, Grouper],
        group_on: Sequence,
        feats: torch.Tensor | None = None,
    ):
        self.names = list(groupers)
        self.rows = torch.stack([g.concept(group_on, feats) for g in groupers.values()])


class ImageNetConcepts(ConceptSpace):
    """
    The rows of clean-fid's inception classifier as a concept space.

    clean-fid's cached features ARE the input to this final linear map W
    (`base.output`), so the ImageNet logits are exactly `features @ W.T + b`.
    So the logit of class i changes by (W @ w)[i] as you travel along a unit direction w:
    W @ w is the direction's ImageNet-class signature. (Kynkaanniemi et al. 2023
    note FID's features are one affine map from these logits.)

    self.rows are the unit-normalized rows, so a decompose() ranks class directions,
    not their weight norms.
    """

    def __init__(self, device):
        self.W = InceptionV3W("/tmp").eval().base.output.weight.detach().double()
        self.W = self.W.to(device)
        labels = imagenet_labels()
        labels += [f"pad{i}" for i in range(self.W.shape[0] - len(labels))]  # -> 1008
        self.names = labels
        self.rows = F.normalize(self.W, dim=1)  # [1008, 2048] unit rows

    def signature(self, w, k=5):
        """
        Name a data-driven direction by ImageNet classes.

        Returns the top-k classes by |logit response| W @ w, the classes whose logits
        move most as you travel along w. Reaches directions the attribute vocabulary
        might miss.
        """
        resp = self.W @ (w / w.norm())  # [1008] logit response along w
        return [(self.names[i], resp[i].item()) for i in resp.abs().topk(k).indices]
