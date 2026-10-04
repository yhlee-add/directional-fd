import os
import random
from abc import ABC, abstractmethod

from accelerate import Accelerator
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from core.utils.layout import sample_path, sample_dir, count_samples

# FID
from cleanfid import fid, features

# IS
from torchmetrics.image.fid import NoTrainInceptionV3
from torchvision.io import read_image

# CMMD
from core.vendor.cmmd import io_util, embedding

# DINOv2
from transformers import Dinov2Model, AutoImageProcessor
from diffusers.utils import load_image

# FVD
from torchcodec.decoders import VideoDecoder
from huggingface_hub import hf_hub_download
from cdfvd.utils.data_utils import preprocess
from cdfvd.third_party.i3d.utils import load_i3d_model, preprocess_i3d
from cdfvd.third_party.VideoMAEv2.utils import load_videomae_model, preprocess_videomae

# Protein geometry
import biotite.structure as bs
import biotite.structure.io.pdb as pdb
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

accelerator = Accelerator()


def index_loader(num_samples: int, batch_size: int) -> DataLoader:
    return accelerator.prepare(DataLoader(range(num_samples), batch_size=batch_size))


class Extractor(ABC):
    """
    Base class for all feature extractors.
    """

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__.lower().removesuffix("extractor")

    @abstractmethod
    def extract(self, path: str, num_samples: int, batch_size: int) -> torch.Tensor:
        """
        Extract features from at max `num_samples` files directly under `path`.
        """
        pass

    @abstractmethod
    def extract_all(self, path: str) -> torch.Tensor:
        """
        Extract features from every sample under `path`, the folder that stores
        a run's results or a dataset, laid out as the domain's contract.
        """
        pass


class ImageExtractor(Extractor):
    """
    Images are stored as `<path>/png/000000.png`.
    """

    def extract_all(self, path: str) -> torch.Tensor:
        return self.extract(path, count_samples(path, "png"), 30)


class InceptionExtractor(ImageExtractor):

    def __init__(self):
        self.model = features.build_feature_extractor(
            "clean", accelerator.device, use_dataparallel=False
        )

    @torch.no_grad()
    def extract(self, path: str, num_samples: int, batch_size: int) -> torch.Tensor:
        results = []

        for batch_idx in tqdm(
            index_loader(num_samples, batch_size),
            disable=not accelerator.is_main_process,
        ):
            feats = fid.get_files_features(
                [sample_path(path, i, "png") for i in batch_idx],
                self.model,
                num_workers=0,
                batch_size=batch_size,
                device=accelerator.device,
                verbose=False,
            )
            results.append(
                accelerator.gather_for_metrics(
                    torch.tensor(feats, device=accelerator.device)
                ).cpu()
            )

        return torch.cat(results)


class InceptionLogitsExtractor(ImageExtractor):

    def __init__(self, model_path: str | None = None):
        self.model = NoTrainInceptionV3(
            name="inception-v3-compat",
            features_list=["logits_unbiased"],
            feature_extractor_weights_path=model_path,
        ).to(accelerator.device)

    @torch.no_grad()
    def extract(self, path: str, num_samples: int, batch_size: int) -> torch.Tensor:
        results = []

        for batch_idx in tqdm(
            index_loader(num_samples, batch_size),
            disable=not accelerator.is_main_process,
        ):
            images = torch.stack(
                [read_image(sample_path(path, i, "png")) for i in batch_idx]
            ).to(accelerator.device)
            results.append(accelerator.gather_for_metrics(self.model(images)).cpu())

        return torch.cat(results)


class ClipExtractor(ImageExtractor):

    def __init__(self, model_key: str = "openai/clip-vit-large-patch14-336"):
        self.model = embedding.ClipEmbeddingModel(model_key)

    @torch.no_grad()
    def extract(self, path: str, num_samples: int, batch_size: int) -> torch.Tensor:
        return io_util.compute_embeddings_for_dir(
            sample_dir(path, "png"),
            self.model,
            batch_size=batch_size,
            max_count=num_samples,
        )


class Dinov2Extractor(ImageExtractor):

    def __init__(
        self,
        model_key: str = "facebook/dinov2-large",
    ):
        self.model = Dinov2Model.from_pretrained(model_key).to(accelerator.device)
        self.processor = AutoImageProcessor.from_pretrained(model_key)

    @torch.no_grad()
    def extract(self, path: str, num_samples: int, batch_size: int) -> torch.Tensor:
        results = []

        for batch_idx in tqdm(
            index_loader(num_samples, batch_size),
            disable=not accelerator.is_main_process,
        ):
            images = [load_image(sample_path(path, i, "png")) for i in batch_idx]
            inputs = self.processor(images=images, return_tensors="pt")

            # TODO handle other pooling strategies such as avg
            output = self.model(**inputs.to(accelerator.device)).pooler_output
            results.append(accelerator.gather_for_metrics(output).cpu())

        return torch.cat(results)


class VideoExtractor(Extractor):
    """
    Clips are stored as `<path>/mkv/000000.mkv`: FFV1 in Matroska, a lossless
    intra-only codec, so `VideoDecoder` returns the exact generated frames with no
    B-frame reorder shifting the decoded window.

    One random `sequence_length`-frame clip is drawn per video and cropped to
    `resolution` by cd-fvd's `preprocess`, matching the random clip and crop their
    published reference statistics are computed at. The draw is seeded from
    `(seed, index)` alone, so each video's clip is a pure function of its identity:
    reproducible regardless of GPU count, visitation order, or dataloader workers,
    where cd-fvd draws from the shared RNG inside reseeded workers and so pins its
    stats to an unrecorded `num_workers`.
    """

    def __init__(
        self, sequence_length: int = 16, resolution: int = 128, seed: int = 0
    ) -> None:
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.seed = seed

    @abstractmethod
    def features(self, clips: np.ndarray) -> torch.Tensor:
        """
        Network features for a batch of uint8 clips shaped (B, T, H, W, C).
        """
        pass

    def load_clip(self, file: str, index: int) -> np.ndarray:
        # String seed so the draw is stable across machines (random hashes it
        # with SHA-512, unlike PYTHONHASHSEED-sensitive builtin hash()).
        dec = VideoDecoder(file, dimension_order="NHWC")
        start = random.Random(f"{self.seed}:{int(index)}").randint(
            0, max(0, dec.metadata.num_frames - self.sequence_length)
        )
        clip = dec[start : start + self.sequence_length]
        video = preprocess(clip, self.resolution, self.sequence_length)["video"]

        # CTHW in [0, 1] back to the uint8 THWC that both preprocessors take.
        return (video * 255).byte().permute(1, 2, 3, 0).numpy()

    @torch.no_grad()
    def extract(self, path: str, num_samples: int, batch_size: int) -> torch.Tensor:
        results = []

        for batch_idx in tqdm(
            index_loader(num_samples, batch_size),
            disable=not accelerator.is_main_process,
        ):
            clips = np.stack(
                [self.load_clip(sample_path(path, i, "mkv"), i) for i in batch_idx]
            )
            results.append(accelerator.gather_for_metrics(self.features(clips)).cpu())

        return torch.cat(results)

    def extract_all(self, path: str) -> torch.Tensor:
        # Clips are far heavier than images, hence the smaller default batch.
        return self.extract(path, count_samples(path, "mkv"), 8)


class I3DExtractor(VideoExtractor):
    """
    I3D logits (400-d) from the Kinetics-400 action classifier for FVD.
    """

    def __init__(self, ckpt_path: str | None = None, **kwargs) -> None:
        """
        Defaults to the torch hub cache, populated by `scripts/download_pretrained.py`;
        cd-fvd's loader otherwise fetches it there with an unchecked download.
        """
        super().__init__(**kwargs)
        ckpt_path = ckpt_path or os.path.join(
            torch.hub.get_dir(), "checkpoints", "i3d_pretrained_400.pt"
        )
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        self.model = load_i3d_model(accelerator.device, ckpt_path)

    def features(self, clips: np.ndarray) -> torch.Tensor:
        return self.model(preprocess_i3d(clips).to(accelerator.device))


class VideoMAEExtractor(VideoExtractor):
    """
    VideoMAE-v2 features (1408-d), self-supervised then fine-tuned on SSv2.
    Sensitive to temporal quality where the I3D logits are nearly blind to it.
    """

    repo = "OpenGVLab/InternVideoMAE_models"
    filename = "mae-g/vit_g_hybrid_pt_1200e_ssv2_ft.pth"

    def __init__(self, ckpt_path: str | None = None, **kwargs) -> None:
        """
        Defaults to the hub cache, which verifies its download; cd-fvd's own
        downloader writes the response body without checking its status, and so
        caches a CDN error page as the checkpoint.
        """
        super().__init__(**kwargs)
        ckpt_path = ckpt_path or hf_hub_download(self.repo, self.filename)
        self.model = load_videomae_model(accelerator.device, ckpt_path).eval()

    def features(self, clips: np.ndarray) -> torch.Tensor:
        return self.model.forward_features(
            preprocess_videomae(clips).to(accelerator.device)
        )


class ProteinExtractor(Extractor):
    """
    Proteins are stored as `<path>/pdb/000000.pdb`.
    """

    def extract_all(self, path: str) -> torch.Tensor:
        return self.extract(path, count_samples(path, "pdb"), 1)


class ESM3Extractor(ProteinExtractor):
    """
    ESM3 structure-token embeddings (mean-pooled, 1536-d) for protein FID, matching
    the reference embeddings distributed with `ESMTrainRef`.

    Drives protfid's model one structure at a time rather than through
    `FID.compute_embeddings`, whose `batch_generator` never yields its final batch
    (silently dropping the shortest leftover structures), returns rows in
    length-sorted order with the pdb identity discarded, and filters out any chain
    longer than the batch budget. In our implementation, row i is file i.
    """

    def __init__(self) -> None:
        from core.vendor.protfid.fid import FID

        self.model = FID()
        self.model.encoder.to(accelerator.device)
        self.model.model.to(accelerator.device)

    @torch.no_grad()
    def _embed_one(self, chain) -> torch.Tensor:
        coords, plddt, ridx = (
            x.to(accelerator.device) for x in chain.to_structure_encoder_inputs()
        )
        _, st = self.model.encoder.encode(coords, residue_index=ridx)

        coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)
        plddt = F.pad(plddt, (1, 1), value=0)
        st = F.pad(st, (1, 1), value=0)
        st[:, 0], st[:, -1] = 4098, 4097  # protfid BOS/EOS structure tokens

        out = self.model.model.forward(
            structure_coords=coords, per_res_plddt=plddt, structure_tokens=st
        )
        return out.embeddings[0].mean(0).cpu()

    @torch.no_grad()
    def extract(self, path: str, num_samples: int, batch_size: int) -> torch.Tensor:
        """
        Files follow the <path>/pdb/000000.pdb layout, so index i is row i; an unparseable
        structure becomes a NaN row rather than a skip, keeping that alignment.
        Callers reducing over rows (FID mean/covariance) should drop NaN first.
        """
        if batch_size != 1:
            raise ValueError("Protein is embedded one file at a time")

        # TODO support multi-gpu
        assert accelerator.num_processes == 1

        from core.vendor.protfid.fid import load_chain

        embeddings = []
        for i in tqdm(range(num_samples), desc="ESM3"):
            res = load_chain(sample_path(path, i, "pdb"))
            embeddings.append(self._embed_one(res[2]) if res[0] else None)

        good = [e for e in embeddings if e is not None]
        if not good:
            raise RuntimeError(f"No parseable pdb structure under {path}")

        nan = torch.full_like(good[0], float("nan"))
        failed = len(embeddings) - len(good)
        if failed:
            print(f"ESM3Extractor: {failed} of {num_samples} pdb unparseable")

        return torch.stack([e if e is not None else nan for e in embeddings])


class GeometryExtractor(ProteinExtractor):
    """
    Whole-structure geometry from each pdb (biotite P-SEA), one row per structure in
    `FIELDS` order. A residue is one chain unit; its CA (alpha carbon) is the backbone
    atom, so the CA coordinates trace the chain as a point cloud. A residue whose CA
    was not resolved simply drops out of the CA-based metrics.

    Size and shape:
      n         chain length, the number of residues.
      rg        radius of gyration: the rms distance of CA atoms from their centroid,
                in angstroms, which grows with both size and how extended the shape is.
      glob      globularity rg / n**(1/3): compactness with the size trend divided
                out (a solid ball has rg proportional to N**(1/3)). Low is ball-like,
                high is extended and stringy.

    Secondary structure (per-residue backbone shape, the three fractions sum to 1):
      helix     fraction in helices, where the backbone coils into a local spring.
      strand    fraction in strands, extended segments that pair side by side into
                sheets and demand long-range coordination.
      coil      fraction in everything else, the turns and loops between them.
      n_seg     contiguous same-type runs per residue, restarting at chain breaks and
                gaps: how often the SSE label switches, a fragmentation measure.

    Topology:
      co        contact order: over intra-chain CA pairs within 8A, their mean
                separation along the sequence (by residue id), normalized by length.
                Low is local (a helix bundle), high folds back to touch distant
                residues (beta sheets).

    Physical plausibility:
      clash     heavy-atom pairs closer than 2.5A that are not same-chain bonded
                neighbors, per residue. A rough overlap proxy; real structures have
                almost none, so it flags bad generated geometry.
      bond_dev  std of CA-CA spacing over genuine peptide neighbors (same chain,
                consecutive residue id; fixed near 3.8A). A complete backbone varies
                little; much higher means stretches or breaks. Measured only over real
                neighbors, so chain breaks and missing residues no longer inflate it.
    """

    FIELDS = (
        "n",
        "rg",
        "glob",
        "helix",
        "strand",
        "coil",
        "n_seg",
        "co",
        "clash",
        "bond_dev",
    )

    def extract(self, path: str, num_samples: int, batch_size: int) -> torch.Tensor:
        rows = [
            self.geometry(sample_path(path, i, "pdb"))
            for i in tqdm(range(num_samples), desc="geometry")
        ]
        return torch.tensor(rows)

    @staticmethod
    def residue_index(atoms: bs.AtomArray) -> np.ndarray:
        """
        Per-atom residue index: res_id with real gaps kept, but each insertion-coded
        residue (same res_id, advancing ins_code) spending one step so it reads as a
        genuine neighbor rather than a duplicate. Equals res_id when no insertion
        codes are present, so gaps and chain breaks are never glued shut.

        A step of 0 (insertion code) or 1 (normal) is treated as continuous, matching
        biotite's own res_id continuity rule (see check_res_id_continuity, which
        annotate_sse uses to place its discontinuities).
        """
        starts = bs.get_residue_starts(atoms)
        delta = np.diff(atoms.res_id[starts])
        delta = np.where(delta == 0, 1, delta)  # insertion step -> +1, real gaps kept
        per_res = atoms.res_id[starts[0]] + np.concatenate([[0], np.cumsum(delta)])
        return bs.spread_residue_wise(atoms, per_res)

    @staticmethod
    def geometry(path: str) -> tuple[float, ...]:
        arr: bs.AtomArray = pdb.PDBFile.read(path).get_structure(model=1)

        prot = arr[bs.filter_amino_acids(arr)]  # drop water/ligands/ions
        is_ca = prot.atom_name == "CA"  # Alpha carbons
        # P-SEA labels one residue at a time; spread to atoms and keep each residue's
        # CA, so a residue whose CA was not resolved drops out and stays aligned to ca.
        sse = bs.spread_residue_wise(prot, bs.annotate_sse(prot))[is_ca]

        prot.res_id = GeometryExtractor.residue_index(prot)  # heal insertion codes
        ca = prot[is_ca]

        ### Size and shape
        n = len(ca)
        rg = float(bs.gyration_radius(ca))
        glob = rg / n ** (1 / 3)

        ### Secondary structure
        helix = (sse == "a").mean()
        strand = (sse == "b").mean()
        coil = (sse == "c").mean()
        # a genuine peptide neighbor is same-chain with the next residue id
        adj = (np.diff(ca.res_id) == 1) & (ca.chain_id[1:] == ca.chain_id[:-1])
        n_seg = int(1 + np.sum((sse[1:] != sse[:-1]) | ~adj)) / n

        ### Topology
        # indices such that 0 <= i < j < n
        i, j = np.triu_indices(n, 1)
        # scipy pdist is distances condensed in triu_indices(n, 1) order
        contacts = (ca.chain_id[i] == ca.chain_id[j]) & (pdist(ca.coord) < 8.0)
        # sequence separation, intra-chain only
        separation = ca.res_id[j] - ca.res_id[i]
        co = separation[contacts].mean() / n if contacts.any() else 0.0

        ### Physical plausibility
        heavy = prot[prot.element != "H"]
        cid, rid = heavy.chain_id, heavy.res_id
        a, b = cKDTree(heavy.coord).query_pairs(2.5, output_type="ndarray").T
        # a clash is a close pair that is not a same-chain covalent neighbor
        clash = (
            np.count_nonzero((cid[a] != cid[b]) | (np.abs(rid[a] - rid[b]) >= 2)) / n
        )
        ca_distances = np.linalg.norm(np.diff(ca.coord, axis=0), axis=1)
        bond_dev = float(ca_distances[adj].std() if adj.any() else 0.0)

        # tuple in FIELDS order
        return n, rg, glob, helix, strand, coil, n_seg, co, clash, bond_dev
