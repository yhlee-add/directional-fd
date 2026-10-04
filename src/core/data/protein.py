import os
import io
import tarfile
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from core.data.base import Dataset
from core.utils.layout import sample_path


class ESMTrainRef(Dataset):
    """
    Protein FID reference set: precomputed ESM3 embeddings of the paper's PDB
    reference (FoldSeek cluster representatives from ESM3's training data,
    Faltings et al., arXiv 2505.08041).

    Cached as `esm3.npy` (the `ESM3Extractor` feature cache) so the protein FID metric
    reads it directly without materializing any PDB structures or running the extractor.

    Also as the generation-conditioning dataset, unconditional protein generation has
    no prompts, so `prepare` writes an index-only `prompts.csv` from which the pipeline
    sources per-sample seeds.
    """

    ckpt_url = "https://zenodo.org/records/15660186/files/esm_train_ref.ckpt"

    def prepare(self) -> None:
        os.makedirs(self.get_path(), exist_ok=True)
        if not os.path.exists(self.get_csv_path()):
            ref = torch.hub.load_state_dict_from_url(self.ckpt_url, map_location="cpu")
            np.save(
                os.path.join(self.get_path(), "esm3.npy"),
                torch.as_tensor(ref).float().numpy(),
            )
            pd.DataFrame({"index": range(len(ref))}).to_csv(
                self.get_csv_path(), index=False
            )


class CATHDomains(Dataset):
    """
    CATH v4.3.0 S40 non-redundant domain set as a labeled real-protein reference.

    Each domain's structure is written as `<path>/pdb/<index>.pdb` and its CATH
    classification as the matching row of `prompts.csv`.

    Non-redundant at 40% sequence identity (~31k domains) and the standard real
    reference in protein backbone generation (Genie2, FrameFlow, TopoDiff).
    """

    cls_url = (
        "https://download.cathdb.info/cath/releases/all-releases/v4_3_0"
        "/cath-classification-data/cath-domain-list-v4_3_0.txt"
    )
    pdb_url = (
        "https://download.cathdb.info/cath/releases/all-releases/v4_3_0"
        "/non-redundant-data-sets/cath-dataset-nonredundant-S40-v4_3_0.pdb.tgz"
    )

    @staticmethod
    def cached(url: str) -> str:
        cache = os.path.join(torch.hub.get_dir(), os.path.basename(url))
        if not os.path.exists(cache):
            torch.hub.download_url_to_file(url, cache)
        return cache

    def prepare(self) -> None:
        if os.path.exists(self.get_csv_path()):  # csv is written last, so it means done
            return

        cols = "domain c a t h s35 s60 s95 s100 s100c length resolution".split()
        dtypes = defaultdict(lambda: "int", domain="str", resolution="float")
        cath = pd.read_csv(
            self.cached(self.cls_url),
            sep=r"\s+",
            comment="#",
            names=cols,
            dtype=dtypes,
            engine="python",
        ).set_index("domain")

        with open(self.cached(self.pdb_url), "rb") as f:
            tgz = io.BytesIO(f.read())

        with tarfile.open(fileobj=tgz) as tar:
            # tarball holds one pdb file per domain, named by its domain id
            members = {
                os.path.basename(m.name): m for m in tar.getmembers() if m.isfile()
            }
            domains = sorted(members)
            for i, domain in enumerate(tqdm(domains, desc="CATH S40 domains")):
                with open(
                    sample_path(self.get_path(), i, "pdb", write=True), "wb"
                ) as f:
                    f.write(tar.extractfile(members[domain]).read())

        cath.loc[domains].to_csv(self.get_csv_path())
