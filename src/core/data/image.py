import ast
import os
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from core.utils.layout import sample_path
from core.data.base import HuggingFaceDataset, LocalDataset


class ImageDataset(HuggingFaceDataset):
    def __init__(
        self,
        hf_key: str,
        split: str = "train",
        image_key: str = "image",
        prompt_key: str = "text",
    ) -> None:
        super().__init__(hf_key, split)
        self.image_key = image_key
        self.prompt_key = prompt_key

    def convert_dataset(self, dataset) -> None:
        path = self.get_path()
        prompts = []

        futures = deque()
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
            for i, x in enumerate(tqdm(dataset)):
                prompts.append(x[self.prompt_key])
                futures.append(
                    ex.submit(
                        x[self.image_key].save, sample_path(path, i, "png", write=True)
                    )
                )
                if len(futures) >= 64:
                    futures.popleft().result()
            for f in futures:
                f.result()

        pd.DataFrame({"prompts": prompts}).to_csv(self.get_csv_path(), index=False)


class COCO30K(ImageDataset):

    def __init__(self, hf_key: str = "sayakpaul/coco-30-val-2014") -> None:
        super().__init__(hf_key, prompt_key="caption")


class MJHQ30K(ImageDataset):

    def __init__(self, hf_key: str = "xingjianleng/mjhq30k") -> None:
        super().__init__(hf_key, split="test")


class CelebAHQ(ImageDataset):

    def __init__(self, hf_key: str = "oftverse/control-celeba-hq") -> None:
        super().__init__(hf_key, split="train+test")


class ImageNet(ImageDataset):

    def __init__(self, hf_key: str = "BenSchneider/imagenet-val") -> None:
        super().__init__(hf_key, split="test", prompt_key="caption")


class FFHQ512(ImageDataset):

    def __init__(self, hf_key: str = "Ryan-sjtu/ffhq512-caption") -> None:
        super().__init__(hf_key)


class Flickr30K(ImageDataset):

    def __init__(self, hf_key: str = "nlphuji/flickr30k") -> None:
        super().__init__(hf_key, split="test", prompt_key="caption")

    def sources(self):
        return list(self._iter_snapshot())

    def _iter_snapshot(self):
        # Repo is not in parquet layout, so read its raw files instead of
        # going through load_dataset.
        df = pd.read_csv(self.resolve_snapshot("flickr_annotations_30k.csv"))
        with zipfile.ZipFile(self.resolve_snapshot("flickr30k-images.zip")) as zf:
            for row in df.itertuples():
                with zf.open(f"flickr30k-images/{row.filename}") as f:
                    img = Image.open(BytesIO(f.read())).copy()
                yield {"image": img, "caption": ast.literal_eval(row.raw)[0]}


class DrawBench(LocalDataset):
    pass
