import glob
import io
import json
import os
import re
import random
import subprocess
import tarfile
import tempfile
import zipfile
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from typing import Iterator

import numpy as np
import pandas as pd
from cdfvd.utils.data_utils import preprocess
from PIL import Image
from torchcodec.decoders import VideoDecoder
from tqdm.auto import tqdm

from core.data.base import HuggingFaceDataset
from core.utils.layout import count_samples, sample_path


def write_ffv1(frames: np.ndarray, path: str, fps: float) -> None:
    """
    Encode a THWC uint8 clip to lossless FFV1 in Matroska at `path`.

    Piping raw rgb24 into ffmpeg round-trips bit-exactly through `VideoDecoder`.
    Encoding via torchvision's `write_video` would not, since it uses yuv420p.

    Slice threading is ffv1's only parallelism: the default single slice encodes
    on one core, so we use 16 slices for faster encoding.
    """
    t, h, w, _ = frames.shape
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error"]
        + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps)]
        + ["-i", "pipe:0", "-c:v", "ffv1", "-slices", "16", "-slicecrc", "0", path],
        input=np.ascontiguousarray(frames, dtype=np.uint8).tobytes(),
        check=True,
    )


def ram_zip(path: str) -> zipfile.ZipFile:
    """
    Open an archive off an in-memory copy.

    Members read out of order seek across the archive, and interleaving those
    seeks with the encoder's writes starves both on a spinning disk. BytesIO
    shares the bytes instead of copying, so this costs the archive's size once.
    """
    with open(path, "rb") as f:
        return zipfile.ZipFile(io.BytesIO(f.read()))


class VideoDataset(HuggingFaceDataset):
    """
    Reference video set following cd-fvd's protocol: `num_clips` sources, each cut
    to one deterministic `num_frames`-frame window and re-encoded losslessly to
    `<name>/mkv/000000.mkv` (FFV1) with a row-aligned `prompts.csv`. The window is
    stored at the source's own resolution; the extractor crops to its `resolution`
    at read time.

    Each source carries its own fps, kept as-is on the stored clip. It is cosmetic
    to FVD (which reads by frame index) and reference clips never reach the
    fps-sensitive VBench, but storing a source's true rate keeps the clip honest.
    """

    def __init__(
        self,
        hf_key: str,
        split: str = "train",
        num_clips: int = 2048,
        num_frames: int = 16,
    ) -> None:
        super().__init__(hf_key, split)
        self.num_clips = num_clips  # cd-fvd's per-dataset clip budget
        self.num_frames = num_frames  # matches the extractor's sequence_length

    def convert_dataset(self, dataset) -> None:
        path = self.get_path()
        prompts = []
        for i, (frames, prompt, fps) in tqdm(
            zip(range(self.num_clips), dataset),
            total=self.num_clips,
            desc=self.get_name(),
        ):
            write_ffv1(frames, sample_path(path, i, "mkv", write=True), fps)
            prompts.append(prompt)
        pd.DataFrame({"prompts": prompts}).to_csv(self.get_csv_path(), index=False)

    @abstractmethod
    def sources(self) -> Iterator[tuple[np.ndarray, str, float]]:
        """
        Yield `(THWC uint8 clip of num_frames frames, prompt, fps)` per source
        video, in a deterministic order so the first `num_clips` are a stable,
        representative subsample.
        """

    def window(self, seq: Sequence, key: str) -> Sequence:
        """
        A deterministic `num_frames`-slice of `seq` -- a list of frame files or a
        range of frame indices -- a pure function of `key`. `random` seeds from the
        string key via SHA-512, so the draw is stable across machines.
        """
        start = random.Random(key).randint(0, max(0, len(seq) - self.num_frames))
        return seq[start : start + self.num_frames]

    def read_clip(self, file: str, key: str) -> tuple[np.ndarray, float]:
        """
        Decode only the deterministic `num_frames` window of `file`, seeking to it
        instead of decoding the whole clip. torchcodec's frame-accurate seek yields
        the same frames a full decode would, so the clip is a pure function of `key`.
        """
        dec = VideoDecoder(file, dimension_order="NHWC")
        idx = self.window(range(dec.metadata.num_frames), key)
        return dec[idx.start : idx.stop].numpy(), dec.metadata.average_fps


def decamel(name: str) -> str:
    return re.sub(r"(?=[A-Z])", " ", name).lower().strip()


class UCF101(VideoDataset):
    """
    UCF-101 action clips from the HF zip mirror, laid out inside the archive as
    `UCF-101/<Class>/v_<Class>_g##_c##.avi`. Prompt is the de-camel-cased class name,
    the zero-shot text-to-video convention.

    Sources are deterministically shuffled so using first N prompts for generation is
    class-representative rather than one class.
    """

    def __init__(self, hf_key: str = "quchenyuan/UCF101-ZIP") -> None:
        super().__init__(hf_key)

    def sources(self) -> Iterator[tuple[np.ndarray, str, float]]:
        with ram_zip(self.resolve_snapshot("UCF-101.zip")) as zf:
            names = sorted(n for n in zf.namelist() if n.lower().endswith(".avi"))
            random.Random(0).shuffle(names)
            for name in names:
                with tempfile.NamedTemporaryFile(suffix=".avi") as tmp:
                    tmp.write(zf.read(name))
                    tmp.flush()
                    clip, fps = self.read_clip(tmp.name, name)
                prompt = decamel(os.path.basename(os.path.dirname(name)))
                yield clip, prompt, fps


def frame_index(path: str) -> int:
    return int(os.path.basename(path).split(".")[0].split("_")[-1])


class FrameVideoDataset(VideoDataset):
    """
    StyleGAN-V-style frame dumps: one zip of per-video folders holding numbered
    frame images (the exact nesting varies by dataset). Every leaf folder (images
    grouped by parent dir) with at least `num_frames` frames is one video, its
    frames ordered by the trailing frame number (`frame_index`).

    Only folders under the `split` path segment are kept: reference statistics
    use the train split, matching the cd-fvd convention (and keeping test data out
    of the reference). Folders are deterministically shuffled so the first
    `num_clips` are a representative subsample, not one alphabetical run.

    The `num_frames` window is taken over the frame files, so only those images
    are decoded. Frame dumps carry no source fps, so a nominal cosmetic rate is
    stored.
    """

    def __init__(
        self,
        hf_key: str,
        archive: str,
        prompt: str,
        split: str = "train",
        fps: float = 30.0,
    ) -> None:
        super().__init__(hf_key, split)
        self.archive = archive
        self.prompt = prompt
        self.fps = fps

    def sources(self) -> Iterator[tuple[np.ndarray, str, float]]:
        with zipfile.ZipFile(self.resolve_snapshot(self.archive)) as zf:
            folders = defaultdict(list)
            for n in zf.namelist():
                if n.lower().endswith((".jpg", ".jpeg", ".png")):
                    folders[os.path.dirname(n)].append(n)
            names = sorted(folders)
            random.Random(0).shuffle(names)
            for folder in names:
                if self.split not in folder.split("/"):
                    continue
                if len(folders[folder]) < self.num_frames:
                    continue
                files = self.window(sorted(folders[folder], key=frame_index), folder)
                clip = np.stack(
                    [
                        np.asarray(Image.open(io.BytesIO(zf.read(f))).convert("RGB"))
                        for f in files
                    ]
                )
                yield clip, self.prompt, self.fps


class SkyTimelapse(FrameVideoDataset):

    def __init__(self, hf_key: str = "maxin-cn/SkyTimelapse") -> None:
        super().__init__(
            hf_key,
            archive="sky_timelapse.zip",
            prompt="a time-lapse video of the sky",
            split="sky_train",
        )


class FaceForensics(FrameVideoDataset):

    def __init__(self, hf_key: str = "maxin-cn/FaceForensics") -> None:
        super().__init__(
            hf_key,
            archive="FaceForensics.zip",
            prompt="a person talking to the camera",
        )


class TaichiHD(FrameVideoDataset):

    def __init__(self, hf_key: str = "maxin-cn/Taichi-HD") -> None:
        super().__init__(
            hf_key,
            archive="taichi-256.zip",
            prompt="a person performing taichi",
        )

    def resolve_snapshot(self, *args: str) -> str:
        """
        Taichi-HD ships as a `zip -s` split (`taichi-256-part.z01`..`.z04` + `.zip`)
        that zipfile cannot read; reassemble the parts into one archive beside them.
        """
        part = super().resolve_snapshot("taichi-256-part.zip")
        whole = os.path.join(os.path.dirname(part), self.archive)
        if not os.path.exists(whole):
            subprocess.run(["zip", "-s", "0", part, "--out", whole], check=True)
        return os.path.join(os.path.dirname(part), *args)


class ChainedFile:
    """Read a byte stream split byte-wise across `paths` as one file object."""

    def __init__(self, paths: list[str]) -> None:
        self.paths = iter(paths)
        self.f = open(next(self.paths), "rb")

    def read(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.f.read(n - len(buf))
            if chunk:
                buf += chunk
                continue
            self.f.close()
            try:
                self.f = open(next(self.paths), "rb")
            except StopIteration:
                break
        return buf


class SomethingSomethingV2(VideoDataset):
    """
    20bn-something-something-v2 clips from the HF mirror of the official release:
    one gzip-tar of `<id>.webm` files split byte-wise into 20 `cat`-and-untar
    parts, labelled by `train.json`. Prompt is the clip's native instantiated
    caption (its `label`), the templated SSv2 convention.

    Only train ids are kept (the reference split); the val/test webms sharing the
    archive are skipped. The parts must be streamed as one gzip stream (each part
    but the first is not independently valid), so clips arrive in the archive's
    own already-id-shuffled order, which makes the first N prompts representative
    across templates without a reorder we could not do on a stream anyway.
    """

    def __init__(self, hf_key: str = "morpheushoc/something-something-v2") -> None:
        super().__init__(hf_key)

    def sources(self) -> Iterator[tuple[np.ndarray, str, float]]:
        with open(self.resolve_snapshot("train.json")) as f:
            labels = {e["id"]: e["label"] for e in json.load(f)}
        parts = sorted(
            glob.glob(self.resolve_snapshot("videos", "20bn-something-something-v2-*"))
        )
        with tarfile.open(fileobj=ChainedFile(parts), mode="r|gz") as tar:
            for member in tar:
                id = os.path.splitext(os.path.basename(member.name))[0]
                if id not in labels:
                    continue
                with tempfile.NamedTemporaryFile(suffix=".webm") as tmp:
                    tmp.write(tar.extractfile(member).read())
                    tmp.flush()
                    clip, fps = self.read_clip(tmp.name, id)
                if len(clip) < self.num_frames:
                    continue
                yield clip, labels[id], fps


DISTORTIONS = {
    "elastic": "elastic_transform",
    "motion": "motion_blur",
}  # distortion family -> imagecorruptions name


def distort(
    clip: np.ndarray, corruption: str, severity: int, redraw: int, seed: int
) -> np.ndarray:
    """
    An ImageNet-C corruption applied per frame at `severity` (1-5), its random field
    redrawn every `redraw` frames. All frames in a redraw-group share a seed, so the
    warp/blur is coherent within the group: `redraw >= len(clip)` reuses one field for
    the whole clip (spatial), `redraw == 1` redraws every frame (full flicker).

    `imagecorruptions.corrupt` draws from numpy's global RNG and takes no seed, so
    seeding it per group is what pins the field; the draw is a pure function of
    (seed, group), independent of visitation order. Seeding mutates global numpy state,
    which is acceptable in this offline materialization.
    """
    from imagecorruptions import corrupt

    name = DISTORTIONS[corruption]
    frames = []
    for t, frame in enumerate(clip):
        np.random.seed(seed * len(clip) + t // redraw)  # unique per (clip, group)
        frames.append(corrupt(frame, severity, corruption_name=name))
    return np.stack(frames).astype(np.uint8)


class DistortedVideoDataset(VideoDataset):
    """
    A distorted copy of a reference set's clips: each stored clip, cropped to the
    extractor's `resolution` view, corrupted by one ImageNet-C family at one severity
    with its field redrawn every `redraw` frames (`distort`). Written to
    `samples/probes/<reference>_<family>_k<redraw>_s<severity>/`, a sibling of any
    reference set, so `VideoExtractor` reads it with no special-casing.

    Holds a `VideoDataset` rather than subclassing one (has-a, not is-a), so it
    distorts any source set with no per-dataset code: the reference supplies clean clips,
    this supplies the distortion.
    """

    def __init__(
        self,
        reference: VideoDataset,
        corruption: str,
        severity: int = 3,
        redraw: int = 1,
        resolution: int = 128,
    ) -> None:
        super().__init__(
            reference.hf_key,
            reference.split,
            reference.num_clips,
            reference.num_frames,
        )
        self.reference = reference
        self.corruption = corruption
        self.severity = severity
        self.redraw = redraw
        self.resolution = resolution

    def get_name(self) -> str:
        ref = self.reference.get_name()
        return f"{ref}_{self.corruption}_k{self.redraw}_s{self.severity}"

    def get_path(self) -> str:
        return os.path.join("samples", "probes", self.get_name())

    def prepare(self) -> None:
        self.reference.prepare()  # materialize the clean reference set first
        super().prepare()

    def read_clean(self, file: str) -> tuple[np.ndarray, float]:
        """
        The reference clip as the extractor sees it: the `resolution` crop only, since
        reference clips are already `num_frames` windows with nothing left to re-window.
        Corruptions are applied at this size, where ImageNet-C's absolute-pixel
        severities are defined.
        """
        dec = VideoDecoder(file, dimension_order="NHWC")
        video = preprocess(dec[0 : self.num_frames], self.resolution, self.num_frames)
        clip = (video["video"] * 255).byte().permute(1, 2, 3, 0).numpy()
        return clip, dec.metadata.average_fps

    def sources(self) -> Iterator[tuple[np.ndarray, str, float]]:
        ref_path = self.reference.get_path()
        prompts = pd.read_csv(self.reference.get_csv_path())["prompts"].tolist()

        for i in range(min(self.num_clips, count_samples(ref_path, "mkv"))):
            clean, fps = self.read_clean(sample_path(ref_path, i, "mkv"))
            clip = distort(clean, self.corruption, self.severity, self.redraw, i)
            yield clip, prompts[i], fps
