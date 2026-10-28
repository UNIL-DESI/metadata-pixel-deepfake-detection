#!/usr/bin/env python3
"""Build a tiny synthetic deepfake dataset for smoke tests.

Real videos show scikit-image's astronaut portrait drifting slowly; fake ones
are the same footage with the face region blended with a colour-shifted,
blurred copy, written at a different frame rate. The ZIP mirrors the Hugging
Face archive layout (<dataset>/<Real|Fake>/<video>.mp4), so it can go through
scripts/prepare_dataset_hf.py --local-zip exactly like the real data.
"""
from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from skimage import data

DATASETS = ("FF++", "Celeb-DF", "DFDC")
# Face box of the astronaut image (512x512), found with OpenCV's Haar cascade.
FACE = (165, 55, 120, 120)


def _frames(fake: bool, seed: int, count: int, size=(320, 320)):
    rng = np.random.default_rng(seed)
    base = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    if fake:
        x, y, w, h = FACE
        face = base[y:y + h, x:x + w].astype(np.float32)
        swapped = cv2.GaussianBlur(face, (0, 0), 3) * (0.85, 1.0, 1.15)
        base = base.copy()
        base[y:y + h, x:x + w] = np.clip(0.4 * face + 0.6 * swapped, 0, 255).astype(np.uint8)
    for i in range(count):
        dx, dy = 6 * np.sin(i / 5 + seed), 4 * np.cos(i / 7 + seed)
        shift = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(base, shift, (512, 512), borderMode=cv2.BORDER_REFLECT)
        noise = rng.normal(0, 4, frame.shape)
        yield cv2.resize(np.clip(frame + noise, 0, 255).astype(np.uint8), size)


def write_video(path: Path, fake: bool, seed: int, frames: int = 40) -> None:
    fps = 24 if fake else 30
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 320))
    for frame in _frames(fake, seed, frames):
        writer.write(frame)
    writer.release()


def build_zip(out_zip: Path, videos_per_class: int = 4) -> Path:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(out_zip, "w") as archive:
        seed = 0
        for dataset in DATASETS:
            for label in ("Real", "Fake"):
                for i in range(videos_per_class):
                    seed += 1
                    video = Path(tmp) / f"{dataset}_{label}_{i}.mp4"
                    write_video(video, fake=label == "Fake", seed=seed)
                    # File names are unique across datasets, as in the real data:
                    # the stage-by-stage pipeline keys videos by file name.
                    slug = dataset.lower().replace("+", "p").replace("-", "")
                    archive.write(video, f"Deepfakes/{dataset}/{label}/{slug}_{label.lower()}_{i:03d}.mp4")
    return out_zip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="ZIP file to write")
    parser.add_argument("--videos-per-class", type=int, default=4)
    args = parser.parse_args()
    print("Wrote", build_zip(args.output, args.videos_per_class))


if __name__ == "__main__":
    main()
