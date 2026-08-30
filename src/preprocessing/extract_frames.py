"""Extract a fixed number of frames from each video."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


def extract_frames(
    video_path: str | Path,
    save_folder: str | Path,
    num_frames: int = 25,
    image_ext: str = ".jpg",
) -> bool:
    """Extract uniformly distributed frames from a video.

    This is a fixed-number strategy, not FPS-based extraction.
    """
    video_path = Path(video_path)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"[WARN] Cannot read video or video has zero frames: {video_path}")
        cap.release()
        return False

    # Use the center of each segment to avoid repeatedly selecting frame 0.
    frame_indices = [
        min(int((i + 0.5) * total_frames / num_frames), total_frames - 1)
        for i in range(num_frames)
    ]

    saved = 0
    for idx, frame_no in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        success, frame = cap.read()

        if success and frame is not None:
            frame_name = f"frame_{idx:03d}{image_ext}"
            cv2.imwrite(str(save_folder / frame_name), frame)
            saved += 1

    cap.release()
    return saved > 0


def extract_frames_from_catalog(
    catalog_csv: str | Path,
    output_dir: str | Path,
    num_frames: int = 25,
) -> None:
    """Extract frames for every video in the dataset catalog."""
    catalog_csv = Path(catalog_csv)
    output_dir = Path(output_dir)

    df = pd.read_csv(catalog_csv)
    print(f"Total videos: {len(df)}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting frames"):
        video_path = Path(row["filepath"])
        label = row["label"]
        video_name = row.get("video_name", video_path.stem)

        save_path = output_dir / label / video_name
        extract_frames(video_path, save_path, num_frames=num_frames)

    print("Finished extracting frames.")
