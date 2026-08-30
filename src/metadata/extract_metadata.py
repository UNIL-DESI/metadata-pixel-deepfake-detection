"""Extract file-level/video-level metadata."""

from __future__ import annotations

from pathlib import Path
import json
import subprocess

import cv2
import pandas as pd
from tqdm import tqdm


def extract_metadata_opencv(video_path: str | Path) -> dict:
    """Extract basic video metadata using OpenCV."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    duration = frame_count / fps if fps and fps > 0 else None

    cap.release()

    return {
        "video_path": str(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration,
        "file_size_bytes": video_path.stat().st_size if video_path.exists() else None,
    }


def extract_metadata_ffprobe(video_path: str | Path) -> dict:
    """Extract richer metadata using ffprobe if it is installed."""
    video_path = Path(video_path)

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        return {}

    data = json.loads(result.stdout)
    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        {},
    )
    fmt = data.get("format", {})

    return {
        "codec_name": video_stream.get("codec_name"),
        "codec_long_name": video_stream.get("codec_long_name"),
        "pix_fmt": video_stream.get("pix_fmt"),
        "bit_rate": video_stream.get("bit_rate") or fmt.get("bit_rate"),
        "format_name": fmt.get("format_name"),
        "format_long_name": fmt.get("format_long_name"),
    }


def extract_metadata_from_catalog(
    catalog_csv: str | Path,
    output_csv: str | Path,
    use_ffprobe: bool = True,
) -> pd.DataFrame:
    """Extract metadata for all videos listed in the catalog."""
    df = pd.read_csv(catalog_csv)
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting metadata"):
        video_path = row["filepath"]

        metadata = {
            "label": row.get("label"),
            "dataset": row.get("dataset"),
            "video_name": row.get("video_name", Path(video_path).stem),
        }
        metadata.update(extract_metadata_opencv(video_path))

        if use_ffprobe:
            metadata.update(extract_metadata_ffprobe(video_path))

        rows.append(metadata)

    metadata_df = pd.DataFrame(rows)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(output_csv, index=False)

    print(f"Saved metadata: {output_csv}")
    print(f"Shape: {metadata_df.shape}")

    return metadata_df
