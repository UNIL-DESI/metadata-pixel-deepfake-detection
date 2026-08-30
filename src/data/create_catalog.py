"""Create a dataset inventory CSV from the video folders."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


DEFAULT_FOLDERS: Dict[str, Tuple[str, str]] = {
    "FaceForensics": ("RealVideos", "FakeVideos"),
    "Celeb-DF": ("RealVideos", "FakeVideos"),
    "DFDC_Selected": ("RealVideos", "FakeVideos"),
}

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def create_dataset_catalog(
    base_dir: str | Path,
    output_csv: str | Path | None = None,
    folders: Dict[str, Tuple[str, str]] | None = None,
    video_extensions: Iterable[str] = VIDEO_EXTENSIONS,
) -> pd.DataFrame:
    """Scan dataset folders and create a video-level inventory.

    Expected layout:
        base_dir/DatasetName/RealVideos/*.mp4
        base_dir/DatasetName/FakeVideos/*.mp4
    """
    base_dir = Path(base_dir)
    folders = folders or DEFAULT_FOLDERS
    video_extensions = tuple(ext.lower() for ext in video_extensions)

    rows = []

    for dataset, (real_folder, fake_folder) in folders.items():
        for folder_name, label in [(real_folder, "real"), (fake_folder, "fake")]:
            folder_path = base_dir / dataset / folder_name

            if not folder_path.exists():
                print(f"[WARN] Folder not found: {folder_path}")
                continue

            for file_path in sorted(folder_path.iterdir()):
                if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                    rows.append(
                        {
                            "filepath": str(file_path),
                            "dataset": dataset,
                            "label": label,
                            "video_name": file_path.stem,
                        }
                    )

    df = pd.DataFrame(rows)

    if output_csv is None:
        output_csv = base_dir / "dataset_inventory.csv"

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Saved catalog: {output_csv}")
    print(f"Total videos: {len(df)}")

    if not df.empty:
        print(df["dataset"].value_counts())
        print(df["label"].value_counts())

    return df
