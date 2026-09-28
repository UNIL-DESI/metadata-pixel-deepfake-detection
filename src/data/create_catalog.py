"""Create a dataset inventory CSV from the video folders."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


# ---> ADD DEEPERFORENSICS HERE <---
DEFAULT_FOLDERS: Dict[str, Tuple[str, str]] = {
    "FF++": ("Real", "Fake"),
    "Celeb-DF": ("Real", "Fake"),
    "DFDC": ("Real", "Fake"),
    "DeeperForensics": ("Real", "Fake"),  
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
        # Iterate over Real and Fake subdirectories
        for folder_name, label in [(real_folder, "real"), (fake_folder, "fake")]:
            folder_path = base_dir / dataset / folder_name

            # Fallback for lowercase folder names if "Real" or "Fake" isn't found
            if not folder_path.exists():
                folder_path_lower = base_dir / dataset / folder_name.lower()
                if folder_path_lower.exists():
                    folder_path = folder_path_lower
                else:
                    print(f"[WARN] Folder not found: {folder_path}")
                    continue

            valid_videos = [f for f in sorted(folder_path.iterdir()) if f.is_file() and f.suffix.lower() in video_extensions]

            for file_path in valid_videos:
                rows.append(
                    {
                        "filepath": str(file_path).replace("\\", "/"), # Ensure forward slashes
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