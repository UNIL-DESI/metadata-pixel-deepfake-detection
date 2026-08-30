"""Extract full-frame, face-crop, and expanded-face regions from frames."""

from __future__ import annotations

from pathlib import Path
import glob

import cv2
import pandas as pd
from tqdm import tqdm

from preprocessing.detect_faces import (
    detect_largest_face,
    expand_bbox,
    get_haar_face_detector,
)
from preprocessing.resize_faces import resize_and_save


def extract_multi_regions_from_frame(
    frame_path: str | Path,
    output_video_folder: str | Path,
    frame_name: str,
    label: str,
    video_name: str,
    log_rows: list[dict],
    face_detector=None,
    size: tuple[int, int] = (224, 224),
    expanded_scale: float = 1.5,
    
) -> None:
    """Extract and save full, face, and expanded-face regions from one frame."""
    frame_path = Path(frame_path)
    output_video_folder = Path(output_video_folder)

    frame = cv2.imread(str(frame_path))

    if frame is None:
        log_rows.append(
            {
                "label": label,
                "video_name": video_name,
                "frame_name": frame_name,
                "frame_path": str(frame_path),
                "face_detected": 0,
                "status": "corrupted_frame",
            }
        )
        return

    height, width = frame.shape[:2]

    full_dir = output_video_folder / "full"
    face_dir = output_video_folder / "face"
    expanded_dir = output_video_folder / "expanded"

    full_save_path = full_dir / frame_name
    resize_and_save(frame, full_save_path, size=size)

    face_box = detect_largest_face(frame, face_detector=face_detector)

    if face_box is None:
        log_rows.append(
            {
                "label": label,
                "video_name": video_name,
                "frame_name": frame_name,
                "frame_path": str(frame_path),
                "full_path": str(full_save_path),
                "face_path": None,
                "expanded_path": None,
                "face_detected": 0,
                "status": "no_face_detected",
            }
        )
        return

    x, y, fw, fh = face_box

    face_crop = frame[y : y + fh, x : x + fw]
    face_save_path = face_dir / frame_name
    resize_and_save(face_crop, face_save_path, size=size)

    ex1, ey1, ex2, ey2 = expand_bbox(
        x,
        y,
        fw,
        fh,
        img_width=width,
        img_height=height,
        scale=expanded_scale,
    )

    expanded_crop = frame[ey1:ey2, ex1:ex2]
    expanded_save_path = expanded_dir / frame_name
    resize_and_save(expanded_crop, expanded_save_path, size=size)

    log_rows.append(
        {
            "label": label,
            "video_name": video_name,
            "frame_name": frame_name,
            "frame_path": str(frame_path),
            "full_path": str(full_save_path),
            "face_path": str(face_save_path),
            "expanded_path": str(expanded_save_path),
            "face_detected": 1,
            "x": int(x),
            "y": int(y),
            "w": int(fw),
            "h": int(fh),
            "expanded_x1": int(ex1),
            "expanded_y1": int(ey1),
            "expanded_x2": int(ex2),
            "expanded_y2": int(ey2),
            "status": "success",
        }
    )


def process_all_multi_regions(
    input_frames_folder: str | Path,
    output_regions_folder: str | Path,
    labels: list[str] | None = None,
    size: tuple[int, int] = (224, 224),
    expanded_scale: float = 1.5,
    video_limit: int | None = None,
) -> pd.DataFrame:
    """Process all extracted frames and create the multi-region log CSV."""
    input_frames_folder = Path(input_frames_folder)
    output_regions_folder = Path(output_regions_folder)
    output_regions_folder.mkdir(parents=True, exist_ok=True)

    labels = labels or ["real", "fake"]
    face_detector = get_haar_face_detector()
    log_rows: list[dict] = []

    for label in labels:
        label_folder = input_frames_folder / label

        if not label_folder.exists():
            print(f"[WARN] Folder not found: {label_folder}")
            continue

        video_folders = sorted([p for p in label_folder.iterdir() if p.is_dir()])
        if video_limit is not None:
            video_folders = video_folders[:video_limit]

        for video_folder in tqdm(video_folders, desc=f"Processing {label}"):
            video_name = video_folder.name
            output_video_folder = output_regions_folder / label / video_name

            # Create output directories once per video
            (output_video_folder / "full").mkdir(parents=True, exist_ok=True)
            (output_video_folder / "face").mkdir(parents=True, exist_ok=True)
            (output_video_folder / "expanded").mkdir(parents=True, exist_ok=True)

            frame_paths = sorted(
                glob.glob(str(video_folder / "*.jpg"))
                + glob.glob(str(video_folder / "*.png"))
                + glob.glob(str(video_folder / "*.jpeg"))
        )
            for frame_path in frame_paths:
                frame_name = Path(frame_path).name

                extract_multi_regions_from_frame(
                    frame_path=frame_path,
                    output_video_folder=output_video_folder,
                    frame_name=frame_name,
                    label=label,
                    video_name=video_name,
                    log_rows=log_rows,
                    face_detector=face_detector,
                    size=size,
                    expanded_scale=expanded_scale,
                )

    log_df = pd.DataFrame(log_rows)
    log_csv_path = output_regions_folder / "multi_region_extraction_log.csv"
    log_df.to_csv(log_csv_path, index=False)

    print("Multi-region extraction finished.")
    print(f"Log saved to: {log_csv_path}")

    if not log_df.empty:
        print(log_df["face_detected"].value_counts(dropna=False))

    return log_df
