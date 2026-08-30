"""High-level handcrafted feature extraction from multi-region frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from features.common import is_valid_path
from features.extract_color import (
    create_zero_color_features,
    extract_color_features,
)
from features.extract_dct import (
    create_zero_dct_features,
    extract_dct_features,
)
from features.extract_edge import (
    create_zero_edge_features,
    extract_edge_features,
)
from features.extract_glcm import (
    create_zero_glcm_features,
    extract_glcm_features,
)
from features.extract_lbp import (
    create_zero_lbp_features,
    extract_lbp_features,
)


def create_zero_features(prefix: str) -> dict:
    """Create zero-filled features when a region is missing."""
    features = {}
    features.update(create_zero_lbp_features(prefix))
    features.update(create_zero_glcm_features(prefix))
    features.update(create_zero_dct_features(prefix))
    features.update(create_zero_color_features(prefix))
    features.update(create_zero_edge_features(prefix))
    return features


def extract_region_features(
    image_path,
    prefix: str,
    size: tuple[int, int] = (224, 224),
) -> dict:
    """Extract all handcrafted features from one image region."""
    if not is_valid_path(image_path):
        features = create_zero_features(prefix)
        features[f"{prefix}_missing"] = 1
        return features

    img_bgr = cv2.imread(str(image_path))

    if img_bgr is None:
        features = create_zero_features(prefix)
        features[f"{prefix}_missing"] = 1
        return features

    img_bgr = cv2.resize(img_bgr, size)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    features = {}
    features.update(extract_lbp_features(gray, prefix))
    features.update(extract_glcm_features(gray, prefix))
    features.update(extract_dct_features(gray, prefix))
    features.update(extract_color_features(rgb, prefix))
    features.update(extract_edge_features(gray, prefix))
    features[f"{prefix}_missing"] = 0

    return features


def extract_features_for_frame(row: pd.Series) -> dict:
    """Extract features for full, face, and expanded regions of one frame."""
    features = {
        "label": row["label"],
        "video_name": row["video_name"],
        "frame_name": row["frame_name"],
        "face_detected": row["face_detected"],
    }

    features.update(extract_region_features(row.get("full_path", None), prefix="full"))
    features.update(extract_region_features(row.get("face_path", None), prefix="face"))
    features.update(
        extract_region_features(row.get("expanded_path", None), prefix="expanded")
    )

    return features


def extract_frame_level_features_from_log(
    log_csv: str | Path,
    output_csv: str | Path,
    limit: int | None = None,
) -> pd.DataFrame:
    """Extract handcrafted frame-level features from a multi-region log CSV."""
    log_df = pd.read_csv(log_csv)

    if limit is not None:
        log_df = log_df.head(limit)

    all_features = []

    for _, row in tqdm(
        log_df.iterrows(),
        total=len(log_df),
        desc="Extracting handcrafted features",
    ):
        all_features.append(extract_features_for_frame(row))

    feature_df = pd.DataFrame(all_features)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_csv, index=False)

    print(f"Saved frame-level features: {output_csv}")
    print(f"Shape: {feature_df.shape}")

    return feature_df
