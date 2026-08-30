"""Gray-Level Co-occurrence Matrix texture features."""

from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops


GLCM_PROPS = [
    "contrast",
    "correlation",
    "energy",
    "homogeneity",
    "dissimilarity",
]


def extract_glcm_features(gray, prefix: str, levels: int = 64) -> dict:
    """Extract GLCM statistical features from a grayscale image."""
    features = {}

    gray_q = np.floor(gray / 256 * levels).astype(np.uint8)
    gray_q[gray_q >= levels] = levels - 1

    distances = [1, 2]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm = graycomatrix(
        gray_q,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    for prop in GLCM_PROPS:
        values = graycoprops(glcm, prop)
        features[f"{prefix}_glcm_{prop}_mean"] = float(np.mean(values))
        features[f"{prefix}_glcm_{prop}_std"] = float(np.std(values))

    eps = 1e-12
    glcm_entropy = -np.sum(glcm * np.log2(glcm + eps), axis=(0, 1))

    features[f"{prefix}_glcm_entropy_mean"] = float(np.mean(glcm_entropy))
    features[f"{prefix}_glcm_entropy_std"] = float(np.std(glcm_entropy))

    return features


def create_zero_glcm_features(prefix: str) -> dict:
    features = {}

    for name in GLCM_PROPS:
        features[f"{prefix}_glcm_{name}_mean"] = 0.0
        features[f"{prefix}_glcm_{name}_std"] = 0.0

    features[f"{prefix}_glcm_entropy_mean"] = 0.0
    features[f"{prefix}_glcm_entropy_std"] = 0.0

    return features
