"""Local Binary Pattern texture features."""

from __future__ import annotations

import numpy as np
from skimage.feature import local_binary_pattern


def extract_lbp_features(gray, prefix: str, P: int = 8, R: int = 1) -> dict:
    """Extract uniform LBP histogram features from a grayscale image."""
    features = {}

    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    n_bins = P + 2

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_bins + 1),
        range=(0, n_bins),
        density=True,
    )

    for i, value in enumerate(hist):
        features[f"{prefix}_lbp_bin_{i}"] = float(value)

    return features


def create_zero_lbp_features(prefix: str, bins: int = 10) -> dict:
    return {f"{prefix}_lbp_bin_{i}": 0.0 for i in range(bins)}
