"""Edge and boundary features."""

from __future__ import annotations

import cv2
import numpy as np


def extract_edge_features(gray, prefix: str) -> dict:
    """Extract Sobel magnitude and Canny edge-density features."""
    features = {}

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edges = cv2.Canny(gray, 100, 200)

    features[f"{prefix}_sobel_mean"] = float(np.mean(magnitude))
    features[f"{prefix}_sobel_std"] = float(np.std(magnitude))
    features[f"{prefix}_canny_edge_density"] = float(
        np.count_nonzero(edges) / edges.size
    )

    return features


def create_zero_edge_features(prefix: str) -> dict:
    return {
        f"{prefix}_sobel_mean": 0.0,
        f"{prefix}_sobel_std": 0.0,
        f"{prefix}_canny_edge_density": 0.0,
    }
