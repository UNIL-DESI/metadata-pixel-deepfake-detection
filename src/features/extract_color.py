"""RGB color statistics."""

from __future__ import annotations

import numpy as np


def extract_color_features(rgb, prefix: str) -> dict:
    """Extract mean, std, and variance for RGB channels."""
    features = {}
    rgb_float = rgb.astype(np.float32) / 255.0

    for i, channel in enumerate(["r", "g", "b"]):
        channel_values = rgb_float[:, :, i].ravel()
        features[f"{prefix}_color_{channel}_mean"] = float(np.mean(channel_values))
        features[f"{prefix}_color_{channel}_std"] = float(np.std(channel_values))
        features[f"{prefix}_color_{channel}_var"] = float(np.var(channel_values))

    return features


def create_zero_color_features(prefix: str) -> dict:
    features = {}

    for channel in ["r", "g", "b"]:
        features[f"{prefix}_color_{channel}_mean"] = 0.0
        features[f"{prefix}_color_{channel}_std"] = 0.0
        features[f"{prefix}_color_{channel}_var"] = 0.0

    return features
