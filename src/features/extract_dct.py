"""DCT frequency-domain features."""

from __future__ import annotations

import numpy as np
from scipy.fftpack import dct
from scipy.stats import entropy as scipy_entropy


DCT_FEATURE_NAMES = [
    "low_energy",
    "mid_energy",
    "high_energy",
    "total_energy",
    "high_low_ratio",
    "mid_low_ratio",
    "ac_mean_abs",
    "ac_std",
    "ac_var",
    "ac_entropy",
    "dc_mean",
    "dc_std",
    "dc_var",
    "high_freq_sparsity",
    "nonzero_ratio",
    "low_energy_block_std",
    "mid_energy_block_std",
    "high_energy_block_std",
]


def block_dct_8x8(gray_float) -> np.ndarray:
    """Apply 8x8 block DCT to a grayscale float image in range 0-1."""
    height, width = gray_float.shape

    h8 = (height // 8) * 8
    w8 = (width // 8) * 8

    gray_float = gray_float[:h8, :w8]

    blocks = []

    for y in range(0, h8, 8):
        for x in range(0, w8, 8):
            block = gray_float[y : y + 8, x : x + 8]
            dct_block = dct(dct(block.T, norm="ortho").T, norm="ortho")
            blocks.append(dct_block)

    return np.array(blocks)


def extract_dct_features(gray, prefix: str) -> dict:
    """Extract DCT energy and coefficient statistics."""
    features = {}
    gray_float = gray.astype(np.float32) / 255.0
    dct_blocks = block_dct_8x8(gray_float)

    if len(dct_blocks) == 0:
        return create_zero_dct_features(prefix)

    idx = np.add.outer(np.arange(8), np.arange(8))

    dc_mask = idx == 0
    low_mask = (idx > 0) & (idx <= 2)
    mid_mask = (idx > 2) & (idx <= 5)
    high_mask = idx > 5

    dc_coeffs = dct_blocks[:, dc_mask].ravel()
    low_coeffs = dct_blocks[:, low_mask].ravel()
    mid_coeffs = dct_blocks[:, mid_mask].ravel()
    high_coeffs = dct_blocks[:, high_mask].ravel()
    ac_coeffs = dct_blocks[:, ~dc_mask].ravel()

    eps = 1e-8

    low_energy = np.sum(low_coeffs**2)
    mid_energy = np.sum(mid_coeffs**2)
    high_energy = np.sum(high_coeffs**2)
    total_energy = low_energy + mid_energy + high_energy

    features[f"{prefix}_dct_low_energy"] = float(low_energy)
    features[f"{prefix}_dct_mid_energy"] = float(mid_energy)
    features[f"{prefix}_dct_high_energy"] = float(high_energy)
    features[f"{prefix}_dct_total_energy"] = float(total_energy)

    features[f"{prefix}_dct_high_low_ratio"] = float(high_energy / (low_energy + eps))
    features[f"{prefix}_dct_mid_low_ratio"] = float(mid_energy / (low_energy + eps))

    features[f"{prefix}_dct_ac_mean_abs"] = float(np.mean(np.abs(ac_coeffs)))
    features[f"{prefix}_dct_ac_std"] = float(np.std(ac_coeffs))
    features[f"{prefix}_dct_ac_var"] = float(np.var(ac_coeffs))

    hist, _ = np.histogram(np.abs(ac_coeffs), bins=32, density=True)
    features[f"{prefix}_dct_ac_entropy"] = float(scipy_entropy(hist + eps))

    features[f"{prefix}_dct_dc_mean"] = float(np.mean(dc_coeffs))
    features[f"{prefix}_dct_dc_std"] = float(np.std(dc_coeffs))
    features[f"{prefix}_dct_dc_var"] = float(np.var(dc_coeffs))

    threshold = 1e-3

    features[f"{prefix}_dct_high_freq_sparsity"] = float(
        np.mean(np.abs(high_coeffs) < threshold)
    )
    features[f"{prefix}_dct_nonzero_ratio"] = float(
        np.mean(np.abs(ac_coeffs) > threshold)
    )

    block_low_energy = np.sum(dct_blocks[:, low_mask] ** 2, axis=1)
    block_mid_energy = np.sum(dct_blocks[:, mid_mask] ** 2, axis=1)
    block_high_energy = np.sum(dct_blocks[:, high_mask] ** 2, axis=1)

    features[f"{prefix}_dct_low_energy_block_std"] = float(np.std(block_low_energy))
    features[f"{prefix}_dct_mid_energy_block_std"] = float(np.std(block_mid_energy))
    features[f"{prefix}_dct_high_energy_block_std"] = float(np.std(block_high_energy))

    return features


def create_zero_dct_features(prefix: str) -> dict:
    return {f"{prefix}_dct_{name}": 0.0 for name in DCT_FEATURE_NAMES}
