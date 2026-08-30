"""Image resizing helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def resize_image(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize an image to the selected model/feature extraction input size."""
    return cv2.resize(image, size)


def resize_and_save(
    image: np.ndarray,
    save_path: str | Path,
    size: tuple[int, int] = (224, 224),
) -> None:
    save_path = Path(save_path)
    image_resized = resize_image(image, size=size)
    cv2.imwrite(str(save_path), image_resized)
