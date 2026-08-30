"""Face detection and bounding-box utilities."""

from __future__ import annotations

import cv2
import numpy as np


def get_haar_face_detector() -> cv2.CascadeClassifier:
    """Create the OpenCV Haar Cascade face detector."""
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def expand_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    img_width: int,
    img_height: int,
    scale: float = 1.5,
) -> tuple[int, int, int, int]:
    """Expand a face bounding box while keeping it inside the image."""
    center_x = x + w // 2
    center_y = y + h // 2

    new_w = int(w * scale)
    new_h = int(h * scale)

    new_x = max(0, center_x - new_w // 2)
    new_y = max(0, center_y - new_h // 2)

    new_x2 = min(img_width, new_x + new_w)
    new_y2 = min(img_height, new_y + new_h)

    return new_x, new_y, new_x2, new_y2


def detect_largest_face(
    frame: np.ndarray,
    face_detector: cv2.CascadeClassifier | None = None,
    min_size: tuple[int, int] = (40, 40),
) -> tuple[int, int, int, int] | None:
    """Detect faces and return the largest face box."""
    if face_detector is None:
        face_detector = get_haar_face_detector()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=min_size,
    )

    if len(faces) == 0:
        return None

    return max(faces, key=lambda box: box[2] * box[3])
