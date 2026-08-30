"""Common helpers for feature extraction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def is_valid_path(path) -> bool:
    """Return True if a path exists and is usable."""
    if path is None:
        return False
    if pd.isna(path):
        return False
    if not isinstance(path, str):
        return False
    if not Path(path).exists():
        return False
    return True
