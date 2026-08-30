from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from features.metadata_extractor import (
    extract_metadata_from_catalog
)


# Dataset catalog
CATALOG_PATH = Path(
    "G:/My Drive/DeepfakeResearch/dataset_inventory.csv"
)

# Metadata feature output
OUTPUT_PATH = Path(
    "G:/My Drive/DeepfakeResearch/features/metadata/metadata_features.csv"
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata features from dataset."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of videos to process."
    )

    args = parser.parse_args()

    extract_metadata_from_catalog(
        catalog_csv=CATALOG_PATH,
        output_csv=OUTPUT_PATH,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()