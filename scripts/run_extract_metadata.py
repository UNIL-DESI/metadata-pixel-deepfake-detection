from __future__ import annotations

import static_ffmpeg
static_ffmpeg.add_paths()  # This magically makes "ffprobe" available to the system

import subprocess
import argparse
import sys
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from features.metadata_extractor import (
    extract_metadata_from_catalog
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata features from dataset."
    )

    parser.add_argument(
        "--config",
        default="pilot.yaml",
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of videos to process."
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists() and Path("config").joinpath(args.config).exists():
        config_path = Path("config") / args.config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    metadata_config = config["features"]["metadata"]

    catalog_path = Path(metadata_config["catalog"])
    output_path = Path(metadata_config["output"])

    extract_metadata_from_catalog(
        catalog_csv=catalog_path,
        output_csv=output_path,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()