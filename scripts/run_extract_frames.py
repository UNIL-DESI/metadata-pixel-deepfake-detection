from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path

# Add src to path so internal package imports work
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing.extract_frames import extract_frames_from_catalog


def main():
    # Step 1: Parse the --config argument first
    parser = argparse.ArgumentParser(description="Extract frames from video catalog")
    parser.add_argument(
        "--config",
        default="pilot.yaml",
        help="Path to the configuration file (e.g., pilot.yaml or full.yaml)"
    )
    
    # Parse only the config argument initially
    args, _ = parser.parse_known_args()

    # Step 2: Load the specified configuration file
    config_path = Path(args.config)
    if not config_path.exists() and Path("config").joinpath(args.config).exists():
        config_path = Path("config") / args.config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    default_catalog = config["catalog"]["file"]
    default_output_dir = config["frames"]["output_dir"]
    default_num_frames = config["frames"]["frames_per_video"]

    # Step 3: Add the rest of the arguments using defaults from the loaded config
    parser.add_argument(
        "--catalog",
        default=default_catalog,
        help="Dataset catalog CSV file"
    )

    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        help="Directory to save extracted frames"
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=default_num_frames,
        help="Number of frames extracted per video"
    )

    args = parser.parse_args()

    # Step 4: Run the function with resolved arguments
    extract_frames_from_catalog(
        catalog_csv=args.catalog,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()