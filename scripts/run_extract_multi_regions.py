from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing.multi_region_extraction import (
    process_all_multi_regions,
)


def main():
    # Step 1: Parse the --config argument first
    parser = argparse.ArgumentParser(
        description="Extract full-frame, face, and expanded-face regions"
    )
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

    # Default values extracted from the loaded YAML
    default_frames_dir = config["regions"]["input_dir"]
    default_output_dir = config["regions"]["output_dir"]

    default_width = config["preprocessing"]["resize_width"]
    default_height = config["preprocessing"]["resize_height"]

    default_expanded_scale = config["preprocessing"]["expanded_face_scale"]

    # Step 3: Add the rest of the arguments using defaults from the loaded config
    parser.add_argument(
        "--frames-dir",
        default=default_frames_dir,
        help="Input extracted frames directory"
    )

    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        help="Output directory for multi-region frames"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=default_width,
        help="Output image width"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=default_height,
        help="Output image height"
    )

    parser.add_argument(
        "--expanded-scale",
        type=float,
        default=default_expanded_scale,
        help="Expansion scale for expanded face region"
    )

    parser.add_argument(
        "--video-limit",
        type=int,
        default=None,
        help="Maximum number of videos to process"
    )
    
    args = parser.parse_args()

    # Step 4: Run the core processing function
    process_all_multi_regions(
        input_frames_folder=args.frames_dir,
        output_regions_folder=args.output_dir,
        size=(args.width, args.height),
        expanded_scale=args.expanded_scale,
        video_limit=args.video_limit,
    )


if __name__ == "__main__":
    main()