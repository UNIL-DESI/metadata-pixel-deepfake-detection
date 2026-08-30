from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path


# Load configuration
CONFIG_FILE = Path("config/pilot.yaml")

with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)


# Default values from YAML
default_frames_dir = config["regions"]["input_dir"]
default_output_dir = config["regions"]["output_dir"]

default_width = config["preprocessing"]["resize_width"]
default_height = config["preprocessing"]["resize_height"]

default_expanded_scale = config["preprocessing"]["expanded_face_scale"]


# Add src directory to Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


from preprocessing.multi_region_extraction import (
    process_all_multi_regions,
)


def main():

    parser = argparse.ArgumentParser(
        description="Extract full-frame, face, and expanded-face regions"
    )

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


    process_all_multi_regions(
        input_frames_folder=args.frames_dir,
        output_regions_folder=args.output_dir,
        size=(args.width, args.height),
        expanded_scale=args.expanded_scale,
        video_limit=args.video_limit,
    )


if __name__ == "__main__":
    main()