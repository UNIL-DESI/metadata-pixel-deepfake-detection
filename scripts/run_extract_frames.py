from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path


CONFIG_FILE = Path("config/pilot.yaml")

with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)


default_catalog = config["catalog"]["file"]
default_output_dir = config["frames"]["output_dir"]
default_num_frames = config["frames"]["frames_per_video"]


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing.extract_frames import extract_frames_from_catalog


def main():

    parser = argparse.ArgumentParser()

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


    extract_frames_from_catalog(
        catalog_csv=args.catalog,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()