from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from features.extract_handcrafted import (
    extract_frame_level_features_from_log,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--config",
            default="pilot.yaml",
            help="Path to the configuration file (e.g., pilot.yaml or full.yaml)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override processing limit"
    )

    args = parser.parse_args()


    # Load YAML
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)


    handcrafted_config = config["features"]["handcrafted"]


    log_csv = handcrafted_config["input_log"]
    output_csv = handcrafted_config["frame_output"]

    limit = (
        args.limit
        if args.limit is not None
        else handcrafted_config.get("limit")
    )


    print("Input log:")
    print(log_csv)

    print("Output:")
    print(output_csv)

    print("Limit:")
    print(limit)


    extract_frame_level_features_from_log(
        log_csv=log_csv,
        output_csv=output_csv,
        limit=limit,
    )


if __name__ == "__main__":
    main()