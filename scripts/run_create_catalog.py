from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path

# Add src to path so internal package imports work
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data.create_catalog import create_dataset_catalog


def main():
    # Step 1: Parse the --config argument first
    parser = argparse.ArgumentParser(description="Create dataset catalog")
    parser.add_argument(
        "--config",
        default="pilot.yaml",
        help="Path to the configuration file (e.g., pilot.yaml or full.yaml)"
    )
    
    # Parse only the config argument initially so we know which file to open
    args, _ = parser.parse_known_args()

    # Step 2: Load the specified configuration file
    config_path = Path(args.config)
    if not config_path.exists() and Path("config").joinpath(args.config).exists():
        config_path = Path("config") / args.config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    default_base_dir = config["dataset"]["base_dir"]
    default_output_csv = config["catalog"]["file"]

    # Step 3: Add the rest of the arguments using defaults from the loaded config
    parser.add_argument(
        "--base-dir",
        default=default_base_dir,
        help="Dataset root directory"
    )

    parser.add_argument(
        "--output-csv",
        default=default_output_csv,
        help="Path to output catalog CSV"
    )

    args = parser.parse_args()

    # Step 4: Run the function with the resolved paths
    create_dataset_catalog(
        args.base_dir,
        args.output_csv
    )


if __name__ == "__main__":
    main()