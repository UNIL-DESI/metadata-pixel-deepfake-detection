from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path


CONFIG_FILE = Path("config/pilot.yaml")

with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

default_base_dir = config["dataset"]["base_dir"]


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data.create_catalog import create_dataset_catalog


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-dir",
        default=default_base_dir,
        help="Dataset root directory"
    )

    parser.add_argument(
        "--output-csv",
        default=None
    )

    args = parser.parse_args()

    create_dataset_catalog(
        args.base_dir,
        args.output_csv
    )


if __name__ == "__main__":
    main()
