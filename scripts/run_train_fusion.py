from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from deepfake_detection.models.train_fusion import train_random_forest


def main():
    parser = argparse.ArgumentParser(description="Train random forest model")
    parser.add_argument(
        "--config",
        default="pilot.yaml",
        help="Path to YAML configuration file"
    )
    
    args, _ = parser.parse_known_args()

    config_path = Path(args.config)
    if not config_path.exists() and Path("config").joinpath(args.config).exists():
        config_path = Path("config") / args.config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_dir = config["dataset"]["base_dir"]
    default_features_csv = str(Path(base_dir) / "features" / "selected" / "train_selected_160.csv")
    default_model_output = str(Path(base_dir) / "models" / "pilot" / "random_forest.joblib")

    parser.add_argument(
        "--features-csv",
        default=default_features_csv,
        help="Path to training features CSV"
    )
    parser.add_argument(
        "--model-output",
        default=default_model_output,
        help="Path to save trained model output"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size"
    )
    
    args = parser.parse_args()

    train_random_forest(
        features_csv=args.features_csv,
        model_output=args.model_output,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()