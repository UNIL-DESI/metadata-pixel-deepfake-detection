from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from deepfake_detection.models.train_fusion import train_random_forest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    train_random_forest(
        features_csv=args.features_csv,
        model_output=args.model_output,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
