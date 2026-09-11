from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

sys.path.append(
    str(
        Path(__file__)
        .resolve()
        .parents[1]
        / "src"
    )
)

from features.cnn_models import get_cnn_model
from features.cnn_feature_extractor import extract_features


def collect_images(folder):
    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png"
    ]

    images = []

    for ext in extensions:
        images.extend(
            Path(folder).rglob(ext)
        )

    return sorted(images)


def main():
    parser = argparse.ArgumentParser(description="Extract CNN features from images")
    parser.add_argument(
        "--config",
        default="config/pilot.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cnn_config = config["features"]["cnn_features"]

    model_name = cnn_config["model"]

    print(
        f"Loading {model_name}"
    )

    model = get_cnn_model(
        model_name
    )

    input_dir = Path(
        cnn_config["input_dir"]
    )

    output_dir = Path(
        cnn_config["output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    images = collect_images(
        input_dir
    )

    print(
        f"Images found: {len(images)}"
    )

    frame_features = extract_features(
        model=model,
        image_paths=images,
        model_name=model_name,
        image_size=cnn_config["input_size"]
    )

    frame_csv = (
        output_dir /
        f"{model_name}_frame_features.csv"
    )

    frame_features.to_csv(
        frame_csv,
        index=False
    )

    print(
        f"\nFrame-level CNN features saved to:\n{frame_csv}"
    )

    print(
        "CNN feature extraction completed."
    )


if __name__ == "__main__":
    main()