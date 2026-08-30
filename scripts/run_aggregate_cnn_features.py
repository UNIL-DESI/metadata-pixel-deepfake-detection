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

from features.aggregate_cnn_features import (
    aggregate_cnn_features
)


def main():

    # Load configuration
    with open(
        "config/pilot.yaml",
        "r"
    ) as f:

        config = yaml.safe_load(f)

    cnn_config = config["features"]["cnn_features"]

    model_name = cnn_config["model"]

    output_dir = Path(
        cnn_config["output_dir"]
    )

    frame_csv = (
        output_dir /
        f"{model_name}_frame_features.csv"
    )

    video_csv = (
        output_dir /
        f"{model_name}_video_features.csv"
    )

    print("=" * 60)
    print("CNN Video Feature Aggregation")
    print("=" * 60)

    print(f"Reading frame-level features:\n{frame_csv}")

    video_features = aggregate_cnn_features(
        frame_csv
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    video_features.to_csv(
        video_csv,
        index=False
    )

    print(f"\nSaved video-level features:\n{video_csv}")

    print("\nSummary")
    print("-" * 60)
    print(f"Videos processed : {len(video_features)}")
    print(f"Output columns   : {len(video_features.columns)}")

    print("\nCNN feature aggregation completed successfully.")


if __name__ == "__main__":
    main()