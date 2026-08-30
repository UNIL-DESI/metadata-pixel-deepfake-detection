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


from features.aggregate_handcrafted_features import (
    aggregate_handcrafted_features
)



def main():

    with open(
        "config/pilot.yaml",
        "r"
    ) as f:

        config = yaml.safe_load(f)


    handcrafted_config = (
        config["features"]["handcrafted"]
    )


    frame_csv = Path(
        handcrafted_config["frame_output"]
    )


    video_csv = Path(
        handcrafted_config["video_output"]
    )


    print("=" * 60)
    print("Handcrafted Feature Aggregation")
    print("=" * 60)


    print(
        "Input:"
    )

    print(frame_csv)


    video_features = (
        aggregate_handcrafted_features(
            frame_csv
        )
    )


    video_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    video_features.to_csv(
        video_csv,
        index=False
    )


    print()
    print(
        "Saved video-level handcrafted features:"
    )

    print(video_csv)


    print()
    print("Summary")
    print("-" * 60)

    print(
        f"Videos processed : {len(video_features)}"
    )

    print(
        f"Output columns   : {len(video_features.columns)}"
    )


    print()
    print(
        "Handcrafted feature aggregation completed."
    )



if __name__ == "__main__":
    main()