from pathlib import Path

import pandas as pd


def aggregate_handcrafted_features(
        frame_csv: str | Path
) -> pd.DataFrame:
    """
    Convert frame-level handcrafted features
    into video-level features.

    Aggregation:
        Mean + Standard deviation

    Input:
        handcrafted_frame_features.csv

    Output:
        One row per video
    """

    frame_csv = Path(frame_csv)

    print("Reading frame-level handcrafted features...")
    
    df = pd.read_csv(
        frame_csv,
        low_memory=False
    )


    # Identify feature columns
    exclude_columns = [
        "label",
        "video_name",
        "frame_name",
        "face_detected"
    ]


    feature_columns = [
        col for col in df.columns
        if col not in exclude_columns
    ]


    print(
        f"Feature columns found: {len(feature_columns)}"
    )


    # Aggregate by video
    aggregated = (
        df.groupby(
            [
                "label",
                "video_name"
            ]
        )[feature_columns]
        .agg(
            [
                "mean",
                "std"
            ]
        )
    )


    # Flatten multi-level column names

    aggregated.columns = [
        f"{feature}_{stat}"
        for feature, stat in aggregated.columns
    ]


    aggregated = (
        aggregated
        .reset_index()
    )


    # Replace NaN standard deviation
    # (happens when a video has only one frame)

    aggregated = aggregated.fillna(0)


    print(
        "Video-level handcrafted feature shape:",
        aggregated.shape
    )


    return aggregated