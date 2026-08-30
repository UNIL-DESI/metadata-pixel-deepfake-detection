from pathlib import Path

import pandas as pd


def aggregate_cnn_features(
        frame_csv: str | Path
) -> pd.DataFrame:
    """
    Convert frame-level CNN features into
    video-level CNN features.

    Aggregation:
        Mean + Standard deviation

    Input:
        mobilenetv2_frame_features.csv

    Output:
        One row per video
    """

    frame_csv = Path(frame_csv)

    print(
        "Reading frame-level CNN features..."
    )

    df = pd.read_csv(
        frame_csv,
        low_memory=False,
        dtype={
            "label": str,
            "video_id": str,
            "region": str,
            "frame": str
        }
    )


    feature_columns = [
        col for col in df.columns
        if col.startswith("cnn_")
    ]


    print(
        f"CNN feature columns found: {len(feature_columns)}"
    )


    # Aggregate frames by video and region

    aggregated = (
        df.groupby(
            [
                "label",
                "video_id",
                "region"
            ]
        )[feature_columns]
        .agg(
            [
                "mean",
                "std"
            ]
        )
    )


    # Flatten column names

    aggregated.columns = [
        f"{feature}_{stat}"
        for feature, stat in aggregated.columns
    ]


    aggregated = (
        aggregated
        .reset_index()
    )


    aggregated = aggregated.fillna(0)



  
    # Convert regions into columns
    id_columns = [
        "label",
        "video_id"
    ]


    feature_columns = [
        col for col in aggregated.columns
        if col not in id_columns + ["region"]
    ]


    region_wide = (
        aggregated
        .pivot_table(
            index=id_columns,
            columns="region",
            values=feature_columns,
            fill_value=0
        )
    )


    # Flatten multi-index columns

    region_wide.columns = [
        f"{region}_{feature}"
        for feature, region in region_wide.columns
    ]


    region_wide = (
        region_wide
        .reset_index()
    )


    print(
        "Video-level CNN feature shape:",
        region_wide.shape
    )


    return region_wide