from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train, validation, and test sets")
    parser.add_argument(
        "--config",
        default="pilot.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists() and Path("config").joinpath(args.config).exists():
        config_path = Path("config") / args.config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    
    inventory_path = config["catalog"]["file"]
    output_dir = "data/splits"

    random_state = 42
    val_size = 0.15
    test_size = 0.15

    print("=" * 70)
    print("DATASET SPLITTING")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading dataset inventory...")
    print(inventory_path)

    df = pd.read_csv(inventory_path)

    print(f"\nTotal videos: {len(df)}")
    print("\nOriginal label distribution:")
    print(df["label"].value_counts())

    df["media_id"] = (
        df["video_name"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.mp4$", "", regex=True)
    )

    duplicates = df["media_id"].duplicated().sum()

    if duplicates > 0:
        raise ValueError(
            f"Found {duplicates} duplicate media IDs."
        )

    print(f"\nUnique media IDs: {df['media_id'].nunique()}")

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=df["label"],
        random_state=random_state
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=random_state
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("\n" + "=" * 70)
    print("SPLIT RESULTS")
    print("=" * 70)

    print(f"\nTrain:      {len(train_df)} videos")
    print(f"Validation: {len(val_df)} videos")
    print(f"Test:       {len(test_df)} videos")

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts())

    print("\nValidation label distribution:")
    print(val_df["label"].value_counts())

    print("\nTest label distribution:")
    print(test_df["label"].value_counts())

    total = len(train_df) + len(val_df) + len(test_df)

    if total != len(df):
        raise ValueError(
            f"Split total mismatch: {total} != {len(df)}"
        )

    train_ids = set(train_df["media_id"])
    val_ids = set(val_df["media_id"])
    test_ids = set(test_df["media_id"])

    if train_ids & val_ids:
        raise ValueError("Train/Validation overlap detected.")

    if train_ids & test_ids:
        raise ValueError("Train/Test overlap detected.")

    if val_ids & test_ids:
        raise ValueError("Validation/Test overlap detected.")

    print("\n✓ No ID overlap between splits")

    train_path = os.path.join(output_dir, "train_inventory.csv")
    val_path = os.path.join(output_dir, "validation_inventory.csv")
    test_path = os.path.join(output_dir, "test_inventory.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    train_ids_path = os.path.join(output_dir, "train_ids.csv")
    val_ids_path = os.path.join(output_dir, "validation_ids.csv")
    test_ids_path = os.path.join(output_dir, "test_ids.csv")

    train_df[["media_id"]].to_csv(train_ids_path, index=False)
    val_df[["media_id"]].to_csv(val_ids_path, index=False)
    test_df[["media_id"]].to_csv(test_ids_path, index=False)

    summary = pd.DataFrame({
        "split": ["train", "validation", "test"],
        "percentage": [70, 15, 15],
        "videos": [
            len(train_df),
            len(val_df),
            len(test_df)
        ],
        "real": [
            (train_df["label"] == "real").sum(),
            (val_df["label"] == "real").sum(),
            (test_df["label"] == "real").sum()
        ],
        "fake": [
            (train_df["label"] == "fake").sum(),
            (val_df["label"] == "fake").sum(),
            (test_df["label"] == "fake").sum()
        ]
    })

    summary_path = os.path.join(output_dir, "split_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("DATASET SPLITTING COMPLETED")
    print("=" * 70)

    print("\nFiles created:")
    print(train_path)
    print(val_path)
    print(test_path)
    print(train_ids_path)
    print(val_ids_path)
    print(test_ids_path)
    print(summary_path)

    print("\nSplit directory:")
    print(output_dir)


if __name__ == "__main__":
    main()