from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def standardize_id(value):
    return str(value).strip()


def check_unique(df, name):
    duplicates = df["media_id"].duplicated().sum()

    print(f"{name}: {len(df)} rows")
    print(f"{name}: {df['media_id'].nunique()} unique IDs")

    if duplicates > 0:
        raise ValueError(
            f"{name} contains {duplicates} duplicate media_id values."
        )


def main():
    parser = argparse.ArgumentParser(description="Align feature sets")
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

    handcrafted_path = "features/handcrafted/handcrafted_video_features.csv"
    
    cnn_model = config["features"]["cnn_features"]["model"]
    cnn_path = f"features/CNNFeatures/{cnn_model}_video_features.csv"
    
    metadata_path = "features/metadata/metadata_preprocessed/metadata_feature_matrix.csv"
    
    inventory_path = config["catalog"]["file"]
    
    output_dir = "features/aligned"

    print("=" * 70)
    print("FEATURE ALIGNMENT")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    print("\nChecking input files...")

    for path, name in [
        (handcrafted_path, "Handcrafted"),
        (cnn_path, "CNN"),
        (metadata_path, "Metadata"),
        (inventory_path, "Dataset Inventory"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} file not found:\n{path}"
            )

        print(f"✓ {name}: {path}")

    print("\nLoading feature files...")

    handcrafted = pd.read_csv(handcrafted_path)
    cnn = pd.read_csv(cnn_path)
    metadata = pd.read_csv(metadata_path)
    inventory = pd.read_csv(inventory_path)

    print(f"Handcrafted shape: {handcrafted.shape}")
    print(f"CNN shape:         {cnn.shape}")
    print(f"Metadata shape:    {metadata.shape}")
    print(f"Inventory shape:   {inventory.shape}")

    print("\nStandardizing identifier columns...")

    if "video_name" not in handcrafted.columns:
        raise ValueError(
            "Handcrafted file does not contain 'video_name'."
        )

    handcrafted["media_id"] = (
        handcrafted["video_name"]
        .astype(str)
        .str.strip()
    )

    if "video_id" not in cnn.columns:
        raise ValueError(
            "CNN file does not contain 'video_id'."
        )

    cnn["media_id"] = (
        cnn["video_id"]
        .astype(str)
        .str.strip()
    )

    if "media_id" not in metadata.columns:
        raise ValueError(
            "Metadata file does not contain 'media_id'."
        )

    metadata["media_id"] = (
        metadata["media_id"]
        .astype(str)
        .str.strip()
    )

    if "video_name" not in inventory.columns:
        raise ValueError(
            "Dataset inventory does not contain 'video_name'."
        )

    if "label" not in inventory.columns:
        raise ValueError(
            "Dataset inventory does not contain 'label'."
        )

    inventory["media_id"] = (
        inventory["video_name"]
        .astype(str)
        .str.strip()
    )

    inventory["label"] = (
        inventory["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    print("\nChecking feature IDs...")

    check_unique(handcrafted, "Handcrafted")
    check_unique(cnn, "CNN")
    check_unique(metadata, "Metadata")
    check_unique(inventory, "Inventory")

    hc_ids = set(handcrafted["media_id"])
    cnn_ids = set(cnn["media_id"])
    meta_ids = set(metadata["media_id"])
    inventory_ids = set(inventory["media_id"])

    print("\nID comparison:")
    print("Handcrafted/CNN same:", hc_ids == cnn_ids)
    print("Handcrafted/Metadata same:", hc_ids == meta_ids)
    print("Handcrafted/Inventory same:", hc_ids == inventory_ids)

    if not (
        hc_ids == cnn_ids
        and hc_ids == meta_ids
        and hc_ids == inventory_ids
    ):
        print("\nID differences detected.")
        raise ValueError(
            "The feature files do not contain exactly the same IDs."
        )

    print("✓ All IDs match.")

    print("\nChecking authoritative labels...")
    valid_labels = {"real", "fake"}
    inventory_labels = set(inventory["label"])

    if not inventory_labels.issubset(valid_labels):
        raise ValueError(
            f"Unexpected labels found: {inventory_labels}"
        )

    print(
        inventory["label"]
        .value_counts()
        .to_string()
    )

    labels = inventory[["media_id", "label"]].copy()

    handcrafted = handcrafted.sort_values("media_id").reset_index(drop=True)
    cnn = cnn.sort_values("media_id").reset_index(drop=True)
    metadata = metadata.sort_values("media_id").reset_index(drop=True)
    labels = labels.sort_values("media_id").reset_index(drop=True)

    if not (
        handcrafted["media_id"].equals(labels["media_id"])
        and cnn["media_id"].equals(labels["media_id"])
        and metadata["media_id"].equals(labels["media_id"])
    ):
        raise ValueError(
            "Feature files are not aligned after sorting."
        )

    print("\n✓ Feature ordering verified.")

    handcrafted["label"] = labels["label"]
    cnn["label"] = labels["label"]
    metadata["label"] = labels["label"]

    hc_output = os.path.join(output_dir, "aligned_handcrafted.csv")
    cnn_output = os.path.join(output_dir, "aligned_cnn.csv")
    metadata_output = os.path.join(output_dir, "aligned_metadata.csv")

    handcrafted.to_csv(hc_output, index=False)
    cnn.to_csv(cnn_output, index=False)
    metadata.to_csv(metadata_output, index=False)

    print("\nCreating combined feature matrix...")

    hc_features = handcrafted.drop(
        columns=["video_name", "label"],
        errors="ignore"
    ).copy()

    cnn_features = cnn.drop(
        columns=["video_id", "label"],
        errors="ignore"
    ).copy()

    metadata_features = metadata.drop(
        columns=["label"],
        errors="ignore"
    ).copy()

    hc_features = hc_features.set_index("media_id")
    cnn_features = cnn_features.set_index("media_id")
    metadata_features = metadata_features.set_index("media_id")
    labels_indexed = labels.set_index("media_id")

    hc_features = hc_features.add_prefix("hc_")
    cnn_features = cnn_features.add_prefix("cnn_")
    metadata_features = metadata_features.add_prefix("meta_")

    combined = pd.concat(
        [
            labels_indexed,
            hc_features,
            cnn_features,
            metadata_features,
        ],
        axis=1
    )

    combined = combined.reset_index()

    print("\nFinal combined dataset:")
    print(f"Rows:    {combined.shape[0]}")
    print(f"Columns: {combined.shape[1]}")

    print("\nLabel distribution:")
    print(
        combined["label"]
        .value_counts()
        .to_string()
    )

    print("\nMissing values:")
    missing = combined.isna().sum().sum()
    print(f"Total missing values: {missing}")

    combined_output = os.path.join(
        output_dir,
        "aligned_all_features.csv"
    )
    combined.to_csv(combined_output, index=False)

    report_output = os.path.join(
        output_dir,
        "alignment_report.txt"
    )

    with open(
        report_output,
        "w",
        encoding="utf-8"
    ) as f:
        f.write("FEATURE ALIGNMENT REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Samples: {combined.shape[0]}\n")
        f.write(f"Total columns: {combined.shape[1]}\n\n")
        f.write("Label distribution:\n")
        f.write(
            combined["label"]
            .value_counts()
            .to_string()
        )
        f.write("\n\n")
        f.write(f"Missing values: {missing}\n")

    print("\n" + "=" * 70)
    print("FEATURE ALIGNMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()