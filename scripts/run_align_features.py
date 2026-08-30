import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

HANDCRAFTED_PATH = (
    r"G:\My Drive\DeepfakeResearch\features\handcrafted"
    r"\handcrafted_video_features.csv"
)

CNN_PATH = (
    r"G:\My Drive\DeepfakeResearch\features\CNNFeatures"
    r"\mobilenetv2_video_features.csv"
)

METADATA_PATH = (
    r"G:\My Drive\DeepfakeResearch\features\metadata"
    r"\metadata_preprocessed\metadata_feature_matrix.csv"
)

INVENTORY_PATH = (
    r"G:\My Drive\DeepfakeResearch\dataset_inventory.csv"
)

OUTPUT_DIR = (
    r"G:\My Drive\DeepfakeResearch\features\aligned"
)


# ============================================================
# HELPERS
# ============================================================

def standardize_id(value):
    """
    Convert ID to a clean string.
    """
    return str(value).strip()


def check_unique(df, name):
    duplicates = df["media_id"].duplicated().sum()

    print(f"{name}: {len(df)} rows")
    print(f"{name}: {df['media_id'].nunique()} unique IDs")

    if duplicates > 0:
        raise ValueError(
            f"{name} contains {duplicates} duplicate media_id values."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FEATURE ALIGNMENT")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # 1. Check files
    # --------------------------------------------------------

    print("\nChecking input files...")

    for path, name in [
        (HANDCRAFTED_PATH, "Handcrafted"),
        (CNN_PATH, "CNN"),
        (METADATA_PATH, "Metadata"),
        (INVENTORY_PATH, "Dataset Inventory"),
    ]:

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} file not found:\n{path}"
            )

        print(f"✓ {name}: {path}")

    # --------------------------------------------------------
    # 2. Load files
    # --------------------------------------------------------

    print("\nLoading feature files...")

    handcrafted = pd.read_csv(HANDCRAFTED_PATH)
    cnn = pd.read_csv(CNN_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    inventory = pd.read_csv(INVENTORY_PATH)

    print(f"Handcrafted shape: {handcrafted.shape}")
    print(f"CNN shape:         {cnn.shape}")
    print(f"Metadata shape:    {metadata.shape}")
    print(f"Inventory shape:   {inventory.shape}")

    # --------------------------------------------------------
    # 3. Standardize identifier columns
    # --------------------------------------------------------

    print("\nStandardizing identifier columns...")

    # Handcrafted
    if "video_name" not in handcrafted.columns:
        raise ValueError(
            "Handcrafted file does not contain 'video_name'."
        )

    handcrafted["media_id"] = (
        handcrafted["video_name"]
        .astype(str)
        .str.strip()
    )

    # CNN
    if "video_id" not in cnn.columns:
        raise ValueError(
            "CNN file does not contain 'video_id'."
        )

    cnn["media_id"] = (
        cnn["video_id"]
        .astype(str)
        .str.strip()
    )

    # Metadata
    if "media_id" not in metadata.columns:
        raise ValueError(
            "Metadata file does not contain 'media_id'."
        )

    metadata["media_id"] = (
        metadata["media_id"]
        .astype(str)
        .str.strip()
    )

    # Inventory
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

    # --------------------------------------------------------
    # 4. Check IDs
    # --------------------------------------------------------

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

    print(
        "Handcrafted/CNN same:",
        hc_ids == cnn_ids
    )

    print(
        "Handcrafted/Metadata same:",
        hc_ids == meta_ids
    )

    print(
        "Handcrafted/Inventory same:",
        hc_ids == inventory_ids
    )

    if not (
        hc_ids == cnn_ids
        and hc_ids == meta_ids
        and hc_ids == inventory_ids
    ):
        print("\nID differences detected.")

        print(
            "HC missing:",
            sorted(hc_ids - cnn_ids)[:10]
        )

        print(
            "CNN missing:",
            sorted(cnn_ids - hc_ids)[:10]
        )

        print(
            "Metadata missing:",
            sorted(meta_ids - hc_ids)[:10]
        )

        print(
            "Inventory missing:",
            sorted(inventory_ids - hc_ids)[:10]
        )

        raise ValueError(
            "The feature files do not contain exactly the same IDs."
        )

    print("✓ All 300 IDs match.")

    # --------------------------------------------------------
    # 5. Check inventory labels
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 6. Create label table
    # --------------------------------------------------------

    labels = inventory[["media_id", "label"]].copy()

    # --------------------------------------------------------
    # 7. Sort all datasets by media_id
    # --------------------------------------------------------

    handcrafted = handcrafted.sort_values(
        "media_id"
    ).reset_index(drop=True)

    cnn = cnn.sort_values(
        "media_id"
    ).reset_index(drop=True)

    metadata = metadata.sort_values(
        "media_id"
    ).reset_index(drop=True)

    labels = labels.sort_values(
        "media_id"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 8. Verify exact ordering
    # --------------------------------------------------------

    if not (
        handcrafted["media_id"].equals(labels["media_id"])
        and cnn["media_id"].equals(labels["media_id"])
        and metadata["media_id"].equals(labels["media_id"])
    ):
        raise ValueError(
            "Feature files are not aligned after sorting."
        )

    print("\n✓ Feature ordering verified.")

    # --------------------------------------------------------
    # 9. Add authoritative labels
    # --------------------------------------------------------

    handcrafted["label"] = labels["label"]
    cnn["label"] = labels["label"]
    metadata["label"] = labels["label"]

    # --------------------------------------------------------
    # 10. Save aligned individual datasets
    # --------------------------------------------------------

    hc_output = os.path.join(
        OUTPUT_DIR,
        "aligned_handcrafted.csv"
    )

    cnn_output = os.path.join(
        OUTPUT_DIR,
        "aligned_cnn.csv"
    )

    metadata_output = os.path.join(
        OUTPUT_DIR,
        "aligned_metadata.csv"
    )

    handcrafted.to_csv(
        hc_output,
        index=False
    )

    cnn.to_csv(
        cnn_output,
        index=False
    )

    metadata.to_csv(
        metadata_output,
        index=False
    )

    # --------------------------------------------------------
    # 11. Create combined feature matrix
    # --------------------------------------------------------

    print("\nCreating combined feature matrix...")

    # Keep media_id temporarily so that all feature sets
    # can be aligned using the same identifier.

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


    # Use media_id as the common index.
    hc_features = hc_features.set_index("media_id")
    cnn_features = cnn_features.set_index("media_id")
    metadata_features = metadata_features.set_index("media_id")

    labels_indexed = labels.set_index("media_id")


    # Prefix feature names so we know which feature
    # source they came from.

    hc_features = hc_features.add_prefix("hc_")
    cnn_features = cnn_features.add_prefix("cnn_")
    metadata_features = metadata_features.add_prefix("meta_")


    # Combine all feature groups.
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

    # --------------------------------------------------------
    # 12. Final validation
    # --------------------------------------------------------

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

    if missing > 0:
        print(
            "\nWARNING: Missing values detected."
        )

    # --------------------------------------------------------
    # 13. Save combined dataset
    # --------------------------------------------------------

    combined_output = os.path.join(
        OUTPUT_DIR,
        "aligned_all_features.csv"
    )

    combined.to_csv(
        combined_output,
        index=False
    )

    # --------------------------------------------------------
    # 14. Save report
    # --------------------------------------------------------

    report_output = os.path.join(
        OUTPUT_DIR,
        "alignment_report.txt"
    )

    with open(
        report_output,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("FEATURE ALIGNMENT REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(
            f"Samples: {combined.shape[0]}\n"
        )

        f.write(
            f"Total columns: {combined.shape[1]}\n\n"
        )

        f.write(
            "Label distribution:\n"
        )

        f.write(
            combined["label"]
            .value_counts()
            .to_string()
        )

        f.write("\n\n")

        f.write(
            f"Missing values: {missing}\n"
        )

        f.write("\nInput shapes:\n")
        f.write(
            f"Handcrafted: {handcrafted.shape}\n"
        )
        f.write(
            f"CNN: {cnn.shape}\n"
        )
        f.write(
            f"Metadata: {metadata.shape}\n"
        )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE ALIGNMENT COMPLETED")
    print("=" * 70)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nFiles created:")
    print(hc_output)
    print(cnn_output)
    print(metadata_output)
    print(combined_output)
    print(report_output)


if __name__ == "__main__":
    main()