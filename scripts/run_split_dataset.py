import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

INVENTORY_PATH = r"G:\My Drive\DeepfakeResearch\dataset_inventory.csv"

OUTPUT_DIR = r"G:\My Drive\DeepfakeResearch\features\splits"

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DATASET SPLITTING")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Load inventory
    # --------------------------------------------------------

    print("\nLoading dataset inventory...")
    print(INVENTORY_PATH)

    df = pd.read_csv(INVENTORY_PATH)

    print(f"\nTotal videos: {len(df)}")
    print("\nOriginal label distribution:")
    print(df["label"].value_counts())

    # --------------------------------------------------------
    # Standardize video ID
    # --------------------------------------------------------

    df["media_id"] = (
        df["video_name"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.mp4$", "", regex=True)
    )

    # --------------------------------------------------------
    # Check duplicate IDs
    # --------------------------------------------------------

    duplicates = df["media_id"].duplicated().sum()

    if duplicates > 0:
        raise ValueError(
            f"Found {duplicates} duplicate media IDs."
        )

    print(f"\nUnique media IDs: {df['media_id'].nunique()}")

    # --------------------------------------------------------
    # First split:
    # 70% Train
    # 30% Temporary
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_SIZE + TEST_SIZE),
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Second split:
    # Temporary 30%
    # → 15% Validation
    # → 15% Test
    #
    # Therefore:
    # test_size = 0.5
    # --------------------------------------------------------

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Reset indexes
    # --------------------------------------------------------

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # --------------------------------------------------------
    # Print split information
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Verify total
    # --------------------------------------------------------

    total = len(train_df) + len(val_df) + len(test_df)

    if total != len(df):
        raise ValueError(
            f"Split total mismatch: {total} != {len(df)}"
        )

    # --------------------------------------------------------
    # Verify no overlap
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save complete split files
    # --------------------------------------------------------

    train_path = os.path.join(
        OUTPUT_DIR,
        "train_inventory.csv"
    )

    val_path = os.path.join(
        OUTPUT_DIR,
        "validation_inventory.csv"
    )

    test_path = os.path.join(
        OUTPUT_DIR,
        "test_inventory.csv"
    )

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # --------------------------------------------------------
    # Save ID-only files
    # --------------------------------------------------------

    train_ids_path = os.path.join(
        OUTPUT_DIR,
        "train_ids.csv"
    )

    val_ids_path = os.path.join(
        OUTPUT_DIR,
        "validation_ids.csv"
    )

    test_ids_path = os.path.join(
        OUTPUT_DIR,
        "test_ids.csv"
    )

    train_df[["media_id"]].to_csv(
        train_ids_path,
        index=False
    )

    val_df[["media_id"]].to_csv(
        val_ids_path,
        index=False
    )

    test_df[["media_id"]].to_csv(
        test_ids_path,
        index=False
    )

    # --------------------------------------------------------
    # Save split summary
    # --------------------------------------------------------

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

    summary_path = os.path.join(
        OUTPUT_DIR,
        "split_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

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
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()